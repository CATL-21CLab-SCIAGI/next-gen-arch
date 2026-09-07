"""Single-GPU, cache-free sampling of the qualified pretrained additive model.

Reuse AutoModel's checkpoint adapter and single-owner PLE implementation. Only
the lookup table executes on CPU (Accelerate's device hook); no quantization,
attention replacement, distributed training state, or installed-source edits.
This entry can run by filename with the training source snapshot on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


@dataclass(frozen=True)
class SamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int = 42
    gpu_memory_fraction: float = 0.92
    gpu_headroom_gib: float = 8.0

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < self.top_p <= 1 or not 0 < self.gpu_memory_fraction <= 1:
            raise ValueError("probability and memory fractions must be in (0, 1]")
        if not math.isfinite(self.gpu_headroom_gib) or self.gpu_headroom_gib <= 0:
            raise ValueError("positive GPU headroom is required")


def emit(event: str, **values) -> None:
    print(json.dumps({"event": event, "time_unix": time.time(), **values}, allow_nan=False), flush=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def restore_adapters(adapters: dict, checkpoint: Path) -> str:
    """Reshard only adapter tensors, rejecting missing/extra keys and wrong shapes.

    This is an inference subset load, not optimizer/RNG continuation. Its digest
    describes the full restored adapters, not the training rank-state digests.
    """
    from archlab.automodel.checkpointing import state_digest
    from archlab.automodel.loading import assert_loaded_weights_finite, poison_weights_before_load

    if torch.distributed.is_initialized():
        raise RuntimeError("this sampler requires one unsharded inference process")
    container = torch.nn.ModuleDict(adapters)
    poison_weights_before_load(container)
    payload = {"adapters": {name: module.state_dict() for name, module in adapters.items()}}
    expected = {f"adapters.{layer}.{name}": value
                for layer, state in payload["adapters"].items() for name, value in state.items()}
    saved = dcp.FileSystemReader(checkpoint / "state").read_metadata().state_dict_metadata
    actual_keys = {name for name in saved if name.startswith("adapters.")}
    if actual_keys != expected.keys():
        raise ValueError("adapter checkpoint key coverage mismatch")
    for name, tensor in expected.items():
        if saved[name].size != tensor.shape or saved[name].properties.dtype != tensor.dtype:
            raise ValueError(f"adapter checkpoint shape/dtype mismatch: {name}")
    dcp.load(payload, checkpoint_id=checkpoint / "state")
    for name, module in adapters.items():
        module.load_state_dict(payload["adapters"][name], strict=True)
    assert_loaded_weights_finite(container)
    return state_digest(payload["adapters"])


def attach_cpu_lookup_hook(table: torch.nn.Module) -> None:
    """Execute the original lookup on CPU and return its output to its caller."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    add_hook_to_module(table, AlignDevicesHook(execution_device="cpu", io_same_device=True))


def place_for_sampling(model: torch.nn.Module, config: SamplingConfig) -> dict:
    from accelerate.utils import set_module_tensor_to_device
    from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
        Qwen3_8_FlashNextOwnerShardedEmbedding,
    )

    tables = {name: module for name, module in model.named_modules()
              if isinstance(module, Qwen3_8_FlashNextOwnerShardedEmbedding)}
    if len(tables) != 1 or any(m.process_group is not None for m in tables.values()):
        raise ValueError("expected one complete single-owner PLE table")
    cpu_names = {f"{name}.{key}" for name, module in tables.items()
                 for key, _ in module.named_parameters()}
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    gpu_bytes = sum(t.numel() * t.element_size() for n, t in tensors.items() if n not in cpu_names)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    available = min(free_bytes, int(total_bytes * config.gpu_memory_fraction))
    if gpu_bytes + config.gpu_headroom_gib * 2**30 > available:
        raise MemoryError(f"model requires {gpu_bytes} GPU bytes plus headroom, only {available} available")
    emit("gpu_placement_begin", model_gpu_bytes=gpu_bytes, free_gpu_bytes=free_bytes)
    # Drop the pre-placement tensor dictionary before sampling: retaining its
    # tensors would keep a redundant CPU copy of every GPU parameter alive.
    names = tuple(tensors)
    del tensors
    for name in names:
        set_module_tensor_to_device(model, name, "cpu" if name in cpu_names else "cuda:0")
    for table in tables.values():
        attach_cpu_lookup_hook(table)
    torch.cuda.synchronize()
    return {"cpu_lookup_modules": list(tables), "gpu_parameter_bytes": gpu_bytes,
            "allocated_gpu_bytes": torch.cuda.memory_allocated(), "free_gpu_bytes_before": free_bytes}


def build_model(base: Path, checkpoint: Path, config: SamplingConfig):
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype
    from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextConfig
    from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
        QWEN3_8_FLASH_NEXT_NGRAM_PADDED_ROWS,
        Qwen3_8_FlashNextEngramTableConfig,
    )
    from nemo_automodel.components.models.qwen3_8_flash_next.model import (
        Qwen3_8_FlashNextForConditionalGeneration,
    )

    from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig
    from archlab.automodel.checkpointing import read_training_checkpoint
    from archlab.automodel.loading import (
        assert_loaded_weights_finite,
        audit_checkpoint_keys,
        poison_weights_before_load,
        rebuild_nonpersistent_buffers,
    )
    from archlab.automodel.simplicial import install_simplicial_modules

    metadata = json.loads((checkpoint / "COMPLETE.json").read_text())
    read_training_checkpoint(checkpoint, metadata["contract"])
    for name, expected in metadata["contract"]["pretrained_metadata_sha256"].items():
        if sha256(base / name) != expected:
            raise ValueError(f"pretrained metadata changed: {name}")
    model_config = Qwen3_8_FlashNextConfig.from_pretrained(base, local_files_only=True, language_model_only=True)
    text_config = model_config.text_config
    table_config = Qwen3_8_FlashNextEngramTableConfig(
        num_embeddings=QWEN3_8_FLASH_NEXT_NGRAM_PADDED_ROWS,
        embedding_dim=text_config.ple_embed_dim // ((text_config.ngram_size - 1) * text_config.heads_per_ngram),
        initializer_range=text_config.initializer_range)
    backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32", experts="torch_mm",
                            dispatcher="torch", rope_fusion=False, gate_precision="float32",
                            enable_hf_state_dict_adapter=True)
    with torch.device("meta"):
        model = Qwen3_8_FlashNextForConditionalGeneration(
            model_config, backend=backend, engram_table_config=table_config,
            moe_overrides={"aux_loss_coeff": 0.0})
    audit = audit_checkpoint_keys(model, base)
    emit("checkpoint_key_audit", **audit)
    cast_model_to_dtype(model, torch.bfloat16, skip_modules=("_fp32_params",))
    model.requires_grad_(False)
    model._skip_init_weights_on_load = True
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    rebuild_nonpersistent_buffers(model, torch.device("cpu"))
    emit("base_poison_begin")
    poison_weights_before_load(model)
    checkpointer = CheckpointingConfig(checkpoint_dir="", model_repo_id=str(base), save_consolidated=False,
                                      dequantize_base_checkpoint=False, is_async=False).build(
        dp_rank=0, tp_rank=0, pp_rank=0)
    emit("base_load_begin")
    checkpointer.load_base_model(model, torch.device("cpu"), None, str(base))
    rebuild_nonpersistent_buffers(model, torch.device("cpu"))
    emit("base_load_check_begin")
    elements = assert_loaded_weights_finite(model)
    emit("base_load_end", elements=elements)
    adapter_config = SimplicialAdapterConfig(**metadata["contract"]["adapter"])
    adapters = install_simplicial_modules(model, adapter_config, backend="triton", dtype=torch.float32)
    adapter_digest = restore_adapters(adapters, checkpoint)
    adapter_count = sum(p.numel() for module in adapters.values() for p in module.parameters())
    if adapter_count != len(adapters) * adapter_config.parameter_count():
        raise ValueError("unexpected adapter parameter count")
    for module in adapters.values():
        module.to(dtype=torch.bfloat16)  # Same compute precision as training FSDP.
    emit("adapters_loaded", layers=list(adapters), parameters=adapter_count, sha256=adapter_digest)
    model.requires_grad_(False)
    model.eval()
    placement = place_for_sampling(model, config)
    emit("model_ready", **placement)
    return model, {"checkpoint_audit": audit, "step": metadata["cursor"], "adapters_fp32_sha256": adapter_digest,
                   "adapter_parameters": adapter_count, "placement": placement}


@torch.inference_mode()
def generate(model, tokenizer, prompt: str, config: SamplingConfig, *, seed: int) -> dict:
    from transformers import TemperatureLogitsWarper, TopPLogitsWarper

    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    original_length = len(tokens)
    if not tokens or original_length + config.max_new_tokens > 16384:
        raise ValueError("prompt must be nonempty and fit the trained 16K context")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    top_p = TopPLogitsWarper(config.top_p)
    temperature = TemperatureLogitsWarper(config.temperature) if config.temperature else None
    started = time.monotonic()
    for index in range(config.max_new_tokens):
        ids = torch.tensor([tokens], device="cuda", dtype=torch.long)
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=1, output_hidden_states=False).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite generation logits")
        if temperature is None:
            token = logits.argmax(-1).item()
        else:
            probabilities = top_p(ids, temperature(ids, logits)).softmax(-1)
            token = torch.multinomial(probabilities, 1, generator=generator).item()
        tokens.append(token)
        if index % 16 == 0:
            emit("generation_progress", generated_tokens=index + 1, elapsed_seconds=time.monotonic() - started)
        if token == tokenizer.eos_token_id:
            break
    return {"prompt": prompt, "continuation": tokenizer.decode(tokens[original_length:], skip_special_tokens=True),
            "token_ids": tokens, "prompt_tokens": original_length, "new_tokens": len(tokens) - original_length,
            "stopped_at_eos": tokens[-1] == tokenizer.eos_token_id, "seed": seed,
            "elapsed_seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    config = SamplingConfig(max_new_tokens=args.max_new_tokens)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(16)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
    import nemo_automodel
    from transformers import AutoTokenizer

    from archlab.automodel.checkpointing import write_json
    from archlab.automodel.runtime import configure_frozen_gdn_runtime
    from archlab.automodel.simplicial import UPSTREAM_COMMIT
    from archlab.prompts import load_prompts

    upstream = Path(nemo_automodel.__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != UPSTREAM_COMMIT or dirty:
        raise RuntimeError("sampling requires the qualified unmodified upstream pin")
    metadata = json.loads((args.checkpoint / "COMPLETE.json").read_text())
    source_checks = {}
    for name in ("archlab.automodel.simplicial", "archlab.architectures.simplicial_adapter",
                 "archlab.architectures.simplicial_attention", "archlab.automodel.loading"):
        module = __import__(name, fromlist=["__file__"])
        path = Path(module.__file__)
        relative = str(Path(*name.split(".")[1:]).with_suffix(".py"))
        digest = sha256(path)
        if metadata["contract"]["runtime"]["project_source_sha256"][relative] != digest:
            raise ValueError(f"sample from the exact training source snapshot: {name}")
        source_checks[name] = {"path": str(path), "sha256": digest}
    report = {"config": asdict(config), "hostname": socket.gethostname(), "pid": os.getpid(),
              "python": sys.executable, "python_version": sys.version, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "upstream_commit": revision, "source_checks": source_checks,
              "training_container": metadata["contract"]["runtime"]["container_image"],
              "sampling_environment": "existing user-approved DSW venv; not the DLC training container",
              "packages": {name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "accelerate", "safetensors", "fla-core", "triton")},
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_complete_sha256": sha256(args.checkpoint / "COMPLETE.json"),
              "prompts_sha256": sha256(args.prompts), "decoding": "raw continuation; no chat template; no cache",
              "gdn_runtime": configure_frozen_gdn_runtime(), "samples": [], "complete": False}
    write_json(args.output / "samples.json", report)
    emit("runtime", **{k: v for k, v in report.items() if k not in ("gdn_runtime", "source_checks")})
    model, model_report = build_model(args.base, args.checkpoint, config)
    report.update(model_report)
    write_json(args.output / "samples.json", report)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    for index, prompt in enumerate(load_prompts(args.prompts)):
        emit("prompt_begin", id=prompt.id)
        sample = generate(model, tokenizer, prompt.text, config, seed=config.seed + index)
        report["samples"].append({"id": prompt.id, **sample})
        write_json(args.output / "samples.json", report)
        emit("prompt_complete", id=prompt.id, **sample)
    report["complete"] = True
    report["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
    write_json(args.output / "samples.json", report)
    emit("complete", output=str(args.output), samples=len(report["samples"]))


if __name__ == "__main__":
    main()

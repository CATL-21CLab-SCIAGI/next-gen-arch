"""Sample a native Flash-Next checkpoint as one complete DP-only replica.

Full-prefix recomputation is intentional: GDN and PLE do not implement a shared
incremental inference-cache contract. This is a qualitative check, not a serving
benchmark, and never loads or changes optimizer state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

import torch

from archlab.architectures.qwen38_flash_next_full import Qwen38FlashNextFullConfig
from archlab.megatron.backend import validate_runtime
from archlab.megatron.qwen38_flash_next_full_train import (
    _assert_dp_only_groups,
    _atomic_json,
    _megatron_argv,
    build_model,
)
from archlab.megatron.qwen38_flash_next_full_train import (
    _parser as trainer_parser,
)
from archlab.prompts import load_prompts


def select_token(logits: torch.Tensor, *, temperature: float, top_p: float) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise RuntimeError("sampling logits contain nonfinite values")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_probs, indices = probabilities.sort(descending=True, dim=-1)
    excluded = sorted_probs.cumsum(dim=-1) - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(excluded, 0)
    sampled = torch.multinomial(sorted_probs, num_samples=1)
    return indices.gather(-1, sampled)


def sampling_config(variant: str) -> Qwen38FlashNextFullConfig:
    if variant == "1b-depth48-no-mtp":
        return Qwen38FlashNextFullConfig.billion_depth48_no_mtp()
    if variant == "w320-e32-depth48-no-mtp":
        return Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()
    raise ValueError(f"unsupported sampling model variant: {variant}")


def _sampling_argv(trainer, config, *, checkpoint_step=None, attention_backend="auto") -> list[str]:
    # Inference has no optimizer/backward pass and must not require Apex's
    # fused gradient-accumulation extension just to construct the output head.
    argv = _megatron_argv(trainer, config) + [
        "--no-load-optim",
        "--no-load-rng",
        "--no-gradient-accumulation-fusion",
    ]
    if checkpoint_step is not None:
        if checkpoint_step < 1:
            raise ValueError("checkpoint step must be positive")
        argv.extend(["--ckpt-step", str(checkpoint_step)])
    if attention_backend != "auto":
        if attention_backend not in ("unfused", "flash", "fused"):
            raise ValueError("unsupported attention backend")
        argv.extend(["--attention-backend", attention_backend])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path(__file__).parents[1] / "prompts" / "backbone_validation.yaml",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-variant", default="1b-depth48-no-mtp",
                        choices=("1b-depth48-no-mtp", "w320-e32-depth48-no-mtp"))
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--attention-backend", choices=("auto", "unfused", "flash", "fused"),
                        default="auto")
    parser.add_argument("--memory-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite samples: {args.output}")
    if args.max_new_tokens < 1 or args.temperature < 0 or not 0 < args.top_p <= 1:
        raise ValueError("invalid generation controls")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("invalid CUDA memory fraction")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("sampling uses exactly one complete model replica")
    if not (args.checkpoint_root / "latest_checkpointed_iteration.txt").is_file():
        raise ValueError("sampling requires a completed checkpoint marker")

    config = sampling_config(args.model_variant)
    contract_file = args.checkpoint_root.parent / "RUN_CONTRACT.json"
    if contract_file.is_file():
        stored = json.loads(contract_file.read_text())
        if stored.get("model_config") != config.to_dict():
            raise ValueError("selected model does not match the checkpoint's training contract")
    if args.checkpoint_step is not None:
        selected = args.checkpoint_root / f"iter_{args.checkpoint_step:07d}"
        if not (selected / ".metadata").is_file():
            raise ValueError("selected checkpoint has no completed native metadata")
    trainer = trainer_parser().parse_args(
        [
            "--data-root",
            str(args.tokenizer),
            "--tokenizer",
            str(args.tokenizer),
            "--run-dir",
            str(args.output.parent / "sampling-runtime"),
            "--load-dir",
            str(args.checkpoint_root),
            "--model-variant",
            args.model_variant,
            "--parallelism",
            "dp-only",
            "--global-batch-size",
            "1",
            "--micro-batch-size",
            "1",
            "--probe-steps",
            "1",
        ]
    )
    sys.argv = _sampling_argv(trainer, config, checkpoint_step=args.checkpoint_step,
                              attention_backend=args.attention_backend)
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.module import Float16Module
    from megatron.training.arguments import (
        core_transformer_config_from_args,
        parse_args,
        validate_args,
    )
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import initialize_megatron
    from tokenizers import Tokenizer

    native = validate_args(parse_args())
    native.tensorboard_dir = None
    set_global_variables(native)
    initialize_megatron()
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    topology = _assert_dp_only_groups(groups)
    native_config = core_transformer_config_from_args(native)
    model = Float16Module(native_config, build_model(config, native_config, groups).cuda())
    iteration, _ = load_checkpoint([model], None, None, strict=True)
    if args.checkpoint_step is not None and iteration != args.checkpoint_step:
        raise RuntimeError("native loader did not restore the requested checkpoint step")
    model.eval()
    torch.manual_seed(args.seed)
    tokenizer = Tokenizer.from_file(str(args.tokenizer / "tokenizer.json"))
    records = []
    container = {
        key: os.environ[key]
        for key in (
            "NVIDIA_PRODUCT_NAME",
            "NVIDIA_PYTORCH_VERSION",
            "NVIDIA_BUILD_ID",
            "CUDA_VERSION",
            "NGA_CONTAINER_DIGEST",
        )
        if key in os.environ
    }
    evidence = {
        "checkpoint_root": str(args.checkpoint_root),
        "iteration": iteration,
        "model_variant": args.model_variant,
        "model_config": config.to_dict(),
        "attention_backend": args.attention_backend,
        "cuda_memory_fraction": args.memory_fraction,
        "parallelism": topology,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "decoding": "full-prefix recomputation; no chat template",
        "gradient_accumulation_fusion": False,
        "host": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "container_identity": container,
        "sampling_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prompts_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (args.tokenizer / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "started_at_unix": time.time(),
        "samples": records,
        "runtime": validate_runtime(require_pretrain=False),
        "flash_linear_attention": importlib.metadata.version("flash-linear-attention"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    with torch.inference_mode():
        for prompt in load_prompts(args.prompts):
            initial = tokenizer.encode(prompt.text, add_special_tokens=False).ids
            tokens = torch.tensor([initial], device="cuda", dtype=torch.long)
            generated = []
            for _ in range(args.max_new_tokens):
                positions = torch.arange(tokens.size(1), device=tokens.device).expand_as(tokens)
                logits = model(tokens, positions, None)
                token = select_token(
                    logits[:, -1, :], temperature=args.temperature, top_p=args.top_p
                )
                token_id = token.item()
                generated.append(token_id)
                if token_id == config.eos_token_id:
                    break
                tokens = torch.cat((tokens, token), dim=-1)
            record = {
                "id": prompt.id,
                "prompt": prompt.text,
                "token_ids": generated,
                "continuation": tokenizer.decode(generated, skip_special_tokens=True),
                "finish_reason": "eos" if generated[-1] == config.eos_token_id else "length",
            }
            records.append(record)
            _atomic_json(args.output, evidence)
            print(record, flush=True)
    evidence["completed_at_unix"] = time.time()
    evidence["completed"] = True
    _atomic_json(args.output, evidence)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

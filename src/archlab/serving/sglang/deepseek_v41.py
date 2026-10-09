"""Full-checkpoint V4.1 extension for the pinned SGLang runtime.

Register with SGLANG_EXTERNAL_MODEL_PACKAGE=archlab.serving.sglang. The native
architecture name is intentionally retained: SGLang uses it for CSA2 backend
selection. This module overrides that registry entry only inside this worker.
The launcher must verify the resolved model class before loading any weights.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig, V41SimplicialAdapter
from archlab.architectures.deepseek_v41_normal_adapter import V41NormalAttentionAdapter
from archlab.serving.sglang_v41_adapter_bridge import install_adapter_boundary
from archlab.serving.sglang_v41_engram import BF16EngramEmbedding
from archlab.serving.sglang_v41_moe import install_native_moe
from archlab.serving.sglang_v41_padding import (
    install_hash_padding_boundary,
    install_live_source_boundary,
)
from archlab.serving.v41_checkpoint_inventory import V41CheckpointInventory
from archlab.serving.v41_direct_checkpoint import iter_weights
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM as NativeV41
from sglang.srt.runtime_context import get_parallel, get_serving


class DeepseekV4ForCausalLM(NativeV41):
    def __init__(self, config, quant_config=None, prefix=""):
        # A deployment-specific, reviewed runtime fingerprint is mandatory.
        expected = os.environ.get("ARCHLAB_SGLANG_MODEL_SHA256")
        actual = hashlib.sha256(Path(inspect.getfile(NativeV41)).read_bytes()).hexdigest()
        if not expected or actual != expected:
            raise ValueError("SGLang model source differs from the reviewed runtime fingerprint")
        parallel = get_parallel()
        self.archlab_tp_rank = parallel.tp_rank
        geometry = (parallel.tp_size, parallel.moe_ep_size, parallel.attn_dp_size,
                    parallel.attn_cp_size, parallel.pp_group.world_size)
        allowed = {(8, 8, 1, 1, 1)}
        if os.environ.get("ARCHLAB_RL_OFFLOAD_POLICY") == "forbidden":
            from sglang.srt.runtime_context import get_exec
            if get_exec().features.enable_weights_cpu_backup:
                raise ValueError("resident rollout forbids CPU weight backups")
            allowed.add((32, 32, 4, 1, 1))
        if geometry not in allowed:
            raise ValueError("unsupported full-checkpoint serving geometry")
        metadata = getattr(config, "archlab", None)
        if not isinstance(metadata, dict) or metadata.get("variant") not in ("normal", "simplicial"):
            raise ValueError("missing full fine-tuned checkpoint metadata")
        stock_fp8 = os.environ.get("ARCHLAB_MILES_STOCK_FP8") == "1"
        if stock_fp8 and (quant_config is None or quant_config.get_name() != "fp8"):
            raise ValueError("stock FP8 rollout requires the native FP8 quantization config")
        if not stock_fp8 and (quant_config is not None or getattr(config, "quantization_config", None)):
            raise ValueError("full fine-tuned weights must not be requantized")
        if config.model_type != "deepseek_v41" or not config.hc_pre_from_prev_sublayer:
            raise ValueError("expected native V4.1 carried pre-mix architecture")
        # The trained gate stores BF16 weights but computes its linear in FP32.
        config.router_fp32 = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        head_loader = getattr(self.lm_head.weight, "weight_loader", None)
        self.lm_head.float()
        if head_loader is not None:
            self.lm_head.weight.weight_loader = head_loader
        self.archlab_metadata = metadata
        self.archlab_adapters = {}
        self.archlab_cache_states = {}
        for layer in self.model.layers:
            install_live_source_boundary(layer.self_attn)
            if not stock_fp8:
                install_native_moe(layer.mlp, tp_rank=parallel.tp_rank, tp_size=parallel.tp_size,
                                   all_reduce=tensor_model_parallel_all_reduce)
        device = self.model.embed_tokens.weight.device
        variant = metadata["variant"]
        cls = V41NormalAttentionAdapter if variant == "normal" else V41SimplicialAdapter
        adapter_config = V41AdapterConfig(width=config.hidden_size, streams=config.hc_mult)
        for index in metadata["adapter_layers"]:
            with torch.device("cpu"):
                before = torch.get_default_dtype()
                try:
                    torch.set_default_dtype(torch.float32)
                    adapter = cls(adapter_config, backend="reference")
                finally:
                    torch.set_default_dtype(before)
            adapter = adapter.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            self.archlab_adapters[index] = adapter
            self.archlab_cache_states[index] = install_adapter_boundary(
                self.model.layers[index], adapter, tp_size=8, max_slots=32)
        for index, rows in zip(config.engram_layer_ids, config.engram_num_embeddings, strict=True):
            self.model.layers[index].engram.embed = BF16EngramEmbedding(
                rows, config.engram_head_dim, tp_rank=parallel.tp_rank, tp_size=parallel.tp_size,
                all_reduce=tensor_model_parallel_all_reduce, device=device)
        if stock_fp8 or os.environ.get("ARCHLAB_RL_OFFLOAD_POLICY") == "forbidden":
            # Deterministic buffers are not refreshed by the policy stream.
            # Retain them on GPU outside the discardable weight allocation pool.
            from torch_memory_saver import torch_memory_saver
            with torch_memory_saver.disable():
                for buffer in self.buffers():
                    if buffer.device.type == "cuda":
                        buffer.data = buffer.detach().clone()
        install_hash_padding_boundary(self.model.engram_hasher)
        self._archlab_loaded = False
        if os.environ.get("ARCHLAB_RL_OFFLOAD_POLICY") == "forbidden":
            # Replacing native FP8 Engram storage with BF16 leaves obsolete
            # allocations in the caching allocator. Return those pages before
            # colocated policy transfer; live parameters remain on GPU.
            torch.cuda.synchronize()
            before_free, total = torch.cuda.mem_get_info()
            torch.cuda.empty_cache()
            after_free, _ = torch.cuda.mem_get_info()
            print("ARCHLAB_SERVING_MEMORY " + json.dumps(dict(
                rank=parallel.tp_rank, parameter_bytes=sum(p.numel() * p.element_size() for p in self.parameters()),
                buffer_bytes=sum(b.numel() * b.element_size() for b in self.buffers()),
                released_cached_bytes=after_free-before_free, free_bytes=after_free,
                total_bytes=total)), flush=True)

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if getattr(self, "_archlab_live_update", None) is None:
            return super().post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def _prewarm_mhc_kernels(self):
        if getattr(self, "_archlab_live_update", None) is None:
            return super()._prewarm_mhc_kernels()

    def finalize_live_weights(self):
        # APE conversion and norm caches must see complete weights exactly once
        # per policy version, not partially initialized streaming buckets.
        super().post_load_weights(is_nextn=False)
        super()._prewarm_mhc_kernels()

    def load_weights(self, weights, is_nextn=False):
        weights = iter(weights)
        first = next(weights, None)
        weights = itertools.chain(() if first is None else (first,), weights)
        if os.environ.get("ARCHLAB_MILES_LIVE_WEIGHTS") == "1" and (
                self._archlab_loaded or getattr(self, "_archlab_live_update", None) is not None
                or (first is not None and first[0] == "archlab_update_begin")):
            from archlab.serving.sglang_v41_live_weights import update

            if is_nextn:
                raise ValueError("draft model updates are unsupported")
            return update(self, weights, super().load_weights)
        if is_nextn or self._archlab_loaded:
            raise ValueError("draft models and partial live weight updates are not qualified")
        if "full_checkpoint" in self.archlab_metadata:
            pointer = list(weights)
            cursor = self.archlab_metadata["checkpoint_cursor"]
            if (len(pointer) != 1 or pointer[0][0] != "archlab_full_checkpoint_pointer"
                    or pointer[0][1].dtype != torch.int64
                    or pointer[0][1].tolist() != [cursor["step"], cursor["supervised_tokens"]]):
                raise ValueError("invalid full-checkpoint loader pointer")
            path = Path(self.archlab_metadata["full_checkpoint"])
            identity = hashlib.sha256((path / "COMPLETE.json").read_bytes()).hexdigest()
            if identity != self.archlab_metadata["complete_sha256"]:
                raise ValueError("full-checkpoint identity changed")
            inventory = V41CheckpointInventory(path)
            if (inventory.marker["cursor"] != cursor
                    or inventory.marker["contract"]["variant"] != self.archlab_metadata["variant"]):
                raise ValueError("checkpoint cursor or variant differs")
            weights = iter_weights(inventory, tp_rank=get_parallel().tp_rank,
                                   engram_rows=dict(zip(self.config.engram_layer_ids,
                                                       self.config.engram_num_embeddings,
                                                       strict=True)))
        expected = {f"layers.{index}.attn_hc.simplicial_adapter.{name}": parameter
                    for index, adapter in self.archlab_adapters.items()
                    for name, parameter in adapter.named_parameters()}
        loaded = set()
        loaded_parameters = set()
        originals = {}
        parameters = dict(self.named_parameters())
        for parameter_name, parameter in parameters.items():
            originals[parameter_name] = getattr(parameter, "weight_loader", None)
            original = originals[parameter_name] or default_weight_loader

            def tracked(*args, _original=original, _name=parameter_name, **kwargs):
                result = _original(*args, **kwargs)
                loaded_parameters.add(_name)
                return result

            parameter.weight_loader = tracked

        def backbone():
            for name, value in weights:
                row_shard = re.fullmatch(r"layers\.(\d+)\.engram\.embed\.weight\.rows\.(\d+)", name)
                if row_shard:
                    layer_id, row_start = map(int, row_shard.groups())
                    if layer_id not in self.config.engram_layer_ids:
                        raise ValueError("row shard belongs to an unknown Engram layer")
                    self.model.layers[layer_id].engram.embed.load_piece(row_start, value)
                    loaded_parameters.add(f"model.layers.{layer_id}.engram.embed.weight")
                elif ".simplicial_adapter." in name:
                    if name not in expected or name in loaded:
                        raise ValueError(f"unexpected or duplicate adapter tensor: {name}")
                    parameter = expected[name]
                    if parameter.shape != value.shape or value.dtype != torch.float32:
                        raise ValueError(f"adapter shape/dtype differs: {name}")
                    loaded.add(name)
                    match = re.fullmatch(r"layers\.(\d+)\.attn_hc\.simplicial_adapter\.(.+)", name)
                    if match is None:
                        raise ValueError("unrecognized adapter registration path")
                    yield f"model.layers.{match[1]}.archlab_adapter.{match[2]}", value
                else:
                    yield name, value

        try:
            result = super().load_weights(backbone(), is_nextn=False)
        finally:
            for name, parameter in parameters.items():
                if originals[name] is None:
                    del parameter.weight_loader
                else:
                    parameter.weight_loader = originals[name]
        if loaded != set(expected):
            raise ValueError("full checkpoint did not load every adapter tensor")
        missing = {name for name in parameters
                   if name not in loaded_parameters
                   and not any(marker in name for marker in (
                       "attn_mqa.k_scale", "attn_mqa.v_scale", "blockscale_swizzled"))}
        if missing:
            raise ValueError(f"engine parameters were not loaded from the full checkpoint: {sorted(missing)}")
        self.validate_derived_state()
        if any(parameter.device.type != "cuda" for parameter in self.parameters()):
            raise ValueError("all inference weights must remain resident on GPU")
        self.requires_grad_(False)
        for layer in self.config.engram_layer_ids:
            self.model.layers[layer].engram.embed.finish_load(str(layer))
        self.eval()
        self._archlab_loaded = True
        print(json.dumps(dict(event="archlab_full_checkpoint_loaded",
                              variant=self.archlab_metadata["variant"],
                              cursor=self.archlab_metadata["checkpoint_cursor"],
                              adapter_tensors=len(loaded), engram_dtype="bfloat16",
                              backbone_parameter_tensors_loaded=len(loaded_parameters),
                              router_compute="float32", lm_head_dtype="float32")), flush=True)
        return result

    def validate_derived_state(self):
        root = Path(get_serving().tokenizer_path)
        for name, filename in self.archlab_metadata["derived_buffer_files"].items():
            if not re.fullmatch(r"engram_hash\.(token_map|primes|offsets|multipliers)", name):
                raise ValueError(f"unexpected derived buffer: {name}")
            path = root / filename
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("buffer path escapes the model directory")
            stored = load_file(path)[name]
            derived = getattr(self.model.engram_hasher, name.split(".")[-1]).detach().cpu()
            if stored.numel() != derived.numel() or not torch.equal(stored.reshape(-1), derived.reshape(-1)):
                raise ValueError(f"engine-derived Engram state differs from training: {name}")

    def forward(self, *args, **kwargs):
        if not self._archlab_loaded:
            raise RuntimeError("inference before complete full-checkpoint loading is prohibited")
        return super().forward(*args, **kwargs)


EntryClass = DeepseekV4ForCausalLM

"""Lossless Hugging Face boundary for the publisher Limite model and preludes.

Import this module through verl's ``model.external_lib``. The exported model is
an ordinary AutoModelForCausalLM; its forward and generation remain owned by the
verified publisher. This is a weights migration, not an optimizer migration.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from functools import wraps
from pathlib import Path
from types import MethodType

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from archlab.architectures.limite_adapter import (
    LimiteAdapterConfig,
    install_adapters,
    set_normal_attention_backward,
    set_normal_attention_kernel,
    set_trainable_mode,
)
from archlab.architectures.limite_loader import load_model, upstream_classes
from archlab.artifacts import atomic_write_json, sha256_file


class LimiteConfig(PretrainedConfig):
    model_type = "archlab_limite"

    def __init__(self, publisher_snapshot="", publisher_config=None, adapter=None,
                 parameter_dtypes=None, normal_kernel="shared", normal_backward="tilelang",
                 trainable_mode="full", backbone_attention="sdpa", **kwargs):
        native = dict(publisher_config or {})
        # Expose native geometry to package schedulers and FLOP estimators.
        metadata = {k: v for k, v in native.items()
                    if k not in {"model_type", "architectures", "auto_map", "torch_dtype", "dtype"}}
        metadata.update(kwargs)
        super().__init__(**metadata)
        self.publisher_snapshot = str(publisher_snapshot)
        self.publisher_config = native
        self.adapter = adapter or {}
        self.parameter_dtypes = parameter_dtypes or {}
        self.normal_kernel = normal_kernel
        self.normal_backward = normal_backward
        self.trainable_mode = trainable_mode
        if backbone_attention not in ("sdpa", "flash_attention_2"):
            raise ValueError("unsupported native backbone attention implementation")
        self.backbone_attention = backbone_attention
        self.architectures = ["ArchlabLimiteForCausalLM"]


def _adapter_config(values):
    return LimiteAdapterConfig(**{k: values[k] for k in
                                  ("variant", "short_window", "seed", "attention_backend")
                                  if k in values})


class LimiteForCausalLM(PreTrainedModel):
    """Keep the publisher untouched, including mixed parameter dtypes and cache."""

    config_class = LimiteConfig
    base_model_prefix = "policy"
    _no_split_modules = ["LimiteDecoderLayer", "AttentionPrelude"]
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        config_class, model_class = upstream_classes(config.publisher_snapshot)
        native = config_class.from_dict(config.publisher_config)
        native._attn_implementation = config.backbone_attention
        self.policy = install_adapters(model_class(native), _adapter_config(config.adapter))
        if self.policy.model.adapter_config != config.adapter:
            raise ValueError("exported adapter geometry differs from publisher construction")
        set_normal_attention_kernel(self.policy, config.normal_kernel)
        set_normal_attention_backward(self.policy, config.normal_backward)
        set_trainable_mode(self.policy, config.trainable_mode)
        self._restore_dtypes()
        self._activation_checkpointing = False
        for layer in (*self.policy.model.base.layers, *self.policy.model.adapters):
            self._bind_checkpointing(layer)

    def _bind_checkpointing(self, layer):
        from torch.utils.checkpoint import checkpoint

        original = layer.forward

        @wraps(original)
        def forward(_layer, *args, **kwargs):
            if self._activation_checkpointing and self.training and torch.is_grad_enabled():
                return checkpoint(original, *args, use_reentrant=False, **kwargs)
            return original(*args, **kwargs)

        layer.forward = MethodType(forward, layer)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if (gradient_checkpointing_kwargs or {}).get("use_reentrant", False):
            raise ValueError("Limite requires non-reentrant activation checkpointing")
        self.policy.model._share_training_masks = True
        self._activation_checkpointing = True

    def gradient_checkpointing_disable(self):
        self._activation_checkpointing = False

    def _restore_dtypes(self):
        parameters = dict(self.policy.named_parameters())
        if set(parameters) != set(self.config.parameter_dtypes):
            raise ValueError("exported parameter dtype inventory differs from model")
        for name, parameter in parameters.items():
            dtype = getattr(torch, self.config.parameter_dtypes[name])
            if parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype=dtype)
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.to(dtype=dtype)

    def _apply(self, fn, recurse=True):
        # verl moves/casts complete modules. Limite's RMSNorm epsilon depends on
        # activation dtype, so global FP32/BF16 conversion changes the model.
        def preserve_dtype(tensor):
            result = fn(tensor)
            if tensor.is_floating_point() and result.dtype != tensor.dtype:
                # Recast the source, never an already rounded BF16 result.
                return tensor.to(device=result.device)
            return result

        return super()._apply(preserve_dtype, recurse=recurse)

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, **kwargs):
        return self.policy(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, **kwargs)

    def generate(self, *args, **kwargs):
        return self.policy.generate(*args, **kwargs)

    def get_input_embeddings(self):
        return self.policy.get_input_embeddings()

    def get_output_embeddings(self):
        return self.policy.get_output_embeddings()

    def get_decoder(self):
        return self.policy.get_decoder()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        """Load local, validated mixed-dtype weights without HF global recasting.

        The 1B model fits on CPU; deliberate CPU construction avoids package
        meta-device initialization changing the publisher's derived buffers.
        The package engine may move it after construction. No remote code is downloaded.
        """
        if model_args:
            raise ValueError("Limite export has no positional construction overrides")
        root = Path(pretrained_model_name_or_path)
        verify_export(root)
        config = config or LimiteConfig.from_pretrained(root, local_files_only=True)
        for key in ("torch_dtype", "dtype", "trust_remote_code", "local_files_only",
                    "_from_auto", "_from_pipeline", "cache_dir", "revision", "token",
                    "use_safetensors", "weights_only", "low_cpu_mem_usage"):
            kwargs.pop(key, None)
        if kwargs.pop("adapter_kwargs", None):
            raise ValueError("Limite export cannot load additional PEFT adapters")
        output_loading_info = kwargs.pop("output_loading_info", False)
        device = kwargs.pop("device_map", "cpu")
        if kwargs or not isinstance(device, (str, torch.device)) or str(device) == "auto":
            raise ValueError(f"unsupported Limite loading options: {sorted(kwargs)}, device={device}")
        with torch.device("cpu"):
            model = cls(config)
        load_safetensors_model(model, root / "model.safetensors", strict=True, device="cpu")
        model.to(device=device)
        model.eval()
        if output_loading_info:
            return model, dict(missing_keys=[], unexpected_keys=[], mismatched_keys=[], error_msgs=[])
        return model


def verify_export(path):
    root = Path(path)
    receipt = json.loads((root / "EXPORT.json").read_text())
    if receipt.get("format") != "archlab-limite-hf-v1":
        raise ValueError("unknown Limite export format")
    if not {"config.json", "model.safetensors"} <= set(receipt["files"]):
        raise ValueError("incomplete Limite export")
    for name, digest in receipt["files"].items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError("export checksum mismatch: " + name)
    return receipt


def _state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_legacy_checkpoint(snapshot, checkpoint, *, device="cpu", target_attention_backend=None):
    """Load verified full weights or adapters over their proven publisher base.

    The earliest adapter receipts predate ``trainable_mode`` and name the frozen
    publisher hash ``frozen_sha256``. A backend migration must be explicit; it
    changes the constructed reduction, never the historical source receipt.
    """
    root = Path(checkpoint)
    receipt = json.loads((root / "COMPLETE.json").read_text())
    full = receipt.get("trainable_mode") == "full"
    if full:
        if not {"backbone.pt", "adapter.pt"} <= set(receipt["files"]):
            raise ValueError("full-weight checkpoint lacks model payloads")
        publisher_hash = receipt.get("base_snapshot_sha256")
    else:
        if (receipt.get("trainable_mode") not in (None, "adapter")
                or "adapter.pt" not in receipt["files"]
                or "backbone.pt" in receipt["files"] or (root / "backbone.pt").exists()):
            raise ValueError("adapter checkpoint requires only adapter model weights")
        publisher_hash = receipt.get("frozen_sha256")
    if not publisher_hash:
        raise ValueError("checkpoint lacks verified publisher provenance")
    for name, digest in receipt["files"].items():
        if Path(name).name != name or sha256_file(root / name) != digest:
            raise ValueError("checkpoint checksum mismatch: " + name)
    target_adapter = dict(receipt["adapter"])
    target_adapter.setdefault("attention_backend", "native")
    if target_attention_backend is not None:
        target_adapter["attention_backend"] = target_attention_backend
    model = install_adapters(load_model(snapshot, attn_implementation="sdpa", device_map=device),
                             _adapter_config(target_adapter))
    if model.model.adapter_config != target_adapter:
        raise ValueError("checkpoint adapter geometry differs from publisher")
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1)
                          .view(torch.uint8).numpy().tobytes())
    if digest.hexdigest() != publisher_hash:
        raise ValueError("checkpoint belongs to another publisher snapshot")
    adapter = torch.load(root / "adapter.pt", map_location="cpu", weights_only=True)
    if full:
        backbone = torch.load(root / "backbone.pt", map_location="cpu", weights_only=True)
        state = {**backbone, **{"model.adapters." + name: value for name, value in adapter.items()}}
        if len(state) != len(backbone) + len(adapter):
            raise ValueError("checkpoint model payloads overlap")
        target = model
    else:
        state, target = adapter, model.model.adapters
    expected = target.state_dict()
    if (set(state) != set(expected)
            or any(value.shape != expected[name].shape or value.dtype != expected[name].dtype
                   for name, value in state.items())):
        raise ValueError("checkpoint tensor inventory, shape or dtype differs from publisher construction")
    target.load_state_dict(state, strict=True)
    set_normal_attention_kernel(model, receipt.get("normal_kernel", "shared"))
    set_normal_attention_backward(model, receipt.get("normal_backward", "tilelang"))
    set_trainable_mode(model, "full")
    return model, receipt


def _copy_publisher_assets(snapshot, destination):
    publisher, root = Path(snapshot), Path(destination)
    download = json.loads((publisher / "DOWNLOAD_VERIFIED.json").read_text())
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "chat_template.jinja", "vocab.json", "merges.txt",
             "generation_config.json"}
    copied = {}
    for item in download["files"]:
        name = item["path"]
        if name not in names:
            continue
        if sha256_file(publisher / name) != item["sha256"]:
            raise ValueError("publisher tokenizer checksum mismatch: " + name)
        target = root / name
        if target.exists() and sha256_file(target) != item["sha256"]:
            raise ValueError("export has conflicting publisher asset: " + name)
        if not target.exists():
            shutil.copyfile(publisher / name, target)
        copied[name] = item["sha256"]
    return copied


def add_publisher_assets(snapshot, destination):
    """Complete an earlier verified weights export without rewriting weights."""
    root = Path(destination)
    receipt = verify_export(root)
    receipt["files"].update(_copy_publisher_assets(snapshot, root))
    atomic_write_json(root / "EXPORT.json", receipt)
    return verify_export(root)


def export_model(model, snapshot, destination, *, source_checkpoint=None):
    """Export every weight, then reload and compare its exact state digest."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    config = LimiteConfig(
        publisher_snapshot=str(Path(snapshot).resolve()), publisher_config=model.config.to_dict(),
        adapter=model.model.adapter_config,
        parameter_dtypes={name: str(p.dtype).removeprefix("torch.")
                          for name, p in model.named_parameters()},
        trainable_mode=model.model.trainable_mode,
        normal_kernel=getattr(model.model, "normal_kernel", "shared"),
        normal_backward=getattr(model.model, "normal_backward", "tilelang"),
        backbone_attention=model.model.base.config._attn_implementation,
    )
    # Avoid constructing a second full model just to serialize an unchanged one.
    wrapper = LimiteForCausalLM.__new__(LimiteForCausalLM)
    PreTrainedModel.__init__(wrapper, config)
    wrapper.policy = model
    config.save_pretrained(root)
    save_safetensors_model(wrapper, root / "model.safetensors")
    _copy_publisher_assets(snapshot, root)
    source = None
    if source_checkpoint is not None:
        source_root = Path(source_checkpoint)
        source = dict(path=str(source_root.resolve()),
                      receipt_sha256=sha256_file(source_root / "COMPLETE.json"))
    receipt = dict(format="archlab-limite-hf-v1", source_checkpoint=source,
                   optimizer_migrated=False, state_sha256=_state_digest(model.state_dict()),
                   files={p.name: sha256_file(p) for p in root.iterdir() if p.is_file()})
    atomic_write_json(root / "EXPORT.json", receipt)
    loaded = LimiteForCausalLM.from_pretrained(root)
    if _state_digest(loaded.policy.state_dict()) != receipt["state_sha256"]:
        (root / "EXPORT.json").unlink()
        raise ValueError("HF export roundtrip changed model weights or dtypes")
    return receipt


# Explicit package extension requested by verl's external_lib setting.
AutoConfig.register(LimiteConfig.model_type, LimiteConfig, exist_ok=True)
AutoModelForCausalLM.register(LimiteConfig, LimiteForCausalLM, exist_ok=True)

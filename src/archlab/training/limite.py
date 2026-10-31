"""Lossless Hugging Face boundary for the publisher Limite model and preludes.

Import this module through verl's ``model.external_lib``. The exported model is
an ordinary AutoModelForCausalLM; its forward and generation remain owned by the
verified publisher. This is a weights migration, not an optimizer migration.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from types import MethodType

import torch
from safetensors.torch import load_model as load_safetensors_model
from safetensors.torch import save_model as save_safetensors_model
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from archlab.architectures.limite_adapter import (
    LimiteAdapterConfig,
    insertion_metadata,
    install_adapters,
    set_normal_attention_backward,
    set_normal_attention_kernel,
    set_trainable_mode,
)
from archlab.architectures.limite_conversion import (
    LimiteConversionConfig,
    conversion_metadata,
    replace_limite_attention,
)
from archlab.architectures.limite_loader import load_model, upstream_classes
from archlab.artifacts import atomic_write_json, sha256_file
from archlab.training.limite_artifacts import verify_export


class LimiteConfig(PretrainedConfig):
    model_type = "archlab_limite"

    def __init__(self, publisher_snapshot="", publisher_config=None, adapter=None,
                 conversion=None,
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
        self.conversion = conversion or {}
        if self.adapter and self.conversion:
            raise ValueError("attention insertion and replacement are distinct experiments")
        self.parameter_dtypes = parameter_dtypes or {}
        self.normal_kernel = normal_kernel
        self.normal_backward = normal_backward
        self.trainable_mode = trainable_mode
        if backbone_attention not in ("sdpa", "flash_attention_2"):
            raise ValueError("unsupported native backbone attention implementation")
        self.backbone_attention = backbone_attention
        self.architectures = ["ArchlabLimiteForCausalLM"]


_ADAPTER_FIELDS = ("variant", "short_window", "seed", "attention_backend",
                   "heads", "feature_dim", "compensation")


def _adapter_config(values):
    return LimiteAdapterConfig(**{k: values[k] for k in _ADAPTER_FIELDS if k in values})


def _same_adapter(constructed, recorded):
    """Exports predating a field omit it; construction then used its default."""
    defaults = asdict(LimiteAdapterConfig())
    return set(recorded) <= set(constructed) and all(
        recorded.get(k, defaults.get(k)) == v for k, v in constructed.items())


class LimiteForCausalLM(PreTrainedModel):
    """Keep the publisher untouched, including mixed parameter dtypes and cache."""

    config_class = LimiteConfig
    base_model_prefix = "policy"
    _no_split_modules = ["LimiteDecoderLayer", "AttentionPrelude", "DeltaPrelude"]
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)
        config_class, model_class = upstream_classes(config.publisher_snapshot)
        native = config_class.from_dict(config.publisher_config)
        native._attn_implementation = config.backbone_attention
        self.policy = model_class(native)
        if config.conversion:
            replace_limite_attention(self.policy, LimiteConversionConfig(**config.conversion))
            if config.trainable_mode != "full":
                raise ValueError("converted agentic bootstrap requires full-weight training")
            layers = self.policy.model.layers
        elif not config.adapter:
            # The unmodified publisher: teacher serving and the no-insertion baseline.
            if config.trainable_mode != "full":
                raise ValueError("an uninserted publisher model trains all of its weights")
            layers = self.policy.model.layers
        else:
            self.policy = install_adapters(self.policy, _adapter_config(config.adapter))
            if not _same_adapter(self.policy.model.adapter_config, config.adapter):
                raise ValueError("exported adapter geometry differs from publisher construction")
            set_normal_attention_kernel(self.policy, config.normal_kernel)
            set_normal_attention_backward(self.policy, config.normal_backward)
            set_trainable_mode(self.policy, config.trainable_mode)
            layers = (*self.policy.model.base.layers, *self.policy.model.adapters)
        self._restore_dtypes()
        self._activation_checkpointing = False
        for layer in layers:
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
        if self.config.adapter:
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
        if self.config.conversion:
            attention_mask, position_ids = self.conversion_inputs(
                input_ids, attention_mask, position_ids,
            )
        return self.policy(input_ids=input_ids, attention_mask=attention_mask,
                           position_ids=position_ids, **kwargs)

    @staticmethod
    def conversion_inputs(input_ids, attention_mask, position_ids):
        """Validate one unpadded document once; kernels own causal/window masks."""
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("conversion training requires one unpadded document per microbatch")
        if attention_mask is not None and (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.shape != input_ids.shape
            or not bool((attention_mask == 1).all())
        ):
            raise ValueError("conversion training does not accept padding or custom masks")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None]
        if position_ids is not None and not torch.equal(position_ids, positions):
            raise ValueError("conversion training requires contiguous document positions")
        return dict(full_attention=None, sliding_attention=None), positions

    def generate(self, *args, **kwargs):
        if self.config.conversion:
            raise ValueError("conversion generation requires a separately qualified cached serving path")
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
    if not _same_adapter(model.model.adapter_config, target_adapter):
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
    # GQA/FA4 kernel choices are TileLang execution settings, not weights; a native
    # retarget (e.g. for serving) uses the native defaults instead of re-applying them.
    retarget_native = target_adapter["attention_backend"] == "native"
    set_normal_attention_kernel(model, "shared" if retarget_native else receipt.get("normal_kernel", "shared"))
    set_normal_attention_backward(
        model, "tilelang" if retarget_native else receipt.get("normal_backward", "tilelang"))
    set_trainable_mode(model, "full")
    return model, receipt


def load_native_checkpoint(snapshot, checkpoint, *, device="cpu"):
    """Load a full-weight native RL checkpoint of the uninserted publisher model."""
    root = Path(checkpoint)
    receipt = json.loads((root / "COMPLETE.json").read_text())
    publisher = json.loads((Path(snapshot) / "DOWNLOAD_VERIFIED.json").read_text())
    identity = receipt.get("publisher_identity", {})
    if receipt.get("model_kind") != "native" or receipt.get("trainable_mode") != "full":
        raise ValueError("expected a full-weight native publisher checkpoint")
    if (identity.get("repo"), identity.get("revision")) != (publisher["repo"], publisher["revision"]):
        raise ValueError("checkpoint belongs to another publisher snapshot")
    if sha256_file(root / "model.pt") != receipt["files"]["model.pt"]:
        raise ValueError("checkpoint checksum mismatch: model.pt")
    model = load_model(snapshot, attn_implementation="sdpa", device_map=device)
    state = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if (set(state) != set(expected)
            or any(value.shape != expected[name].shape or value.dtype != expected[name].dtype
                   for name, value in state.items())):
        raise ValueError("checkpoint tensor inventory, shape or dtype differs from publisher construction")
    model.load_state_dict(state, strict=True)
    return model, receipt


_PUBLISHER_ASSETS = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "added_tokens.json", "chat_template.jinja", "vocab.json", "merges.txt",
                     "generation_config.json"}
_TEMPLATE_ASSETS = ("tokenizer_config.json", "chat_template.jinja")


def _verified_assets(snapshot):
    publisher = Path(snapshot)
    download = json.loads((publisher / "DOWNLOAD_VERIFIED.json").read_text())
    return {item["path"]: (publisher / item["path"], item["sha256"])
            for item in download["files"] if item["path"] in _PUBLISHER_ASSETS}


def _copy_publisher_assets(snapshot, destination, chat_template=None):
    """Copy verified tokenizer assets; ``chat_template`` names a same-tokenizer publisher."""
    root = Path(destination)
    assets = _verified_assets(snapshot)
    if chat_template is not None:
        template = _verified_assets(chat_template)
        if template["tokenizer.json"][1] != assets["tokenizer.json"][1]:
            raise ValueError("chat template publisher uses another tokenizer")
        assets.update({name: template[name] for name in _TEMPLATE_ASSETS})
    copied = {}
    for name, (source, digest) in assets.items():
        if sha256_file(source) != digest:
            raise ValueError("publisher tokenizer checksum mismatch: " + name)
        target = root / name
        if target.exists() and sha256_file(target) != digest:
            raise ValueError("export has conflicting publisher asset: " + name)
        if not target.exists():
            shutil.copyfile(source, target)
        copied[name] = digest
    return copied


def add_publisher_assets(snapshot, destination):
    """Complete an earlier verified weights export without rewriting weights."""
    root = Path(destination)
    receipt = verify_export(root)
    receipt["files"].update(_copy_publisher_assets(snapshot, root))
    atomic_write_json(root / "EXPORT.json", receipt)
    return verify_export(root)


def export_model(model, snapshot, destination, *, source_checkpoint=None, chat_template=None,
                 backbone_attention=None):
    """Export every weight, then reload and compare its exact state digest.

    ``chat_template`` takes the tokenizer config and template of another verified
    publisher with the same tokenizer, e.g. base weights served in Violetto's chat.
    ``backbone_attention`` selects the publisher attention the export is constructed
    with (default: the exported model's); CPU exports cannot load FlashAttention-2.
    """
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=False)
    conversion = getattr(model.model, "conversion_config", None)
    inserted = getattr(model.model, "adapter_config", None)
    config = LimiteConfig(
        publisher_snapshot=str(Path(snapshot).resolve()), publisher_config=model.config.to_dict(),
        adapter=inserted or {},
        conversion=asdict(conversion) if conversion else {},
        parameter_dtypes={name: str(p.dtype).removeprefix("torch.")
                          for name, p in model.named_parameters()},
        trainable_mode=model.model.trainable_mode if inserted else "full",
        normal_kernel=getattr(model.model, "normal_kernel", "shared"),
        normal_backward=getattr(model.model, "normal_backward", "tilelang"),
        backbone_attention=backbone_attention
        or (model.model.base if inserted else model.model).config._attn_implementation,
    )
    # Avoid constructing a second full model just to serialize an unchanged one.
    wrapper = LimiteForCausalLM.__new__(LimiteForCausalLM)
    PreTrainedModel.__init__(wrapper, config)
    wrapper.policy = model
    config.save_pretrained(root)
    save_safetensors_model(wrapper, root / "model.safetensors")
    _copy_publisher_assets(snapshot, root, chat_template)
    source = None
    if source_checkpoint is not None:
        source_root = Path(source_checkpoint)
        source = dict(path=str(source_root.resolve()),
                      receipt_sha256=sha256_file(source_root / "COMPLETE.json"))
    receipt = dict(format="archlab-limite-hf-v1", source_checkpoint=source,
                   optimizer_migrated=False, state_sha256=_state_digest(model.state_dict()),
                   files={p.name: sha256_file(p) for p in root.iterdir() if p.is_file()})
    if conversion:
        receipt["conversion"] = conversion_metadata(model)
    if inserted:
        receipt["insertion"] = insertion_metadata(model)
    if chat_template is not None:
        download = json.loads((Path(chat_template) / "DOWNLOAD_VERIFIED.json").read_text())
        receipt["chat_template"] = dict(repo=download["repo"], revision=download["revision"])
    atomic_write_json(root / "EXPORT.json", receipt)
    loaded = LimiteForCausalLM.from_pretrained(root)
    if _state_digest(loaded.policy.state_dict()) != receipt["state_sha256"]:
        (root / "EXPORT.json").unlink()
        raise ValueError("HF export roundtrip changed model weights or dtypes")
    return receipt


def export_insertion(snapshot, destination, adapter, *, chat_template=None, backbone_attention=None):
    """Export the unmodified publisher with fresh zero-output preludes; only preludes train."""
    model = install_adapters(load_model(snapshot, attn_implementation="sdpa"), adapter)
    receipt = export_model(model, snapshot, destination, chat_template=chat_template,
                           backbone_attention=backbone_attention)
    if receipt["insertion"]["output_state"] != "zero" or receipt["insertion"]["trainable_mode"] != "adapter":
        raise ValueError("a fresh insertion must start at the publisher function with frozen weights")
    return receipt


# Explicit package extension requested by verl's external_lib setting.
AutoConfig.register(LimiteConfig.model_type, LimiteConfig, exist_ok=True)
AutoModelForCausalLM.register(LimiteConfig, LimiteForCausalLM, exist_ok=True)

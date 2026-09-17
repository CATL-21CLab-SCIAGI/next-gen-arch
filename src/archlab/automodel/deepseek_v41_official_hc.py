"""Native HC coefficient rounding on the official V4.1 projection/backward path.

Only each HyperConnection's coefficient-producing forward is extended. Its
FP32 projection, parameters, collapse/expand methods and DeepseekV41Mix output
contract remain official. The verified released TileLang coefficient forward
uses the existing frozen-coefficient PyTorch-equation backward for dInput.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from archlab.automodel.deepseek_v41_autograd import _FrozenNativeMHC
from archlab.automodel.deepseek_v41_runtime import REFERENCE_DIGESTS, load_native_reference

_NATIVE_HC_BY_SOURCE = {}


def _native_hc_from_assets(assets):
    """Reuse the verified native loader without installing any native bridge."""
    inference = Path(assets).resolve(strict=True) / "inference"
    for name, digest in REFERENCE_DIGESTS.items():
        if hashlib.sha256((inference / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"unreviewed native HC source: {name}")
    # Prefix fixtures symlink these exact source files. Import their canonical
    # directory so the verified loader sees the same dependency paths as a
    # later full native-reference import in this process.
    canonical_assets = (inference / "kernel.py").resolve(strict=True).parent.parent
    key = str(canonical_assets)
    if key not in _NATIVE_HC_BY_SOURCE:
        before_dtype, before_device, before_path = torch.get_default_dtype(), torch.get_default_device(), sys.path[:]
        suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
        native = load_native_reference(canonical_assets, module_name=f"_archlab_official_hc_native_{suffix}")
        if (torch.get_default_dtype() != before_dtype or torch.get_default_device() != before_device
                or sys.path != before_path):
            raise RuntimeError("native HC source import changed process dtype/device/import path")
        _NATIVE_HC_BY_SOURCE[key] = native.hc_split_sinkhorn
    return _NATIVE_HC_BY_SOURCE[key]


def _forward_with_native_coefficients(self, hidden_states):
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41Mix

    if hidden_states.ndim != 4 or hidden_states.shape[-2] != self.streams:
        raise ValueError("native HC coefficients require [batch, sequence, streams, hidden]")
    if any(parameter.requires_grad or parameter.dtype != torch.float32
           for parameter in (self.fn, self.scale, self.base)):
        raise ValueError("native HC projection, scale and base must remain frozen FP32")
    with torch.autocast(hidden_states.device.type, enabled=False):
        flat = hidden_states.flatten(2).float()
        mixes = F.linear(flat, self.fn.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        # These 27 scalars are frozen. Independent value copies keep the custom
        # backward's saved coefficients separate from FSDP's unshard storage.
        scale, base = self.scale.detach().clone(), self.base.detach().clone()
        geometry = self.streams, self.iterations, self.eps
        if torch.is_grad_enabled() and mixes.requires_grad:
            values = _FrozenNativeMHC.apply(mixes, scale, base, *geometry, self._archlab_native_hc)
        else:
            values = self._archlab_native_hc(mixes, scale, base, *geometry)
    if any(value.dtype != torch.float32 for value in values):
        raise TypeError("native HC must return FP32 carried coefficients")
    return DeepseekV41Mix(*values)


def install_official_native_hc(model, assets):
    """Attach once after frozen official loading/FSDP, before the first forward."""
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection

    if getattr(model, "_archlab_v41_native_hc_installed", False):
        raise ValueError("official native HC coefficient precision is already installed")
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, DeepseekV41HyperConnection)]
    if not selected:
        raise ValueError("no official V4.1 HyperConnections found")
    for name, module in selected:
        if (module.streams != 4 or module.iterations != 20 or module.eps != 1e-6
                or module.fn.shape[0] != 24 or module.scale.shape != (3,) or module.base.shape != (24,)):
            raise ValueError(f"{name}: HC geometry differs from the reviewed V4.1 coefficients")
        if any(parameter.requires_grad or parameter.dtype != torch.float32
               for parameter in (module.fn, module.scale, module.base)):
            raise ValueError(f"{name}: freeze the FP32 HC projection/scale/base before installation")
        if "forward" in module.__dict__ or hasattr(module, "_archlab_native_hc"):
            raise ValueError(f"{name}: HC forward has already been extended")
    native = _native_hc_from_assets(assets)
    original = dict(model.named_parameters())
    for _, module in selected:
        module._archlab_native_hc = native
        module.forward = MethodType(_forward_with_native_coefficients, module)
    after = dict(model.named_parameters())
    if original.keys() != after.keys() or any(after[name] is not parameter for name, parameter in original.items()):
        raise RuntimeError("native HC installation changed an original parameter or checkpoint key")
    model._archlab_v41_native_hc_installed = True
    return {"implementation": "project-native-hc-coefficients-on-official-projection-v1",
            "modules": [name for name, _ in selected], "projection_dtype": "float32",
            "coefficients_dtype": "float32", "forward": "verified-released-hc_split_sinkhorn",
            "backward": "existing-frozen-native-MHC-PyTorch-equations-first-order",
            "native_source_sha256": dict(REFERENCE_DIGESTS), "original_parameters_preserved": True}

"""Optional official FA4 backward at Limite's explicit native BF16 boundary.

The generic GQA FP32-output interface is unchanged. Only the qualified B300
B8/N2048/H10/KV2/D128 reduction uses FA4; every other geometry delegates to
the existing TileLang backward. Official packages remain external and pinned.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import sys
from functools import cache
from pathlib import Path

import torch

_VERSIONS = {
    "flash-attn-4": "4.0.0b33",
    "nvidia-cutlass-dsl": "4.6.2",
    "apache-tvm-ffi": "0.1.12",
    "quack-kernels": "0.5.3",
    "tilelang": "0.1.15",
}
_UPSTREAM_HASHES = {
    "interface.py": "f498a42080b5c1e475b976ef163e78c90d72ad938f69b54df28d6345118982ed",
    "flash_bwd_sm100.py": "2c2c55e0930fcdaa8a47863c36c7e20fd0261db9a8f693bd8e633c6743671d10",
    "flash_bwd_preprocess.py": "36d4b63b857a9538ea62a96f43ff9c3d429bc8683f6cc8149592376b7a4a0265",
    "flash_bwd_postprocess.py": "c9477532c88fd5695fe253365708cd7ec120b94fd9c3a80801d56a275eb8b051",
}


def _installed_contract():
    versions, distributions, errors = {}, {}, []
    for name, expected in _VERSIONS.items():
        try:
            distribution = metadata.distribution(name)
            distributions[name] = distribution
            versions[name] = distribution.version
            if distribution.version != expected:
                errors.append(f"{name} requires {expected}, found {distribution.version}")
        except metadata.PackageNotFoundError:
            versions[name] = None
            errors.append(f"{name} requires installed version {expected}")
    paths, hashes = {}, {}
    if "flash-attn-4" in distributions:
        cute_root = Path(distributions["flash-attn-4"].locate_file("flash_attn/cute")).resolve()
        paths["flash_attn_cute"] = str(cute_root)
        for filename, expected in _UPSTREAM_HASHES.items():
            path = cute_root / filename
            if not path.is_file():
                errors.append(f"missing official FA4 source {filename}")
                continue
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            if hashes[filename] != expected:
                errors.append(f"official FA4 source hash differs: {filename}")
        for name, module in tuple(sys.modules.items()):
            if name == "flash_attn.cute" or name.startswith("flash_attn.cute."):
                filename = getattr(module, "__file__", None)
                if filename is not None and not Path(filename).resolve().is_relative_to(cute_root):
                    errors.append(f"incompatible FA4 module already imported: {name}")
    for name, package, relative in (
        ("nvidia-cutlass-dsl", "cutlass", "nvidia_cutlass_dsl/dsl_packages/cutlass/__init__.py"),
        ("apache-tvm-ffi", "tvm_ffi", "tvm_ffi/__init__.py"),
        ("quack-kernels", "quack", "quack/__init__.py"),
        ("tilelang", "tilelang", "tilelang/__init__.py"),
    ):
        if name not in distributions:
            continue
        expected_path = Path(distributions[name].locate_file(relative)).resolve()
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
        paths[package] = str(origin) if origin is not None else None
        if origin != expected_path:
            errors.append(f"{package} import path must be {expected_path}, found {origin}")
    contract = {
        "implementation": "fa4-native-bf16-preserved-forward-v3",
        "required_versions": dict(_VERSIONS),
        "versions": versions,
        "resolved_paths": paths,
        "upstream_sha256": hashes,
        "required_upstream_sha256": dict(_UPSTREAM_HASHES),
        "scope": "B300 BF16 B8/N2048/Q10/KV2/D128 scale0.1 window1025/2048 only",
        "forward": "exact shared TileLang FP32 output and log2 LSE; explicit BF16 return",
        "backward": "raw FP32 Delta; scale and q0 guard before BF16 dS; unscaled conversion",
        "fallback": "original qualified GQA for generic FP32/short/W1/tails/long RL",
        "upstream": "https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute",
        "errors": errors,
        "validated": False,
    }
    return contract


def fa4_runtime_contract(*, validate=True):
    """Inspect exact dependencies; optionally select their official namespace.

    The FA4 wheel installs ``flash_attn/cute`` without replacing the container's
    regular ``flash_attn`` package. Selecting the metadata-owned namespace is
    deliberate; an already imported incompatible CuTe module is rejected.
    Cutlass's installed ``dsl_packages`` must be on the launch-time import path.
    No runtime package file or function is changed.
    """
    contract = _installed_contract()
    if not validate:
        return contract
    if contract["errors"]:
        raise RuntimeError("FA4 runtime contract: " + "; ".join(contract["errors"]))
    namespace = importlib.import_module("flash_attn")
    root = str(Path(contract["resolved_paths"]["flash_attn_cute"]).parent)
    paths = [str(path) for path in namespace.__path__]
    if not paths or paths[0] != root:
        namespace.__path__ = [root] + [path for path in paths if path != root]
    contract["resolved_paths"]["flash_attn_namespace"] = namespace.__file__
    contract["validated"] = True
    return contract


def _qualified_gqa(q, k, v, *, scaling, long_window, forward):
    from archlab.architectures.tilelang_gqa import gqa_attention

    return gqa_attention(q, k, v, scaling=scaling, long_window=long_window, forward=forward)


@cache
def _native_function():
    fa4_runtime_contract(validate=True)
    from archlab.architectures._fa4_native import NativeBF16Attention

    return NativeBF16Attention


def native_bf16_gqa_attention(q, k, v, *, scaling, long_window, forward):
    """Return the native BF16 reduction, retaining a raw FP32 saved output.

    The explicit output dtype ensures incoming autograd dO is BF16. A generic
    FP32-output caller must continue to use ``gqa_attention`` directly.
    """
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("native FA4/TileLang attention backward uses FP32 atomics")
    supported = (
        q.is_cuda
        and all(value.dtype == torch.bfloat16 and value.device == q.device for value in (q, k, v))
        and tuple(q.shape) == (8, 2048, 10, 128)
        and tuple(k.shape) == tuple(v.shape) == (8, 2048, 2, 128)
        and scaling == 0.1
        and long_window in (1025, 2048)
    )
    if supported:
        from archlab.architectures.tilelang_attention import normal_attention_forward

        supported = (
            forward is normal_attention_forward
            and torch.cuda.get_device_capability(q.device) == (10, 3)
        )
    if not supported:
        return _qualified_gqa(q, k, v, scaling=scaling, long_window=long_window, forward=forward).to(v.dtype)
    return _native_function().apply(q, k, v, long_window, scaling, forward)

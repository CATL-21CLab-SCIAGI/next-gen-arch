"""Triadic Gated DeltaNet: official GPU dependency and an independent oracle.

The joint key is k2_t outer k_t at one token, with learned feature slices E.
This differs from LinSimp's independent long token j and short anchor c: no
random features or token-pair softmax are used here. Slice decay precedes one
joint delta correction, so the erase term reads all slices together.

GPU execution calls the unmodified MIT-licensed cute-triadic-gdn dependency.
The small differentiable recurrence below is a numerical oracle, not a GPU
training implementation. Runtime packages remain owned by the container.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from functools import cache
from importlib import metadata
from pathlib import Path

import torch
from torch.nn import functional as F

OFFICIAL_REPOSITORY = "https://github.com/OliverSieberling/cute-triadic-gdn"
OFFICIAL_COMMIT = "2caa4098073da92c2ba0d573df6453ceb7dbce45"
OFFICIAL_PACKAGE_VERSION = "1.0.0"
OFFICIAL_SOURCE_SHA256 = "512c5c813e74c5b75be6d392e6ac6d517178d9469812924613a5dd2f7553a0d8"
OFFICIAL_DSL_VERSION = "4.7.1"
SUPPORTED_FEATURE_DIMS = (1, 2, 4, 8, 12, 16)


@cache
def _official_runtime_identity():
    """Metadata alone cannot prove which namespace package Python imported."""
    library_packages = (
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    )
    try:
        versions = {name: metadata.version(name) for name in library_packages}
        cutlass = importlib.import_module("cutlass")
    except ImportError as error:
        raise RuntimeError("The complete isolated CUTLASS DSL 4.7.1 runtime is required") from error
    imported_version = getattr(cutlass, "__version__", None)
    source = Path(cutlass.__file__).resolve()
    if imported_version != OFFICIAL_DSL_VERSION or any(
        version != OFFICIAL_DSL_VERSION for version in versions.values()
    ):
        raise RuntimeError(
            f"Expected actual CUTLASS DSL {OFFICIAL_DSL_VERSION}; imported "
            f"{imported_version} from {source}, package metadata {versions}"
        )
    return {
        "imported_version": imported_version,
        "source_file": str(source),
        "source_init_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "distribution_versions": versions,
        "distribution_locations": {
            name: str(metadata.distribution(name).locate_file("").resolve())
            for name in library_packages
        },
    }


@cache
def _official_package():
    _official_runtime_identity()
    try:
        module = importlib.import_module("cute_triadic_gdn")
    except ImportError as error:
        raise RuntimeError(
            "Install the pinned cute-triadic-gdn dependency in the container "
            f"from {OFFICIAL_REPOSITORY}@{OFFICIAL_COMMIT}; no reference fallback is allowed"
        ) from error
    root = Path(module.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(
            str(path.relative_to(root)).encode()
            + b"\0"
            + hashlib.sha256(path.read_bytes()).hexdigest().encode()
            + b"\n"
        )
    if (
        module.__version__ != OFFICIAL_PACKAGE_VERSION
        or digest.hexdigest() != OFFICIAL_SOURCE_SHA256
    ):
        raise RuntimeError("cute-triadic-gdn source differs from the pinned dependency")
    return module


def official_dependency_contract():
    """Validate installed source once and expose provenance for the run receipt."""
    _official_package()
    return {
        "repository": OFFICIAL_REPOSITORY,
        "commit": OFFICIAL_COMMIT,
        "version": OFFICIAL_PACKAGE_VERSION,
        "source_sha256": OFFICIAL_SOURCE_SHA256,
        "license": "MIT",
        "kernel": "gdn_joint_call",
        "convolution": "conv_split_act_call",
        "resolved_cutlass_dsl": _official_runtime_identity(),
        "runtime_requirements": [
            "Python>=3.12",
            "torch>=2.8",
            "nvidia-cutlass-dsl==4.7.1",
            "cuda-python",
            "triton",
        ],
    }


def _validate(q, k, v, k2, q2, g, beta):
    if q.ndim != 4 or min(q.shape) < 1:
        raise ValueError("q must be nonempty [batch, sequence, heads, key_dim]")
    batch, length, heads, _ = q.shape
    if k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("q/k must match and v must share batch/sequence/heads")
    if k2.ndim != 4 or k2.shape[:3] != (batch, length, heads) or k2.shape[-1] < 1:
        raise ValueError("k2 must be [batch, sequence, heads, feature_dim]")
    if q2.shape != k2.shape or g.shape != k2.shape or beta.shape != q.shape[:3]:
        raise ValueError("q2/g must match k2 and beta must be [batch, sequence, heads]")
    if any(t.device != q.device or not t.is_floating_point() for t in (q, k, v, k2, q2, g, beta)):
        raise ValueError("all inputs must be floating tensors on one device")


def _document_offsets(cu_seqlens, batch, length):
    if cu_seqlens is None:
        return [0, length]
    if batch != 1 or cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError("packed documents require one row and one-dimensional integer offsets")
    offsets = cu_seqlens.tolist()
    if (
        len(offsets) < 2
        or offsets[0] != 0
        or offsets[-1] != length
        or any(left >= right for left, right in zip(offsets, offsets[1:], strict=False))
    ):
        raise ValueError("document offsets must strictly increase from zero to sequence length")
    return offsets


def triadic_gdn_reference(q, k, v, k2, q2, g, beta, *, scale=None, cu_seqlens=None):
    """Direct per-token slice recurrence, preserving FP64 inputs for the oracle.

    S_e <- exp(g_e) S_e
    residual <- v - sum_e k2_e k^T S_e
    S_e <- S_e + beta k2_e k residual^T
    o <- scale sum_e q2_e q^T S_e

    Unlike separate E delta updates, one residual erases the joint memory.
    Packed document boundaries reset every slice before the next token.
    """
    _validate(q, k, v, k2, q2, g, beta)
    batch, length, heads, dim = q.shape
    offsets = _document_offsets(cu_seqlens, batch, length)
    starts = set(offsets[:-1])
    dtype = (
        torch.float64
        if any(t.dtype == torch.float64 for t in (q, k, v, k2, q2, g, beta))
        else torch.float32
    )
    output_dtype = q.dtype
    q, k, v, k2, q2, g, beta = [t.to(dtype) for t in (q, k, v, k2, q2, g, beta)]
    multiplier = dim**-0.5 if scale is None else scale
    if not math.isfinite(multiplier):
        raise ValueError("read scale must be finite")
    state = q.new_zeros(batch, heads, k2.shape[-1], dim, v.shape[-1])
    outputs = []
    for index in range(length):
        if index in starts:
            state = torch.zeros_like(state)
        state = state * g[:, index].exp()[..., None, None]
        predicted = torch.einsum("bhd,bhedv,bhe->bhv", k[:, index], state, k2[:, index])
        residual = v[:, index] - predicted
        state = state + torch.einsum(
            "bh,bhe,bhd,bhv->bhedv", beta[:, index], k2[:, index], k[:, index], residual
        )
        outputs.append(
            multiplier * torch.einsum("bhd,bhedv,bhe->bhv", q[:, index], state, q2[:, index])
        )
    return torch.stack(outputs, dim=1).to(output_dtype)


def triadic_gdn_attention(
    q, k, v, k2, q2, g, beta, *, backend="official", scale=None, cu_seqlens=None
):
    """Explicit backend dispatch; production never silently falls back to Torch."""
    _validate(q, k, v, k2, q2, g, beta)
    if backend == "reference":
        return triadic_gdn_reference(q, k, v, k2, q2, g, beta, scale=scale, cu_seqlens=cu_seqlens)
    if backend != "official":
        raise ValueError("Triadic backend must be official or reference")
    if not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] not in (9, 10):
        raise ValueError("official Triadic kernels require Hopper or datacenter Blackwell")
    if q.shape[-1] != 128 or v.shape[-1] != 128 or k2.shape[-1] not in SUPPORTED_FEATURE_DIMS:
        raise ValueError("official Triadic kernels require D=V=128 and supported feature_dim")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("official Triadic q/k/v must be BF16")
    if any(t.dtype != torch.float32 for t in (k2, q2, g, beta)):
        raise ValueError("official Triadic second axes, gates and beta must be FP32")
    if cu_seqlens is None and q.shape[1] % 64:
        raise ValueError("unpacked official Triadic sequence length must be a multiple of 64")
    return _official_package().gdn_joint_call(
        q, k, v, k2, q2, g, beta, scale=scale, cu_seqlens=cu_seqlens
    )


def causal_conv_reference(x, weight, act_channels, *, cu_seqlens=None):
    """CPU/small-oracle depthwise causal convolution with packed resets."""
    if x.ndim != 3 or weight.ndim != 2 or weight.shape[0] != x.shape[-1]:
        raise ValueError("convolution needs [B,T,C] inputs and [C,W] weights")
    if not 0 <= act_channels <= x.shape[-1] or weight.shape[1] < 1:
        raise ValueError("invalid activation channels or convolution width")
    offsets = _document_offsets(cu_seqlens, x.shape[0], x.shape[1])
    segments = []
    for left, right in zip(offsets, offsets[1:], strict=False):
        segment = F.pad(x[:, left:right].transpose(1, 2), (weight.shape[1] - 1, 0))
        segment = F.conv1d(segment, weight.unsqueeze(1), groups=x.shape[-1]).transpose(1, 2)
        segments.append(
            torch.cat((F.silu(segment[..., :act_channels]), segment[..., act_channels:]), -1)
        )
    return torch.cat(segments, dim=1)


def causal_conv_attention(x, weight, act_channels, *, backend="official", cu_seqlens=None):
    if backend == "reference":
        return causal_conv_reference(x, weight, act_channels, cu_seqlens=cu_seqlens)
    if backend != "official" or not x.is_cuda:
        raise ValueError("official causal convolution requires CUDA; select reference explicitly")
    return _official_package().conv_split_act_call(x, weight, act_channels, cu_seqlens)

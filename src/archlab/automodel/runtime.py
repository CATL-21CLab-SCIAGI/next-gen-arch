"""Narrow in-process compatibility settings for the frozen DLC runtime.

Reuse the upstream FLA restrictions, not replacement kernels:
https://github.com/fla-org/flash-linear-attention/pull/953
https://github.com/fla-org/flash-linear-attention/pull/1000
No installed source files or compiled kernel implementations are edited.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
from pathlib import Path
import socket
import subprocess

import torch


def restrict_autotuner(kernel, *, num_warps: int, num_stages: int | None = None) -> dict:
    """Select existing configurations and invalidate only the in-process cache."""
    tuner = kernel
    while not hasattr(tuner, "configs"):
        if not hasattr(tuner, "fn"):
            raise RuntimeError("unsupported FLA autotuner wrapper")
        tuner = tuner.fn
    before = len(tuner.configs)
    configs = [c for c in tuner.configs if c.num_warps == num_warps
               and (num_stages is None or c.num_stages == num_stages)]
    if not configs:
        raise RuntimeError("upstream safe FLA configurations are absent")
    tuner.configs = configs
    tuner.cache.clear()
    # Never reload a previously benchmarked unsafe selection from disk.
    tuner.cache_results = False
    return {"before_count": before, "after_count": len(configs),
            "configs": [{"kwargs": c.kwargs, "num_warps": c.num_warps, "num_stages": c.num_stages}
                        for c in configs]}


def configure_frozen_gdn_runtime() -> dict:
    """Apply FLA's Blackwell fixes to the audited container at process startup."""
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        return {"applied": False, "capability": capability}
    version = importlib.metadata.version("fla-core")
    if version != "0.4.2":
        raise RuntimeError(f"FLA {version} has not been audited for this runtime restriction")
    specs = (("fla.ops.common.chunk_delta_h", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", None),
             ("fla.ops.gated_delta_rule.wy_fast", "prepare_wy_repr_bwd_kernel", 4))
    evidence = {}
    for module_name, kernel_name, stages in specs:
        module = importlib.import_module(module_name)
        evidence[kernel_name] = {
            "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            **restrict_autotuner(getattr(module, kernel_name), num_warps=2, num_stages=stages),
        }
    return {"applied": True, "capability": capability, "fla_version": version,
            "upstream_restrictions": [953, 1000], "kernels": evidence}


def runtime_provenance(*, ep_size: int) -> dict:
    """Audit the source pin, installed versions and actual node-local EP topology."""
    import nemo_automodel
    import torch.distributed as dist
    from archlab.automodel.simplicial import UPSTREAM_COMMIT

    root = Path(nemo_automodel.__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != UPSTREAM_COMMIT or dirty:
        raise RuntimeError("the qualified, unmodified AutoModel source pin is required")
    image = os.environ.get("NGA_CONTAINER_DIGEST")
    if not image:
        raise RuntimeError("record NGA_CONTAINER_DIGEST (image identity) before launching")
    hosts = [None] * dist.get_world_size()
    dist.all_gather_object(hosts, socket.gethostname())
    if any(len(set(hosts[i:i + ep_size])) != 1 for i in range(0, len(hosts), ep_size)):
        raise ValueError("each EP group must be entirely within one existing node")
    source = Path(__file__).resolve().parent
    return {"upstream": revision, "container_image": image,
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "transformer-engine", "megatron-core", "fla-core", "triton")},
            "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(), "rank_hosts": hosts,
            "project_source_sha256": {str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted([*source.glob("*.py"),
                                                       *source.parent.joinpath("architectures").glob("*.py")])}}

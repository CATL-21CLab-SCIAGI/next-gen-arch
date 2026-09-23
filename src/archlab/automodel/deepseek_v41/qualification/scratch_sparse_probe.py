"""Bounded sparse-kernel correctness checks, independent of training replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    resolved_kernel_packages = select_container_kernel_packages(
        Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"])
    )
    import torch
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention

    from archlab.automodel.deepseek_v41_official_sparse import deterministic_sparse_attention
    from archlab.automodel.deepseek_v41_scratch_high_mfu import high_mfu_sparse_attention

    torch.set_num_threads(1)
    torch.manual_seed(703)
    results = []
    for slots in (17, 65, 512):
        batch, length, heads, dim = 2, 33, 64, 64
        q = torch.randn(batch, length, heads, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(batch, 45, dim, device="cuda", dtype=torch.bfloat16)
        sinks = torch.randn(heads, device="cuda", dtype=torch.float32)
        indices = torch.randint(-1, 45, (batch, length, slots), device="cuda", dtype=torch.int32)
        indices[:, -1] = -1
        indices[:, 0, 1:] = -1
        indices[:, 0, 0] = 0
        indices[:, 2, :4] = 1  # duplicate indices exercise shared-KV accumulation
        dout = torch.randn_like(q)

        def reference(x, k, s, *, batch=batch, indices=indices, dim=dim, length=length):
            selected = k[
                torch.arange(batch, device="cuda")[:, None, None], indices.clamp_min(0).long()
            ]
            logits = torch.einsum("bthd,btkd->bthk", x, selected) * dim**-0.5
            logits = logits.masked_fill(indices[:, :, None, :] < 0, -torch.inf)
            probs = torch.cat(
                (logits, s[None, None, :, None].expand(batch, length, -1, -1)), -1
            ).softmax(-1)[..., :-1]
            return torch.einsum("bthk,btkd->bthd", probs, selected)

        versions = {}
        for name, fn in [
            ("fp32", reference),
            ("official", dsv4_sparse_attention),
            ("deterministic", deterministic_sparse_attention),
            ("batched", high_mfu_sparse_attention),
            ("batched_repeat", high_mfu_sparse_attention),
        ]:
            x, k, s = [
                t.detach().clone().to(torch.float32 if name == "fp32" else t.dtype).requires_grad_()
                for t in (q, kv, sinks)
            ]
            if name == "fp32":
                output = fn(x, k, s)
            else:
                output = fn(
                    x, k, s, indices, dim**-0.5, backend="tilelang", reference_rounding=True
                )
            gradients = torch.autograd.grad(output, (x, k, s), dout.to(output.dtype))
            versions[name] = [output.detach(), *gradients]
        metrics = {}
        for name, values in versions.items():
            metrics[name] = {}
            for field, actual, expected in zip(
                ("output", "dq", "dkv", "dsink"), values, versions["fp32"]
            , strict=False):
                if not bool(actual.isfinite().all()):
                    raise AssertionError(f"{name}/{field} is nonfinite")
                relative = float(
                    (actual.float() - expected).norm() / expected.norm().clamp_min(1e-20)
                )
                metrics[name][field + "_relative_l2_to_fp32"] = relative
                if relative > 0.025:
                    raise AssertionError(f"{name}/{field} relative error {relative} exceeds .025")
        if not torch.equal(versions["official"][0], versions["batched"][0]):
            raise AssertionError("batched forward differs from native inference forward")
        for actual in (versions["batched"][0][:, -1], versions["batched"][1][:, -1]):
            if bool(actual.count_nonzero()):
                raise AssertionError("all-invalid row must have zero output/query gradient")
        metrics["batched_repeat"]["exact_equal"] = {
            name: torch.equal(a, b)
            for name, a, b in zip(
                ("output", "dq", "dkv", "dsink"), versions["batched"], versions["batched_repeat"]
            , strict=False)
        }
        results.append({"shape": [batch, length, heads, dim], "slots": slots, "metrics": metrics})
    report = {
        "passed": True,
        "cases": results,
        "hardware": "NVIDIA B300 (NVML label L20D)",
        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
        "resolved_kernel_packages": resolved_kernel_packages,
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "tilelang", "triton")
        },
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()

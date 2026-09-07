"""DSW preflight for exact-shape simplicial attention, without runtime changes.

Writes partial evidence after each gate. Never starts model training. Use an
unoccupied GPU when interpreting latency; concurrent services are recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from archlab.architectures.simplicial_attention import (
    reference_simplicial,
    simplicial_attention,
)


def _command(*args):
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    return {"returncode": result.returncode, "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()}


def _error(actual, expected):
    difference = actual.detach().float() - expected.detach().float()
    return {"max_absolute": difference.abs().max().item(),
            "relative_l2": (difference.norm() / expected.float().norm().clamp_min(1e-12)).item()}


def correctness_case(n, w1, w2, dtype, positions=None, batch=1, head_dim=32):
    torch.manual_seed(42)
    inputs = [torch.randn(batch, n, h, head_dim, device="cuda", dtype=dtype, requires_grad=True)
              for h in (24, 2, 2, 2, 2)]
    # FP64 oracle avoids silently evaluating a purported FP32 reference with
    # the host's default TF32 matmul mode.
    reference = [x.detach().double().requires_grad_() for x in inputs]
    all_outputs = simplicial_attention(*inputs, w1, w2)
    actual = all_outputs if positions is None else all_outputs[:, positions]
    upstream = torch.randn_like(actual)
    expected = reference_simplicial(*reference, w1, w2, query_positions=positions)
    actual.backward(upstream)
    expected.backward(upstream.double())
    errors = {"output": _error(actual, expected)}
    errors.update({name: _error(x.grad, ref.grad)
                   for name, x, ref in zip(("q", "k1", "k2", "v1", "v2"), inputs, reference)})
    # One-pair windows have exactly zero score gradients; relative error is
    # meaningless at zero, so use an absolute floor for those entries.
    tolerance = 0.025 if dtype == torch.bfloat16 else 0.0001
    passed = all(e["relative_l2"] <= tolerance or e["max_absolute"] <= tolerance * 0.01
                 for e in errors.values())
    if not all(torch.isfinite(x.grad).all().item() for x in inputs):
        passed = False
    # Future perturbation must not change earlier outputs on either key axis.
    split = max(1, n // 2)
    causal = True
    for branch in (1, 2, 3, 4):
        altered = [x.detach().clone() for x in inputs]
        altered[branch][:, split:] += 10
        observed = simplicial_attention(*altered, w1, w2)
        causal = causal and torch.equal(observed[:, :split], all_outputs.detach()[:, :split])
    return {"batch": batch, "sequence": n, "head_dim": head_dim,
            "windows": [w1, w2], "dtype": str(dtype), "oracle_dtype": "float64",
            "query_positions": positions, "errors": errors, "relative_l2_limit": tolerance,
            "future_perturbation_exact": causal, "passed": passed and causal}


def benchmark_case(kind, window, *, sequence, batch, repetitions):
    torch.manual_seed(42)
    inputs = [torch.randn(batch, sequence, h, 32, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True) for h in (24, 2, 2, 2, 2)]
    q, k1, k2, v1, v2 = inputs
    local_mask = None
    implementation = "project-stock-triton"
    if kind == "local":
        indices = torch.arange(sequence, device="cuda")
        distance = indices[:, None] - indices[None, :]
        local_mask = (distance >= 0) & (distance < window[1])
    flash = None
    if kind.startswith("flash"):
        from flash_attn import flash_attn_func

        flash = flash_attn_func
        implementation = "container-flash-attn"
    elif kind != "simplicial":
        implementation = "pytorch-sdpa-default-dispatch"

    def forward():
        if kind == "simplicial":
            return simplicial_attention(*inputs, *window)
        if flash is not None:
            return flash(q, k1, v1, causal=True,
                         window_size=(-1, -1) if kind == "flash-global" else (window[1] - 1, 0))
        # No projection/normalization/gating time is included in any case.
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k1.transpose(1, 2), v1.transpose(1, 2),
            attn_mask=local_mask, is_causal=kind == "global", enable_gqa=True,
        ).transpose(1, 2)

    upstream = torch.randn_like(q)

    def step():
        for x in inputs:
            x.grad = None
        forward().backward(upstream)

    # Compile/warm up before collecting latency or peak allocation.
    for _ in range(4):
        with torch.no_grad():
            forward()
        step()
    torch.cuda.synchronize()
    forward_ms, training_ms = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repetitions):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        with torch.no_grad():
            forward()
        end.record()
        end.synchronize()
        forward_ms.append(begin.elapsed_time(end))
        begin.record()
        step()
        end.record()
        end.synchronize()
        training_ms.append(begin.elapsed_time(end))
    return {"kind": kind, "implementation": implementation, "windows": window,
            "batch": batch, "sequence": sequence, "query_heads": 24, "kv_heads": 2,
            "head_dim": 32, "forward_ms": forward_ms, "forward_backward_ms": training_ms,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "scope": "attention core only; no projections, optimizer, or model training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--include-flash", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    if min(args.sequence, args.batch, args.repetitions) < 1:
        raise ValueError("benchmark dimensions and repetitions must be positive")
    torch.cuda.set_device(0)
    # Protect the co-resident service from an accidental large allocation.
    torch.cuda.set_per_process_memory_fraction(0.25)
    import triton

    packages = {}
    for package in ("torch", "triton", "transformer-engine", "flash-attn", "apex"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "absent"
    source_root = Path(__file__).parents[1]
    source_hashes = {str(path.relative_to(source_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (Path(__file__), source_root / "architectures/simplicial_attention.py",
                                  source_root / "architectures/simplicial_kernels.py")}
    evidence = {"status": "running", "host": socket.gethostname(),
                "python": sys.executable, "python_version": platform.python_version(),
                "torch_version": torch.__version__, "triton_version": triton.__version__,
                "cuda_version": torch.version.cuda, "packages": packages,
                "compiler_target": str(triton.runtime.driver.active.get_current_target()),
                "container": {key: os.environ.get(key) for key in
                              ("NVIDIA_PRODUCT_NAME", "NVIDIA_BUILD_ID", "NVIDIA_PYTORCH_VERSION", "CUDA_VERSION")},
                "gpu": str(torch.cuda.get_device_properties(0)),
                "gpu_processes_before": _command("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"),
                "git_commit": _command("git", "rev-parse", "HEAD"),
                "source_sha256": source_hashes, "started_unix": time.time(),
                "runtime_mutations": False, "training_started": False,
                "correctness": [], "benchmarks": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(evidence, indent=2) + "\n")
        temporary.replace(args.output)

    save()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for n, w1, w2 in ((1, 1, 1), (17, 1, 1), (17, 3, 8), (35, 16, 32), (65, 8, 128), (529, 32, 512)):
                positions = [0, 31, 511, 512, 528] if n == 529 else None
                result = correctness_case(n, w1, w2, dtype, positions, batch=2 if n == 17 else 1)
                evidence["correctness"].append(result)
                save()
                print(json.dumps(result), flush=True)
                if not result["passed"]:
                    raise RuntimeError("simplicial numerical gate failed; no performance or training launch")
        if not args.correctness_only:
            cases = [("global", [0, 0])]
            if args.include_flash:
                cases.append(("flash-global", [0, 0]))
            for w in ([8, 64], [16, 128], [32, 512]):
                cases.extend([("local", w), ("simplicial", w)])
                if args.include_flash:
                    cases.append(("flash-local", w))
            for kind, window in cases:
                result = benchmark_case(kind, window, sequence=args.sequence,
                                        batch=args.batch, repetitions=args.repetitions)
                evidence["benchmarks"].append(result)
                save()
                print(json.dumps(result), flush=True)
        evidence["status"] = "passed"
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        evidence["ended_unix"] = time.time()
        evidence["gpu_processes_after"] = _command("nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader")
        save()


if __name__ == "__main__":
    main()

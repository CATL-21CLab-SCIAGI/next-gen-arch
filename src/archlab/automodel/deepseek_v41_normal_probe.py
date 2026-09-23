"""Full-context numerical oracle and repeatability for the normal adapter core."""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages
    select_container_kernel_packages(args.container_kernel_packages)
    import torch
    from archlab.artifacts import atomic_write_json
    from archlab.architectures.local_attention import deterministic_local_attention, reference_local_attention
    from archlab.automodel.deepseek_v41_official_execution import configure_official_reproducibility, runtime_identity

    configure_official_reproducibility()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(2234)
    sequence = 16384
    positions = [0, 1, 31, 511, 512, 2047, 8191, 16383]
    q = torch.randn(1, sequence, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, sequence, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    inputs = (q, k, v)
    references = tuple(x.detach().float().requires_grad_() for x in inputs)
    grad = torch.randn(1, len(positions), 8, 128, device="cuda", dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    output = deterministic_local_attention(*inputs, 512)
    expected = reference_local_attention(*references, 512, query_positions=positions)
    gradients = torch.autograd.grad(output[:, positions], inputs, grad)
    oracle_gradients = torch.autograd.grad(expected, references, grad.float())
    report = {**runtime_identity(), "hardware": "NVIDIA B300 (NVML L20D mislabel)",
              "sequence": sequence, "query_heads": 8, "kv_heads": 2, "head_dim": 128,
              "window_including_current": 512, "query_positions": positions, "oracle": {}}
    for name, measured, oracle in zip(("output", "dq", "dk", "dv"),
                                     (output[:, positions], *gradients), (expected, *oracle_gradients), strict=True):
        relative = float((measured.float() - oracle).norm() / oracle.norm())
        report["oracle"][name] = {"relative_l2": relative,
                                    "max_abs_error": float((measured.float() - oracle).abs().max())}
        if relative >= .008 or not bool(measured.isfinite().all()):
            raise AssertionError(f"{name} exceeds the 0.8% BF16 oracle envelope: {relative}")
    for _ in range(2):
        repeated = deterministic_local_attention(*inputs, 512)
        repeated_gradients = torch.autograd.grad(repeated[:, positions], inputs, grad)
        for a, b in zip((output, *gradients), (repeated, *repeated_gradients), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    report["exact_repeats"] = 2
    full_grad = torch.randn_like(output)
    for _ in range(3):
        torch.autograd.grad(deterministic_local_attention(*inputs, 512), inputs, full_grad)
    times = []
    for _ in range(8):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.autograd.grad(deterministic_local_attention(*inputs, 512), inputs, full_grad)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    report.update(forward_backward_seconds=times, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  passed=True)
    atomic_write_json(args.output, report, allow_nan=False)
    print(report, flush=True)


if __name__ == "__main__":
    main()

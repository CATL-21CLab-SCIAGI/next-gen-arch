"""Compare deterministic private-query sparse backward with official kernels."""

import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages
    kernels = select_container_kernel_packages(args.container_kernel_packages)
    import torch
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_official_sparse import deterministic_sparse_attention
    from archlab.automodel.deepseek_v41_moe_rounding_probe import _compare

    if args.output.exists():
        raise ValueError("use a fresh probe output")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(839)
    q = torch.randn(1, args.context, args.heads, args.dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, args.context, args.dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(args.heads, device="cuda", dtype=torch.float32)
    count = min(512, args.context)
    positions = torch.arange(args.context, device="cuda")
    idx = (positions - count + 1).clamp_min(0)[:, None] + torch.arange(count, device="cuda")
    idx = idx.masked_fill(idx > positions[:, None], -1).unsqueeze(0).int()
    do = torch.randn_like(q)
    report = {"context": args.context, "heads": args.heads, "dim": args.dim, "kernels": kernels,
              "production_qualification": False, "repeats": [], "passed": False}
    outputs = {}
    for name, fn in (("official", dsv4_sparse_attention), ("deterministic", deterministic_sparse_attention)):
        records = []
        for repeat in range(args.repeats):
            x, k = q.detach().clone().requires_grad_(), kv.detach().clone().requires_grad_()
            start = time.perf_counter()
            y = fn(x, k, sink, idx, args.dim ** -.5, backend="tilelang", reference_rounding=True)
            y.backward(do)
            torch.cuda.synchronize()
            current = (y.detach(), x.grad.detach(), k.grad.detach())
            if repeat == 0:
                outputs[name] = tuple(value.clone() for value in current)
            records.append({"repeat": repeat, "seconds": time.perf_counter() - start,
                            **{field: _compare(value, expected) for field, value, expected in
                               zip(("output", "dq", "dkv"), current, outputs[name], strict=True)}})
        report["repeats"].append({"implementation": name, "results": records})
    report["comparison"] = {field: _compare(actual, expected) for field, actual, expected in
                            zip(("output", "dq", "dkv"), outputs["deterministic"], outputs["official"], strict=True)}
    report["passed"] = (report["comparison"]["output"]["equal"]
        and all(value["relative_l2"] < .01 for value in report["comparison"].values())
        and all(row[key]["equal"] for row in report["repeats"][1]["results"] for key in ("output", "dq", "dkv")))
    report["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    atomic_write_json(args.output, report, allow_nan=False)
    print(json.dumps({"passed": report["passed"], "comparison": report["comparison"],
                      "max_memory_allocated_gib": report["max_memory_allocated_gib"]}), flush=True)
    if not report["passed"]:
        raise AssertionError("deterministic sparse backward did not pass comparison/repetition")


if __name__ == "__main__":
    main()

"""Check repeatability and native equivalence of the FP32 simplicial core."""

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    import torch

    from archlab.architectures.simplicial_attention import simplicial_attention
    from archlab.architectures.simplicial_deterministic import deterministic_simplicial_attention
    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41.qualification.moe_rounding_probe import _compare

    if args.output.exists():
        raise ValueError("use a fresh probe output")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(840)
    values = [torch.randn(1, args.context, heads, 128, device="cuda") for heads in (8, 2, 2, 2, 2)]
    cotangent = torch.randn_like(values[0])
    report = {
        "production_qualification": False,
        "context": args.context,
        "windows": [32, 512],
        "repeats": [],
        "passed": False,
    }
    snapshots = {}
    names = ("output", "dq", "dk1", "dk2", "dv1", "dv2")
    for label, fn in (
        ("atomic", simplicial_attention),
        ("deterministic", deterministic_simplicial_attention),
    ):
        records = []
        for repeat in range(args.repeats):
            inputs = [value.clone().requires_grad_() for value in values]
            start = time.perf_counter()
            output = fn(*inputs, 32, 512)
            output.backward(cotangent)
            torch.cuda.synchronize()
            current = (output.detach(), *[value.grad.detach() for value in inputs])
            if repeat == 0:
                snapshots[label] = tuple(value.clone() for value in current)
            records.append(
                {
                    "repeat": repeat,
                    "seconds": time.perf_counter() - start,
                    **{
                        name: _compare(value, expected)
                        for name, value, expected in zip(
                            names, current, snapshots[label], strict=True
                        )
                    },
                }
            )
        report["repeats"].append({"implementation": label, "results": records})
    report["comparison"] = {
        name: _compare(value, expected)
        for name, value, expected in zip(
            names, snapshots["deterministic"], snapshots["atomic"], strict=True
        )
    }
    report["passed"] = (
        report["comparison"]["output"]["equal"]
        and all(value["relative_l2"] < 1e-4 for value in report["comparison"].values())
        and all(row[name]["equal"] for row in report["repeats"][1]["results"] for name in names)
    )
    report["max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    atomic_write_json(args.output, report, allow_nan=False)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "comparison": report["comparison"],
                "max_memory_allocated_gib": report["max_memory_allocated_gib"],
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise AssertionError(
            "simplicial deterministic backward did not pass native/repetition checks"
        )


if __name__ == "__main__":
    main()

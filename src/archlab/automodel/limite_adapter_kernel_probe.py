"""Measure cache-aware simplicial tiling against its independent FP32 oracle."""

import argparse
import json
from pathlib import Path

import torch

from archlab.architectures.simplicial_attention import reference_simplicial, simplicial_attention
from archlab.architectures.simplicial_packed import packed_simplicial_attention


def milliseconds(run):
    for _ in range(3):
        run()
    times = []
    for _ in range(10):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)[len(times) // 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decode", action="store_true")
    a = p.parse_args()
    torch.manual_seed(8)
    torch.set_num_threads(4)
    if a.decode:
        from archlab.architectures.simplicial_decode import simplicial_decode_attention
        from archlab.architectures.simplicial_packed import packed_simplicial_decode

        xs = [
            torch.randn(4, n, h, 128, device="cuda")
            for n, h in [(1, 10), (16, 2), (8192, 2), (16, 2), (8192, 2)]
        ]
        old = simplicial_decode_attention(*xs)
        new = packed_simplicial_decode(*xs)
        error = float((old - new).abs().max())
        assert error < 1e-4
        results = {
            name: milliseconds(lambda fn=fn: fn(*xs))
            for name, fn in [
                ("legacy", simplicial_decode_attention),
                ("split_packed", packed_simplicial_decode),
            ]
        }
        result = dict(
            passed=True,
            mode="decode",
            context=8192,
            batch=4,
            forward_max_error=error,
            milliseconds=results,
            speedup=results["legacy"] / results["split_packed"],
        )
        a.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return

    def inputs(n):
        return [
            torch.randn(2, n, h, 128, device="cuda", requires_grad=True) for h in [10, 2, 2, 2, 2]
        ]

    x = inputs(129)
    dy = torch.randn_like(x[0])
    ref = reference_simplicial(*x, 16, 1025)
    gr = torch.autograd.grad(ref, x, dy)
    results = []
    for name, fn in [("legacy", simplicial_attention), ("packed", packed_simplicial_attention)]:
        out = fn(*x, 16, 1025)
        gg = torch.autograd.grad(out, x, dy)
        error = float((out - ref).abs().max())
        ge = max(
            float((g - r).norm() / r.norm().clamp_min(1e-6)) for g, r in zip(gg, gr, strict=True)
        )
        assert error < 1e-4 and ge < 1e-4, (error, ge)
        large = inputs(2048)
        grad = torch.randn_like(large[0])

        def run(large=large, grad=grad, fn=fn):
            o = fn(*large, 16, 1025)
            torch.autograd.grad(o, large, grad)

        results.append(
            dict(
                kernel=name,
                forward_max_error=error,
                gradient_relative_error=ge,
                milliseconds=milliseconds(run),
            )
        )
        print(results[-1], flush=True)
    best = min(results, key=lambda r: r["milliseconds"])
    a.output.write_text(
        json.dumps(
            dict(
                passed=True,
                results=results,
                selected=best,
                speedup=results[0]["milliseconds"] / best["milliseconds"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Compare the existing atomic and ordered FP32 simplicial backward paths."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    packages = select_container_kernel_packages(
        Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"])
    )
    import torch

    from archlab.architectures.simplicial_attention import simplicial_attention
    from archlab.architectures.simplicial_deterministic import deterministic_simplicial_attention

    torch.manual_seed(73)
    torch.backends.cuda.matmul.allow_tf32 = False
    cases = []
    for dim in (16, 32):
        for length in (65, 257, 2048):
            inputs = [
                torch.randn(1, length, heads, dim, device="cuda") * 0.5 for heads in (8, 2, 2, 2, 2)
            ]
            grad = torch.randn_like(inputs[0])
            results, times = [], []
            for function in (deterministic_simplicial_attention, simplicial_attention):
                values = [x.detach().clone().requires_grad_() for x in inputs]

                def step(function=function, values=values, grad=grad):
                    out = function(*values, 32, 512)
                    gradients = torch.autograd.grad(out, values, grad)
                    return out.detach(), gradients

                result = step()  # compile outside the measured interval
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(3):
                    result = step()
                torch.cuda.synchronize()
                times.append((time.perf_counter() - started) / 3)
                results.append(result)
            torch.testing.assert_close(results[0][0], results[1][0], atol=0, rtol=0)
            errors = []
            for expected, actual in zip(results[0][1], results[1][1], strict=True):
                assert bool(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, expected, atol=1e-4, rtol=3e-4)
                errors.append(float((actual - expected).norm() / expected.norm().clamp_min(1e-12)))
                assert errors[-1] < 3e-4
            cases.append(
                dict(
                    head_dim=dim,
                    sequence=length,
                    relative_l2_errors=errors,
                    deterministic_seconds=times[0],
                    atomic_seconds=times[1],
                )
            )
    args.output.write_text(json.dumps(dict(passed=True, cases=cases, packages=packages), indent=2))


if __name__ == "__main__":
    main()

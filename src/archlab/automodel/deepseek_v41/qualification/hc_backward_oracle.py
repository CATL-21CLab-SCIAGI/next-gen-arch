"""Check the compiled mHC derivative against the released-equation reference."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch

    from archlab.architectures.deepseek_v41_math import hc_split_sinkhorn
    from archlab.automodel.deepseek_v41_full_boundaries import NativeTrainableHC

    torch.manual_seed(91)
    compiled = torch.compile(hc_split_sinkhorn, fullgraph=True, dynamic=True)
    rows = []
    for sequence in (17, 128, 256):
        leaves = (
            torch.randn(2, sequence, 24, device="cuda", requires_grad=True),
            torch.randn(3, device="cuda", requires_grad=True),
            torch.randn(24, device="cuda", requires_grad=True),
        )
        expected = hc_split_sinkhorn(*leaves)
        actual = NativeTrainableHC.apply(*leaves, 4, 20, 1e-6, hc_split_sinkhorn, compiled)
        upstream = tuple(torch.randn_like(x) for x in expected)
        want = torch.autograd.grad(expected, leaves, upstream)
        got = torch.autograd.grad(actual, leaves, upstream)
        errors = []
        for a, b in zip(got, want, strict=True):
            torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)
            errors.append(float((a - b).abs().max()))
        rows.append({"sequence": sequence, "gradient_max_errors": errors})
    args.output.write_text(json.dumps({"passed": True, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()

"""Bounded production-shape native FP4 in-place quantizer repeatability check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    import torch

    from archlab.architectures.deepseek_v41_torch import rounded_activation
    from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
    from archlab.automodel.deepseek_v41_runtime import (
        load_native_reference,
        select_container_kernel_packages,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    packages = select_container_kernel_packages(Path("/usr/local/lib/python3.12/dist-packages"))
    native = load_native_reference(args.assets, module_name="_archlab_v41_fp4_repeat")
    bridge = install_native_row_padding(native)
    report = {
        "kind": "native-fp4-inplace-production-shape-repeatability",
        "training_launched": False,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
        "kernel_packages": packages,
        "quantization_bridge": bridge,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_kernel_sha256": hashlib.sha256(
            (args.assets / "inference/kernel.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    torch.manual_seed(4091)
    with torch.inference_mode():
        for block_size, scale_dtype in ((16, torch.float8_e4m3fn), (32, torch.float8_e8m0fnu)):
            x = torch.randn(1, 8192, 512, device="cuda", dtype=torch.bfloat16)
            # Include zero groups, which exercise the minimum-scale rule.
            x[:, :32] = 0
            expected = rounded_activation(
                x,
                bits=4,
                block_size=block_size,
                e4m3_scale=scale_dtype == torch.float8_e4m3fn,
                straight_through=False,
            )
            case = {
                "shape": list(x.shape),
                "block_size": block_size,
                "scale_dtype": str(scale_dtype),
                "repeats": [],
            }
            first = None
            for repeat in range(6):
                scratch = torch.empty_like(x).fill_(float("nan") if repeat % 2 else repeat + 1)
                actual = x.clone()
                del scratch
                returned = native.fp4_act_quant(actual, block_size, True, scale_dtype=scale_dtype)
                finite = bool(actual.isfinite().all())
                item = {
                    "repeat": repeat,
                    "inplace_alias_preserved": returned is actual,
                    "finite": finite,
                    "nonfinite_count": int((~actual.isfinite()).sum()),
                    "oracle_equal": torch.equal(actual, expected),
                    "repeat_equal": first is None or torch.equal(actual, first),
                    "mismatch_elements": int((actual != expected).sum()),
                }
                if finite:
                    error = actual.float() - expected.float()
                    item.update(
                        max_abs=float(error.abs().max()),
                        relative_l2=float(error.norm() / expected.float().norm()),
                    )
                if first is None:
                    first = actual.clone()
                case["repeats"].append(item)
                print(
                    json.dumps(
                        {"block_size": block_size, "scale_dtype": str(scale_dtype), **item},
                        allow_nan=False,
                    ),
                    flush=True,
                )
                del actual, returned
            case["passed"] = all(
                item["finite"]
                and item["oracle_equal"]
                and item["repeat_equal"]
                and item["inplace_alias_preserved"]
                for item in case["repeats"]
            )
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
            del x, expected, first
    report["complete"] = True
    report["passed"] = all(case["passed"] for case in report["cases"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

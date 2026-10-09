"""Isolate native-kernel repeatability at real V4.1 dimensions; no training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    import torch
    from safetensors import safe_open

    from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight, hc_split_sinkhorn
    from archlab.architectures.deepseek_v41_torch import (
        query_chunked_sparse_attention,
        rounded_activation,
    )
    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_runtime import (
        load_native_reference,
        select_container_kernel_packages,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    select_container_kernel_packages(Path("/usr/local/lib/python3.12/dist-packages"))
    native = load_native_reference(args.assets, module_name="_archlab_v41_kernel_repeatability")
    mapping = json.loads((args.weights / "model.safetensors.index.json").read_text())["weight_map"]
    report = {
        "training_launched": False,
        "kind": "native-full-shape-kernel-repeatability",
        "cases": [],
    }

    def read(name):
        with (
            torch.device("cpu"),
            safe_open(args.weights / mapping[name], framework="pt", device="cpu") as reader,
        ):
            return reader.get_tensor(name).cuda()

    def compare(name, actual, repeated, expected):
        finite = bool(actual.isfinite().all())
        value = {
            "name": name,
            "shape": list(actual.shape),
            "native_finite": finite,
            "native_repeat_equal": torch.equal(actual, repeated),
            "torch_finite": bool(expected.isfinite().all()),
        }
        if finite and value["torch_finite"]:
            value["relative_l2"] = float(
                (actual.float() - expected.float()).norm()
                / expected.float().norm().clamp_min(1e-20)
            )
            value["max_abs"] = float((actual.float() - expected.float()).abs().max())
            value["repeat_max_abs"] = float((actual.float() - repeated.float()).abs().max())
        report["cases"].append(value)
        print(json.dumps(value, allow_nan=False), flush=True)

    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(4091)
    with torch.no_grad():
        for rows in (17, 32, 33):
            width, block = 5120, 32
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            guard_rows = (rows + 31) // 32 * 32 + 32
            scales = torch.ones(guard_rows, width // block, device="cuda", dtype=torch.float32).to(
                torch.float8_e8m0fnu
            )
            before = scales.view(torch.uint8).clone()
            kernel = native.act_quant.__globals__["act_quant_kernel"](
                width, block, scale_dtype="float8_e8m0fnu", round_scale=True
            )
            kernel(x, out, scales[:rows])
            changed = scales.view(torch.uint8)[rows:] != before[rows:]
            record = {
                "name": f"native_fp8_scale_guard/rows={rows}",
                "out_of_logical_bounds_writes": int(changed.count_nonzero()),
                "modified_guard_rows": int(changed.any(-1).sum()),
            }
            report["cases"].append(record)
            print(json.dumps(record), flush=True)
        for layer in (0, 1, 20, 39):
            for sublayer in ("attn", "ffn"):
                stem = f"layers.{layer}.hc_{sublayer}"
                scale, base = read(stem + "_scale"), read(stem + "_base")
                for magnitude in (1.0, 10.0):
                    mixes = torch.randn(1, 128, 24, device="cuda", dtype=torch.float32) * magnitude
                    first = native.hc_split_sinkhorn(mixes, scale, base)
                    second = native.hc_split_sinkhorn(mixes, scale, base)
                    expected = hc_split_sinkhorn(mixes, scale, base)
                    compare(
                        f"{stem}/mix_std={magnitude}",
                        torch.cat([x.flatten() for x in first]),
                        torch.cat([x.flatten() for x in second]),
                        torch.cat([x.flatten() for x in expected]),
                    )
        for stem in (
            "layers.0.attn.wq_a",
            "layers.0.attn.wq_b",
            "layers.0.ffn.experts.0.w1",
            "layers.0.ffn.experts.0.w2",
            "layers.1.engram.wkv",
        ):
            weight, scale = read(stem + ".weight"), read(stem + ".scale")
            if weight.dtype == torch.int8:
                weight = weight.view(torch.float4_e2m1fn_x2)
            weight.scale = scale
            decoded = dequantize_frozen_weight(weight, scale)
            for rows in (17, 128, 257):
                x = torch.randn(rows, decoded.shape[1], device="cuda", dtype=torch.bfloat16)
                first, second = native.linear(x, weight), native.linear(x, weight)
                expected = torch.nn.functional.linear(rounded_activation(x), decoded)
                compare(f"{stem}/rows={rows}", first, second, expected)
        sink = read("layers.0.attn.attn_sink")
        for length in (128, 1024):
            q = torch.randn(1, length, 64, 512, device="cuda", dtype=torch.bfloat16)
            kv = torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16)
            positions = torch.arange(length, device="cuda")[:, None]
            ids = positions - torch.arange(640, device="cuda").flip(0)[None, :]
            ids = torch.where(ids >= 0, ids, -1).int().unsqueeze(0)
            first = native.sparse_attn(q, kv, sink, ids, 512**-0.5)
            second = native.sparse_attn(q, kv, sink, ids, 512**-0.5)
            expected = query_chunked_sparse_attention(
                q, kv, sink, ids, 512**-0.5, native_rounding=True
            )
            compare(f"sparse/H64/D512/S{length}/K640", first, second, expected)
    atomic_write_json(args.output, report, allow_nan=False)


if __name__ == "__main__":
    main()

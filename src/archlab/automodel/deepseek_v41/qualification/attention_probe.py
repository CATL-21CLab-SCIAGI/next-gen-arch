"""Replay actual layer-zero attention inputs without loading the whole model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    import torch
    from safetensors import safe_open

    from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight
    from archlab.architectures.deepseek_v41_torch import (
        query_chunked_sparse_attention,
        rounded_activation,
    )
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_native_quantization import install_native_row_padding
    from archlab.automodel.deepseek_v41_runtime import (
        load_native_reference,
        select_container_kernel_packages,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    packages = select_container_kernel_packages(Path("/usr/local/lib/python3.12/dist-packages"))
    native = load_native_reference(args.assets, module_name="_archlab_v41_actual_attention")
    raw_quantize = native.act_quant
    quantization_bridge = install_native_row_padding(native)
    clean_quantizers = []
    for block_size in (32, 128):
        for scale_fmt, scale_dtype in ((None, torch.float32), ("ue8m0", torch.float8_e8m0fnu)):
            known = (
                torch.arange(32 * 128, device="cuda", dtype=torch.float32)
                .sin()
                .bfloat16()
                .reshape(32, 128)
            )
            original, original_scales = raw_quantize(known, block_size, scale_fmt, scale_dtype)
            stable, stable_scales = native.act_quant(known, block_size, scale_fmt, scale_dtype)
            record = {
                "block_size": block_size,
                "scale_fmt": scale_fmt,
                "scale_dtype": str(scale_dtype),
                "native_finite": bool(original.float().isfinite().all()),
                "values_equal": torch.equal(original.view(torch.uint8), stable.view(torch.uint8)),
                "scales_equal": torch.equal(original_scales, stable_scales)
                if scale_dtype == torch.float32
                else torch.equal(
                    original_scales.view(torch.uint8), stable_scales.view(torch.uint8)
                ),
            }
            clean_quantizers.append(record)
            print(json.dumps({"event": "clean_native_quantizer", **record}), flush=True)
            if not (record["native_finite"] and record["values_equal"] and record["scales_equal"]):
                raise AssertionError(record)
    config = native.ModelArgs(**json.loads((args.assets / "inference/config.json").read_text()))
    config.max_seq_len, config.max_batch_size = 2048, 1
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    attention = native.Attention(0, config).requires_grad_(False)
    norm = native.RMSNorm(config.dim, config.norm_eps).requires_grad_(False)
    mapping = json.loads((args.weights / "model.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        with (
            torch.device("cpu"),
            safe_open(args.weights / mapping[name], framework="pt", device="cpu") as reader,
        ):
            return reader.get_tensor(name).cuda()

    with torch.no_grad():
        for name, parameter in attention.named_parameters():
            prefix = "layers.0.attn." + name
            source = read(prefix)
            if name == "wo_a.weight":
                source = dequantize_frozen_weight(source, read("layers.0.attn.wo_a.scale"))
            parameter.copy_(source)
        norm.weight.copy_(read("layers.0.attn_norm.weight"))
    data = MathPilot(
        args.pilot, expected_split="train", expected_budget=1_000_000_000, order_seed=2234
    )
    weights = {
        id(module.weight): name
        for name, module in attention.named_modules()
        if hasattr(module, "weight")
    }
    captures, current_linear = {}, []

    def capture(name, tensor):
        captures[name] = tensor.detach().clone()

    linear, quantize, sparse = native.linear, native.act_quant, native.sparse_attn

    def observe_linear(x, weight, bias=None):
        name = weights[id(weight)]
        capture(name + ".input", x)
        current_linear.append(name)
        try:
            output = linear(x, weight, bias)
        finally:
            current_linear.pop()
        capture(name + ".output", output)
        return output

    def observe_quantize(x, *positional, **keywords):
        name = (current_linear[-1] if current_linear else "window_kv") + ".quantize"
        capture(name + ".input", x)
        output = quantize(x, *positional, **keywords)
        if isinstance(output, tuple):
            capture(name + ".values", output[0])
            capture(name + ".scales", output[1])
        else:
            capture(name + ".output", output)
        return output

    def observe_sparse(q, kv, sink, indices, scale):
        for name, tensor in (("q", q), ("kv", kv), ("sink", sink), ("indices", indices)):
            capture("sparse." + name, tensor)
        output = sparse(q, kv, sink, indices, scale)
        capture("sparse.output", output)
        return output

    native.linear, native.act_quant, native.sparse_attn = (
        observe_linear,
        observe_quantize,
        observe_sparse,
    )
    handles = [
        attention.q_norm.register_forward_hook(
            lambda module, values, out: capture("q_norm.output", out)
        ),
        attention.kv_norm.register_forward_hook(
            lambda module, values, out: capture("kv_norm.output", out)
        ),
    ]

    def describe(tensor):
        value = tensor.float()
        finite = value.isfinite()
        clean = value.masked_fill(~finite, 0)
        return {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite": bool(finite.all()),
            "nonfinite_count": int((~finite).sum()),
            "finite_l2": float(clean.norm()),
            "finite_max_abs": float(clean.abs().max()),
        }

    def compare(actual, expected):
        a, b = actual.float(), expected.float()
        valid = a.isfinite() & b.isfinite()
        difference = torch.where(valid, a - b, 0)
        return {
            "equal": torch.equal(a, b),
            "both_finite": bool(valid.all()),
            "same_nonfinite_pattern": bool(torch.equal(a.isnan(), b.isnan())),
            "finite_pair_max_abs": float(difference.abs().max()),
            "finite_pair_relative_l2": float(
                difference.norm() / b.masked_fill(~valid, 0).norm().clamp_min(1e-30)
            ),
        }

    report = {
        "kind": "actual-first-attention-replay",
        "training_launched": False,
        "torch": str(torch.__version__),
        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
        "kernel_packages": packages,
        "order_seed": 2234,
        "quantization_bridge": quantization_bridge,
        "clean_native_quantizers": clean_quantizers,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    with torch.no_grad():
        for sample_id, context in ((1, 128), (0, 2048)):
            inputs, _, _ = data.batch(
                sample_id, device="cpu", smoke_context=context, pad_to_full=True
            )
            with (
                torch.device("cpu"),
                safe_open(
                    args.weights / mapping["embed.weight"], framework="pt", device="cpu"
                ) as reader,
            ):
                embedded = torch.nn.functional.embedding(
                    inputs, reader.get_tensor("embed.weight")
                ).cuda()
            # Identity pre-mix collapses identical embedding streams exactly.
            x = norm(embedded)
            first, case = (
                None,
                {"sample_id": sample_id, "context": context, "input": describe(x), "passes": []},
            )
            for repeat in range(4):
                captures = {}
                output = attention(x, 0)
                capture("attention.output", output)
                stage = {"repeat": repeat, "boundaries": []}
                for name, tensor in captures.items():
                    item = {"name": name, **describe(tensor)}
                    if first is not None:
                        item["comparison"] = compare(tensor, first[name])
                    stage["boundaries"].append(item)
                if first is None:
                    first = captures
                stage["first_nonfinite"] = next(
                    (v["name"] for v in stage["boundaries"] if not v["finite"]), None
                )
                stage["first_unequal"] = next(
                    (
                        v["name"]
                        for v in stage["boundaries"]
                        if not v.get("comparison", {}).get("equal", True)
                    ),
                    None,
                )
                case["passes"].append(stage)
                print(
                    json.dumps(
                        {
                            "event": "attention_replay",
                            "sample_id": sample_id,
                            "context": context,
                            "repeat": repeat,
                            "first_nonfinite": stage["first_nonfinite"],
                            "first_unequal": stage["first_unequal"],
                        }
                    ),
                    flush=True,
                )
            case["linear_oracles"] = {}
            for name in ("wq_a", "wq_b", "wkv", "wo_b"):
                module = getattr(attention, name)
                decoded = dequantize_frozen_weight(module.weight, module.scale)
                expected = torch.nn.functional.linear(
                    rounded_activation(first[name + ".input"]), decoded
                )
                case["linear_oracles"][name] = compare(first[name + ".output"], expected)
                del decoded, expected
            q, kv, sink, indices = (
                first["sparse." + name] for name in ("q", "kv", "sink", "indices")
            )
            expected = query_chunked_sparse_attention(
                q, kv, sink, indices, attention.softmax_scale, native_rounding=True, recompute=False
            )
            case["sparse_to_torch"] = compare(first["sparse.output"], expected)
            case["standalone_sparse_repeats"] = []
            for _repeat in range(4):
                actual = sparse(q, kv, sink, indices, attention.softmax_scale)
                case["standalone_sparse_repeats"].append(compare(actual, first["sparse.output"]))
            repro = args.output.with_name(args.output.stem + f"-rank{sample_id}-s{context}.pt")
            torch.save(
                {
                    "q": q.cpu(),
                    "kv": kv.cpu(),
                    "sink": sink.cpu(),
                    "indices": indices.cpu(),
                    "native_output": first["sparse.output"].cpu(),
                    "scale": attention.softmax_scale,
                },
                repro,
            )
            case["sparse_reproducer"] = str(repro)
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(
                json.dumps(
                    {
                        "event": "case_complete",
                        "sample_id": sample_id,
                        "context": context,
                        "linear_oracles": case["linear_oracles"],
                        "sparse_to_torch": case["sparse_to_torch"],
                        "standalone_sparse_repeats": case["standalone_sparse_repeats"],
                    }
                ),
                flush=True,
            )
            captures = {}
            del first, q, kv, sink, indices, expected, actual, output, embedded, x
    for handle in handles:
        handle.remove()
    report["complete"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

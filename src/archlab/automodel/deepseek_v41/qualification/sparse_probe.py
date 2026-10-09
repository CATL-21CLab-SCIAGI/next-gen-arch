"""Bounded native sparse-kernel repeatability at the full model's tile counts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    import torch
    from safetensors import safe_open

    from archlab.architectures.deepseek_v41_torch import (
        query_chunked_sparse_attention,
        rounded_activation,
    )
    from archlab.automodel.deepseek_v41_runtime import (
        load_native_reference,
        select_container_kernel_packages,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--topks", type=int, nargs="+", default=[128, 512, 640])
    parser.add_argument("--compressed-joint", action="store_true")
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    packages = select_container_kernel_packages(Path("/usr/local/lib/python3.12/dist-packages"))
    native = load_native_reference(args.assets, module_name="_archlab_v41_sparse_repeat")
    mapping = json.loads((args.weights / "model.safetensors.index.json").read_text())["weight_map"]
    name = "layers.0.attn.attn_sink"
    with safe_open(args.weights / mapping[name], framework="pt", device="cpu") as reader:
        sink = reader.get_tensor(name).cuda()
    report = {
        "kind": "native-sparse-tile-repeatability",
        "training_launched": False,
        "container_image": os.environ.get("NGA_CONTAINER_DIGEST"),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "kernel_packages": packages,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": {
            name: hashlib.sha256((args.assets / "inference" / name).read_bytes()).hexdigest()
            for name in ("model.py", "kernel.py")
        },
        "cases": [],
    }

    def metric(first, second):
        finite = bool(first.isfinite().all() and second.isfinite().all())
        result = {"equal": torch.equal(first, second), "both_finite": finite}
        if finite:
            error = first.float() - second.float()
            result.update(
                max_abs=float(error.abs().max()),
                relative_l2=float(error.norm() / second.float().norm().clamp_min(1e-30)),
            )
        return result

    torch.manual_seed(4091)
    with torch.inference_mode():
        for length in args.contexts:
            for topk in args.topks:
                if args.replay:
                    stored = torch.load(args.replay, map_location="cpu", weights_only=True)
                    q, kv = stored["q"].cuda(), stored["kv"].cuda()
                    if q.shape[1] != length or kv.shape[1] != length:
                        raise ValueError(
                            "replay must contain exactly the requested raw-window context"
                        )
                else:
                    q = torch.randn(1, length, 64, 512, device="cuda", dtype=torch.bfloat16)
                    kv = torch.randn(1, length, 512, device="cuda", dtype=torch.bfloat16)
                if args.compressed_joint:
                    compressed = topk - min(length, 128)
                    if compressed < 1 or length % compressed:
                        raise ValueError("joint geometry requires an integer compression ratio")
                    ratio = length // compressed
                    compressed_kv = torch.randn(
                        1, compressed, 512, device="cuda", dtype=torch.bfloat16
                    )
                    compressed_kv = rounded_activation(
                        compressed_kv, bits=4, block_size=16, e4m3_scale=True
                    )
                    kv = torch.cat((kv, compressed_kv), 1)
                    window_ids = native.get_window_topk_idxs(128, 1, length, 0).cuda()
                    visible = (torch.arange(1, length + 1, device="cuda") // ratio)[None, :, None]
                    compressed_ids = torch.arange(compressed, device="cuda")[None, None].expand(
                        1, length, compressed
                    )
                    compressed_ids = torch.where(
                        compressed_ids < visible, compressed_ids + length, -1
                    ).int()
                    ids = torch.cat((window_ids, compressed_ids), -1)
                else:
                    positions = torch.arange(length, device="cuda")[:, None]
                    ids = positions - torch.arange(topk, device="cuda").flip(0)[None, :]
                    ids = torch.where(ids >= 0, ids, -1).int().unsqueeze(0).contiguous()
                first = native.sparse_attn(q, kv, sink, ids, 512**-0.5).clone()
                expected = query_chunked_sparse_attention(
                    q, kv, sink, ids, 512**-0.5, native_rounding=True, recompute=False
                )
                case = {
                    "sequence": length,
                    "topk": topk,
                    "heads": 64,
                    "head_dim": 512,
                    "query_source": str(args.replay) if args.replay else "random",
                    "index_layout": "native-window-plus-synthetic-compressed"
                    if args.compressed_joint
                    else "right-aligned-causal",
                    "native_to_torch": metric(first, expected),
                    "repeats": [],
                }
                del expected
                for repeat in range(args.repeats):
                    # Change dead storage contents and input addresses without
                    # changing input values, so unwritten outputs are visible.
                    scratch = torch.empty_like(q).fill_(float("nan") if repeat % 2 else repeat + 1)
                    new_q, new_kv, new_ids = q.clone(), kv.clone(), ids.clone()
                    del scratch
                    actual = native.sparse_attn(new_q, new_kv, sink, new_ids, 512**-0.5)
                    case["repeats"].append(metric(actual, first))
                    del actual, new_q, new_kv, new_ids
                case["all_repeats_equal"] = all(item["equal"] for item in case["repeats"])
                print(json.dumps(case, allow_nan=False), flush=True)
                report["cases"].append(case)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
                del q, kv, ids, first
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

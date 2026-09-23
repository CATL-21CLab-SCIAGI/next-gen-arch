"""Replay one real V4.1 MoE layer with saved EP-group inputs and native experts.

Only active experts from the selected layer are streamed through one GPU. The
same BF16 expert outputs feed late-FP32 versus early-BF16 shared-expert sums.
This diagnostic does not modify or qualify the production implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import ExitStack
from pathlib import Path


def _compare(actual, expected):
    import torch

    a, b = actual.detach().float(), expected.detach().float()
    delta = a - b
    return {"finite": bool(a.isfinite().all() and b.isfinite().all()),
            "equal": torch.equal(actual, expected), "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
            "differing_elements": int(delta.count_nonzero()), "elements": delta.numel()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--boundaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--first-rank", type=int, default=0)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--official-leaf", action="store_true",
                        help="also run the project FP32 sum leaf on the unchanged official grouped expert kernels")
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import load_native_reference, select_container_kernel_packages

    kernels = select_container_kernel_packages(args.container_kernel_packages)
    import torch
    from safetensors import safe_open

    from archlab.architectures.deepseek_v41_math import dequantize_frozen_weight
    from archlab.artifacts import atomic_write_json, sha256_file
    from archlab.automodel.deepseek_v41_official_execution import runtime_identity

    args.output.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    started = time.perf_counter()
    native = load_native_reference(args.assets, module_name="_archlab_native_moe_rounding_probe")
    config = json.loads((args.assets / "inference/config.json").read_text())
    native_args = native.ModelArgs(**config)
    width, intermediate = config["dim"], config["moe_inter_dim"]
    n_experts = config["n_routed_experts"]
    if n_experts % args.ep_size or args.layer < 0:
        raise ValueError("invalid layer/EP geometry")
    index = args.weights / "model.safetensors.index.json"
    raw = index.read_bytes()
    index_sha = hashlib.sha256(raw).hexdigest()
    verified = json.loads((args.weights / "ARCHLAB_VERIFIED_COPY.json").read_text())
    if index_sha != verified["source_index_sha256"]:
        raise ValueError("checkpoint differs from verified cache index")
    mapping = json.loads(raw)["weight_map"]
    layer = f"layers.{args.layer}.ffn"
    captures = {kind: [] for kind in ("native", "official")}
    files = []
    for rank in range(args.first_rank, args.first_rank + args.ep_size):
        path = args.boundaries / f"rank{rank}-boundaries.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        files.append({"path": str(path), "bytes": path.stat().st_size})
        for kind in captures:
            captures[kind].append({name: saved[kind][f"{layer}.{name}"].clone() for name in
                                   ("input", "output", "gate.weights", "gate.indices", "shared_experts.output")})
        del saved
    data = {}
    for kind, rows in captures.items():
        lengths = [row["input"].numel() // width for row in rows]
        data[kind] = {"lengths": lengths}
        for key in rows[0]:
            trailing = width if key in ("input", "output", "shared_experts.output") else rows[0][key].shape[-1]
            data[kind][key] = torch.cat([row[key].reshape(-1, trailing) for row in rows]).cuda()
    if data["native"]["lengths"] != data["official"]["lengths"]:
        raise ValueError("native and official saved EP windows differ in length")
    active = sorted(set(data["native"]["gate.indices"].flatten().tolist())
                    | set(data["official"]["gate.indices"].flatten().tolist()))
    report = {"kind": "real-layer-native-expert-shared-rounding-diagnostic", "production_qualification": False,
              "runtime": runtime_identity(), "kernels": kernels, "completed": False,
              "source_index_sha256": index_sha, "probe_sha256": sha256_file(Path(__file__)),
              "layer": args.layer, "ep_size": args.ep_size, "first_rank": args.first_rank,
              "boundary_files": files, "active_experts": len(active), "streamed_experts": True,
              "router_weight_placement": "FP32 SwiGLU before BF16 down-projection input",
              "cases": {}}
    atomic_write_json(args.output / "STARTED.json", report, allow_nan=False)
    expert = native.Expert(width, intermediate, dtype=torch.bfloat16,
                           swiglu_limit=config["swiglu_limit"]).cuda().requires_grad_(False)
    official_leaf = None
    if args.official_leaf:
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.moe.config import MoEConfig
        from nemo_automodel.components.moe.layers import MoE

        from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe

        moe_config = MoEConfig(
            n_routed_experts=n_experts, n_shared_experts=1,
            n_activated_experts=config["n_activated_experts"], n_expert_groups=0, n_limited_groups=0,
            train_gate=False, gate_bias_update_factor=0., aux_loss_coeff=0., score_func=config["score_func"],
            route_scale=config["route_scale"], dim=width, inter_dim=intermediate, moe_inter_dim=intermediate,
            norm_topk_prob=native_args.norm_topk_prob, swiglu_limit=config["swiglu_limit"], dtype=torch.bfloat16,
        )
        with torch.device("meta"):
            official_leaf = MoE(moe_config, BackendConfig(experts="torch_mm", dispatcher="torch", linear="torch"))
        official_leaf.to_empty(device="cuda").requires_grad_(False)
        for parameter in official_leaf.parameters():
            parameter.zero_()

        class CapturedRouter(torch.nn.Module):
            def forward(self, x, token_mask=None, cp_mesh=None):
                return self.weights, self.indices, None

        official_leaf.gate = CapturedRouter()
        report["official_leaf"] = install_official_fp32_moe(official_leaf)
    routed = {kind: torch.zeros(args.ep_size, data[kind]["input"].shape[0], width,
                                device="cuda", dtype=torch.float32) for kind in data}
    source_tensors = []
    with ExitStack() as stack:
        readers = {}

        def tensor(name):
            shard = mapping[name]
            if Path(shard).name != shard:
                raise ValueError("checkpoint shard escapes source directory")
            if shard not in readers:
                readers[shard] = stack.enter_context(safe_open(args.weights / shard, framework="pt", device="cpu"))
            source_tensors.append(name)
            return readers[shard].get_tensor(name)

        def load_expert(prefix):
            for projection in ("w1", "w3", "w2"):
                key = f"{prefix}.{projection}"
                weight, scale = tensor(key + ".weight"), tensor(key + ".scale")
                target = getattr(expert, projection).weight
                for start in range(0, weight.shape[0], 1024):
                    end = min(start + 1024, weight.shape[0])
                    factors = scale[start // 32:(end + 31) // 32] if weight.dtype == torch.float8_e4m3fn else scale[start:end]
                    decoded = dequantize_frozen_weight(weight[start:end].cuda(), factors.cuda())
                    target[start:end].copy_(decoded)

        for offset, expert_id in enumerate(active):
            load_expert(f"{layer}.experts.{expert_id}")
            if official_leaf is not None:
                combined = official_leaf.experts.gate_and_up_projs[expert_id]
                combined[:, :intermediate].copy_(expert.w1.weight.T)
                combined[:, intermediate:].copy_(expert.w3.weight.T)
                official_leaf.experts.down_projs[expert_id].copy_(expert.w2.weight.T)
            owner = expert_id // (n_experts // args.ep_size)
            for kind in data:
                case = data[kind]
                token_ids, slots = torch.where(case["gate.indices"] == expert_id)
                if token_ids.numel():
                    output = expert(case["input"][token_ids], case["gate.weights"][token_ids, slots, None])
                    routed[kind][owner, token_ids] += output.float()
            if (offset + 1) % 32 == 0:
                print(json.dumps({"event": "expert_replayed", "count": offset + 1, "total": len(active)}), flush=True)
        load_expert(layer + ".shared_experts")
        if official_leaf is not None:
            official_leaf.shared_experts.gate_proj.weight.copy_(expert.w1.weight)
            official_leaf.shared_experts.up_proj.weight.copy_(expert.w3.weight)
            official_leaf.shared_experts.down_proj.weight.copy_(expert.w2.weight)
        for kind, case in data.items():
            shared = torch.cat([expert(part) for part in case["input"].split(case["lengths"])])
            summed = routed[kind].sum(0)
            late = (summed + shared.float()).bfloat16()
            early = summed.bfloat16() + shared
            owner_rounded = routed[kind].bfloat16().float().sum(0)
            owner_late = (owner_rounded + shared.float()).bfloat16()
            owner_early = owner_rounded.bfloat16() + shared
            compared = {"shared_replay_vs_saved": _compare(shared, case["shared_experts.output"]),
                        "late_sum_vs_saved": _compare(late, case["output"]),
                        "early_sum_vs_saved": _compare(early, case["output"]),
                        "owner_rounded_late_shared_vs_saved": _compare(owner_late, case["output"]),
                        "owner_rounded_early_shared_vs_saved": _compare(owner_early, case["output"]),
                        "early_vs_late_same_expert_outputs": _compare(early, late),
                        "by_rank": []}
            first = 0
            for rank, length in zip(range(args.first_rank, args.first_rank + args.ep_size), case["lengths"], strict=True):
                section = slice(first, first + length)
                compared["by_rank"].append({"rank": rank,
                                            "shared_replay_vs_saved": _compare(shared[section], case["shared_experts.output"][section]),
                                            "late_sum_vs_saved": _compare(late[section], case["output"][section]),
                                            "early_sum_vs_saved": _compare(early[section], case["output"][section]),
                                            "owner_rounded_late_shared_vs_saved": _compare(owner_late[section], case["output"][section]),
                                            "owner_rounded_early_shared_vs_saved": _compare(owner_early[section], case["output"][section]),
                                            "early_vs_late_same_expert_outputs": _compare(early[section], late[section])})
                first += length
            report["cases"][kind] = compared
            if official_leaf is not None:
                official_leaf.gate.weights = case["gate.weights"]
                official_leaf.gate.indices = case["gate.indices"]
                adapted = official_leaf(case["input"])
                compared["official_leaf_vs_native_late_sum"] = _compare(adapted, late)
                compared["official_leaf_vs_saved"] = _compare(adapted, case["output"])
                torch.save(adapted.cpu(), args.output / f"{kind}-official-leaf.pt")
            torch.save({"routed_fp32": summed.cpu(), "per_owner_fp32": routed[kind].cpu(),
                        "shared_bf16": shared.cpu(), "late_sum_bf16": late.cpu(), "early_sum_bf16": early.cpu(),
                        "owner_rounded_late_shared_bf16": owner_late.cpu(),
                        "owner_rounded_early_shared_bf16": owner_early.cpu()},
                       args.output / f"{kind}-replayed.pt")
    torch.cuda.synchronize()
    report.update(completed=True, seconds=time.perf_counter() - started,
                  max_gpu_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  source_tensor_count=len(source_tensors), source_shards=sorted(readers))
    atomic_write_json(args.output / "REPORT.json", report, allow_nan=False)
    print(json.dumps({"event": "moe_rounding_replay_complete", "seconds": report["seconds"],
                      "cases": {kind: values["by_rank"][0] for kind, values in report["cases"].items()}}), flush=True)


if __name__ == "__main__":
    main()

"""Localize official/released V4.1 drift on a real-weight three-layer fixture.

This diagnostic never changes production gates or starts training. Its derived
index refers to read-only original shards; both backbones retain the released
compression/Engram schedules and differ only in active layer count.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import traceback
from contextlib import contextmanager
from pathlib import Path


def prepare_prefix_fixture(*, assets: Path, weights: Path, output: Path, layers: int = 3):
    """Derive an auditable index subset; never copy or modify source weights."""
    assets, weights = assets.resolve(strict=True), weights.resolve(strict=True)
    if not 1 <= layers <= 3:
        raise ValueError("this bounded diagnostic permits one to three original layers")
    raw_index = (weights / "model.safetensors.index.json").read_bytes()
    original_sha = hashlib.sha256(raw_index).hexdigest()
    parent_receipt = (weights / "ARCHLAB_VERIFIED_COPY.json").read_bytes()
    parent = json.loads(parent_receipt)
    if parent["source_index_sha256"] != original_sha:
        raise ValueError("original source index no longer matches its verified cache")
    mapping = json.loads(raw_index)["weight_map"]
    selected = {}
    for name, shard in mapping.items():
        match = re.match(r"layers\.(\d+)\.", name)
        if name in ("embed.weight", "head.weight", "norm.weight") or (match and int(match[1]) < layers):
            if Path(shard).name != shard:
                raise ValueError("source shards must be direct checkpoint children")
            selected[name] = shard
    if not {"embed.weight", "head.weight", "norm.weight"}.issubset(selected):
        raise ValueError("prefix source misses embedding, head, or final norm")
    if output.exists() and any(output.iterdir()):
        existing = json.loads((output / "PREFIX_DERIVATION.json").read_text())
        if (existing["parent_index_sha256"] != original_sha or existing["layers"] != layers
                or existing["parent_checkpoint"] != str(weights) or existing["parent_assets"] != str(assets)):
            raise ValueError("existing prefix fixture belongs to a different source")
        _verify_fixture(output, existing)
        return existing
    output.mkdir(parents=True, exist_ok=True)
    index_bytes = (json.dumps({"metadata": {"archlab_partial_fixture": True},
                              "weight_map": selected}, sort_keys=True, indent=2) + "\n").encode()
    (output / "model.safetensors.index.json").write_bytes(index_bytes)
    links = {}
    for shard in sorted(set(selected.values())):
        source = weights / shard
        if not source.is_file():
            raise FileNotFoundError(source)
        (output / shard).symlink_to(source)
        links[shard] = str(source)
    config = json.loads((weights / "config.json").read_text())
    original_layers = config["text_config"]["num_hidden_layers"]
    config["text_config"]["num_hidden_layers"] = layers
    config["vision_config"]["num_hidden_layers"] = 0
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for source in assets.iterdir():
        if source.is_file() and source.suffix != ".safetensors" and source.name not in (
            "config.json", "model.safetensors.index.json", "ARCHLAB_VERIFIED_COPY.json", "PREFIX_DERIVATION.json"):
            (output / source.name).symlink_to(source)
            links[source.name] = str(source)
    inference = output / "inference"
    inference.mkdir()
    for source in (assets / "inference").iterdir():
        if source.is_file() and source.name != "config.json":
            (inference / source.name).symlink_to(source)
            links[f"inference/{source.name}"] = str(source)
    native = json.loads((assets / "inference/config.json").read_text())
    native["n_layers"] = layers
    (inference / "config.json").write_text(json.dumps(native, indent=2) + "\n")
    generated = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in
                 ("model.safetensors.index.json", "config.json", "inference/config.json")}
    receipt = {
        "kind": "derived-real-weight-prefix-fixture-only", "partial_fixture": True,
        "layers": layers, "original_layers": original_layers,
        "parent_checkpoint": str(weights), "parent_assets": str(assets),
        "parent_index_sha256": original_sha,
        "parent_verification_sha256": hashlib.sha256(parent_receipt).hexdigest(),
        "selected_tensors": len(selected), "original_tensors": len(mapping),
        "generated_sha256": generated, "symlinks": links,
        "source_weight_files_modified": False, "production_qualification": False,
    }
    (output / "PREFIX_DERIVATION.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    # The strict stream loader verifies this derived index; this receipt says
    # explicitly why it is trusted and links the original full-copy evidence.
    (output / "ARCHLAB_VERIFIED_COPY.json").write_text(json.dumps({
        "source_index_sha256": generated["model.safetensors.index.json"],
        "verification_mode": "index-subset-of-verified-cache-with-original-shard-symlinks",
        "partial_fixture": True, "parent_checkpoint": str(weights),
        "parent_index_sha256": original_sha,
        "parent_verification_sha256": receipt["parent_verification_sha256"],
        "derivation_receipt": "PREFIX_DERIVATION.json",
    }, sort_keys=True, indent=2) + "\n")
    return receipt


def _verify_fixture(root, receipt):
    for name, expected in receipt["generated_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"derived prefix file changed: {name}")
    for name, target in receipt["symlinks"].items():
        if not (root / name).is_symlink() or (root / name).resolve(strict=True) != Path(target):
            raise ValueError(f"derived prefix symlink changed: {name}")


def build_official_prefix(fixture: Path, *, ep_size: int = 8):
    """Use public native construction, placement, and checkpoint loading."""
    import torch
    import torch.distributed as dist
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config, MoEParallelizerConfig
    from nemo_automodel.components.distributed.mesh import ParallelismSizes
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from archlab.automodel.deepseek_v41_official_execution import runtime_identity
    from archlab.automodel.deepseek_v41_official_moe import install_official_fp32_moe
    from archlab.automodel.deepseek_v41_official_hc import install_official_native_hc

    identity = runtime_identity()
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                     output_dtype=None, cast_forward_inputs=False)
    setup = DistributedSetup.build(
        strategy=FSDP2Config(mp_policy=precision, reshard_after_forward=True, sequence_parallel=False),
        parallelism_sizes=ParallelismSizes(tp_size=1, pp_size=1, cp_size=1, ep_size=ep_size),
        moe_parallel_config=MoEParallelizerConfig(mp_policy=precision, lm_head_precision=torch.float32,
                                                 reshard_after_forward=True, wrap_outer_model=True),
        activation_checkpointing=False, world_size=dist.get_world_size())
    config = DeepseekV41Config.from_pretrained(fixture, local_files_only=True)
    config.name_or_path = str(fixture)
    if config.text_config.num_hidden_layers > 3 or config.text_config.hidden_size != 5120:
        raise ValueError("expected the full-width bounded prefix fixture")
    model = NeMoAutoModelForCausalLM.from_config(
        config, load_base_model=True, distributed_setup=setup, torch_dtype=torch.bfloat16,
        backend=BackendConfig(attn="tilelang", linear="torch", rms_norm="torch_fp32", experts="torch_mm",
                              dispatcher="torch", gate_precision="float32", rope_fusion=False,
                              fake_balanced_gate=False, enable_hf_state_dict_adapter=True),
        trust_remote_code=False, force_hf=False, use_liger_kernel=False, use_sdpa_patching=False,
        freeze_config={"freeze_modules": [{"glob": "*"}]})
    model.requires_grad_(False).eval()
    identity["moe_precision"] = install_official_fp32_moe(model)
    identity["hc_precision"] = install_official_native_hc(model, fixture)
    return model, setup, identity


@contextmanager
def capture_prefix(model, *, official: bool):
    """Capture complete 128-token boundaries with reversible observation hooks."""
    import torch
    from torch import nn

    captured, handles, methods = {}, [], []
    counts = {}

    def save(key, value):
        if isinstance(value, torch.Tensor):
            count = counts.get(key, 0)
            counts[key] = count + 1
            key = key if count == 0 else f"{key}.call{count}"
            captured[key] = value.detach().cpu().clone()

    def watch(key, module, kind="tensor"):
        if not isinstance(module, nn.Module):
            return

        def before(_module, args):
            # Native Engram executes immediately before Block.forward; the
            # official block owns Engram internally. HC inputs are the common
            # post-Engram boundary, while the two raw block inputs differ.
            if args and kind != "block":
                save(key + ".input", args[0])

        def after(_module, _args, output):
            if kind == "mix":
                for field in ("pre", "post", "comb"):
                    save(key + "." + field, getattr(output, field))
            elif kind == "gate":
                save(key + ".weights", output[0])
                save(key + ".indices", output[1])
            elif kind == "block":
                save(key + ".output", output[0])
                save(key + ".next_pre", output[1])
            else:
                save(key + ".output", getattr(output, "hidden_states", output))

        handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    def observe_method(owner, name, callback):
        original = getattr(owner, name)
        had_own = name in owner.__dict__
        own = owner.__dict__.get(name)

        def observed(*args, **kwargs):
            output = original(*args, **kwargs)
            callback(args, output)
            return output

        setattr(owner, name, observed)
        methods.append((owner, name, had_own, own))

    backbone = model.model if official else model
    layers = list(backbone.layers.values()) if official else list(backbone.layers)
    watch("embed", backbone.embed_tokens if official else backbone.embed)
    watch("engram_hash", getattr(backbone, "engram_hash", None))
    watch("norm", backbone.norm)
    for index, layer in enumerate(layers):
        key = f"layers.{index}"
        watch(key, layer, "block")
        for name in ("attn_norm", "ffn_norm", "attn", "ffn"):
            watch(key + "." + name, getattr(layer, name))
        for name in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_b"):
            watch(key + ".attn." + name, getattr(layer.attn, name, None))
        watch(key + ".ffn.gate", layer.ffn.gate, "gate")
        watch(key + ".ffn.shared_experts", layer.ffn.shared_experts)
        if layer.engram is not None:
            for suffix, module in (("", layer.engram), (".embed", layer.engram.embed), (".wkv", layer.engram.wkv)):
                watch(key + ".engram" + suffix, module)
        if layer.attn.compressor is not None:
            # Native compressor forward also rotates/quantizes its final output;
            # projection/norm boundaries are directly comparable before that.
            for name in ("wkv", "wgate", "norm"):
                watch(key + ".attn.compressor." + name, getattr(layer.attn.compressor, name, None))
        if official:
            for site in ("attn", "ffn"):
                connection = getattr(layer, site + "_hc")
                watch(key + "." + site + "_mix", connection, "mix")
                observe_method(connection, "collapse", lambda args, value, k=key, s=site:
                               save(k + "." + s + "_collapsed", value))
                observe_method(connection, "expand", lambda args, value, k=key, s=site:
                               save(k + "." + s + "_streams", value))
        else:
            counters = {"mix": 0, "collapse": 0, "expand": 0}

            def mix(args, output, k=key, c=counters):
                site = ("attn", "ffn")[c["mix"]]
                c["mix"] += 1
                save(k + "." + site + "_mix.input", args[0])
                for field, value in zip(("pre", "post", "comb"), output, strict=True):
                    save(k + "." + site + "_mix." + field, value)

            def collapse(args, output, k=key, c=counters):
                site = ("attn", "ffn", "final")[c["collapse"]]
                c["collapse"] += 1
                save(k + "." + site + "_collapsed", output)

            def expand(args, output, k=key, c=counters):
                site = ("attn", "ffn")[c["expand"]]
                c["expand"] += 1
                save(k + "." + site + "_streams", output)

            observe_method(layer, "hc_mixes", mix)
            observe_method(layer, "hc_pre", collapse)
            observe_method(layer, "hc_post", expand)
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()
        for owner, name, had_own, own in reversed(methods):
            if had_own:
                setattr(owner, name, own)
            else:
                delattr(owner, name)


def compare_snapshots(reference, actual):
    """Compare matching boundaries, retaining shape/dtype mismatches explicitly."""
    import torch
    from archlab.automodel.deepseek_v41_diagnostic import _compare_tensors

    rows = []
    for key in reference.keys() & actual.keys():
        left, right = reference[key], actual[key]
        row = {"boundary": key, "reference_shape": list(left.shape), "official_shape": list(right.shape),
               "reference_dtype": str(left.dtype), "official_dtype": str(right.dtype)}
        # Flattened token layouts differ for some shared-expert boundaries.
        if left.numel() == right.numel() and left.shape != right.shape:
            right = right.reshape_as(left)
            row["layout_normalized"] = True
        row.update(_compare_tensors(left, right))
        if left.shape == right.shape and left.is_floating_point() and right.is_floating_point():
            row["numeric"] = _compare_tensors(left.float(), right.float())
        if key.endswith(".indices") and left.shape == right.shape:
            row["same_selected_expert_set"] = bool(torch.equal(left.sort(-1).values, right.sort(-1).values))
        rows.append(row)
    return {"comparisons": sorted(rows, key=lambda row: row["boundary"]),
            "reference_execution_order": list(reference), "official_execution_order": list(actual),
            "reference_only": sorted(reference.keys() - actual.keys()),
            "official_only": sorted(actual.keys() - reference.keys())}


def run(args):
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages
    kernels = select_container_kernel_packages(args.container_kernel_packages)
    import torch
    import torch.distributed as dist
    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_official_reference import build_matching_precision_reference, reference_hidden
    from archlab.automodel.deepseek_v41_official_qualification import compare_hidden
    from archlab.automodel.deepseek_v41_training import emit

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120),
                            device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank = dist.get_rank()
    report = {"kind": "three-layer-real-weight-prefix-diagnostic", "production_qualification": False,
              "rank": rank, "kernels": kernels, "completed": False}
    try:
        fixture = args.output / "fixture"
        if rank == 0:
            prepare_prefix_fixture(assets=args.assets, weights=args.weights, output=fixture, layers=args.layers)
        dist.barrier()
        report["fixture"] = json.loads((fixture / "PREFIX_DERIVATION.json").read_text())
        emit("official_prefix_construct_start", fixture=str(fixture))
        model, setup, report["runtime"] = build_official_prefix(fixture, ep_size=args.ep_size)
        pilot = MathPilot(args.pilot, expected_split="train", expected_budget=1_000_000_000)
        inputs, labels, _ = pilot.batch(rank, device="cuda", smoke_context=128, pad_to_full=True)
        report["input_ids"] = inputs.cpu().tolist()
        with torch.no_grad(), capture_prefix(model, official=True) as actual:
            actual_hidden = model(input_ids=inputs, return_hidden_states=True).hidden_states.detach().cpu().clone()
        emit("official_prefix_forward_complete", boundaries=len(actual))
        reference, native, report["reference_loading"] = build_matching_precision_reference(
            assets=fixture, weights=fixture, expert_group=setup.mesh_context.moe_mesh["ep"].get_group(),
            engram_group=dist.group.WORLD, context=128, module_name="_archlab_v41_prefix_reference")
        with capture_prefix(native, official=False) as expected:
            expected_hidden = reference_hidden(native, inputs)
        emit("native_prefix_forward_complete", boundaries=len(expected))
        report.update(compare_snapshots(expected, actual))
        report["logits"] = compare_hidden(expected_hidden, actual_hidden, labels, native.head)
        report["completed"] = True
        if args.save_tensors:
            torch.save({"native": expected, "official": actual}, args.output / f"rank{rank}-boundaries.pt")
        atomic_write_json(args.output / f"rank{rank}.json", report, allow_nan=False)
        emit("prefix_diagnostic_complete", prefix_kl=report["logits"]["native_to_candidate_kl"])
        dist.barrier()
    except BaseException:
        report["error"] = traceback.format_exc()
        atomic_write_json(args.output / f"rank{rank}.json", report, allow_nan=False)
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "weights", "pilot", "output", "container-kernel-packages"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--save-tensors", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Bounded inference driver, executed as a file with checkpoint-pinned PYTHONPATH.

The driver source and the model source have independent recorded identities.
Never import today's training implementation into an archived model process.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archived_module(source, filename):
    path = Path(source) / "src/archlab/automodel" / filename
    spec = importlib.util.spec_from_file_location("reasoning_helper_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_source(source, commit):
    for args, expected in ((["rev-parse", "HEAD"], commit), (["status", "--porcelain"], "")):
        if (
            subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
            != expected
        ):
            raise ValueError("archived model source is not the expected clean commit")


def load_adapter(phase, weights, assets):
    import torch

    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_official_execution import build_official_base

    checkpoint = Path(phase["checkpoint"])
    marker = json.loads((checkpoint / "COMPLETE.json").read_text())
    contract = marker["contract"]
    if marker["cursor"]["step"] != 307 or marker["cursor"]["supervised_tokens"] != 50119869:
        raise ValueError("wrong matched adapter cursor")
    if (
        contract.get("adapter_variant", "simplicial") != phase["variant"]
        or contract["project_commit"] != phase["model_commit"]
    ):
        raise ValueError("wrong adapter variant/source")
    for relative, digest in contract["implementation_sha256"].items():
        if sha(Path(phase["model_source"]) / "src/archlab" / relative) != digest:
            raise ValueError(f"adapter trained source changed: {relative}")
    for name, key in (
        ("config.json", "base_config_sha256"),
        ("model.safetensors.index.json", "checkpoint_index_sha256"),
    ):
        if sha(weights / name) != contract[key]:
            raise ValueError(f"released base identity changed: {name}")
    state_path = checkpoint / "adapter-state.pt"
    if (
        state_path.stat().st_size != marker["state_bytes"]
        or sha(state_path) != marker["state_sha256"]
    ):
        raise ValueError("adapter state checksum mismatch")
    model, setup, loading = build_official_base(weights=weights, assets=assets, ep_size=8)
    for key in ("container_image", "cuda", "nccl", "reproducibility"):
        if loading[key] != contract[key]:
            raise ValueError(f"adapter runtime changed: {key}")
    if loading["packages"] != contract["runtime"]:
        raise ValueError("adapter package runtime changed")
    variant_arg = (
        {"variant": phase["variant"]}
        if "variant" in inspect.signature(install_official_adapters).parameters
        else {}
    )
    if not variant_arg and phase["variant"] != "simplicial":
        raise ValueError("legacy adapter implementation only supports simplicial")
    adapters = install_official_adapters(
        model,
        V41AdapterConfig(**contract.get("adapter_config", {})),
        layer_indices=tuple(contract.get("adapter_layers_0based", (4, 9, 14, 19, 24, 29, 34, 39))),
        device="cuda",
        **variant_arg,
        backend="deterministic" if phase["variant"] == "simplicial" else "flash-attn-deterministic",
    )
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["cursor"] != marker["cursor"]:
        raise ValueError("adapter marker/state cursor mismatch")
    for layer, adapter in adapters.items():
        adapter.load_state_dict(state["adapters"][str(layer)], strict=True)
    if (
        sum(p.numel() for adapter in adapters.values() for p in adapter.parameters())
        != contract["trainable_parameters"]
    ):
        raise ValueError("adapter parameter count differs from trained contract")
    del state
    model.requires_grad_(False)
    model.eval()
    return (
        model,
        setup,
        dict(
            cursor=marker["cursor"],
            state_sha256=marker["state_sha256"],
            optimizer_created=False,
            mesh_change="32 to 16; held-out CE qualification required",
            runtime=loading,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    phase = plan["phases"][args.phase]
    verify_source(phase["model_source"], phase["model_commit"])
    verify_source(plan["helper_source"], plan["helper_commit"])
    if sha(__file__) != plan["driver_sha256"]:
        raise ValueError("evaluation driver changed")
    bundle = Path(plan["bundle"])
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    if (
        sha(bundle / "MANIFEST.json") != plan["manifest_sha256"]
        or sha(bundle / "cases.json") != manifest["cases_sha256"]
    ):
        raise ValueError("sealed public cases changed")
    rows = json.loads((bundle / "cases.json").read_text())
    config = manifest["config"]
    assets, weights = Path(plan["assets"]), Path(plan["weights"])
    for path, key in (
        (assets / "tokenizer.json", "tokenizer_sha256"),
        (assets / "encoding/encoding.py", "encoder_sha256"),
    ):
        if sha(path) != manifest[key]:
            raise ValueError("native tokenizer identity changed")
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(Path(os.environ["ARCHLAB_CONTAINER_KERNEL_PACKAGES"]))
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_official_execution import configure_official_reproducibility

    configure_official_reproducibility()
    torch.set_grad_enabled(False)
    torch.set_num_threads(2)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    free, total = torch.cuda.mem_get_info()
    if free < config["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("insufficient co-location memory headroom")
    torch.cuda.set_per_process_memory_fraction(config["gpu_allocator_limit_gib"] * 2**30 / total)
    torch.manual_seed(42)
    dist.init_process_group(
        "nccl", timeout=datetime.timedelta(minutes=90), device_id=torch.device("cuda", local)
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    output = Path(plan["output"]) / ("tiny-" + phase["name"] if args.tiny else phase["name"])
    output.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    helper = archived_module(plan["helper_source"], "deepseek_v41_live_window.py")
    try:
        if world != 16 or (output / "COMPLETE.json").exists():
            raise ValueError("requires a new 16-rank evaluation output")
        if rank == 0:
            atomic_write_json(
                output / "RUN.json",
                dict(
                    phase=phase,
                    driver_sha256=sha(__file__),
                    helper_commit=plan["helper_commit"],
                    cases_sha256=manifest["cases_sha256"],
                    optimizer_created=False,
                    training_updates_enabled=False,
                    tiny=args.tiny,
                ),
            )
        if args.tiny:
            from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
            from archlab.automodel.deepseek_v41_full_eval_construct import build_eval_shell
            from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters

            model, setup, receipt = build_eval_shell(weights=weights, assets=assets, tiny=True)
            install_official_adapters(
                model,
                V41AdapterConfig(width=256),
                layer_indices=(1, 3, 5),
                device="cuda",
                variant=phase["variant"],
                backend="deterministic"
                if phase["variant"] == "simplicial"
                else "flash-attn-deterministic",
            )
            model.requires_grad_(False)
            model.eval()
            rows = rows[:world]
        elif phase["kind"] == "full":
            marker = json.loads((Path(phase["checkpoint"]) / "COMPLETE.json").read_text())
            if (
                marker["cursor"]["step"] != 4537
                or marker["cursor"]["supervised_tokens"] != phase["expected_tokens"]
            ):
                raise ValueError("wrong matched full checkpoint cursor")
            model, setup, receipt = helper.construct_secondary(
                Path(phase["checkpoint"]),
                phase["variant"],
                weights=weights,
                assets=assets,
                registry=plan["source_registry"],
            )
        else:
            model, setup, receipt = load_adapter(phase, weights, assets)
        atomic_write_json(output / f"rank-{rank:02d}-restore.json", receipt)
        if not args.tiny:
            from archlab.automodel.deepseek_v41_data import MathPilot

            validation = MathPilot(
                Path(plan["validation_pilot"]), expected_split="validation", expected_budget=1000000
            )
            evaluation = archived_module(plan["helper_source"], "deepseek_v41_full_validation.py")
            validation_plan = evaluation.make_plan(validation, 1000000)

            def paced_validation_batches():
                entries = validation_plan["windows"]
                for first in range(0, len(entries), world):
                    tick = time.monotonic()
                    yield evaluation.capped_batch(
                        validation,
                        entries[first + rank] if first + rank < len(entries) else None,
                        "cuda",
                    )
                    torch.cuda.synchronize()
                    delay = torch.tensor(time.monotonic() - tick, device="cuda")
                    dist.all_reduce(delay, op=dist.ReduceOp.MAX)
                    time.sleep(min(300, float(delay) * 9))

            metric = evaluation.evaluate_batches(model, paced_validation_batches())
            if metric["targets"] != 1000000:
                raise ValueError("qualification target count changed")
            metric["plan_sha256"] = validation_plan["sha256"]
            error = metric["loss"] - phase["reference_ce"]
            passed = abs(error) <= config["qualification_ce_atol"]
            if rank == 0:
                atomic_write_json(
                    output / "NUMERICAL_QUALIFICATION.json",
                    dict(
                        passed=passed,
                        metric=metric,
                        reference_ce=phase["reference_ce"],
                        reference=phase["reference"],
                        error=error,
                    ),
                )
            if not passed:
                raise ValueError(f"checkpoint held-out CE does not match: delta={error}")
        tokenizer = AutoTokenizer.from_pretrained(
            assets, local_files_only=True, trust_remote_code=False
        )
        predictions = []
        for first in range(0, len(rows), world):
            if time.monotonic() - began > config["maximum_phase_hours"] * 3600:
                raise TimeoutError("bounded evaluation expired")
            batch = rows[first : first + world]
            row = batch[rank] if rank < len(batch) else batch[0]
            ids = list(row["input_ids"])
            context = max(
                128,
                1
                << (
                    max(len(r["input_ids"]) for r in batch) + config["max_new_tokens"] - 1
                ).bit_length(),
            )
            if context > config["max_context"]:
                raise ValueError("evaluation context overflow")
            generated, done = [], False
            tick = time.monotonic()
            for _ in range(config["max_new_tokens"]):
                logits = helper.next_token_logits(model, ids, context)
                token = int(logits.argmax())
                if not done:
                    generated.append(token)
                    if token == manifest["eos_id"]:
                        done = True
                    else:
                        ids.append(token)
                # EOS ranks continue forwards, preserving all FSDP/EP collectives.
                active = torch.tensor(int(not done), device="cuda")
                dist.all_reduce(active, op=dist.ReduceOp.MAX)
                if not bool(active):
                    break
            torch.cuda.synchronize()
            local_result = dict(
                id=row["id"],
                token_ids=generated,
                text=tokenizer.decode(generated, skip_special_tokens=True),
                finish_reason="eos" if done else "length",
                context=context,
                seconds=time.monotonic() - tick,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            )
            gathered = [None] * world
            dist.all_gather_object(gathered, local_result)
            if rank == 0:
                predictions.extend(gathered[: len(batch)])
                with (output / "predictions.jsonl").open("a") as stream:
                    for item in gathered[: len(batch)]:
                        stream.write(json.dumps(item) + "\n")
                atomic_write_json(
                    output / "PROGRESS.json",
                    dict(completed=len(predictions), total=len(rows), unix=time.time()),
                )
            # Ten percent inference duty cycle initially; the controller can
            # increase rest when the independent trainer slows down.
            if first + world >= len(rows):
                continue
            pacing = Path(plan["output"]) / "PACING.json"
            sleep_packet = [None]
            if rank == 0:
                factor = (
                    json.loads(pacing.read_text()).get("rest_factor", 9) if pacing.exists() else 9
                )
                sleep_packet[0] = min(300, (time.monotonic() - tick) * factor)
            dist.broadcast_object_list(sleep_packet, src=0)
            time.sleep(sleep_packet[0])
        dist.barrier()
        if rank == 0:
            atomic_write_json(
                output / "COMPLETE.json",
                dict(
                    cases_sha256=manifest["cases_sha256"],
                    cases=len(rows),
                    seconds=time.monotonic() - began,
                    tiny=args.tiny,
                    checkpoint=phase["checkpoint"],
                    matched_training_tokens=phase["expected_tokens"],
                    variant=phase["variant"],
                    kind=phase["kind"],
                ),
            )
        del model, setup
    except BaseException:
        atomic_write_json(
            output / f"FAILED-rank-{rank:02d}.json", dict(traceback=traceback.format_exc())
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

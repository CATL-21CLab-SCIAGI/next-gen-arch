"""Qualify the official V4.1 adapter backward and resume on a tiny EP/FSDP mesh.

Launch with torchrun on 8 or 32 GPUs. This uses the production adapter head and
window geometry on a width-256, six-layer random backbone; it does not qualify
pretrained parity, production memory, or the production Muon optimizer.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import socket
import traceback
from dataclasses import asdict
from pathlib import Path


def _tensor_digest(named_tensors):
    """Hash exact local bytes without gathering FSDP-owned base weights."""
    import torch
    from torch.distributed.tensor import DTensor

    digest = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        digest.update(json.dumps([name, list(local.shape), str(local.dtype)]).encode())
        raw = local.detach().contiguous().reshape(-1).view(torch.uint8)
        for chunk in raw.split(8 * 1024 * 1024):
            digest.update(chunk.cpu().numpy().tobytes())
    return digest.hexdigest()


def _gather(value, group=None):
    import torch.distributed as dist

    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _require_all(errors):
    failures = _gather(errors)
    if any(failures):
        grouped = {}
        for rank, messages in enumerate(failures):
            for message in messages:
                grouped.setdefault(message, []).append(rank)
        summary = [{"ranks": ranks, "error": message} for message, ranks in grouped.items()]
        raise RuntimeError(f"distributed tiny-mesh qualification failed: {summary}")


def _fresh_output(path):
    import torch.distributed as dist

    status = [None]
    if dist.get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            status[0] = f"output must be fresh: {error}"
    dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise ValueError(status[0])


def _mesh_receipt(model, setup, ep_size):
    import torch.distributed as dist
    from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram
    from torch.distributed.tensor import DTensor

    mesh = setup.mesh_context
    ep_group = mesh.moe_mesh["ep"].get_group()
    ep_hosts = _gather(socket.gethostname(), group=ep_group)
    ep_shard = mesh.moe_mesh["ep_shard"]
    errors = []
    if len(ep_hosts) != ep_size or len(set(ep_hosts)) != 1:
        errors.append("expert group is not node-local EP8")
    if ep_shard.size() != dist.get_world_size() // ep_size:
        errors.append("expert FSDP mesh has the wrong size")
    engrams = []
    for name, module in model.named_modules():
        if isinstance(module, DeepseekV41Engram):
            table = module.embed
            ranks = ([] if table.process_group is None
                     else dist.get_process_group_ranks(table.process_group))
            weight = table.weight
            if ranks != list(range(dist.get_world_size())) or not isinstance(weight, DTensor):
                errors.append(f"{name} does not have WORLD-owned Engram shards")
            engrams.append({"module": name, "owner_ranks": ranks,
                            "global_shape": list(weight.shape),
                            "local_shape": list(weight.to_local().shape) if isinstance(weight, DTensor)
                            else list(weight.shape)})
    if not engrams:
        errors.append("tiny backbone did not construct an Engram layer")
    _require_all(errors)
    return {"ep_ranks": dist.get_process_group_ranks(ep_group), "ep_hosts": ep_hosts,
            "expert_fsdp_ranks": ep_shard.mesh.tolist(), "engrams": engrams}


def _adapter_receipt(adapters, *, require_nonzero):
    import torch
    from torch.distributed.tensor import DTensor

    named = [(f"{index}.{name}", parameter) for index, adapter in adapters.items()
             for name, parameter in adapter.named_parameters()]
    errors, gradients = [], []
    for name, parameter in named:
        if isinstance(parameter, DTensor) or parameter.dtype != torch.float32:
            errors.append(f"{name}: adapter must remain a replicated FP32 master")
        if parameter.grad is None or not bool(parameter.grad.isfinite().all()):
            errors.append(f"{name}: missing or nonfinite gradient")
        else:
            nonzero = bool(parameter.grad.count_nonzero())
            if require_nonzero and not nonzero:
                errors.append(f"{name}: zero second-update gradient")
            gradients.append({"name": name, "nonzero": nonzero,
                              "max_abs": float(parameter.grad.abs().max())})
    _require_all(errors)
    gradient_sha = _tensor_digest([(name, parameter.grad) for name, parameter in named])
    parameter_sha = _tensor_digest(named)
    copies = _gather({"gradient_sha256": gradient_sha, "adapter_sha256": parameter_sha})
    if any(value != copies[0] for value in copies):
        raise RuntimeError("adapter parameters or globally reduced gradients differ across ranks")
    return {"replicated_fp32_masters": True, "world_gradient_agreement": True,
            "world_parameter_agreement": True, "gradient_sha256": gradient_sha,
            "adapter_sha256": parameter_sha, "gradients": gradients}


def _batch(rank, context):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(2234 + rank)
    tokens = torch.randint(128, 1000, (1, context + 1), generator=generator, device="cpu")
    tokens[0, 0] = 128 + rank  # Different input windows on every data rank.
    # Both slices can already be contiguous and still alias the same token
    # storage. Clone before masking labels so -100 never reaches input_ids.
    inputs, labels = tokens[:, :-1].clone(), tokens[:, 1:].clone()
    labels[:, :8] = -100
    return inputs.cuda(), labels.cuda()


def _diagnostic_comparison(before, expected_metric, expected, actual_metric, actual):
    """Measure variation even when the existing admission tolerance rejects it."""
    import torch

    from archlab.automodel.deepseek_v41_official_training import (
        REPLAY_TOLERANCES,
        compare_replayed_update,
        state_difference_statistics,
    )

    comparison_error = None
    try:
        compare_replayed_update(expected_metric, expected, actual_metric, actual)
    except AssertionError as error:
        comparison_error = str(error)
    failures = _gather(comparison_error)

    def norms(initial, reference, observed):
        error_squared = reference_squared = update_squared = 0.0

        def visit(start, target, value):
            nonlocal error_squared, reference_squared, update_squared
            if isinstance(target, torch.Tensor):
                chunks = [tensor.detach().cpu().reshape(-1).split(1024 * 1024)
                          for tensor in (start, target, value)]
                for a, b, c in zip(*chunks, strict=True):
                    a, b, c = a.double(), b.double(), c.double()
                    error_squared += float((c - b).square().sum())
                    reference_squared += float(b.square().sum())
                    update_squared += float((b - a).square().sum())
            elif isinstance(target, dict):
                for key in target:
                    visit(start[key], target[key], value[key])
            elif isinstance(target, (list, tuple)):
                for a, b, c in zip(start, target, value, strict=True):
                    visit(a, b, c)

        visit(initial, reference, observed)
        return {"error_l2": math.sqrt(error_squared), "reference_l2": math.sqrt(reference_squared),
                "update_l2": math.sqrt(update_squared),
                "global_relative_l2": math.sqrt(error_squared / max(reference_squared, 1e-300)),
                "update_relative_l2": math.sqrt(error_squared / max(update_squared, 1e-300))}

    unique_failures = list(dict.fromkeys(error for error in failures if error is not None))
    return {"within_existing_tolerance": not any(failures),
            "comparison_errors": unique_failures,
            "failing_ranks": [rank for rank, error in enumerate(failures) if error is not None],
            "replay_tolerance": dict(REPLAY_TOLERANCES["updated_state"]),
            "loss_tolerance": dict(REPLAY_TOLERANCES["loss"]),
            "loss": {"expected": expected_metric["loss"], "actual": actual_metric["loss"],
                     "abs_difference": abs(actual_metric["loss"] - expected_metric["loss"])},
            **{component: {**state_difference_statistics(expected[component], actual[component]),
                           **norms(before[component], expected[component], actual[component])}
               for component in ("adapters", "optimizers")}}


def _frozen_base_receipt(model, frozen_names, frozen_sha):
    current = dict(model.named_parameters())
    errors = [name for name in frozen_names if name not in current or current[name].requires_grad
              or current[name].grad is not None]
    _require_all(errors)
    final_sha = _tensor_digest([(name, current[name]) for name in frozen_names])
    _require_all([] if final_sha == frozen_sha else ["frozen local base bytes changed"])
    return {"unchanged": True, "no_gradients": True,
            "local_sha256": frozen_sha, "parameter_tensors": len(frozen_names)}


def qualify(*, weights, assets, output, ep_size=8, context=128, also_512=False,
            diagnose_replay=False, diagnostic_repeats=3, adapter_variant="simplicial"):
    import torch
    import torch.distributed as dist

    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters
    from archlab.automodel.deepseek_v41_official_execution import build_official_base
    from archlab.automodel.deepseek_v41_official_training import (
        REPLAY_PROTOCOL,
        assert_exact_training_state,
        capture_adapter_training_state,
        optimizer_step,
        qualify_checkpoint_replay,
        restore_in_memory_training_state,
        state_difference_statistics,
        training_state_sha256,
    )
    from archlab.automodel.deepseek_v41_training import (
        emit,
        restore_adapter_checkpoint,
        save_adapter_checkpoint,
    )

    if dist.get_world_size() not in (8, 32) or ep_size != 8 or context not in (128, 512):
        raise ValueError("probe requires world 8/32, node-local EP8, and context 128/512")
    model, setup, base_report = build_official_base(weights=weights, assets=assets,
                                                   ep_size=ep_size, tiny=True)
    report = {**base_report, "mesh": _mesh_receipt(model, setup, ep_size),
              "context": context, "optimizer": "AdamW probe only", "passed": False}
    frozen_names = set(dict(model.named_parameters()))
    frozen_sha = _tensor_digest(model.named_parameters())
    config = V41AdapterConfig(width=256)
    adapter_backend = "deterministic" if adapter_variant == "simplicial" else "flash-attn-deterministic"
    adapters = install_official_adapters(model, config, layer_indices=(4,), device="cuda",
                                         backend=adapter_backend, variant=adapter_variant)
    parameters = [parameter for adapter in adapters.values() for parameter in adapter.parameters()]
    optimizers = [torch.optim.AdamW(parameters, lr=1e-5, betas=(0.9, 0.95), eps=1e-8,
                                    weight_decay=0.0, foreach=False)]
    report["adapter"] = {"config": asdict(config), "layers_0based": [4], "backend": adapter_backend,
                         "variant": adapter_variant,
                         "trainable_parameters": sum(parameter.numel() for parameter in parameters)}
    model.train()
    inputs, labels = _batch(dist.get_rank(), context)
    windows = _gather(_tensor_digest([("input_ids", inputs)]))
    _require_all([] if len(set(windows)) == dist.get_world_size() else ["duplicate data-rank inputs"])
    report["unique_rank_windows"] = True
    report["updates"] = []
    for index in range(2):
        metric = optimizer_step(model, optimizers, inputs, labels, learning_rate=1e-5)
        receipt = _adapter_receipt(adapters, require_nonzero=index == 1)
        report["updates"].append({**metric, **receipt})
        emit("official_tiny_mesh_update", step=index + 1, **metric)

    contract = {"probe": "official-v41-tiny-mesh-v3", "world_size": dist.get_world_size(),
                "ep_size": ep_size, "context": context, "adapter": report["adapter"],
                "automodel_commit": base_report["automodel_commit"],
                "replay_protocol": REPLAY_PROTOCOL}
    cursor = {"step": 2, "supervised_tokens": sum(row["supervised_tokens"] for row in report["updates"])}
    checkpoint = output / "checkpoint-step2"
    if not diagnose_replay:
        report["checkpoint_replay"] = qualify_checkpoint_replay(
            model, adapters, optimizers, inputs, labels, path=checkpoint, cursor=cursor, contract=contract)
        report["checkpoint_replay"].update(_adapter_receipt(adapters, require_nonzero=True))
        emit("official_tiny_mesh_checkpoint_replay_passed", protocol=REPLAY_PROTOCOL["version"])
        if also_512 and context != 512:
            long_inputs, long_labels = _batch(dist.get_rank(), 512)
            metric = optimizer_step(model, optimizers, long_inputs, long_labels, learning_rate=1e-5)
            report["context512_update"] = {**metric, **_adapter_receipt(adapters, require_nonzero=True)}
            emit("official_tiny_mesh_context512_update", **metric)
        report["frozen_base"] = _frozen_base_receipt(model, frozen_names, frozen_sha)
        report["passed"] = True
        return report
    saved_state = capture_adapter_training_state(adapters, optimizers)
    saved_sha = training_state_sha256(saved_state)
    save_adapter_checkpoint(checkpoint, adapters, optimizers, cursor, contract)
    expected_metric = optimizer_step(model, optimizers, inputs, labels, learning_rate=1e-5)
    expected_state = capture_adapter_training_state(adapters, optimizers)
    restored_cursor = restore_adapter_checkpoint(checkpoint, adapters, optimizers, contract=contract)
    _require_all([] if restored_cursor == cursor else ["checkpoint changed the data cursor"])
    errors = []
    try:
        restoration = assert_exact_training_state(saved_state, capture_adapter_training_state(adapters, optimizers))
    except AssertionError as error:
        errors.append(f"checkpoint exact restoration: {error}")
    _require_all(errors)
    replay_metric = optimizer_step(model, optimizers, inputs, labels, learning_rate=1e-5)
    replay_state = capture_adapter_training_state(adapters, optimizers)
    if diagnose_replay:
        diagnostic = {"saved_state_sha256": saved_sha, "checkpoint_replay": {
            "exact_state_restoration": True, "exact_restoration": restoration,
            "metric": replay_metric,
            **_diagnostic_comparison(saved_state, expected_metric, expected_state, replay_metric, replay_state),
            **_adapter_receipt(adapters, require_nonzero=True)}, "in_memory_repeats": []}
        for repeat in range(diagnostic_repeats):
            restore_in_memory_training_state(saved_state, adapters, optimizers)
            errors = []
            try:
                exact = assert_exact_training_state(saved_state, capture_adapter_training_state(adapters, optimizers))
            except AssertionError as error:
                errors.append(f"diagnostic in-memory exact restoration: {error}")
            _require_all(errors)
            metric = optimizer_step(model, optimizers, inputs, labels, learning_rate=1e-5)
            state = capture_adapter_training_state(adapters, optimizers)
            measured = {"repeat": repeat + 1, "exact_state_restoration": True,
                        "exact_restoration": exact, "metric": metric,
                        **_diagnostic_comparison(saved_state, expected_metric, expected_state, metric, state),
                        "difference_from_checkpoint_replay": {
                            component: state_difference_statistics(replay_state[component], state[component])
                            for component in ("adapters", "optimizers")},
                        **_adapter_receipt(adapters, require_nonzero=True)}
            diagnostic["in_memory_repeats"].append(measured)
            emit("official_tiny_mesh_replay_diagnostic", repeat=repeat + 1,
                 within_existing_tolerance=measured["within_existing_tolerance"],
                 adapters=measured["adapters"], optimizers=measured["optimizers"], loss=measured["loss"])
        report.update(diagnostic_only=True, passed=False, replay_diagnostic=diagnostic,
                      frozen_base=_frozen_base_receipt(model, frozen_names, frozen_sha))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--container-kernel-packages", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=128, choices=(128, 512))
    parser.add_argument("--also-512", action="store_true")
    parser.add_argument("--adapter-variant", choices=("simplicial", "normal"), default="simplicial")
    parser.add_argument("--diagnose-replay", action="store_true",
                        help="measure rejected replay differences; never publish a passing qualification")
    parser.add_argument("--diagnostic-repeats", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    from archlab.automodel.deepseek_v41_runtime import select_container_kernel_packages

    select_container_kernel_packages(args.container_kernel_packages)
    import torch
    import torch.distributed as dist

    from archlab.artifacts import atomic_write_json, sha256_file
    from archlab.automodel.deepseek_v41_official_execution import official_core_hashes, runtime_identity

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120),
                            device_id=torch.device("cuda", local_rank))
    receipt = {"rank": dist.get_rank(), "world_size": dist.get_world_size(),
               "hostname": socket.gethostname(), "passed": False}
    fresh_output = False
    try:
        _fresh_output(args.output)
        fresh_output = True
        receipt.update(runtime_identity())
        receipt["implementation_sha256"] = official_core_hashes()
        receipt["probe_sha256"] = sha256_file(Path(__file__))
        receipt.update(qualify(weights=args.weights, assets=args.assets, output=args.output,
                               ep_size=args.ep_size, context=args.context, also_512=args.also_512,
                               diagnose_replay=args.diagnose_replay, diagnostic_repeats=args.diagnostic_repeats,
                               adapter_variant=args.adapter_variant))
        suffix = "-diagnostic" if args.diagnose_replay else ""
        atomic_write_json(args.output / f"rank{dist.get_rank()}{suffix}.json", receipt, allow_nan=False)
        dist.barrier()
        if dist.get_rank() == 0:
            marker = "DIAGNOSTIC_COMPLETE.json" if args.diagnose_replay else "COMPLETE.json"
            atomic_write_json(args.output / marker,
                              {"passed": not args.diagnose_replay, "diagnostic_only": args.diagnose_replay,
                               "world_size": dist.get_world_size(),
                               "rank_receipts": [f"rank{rank}{suffix}.json" for rank in range(dist.get_world_size())]},
                              allow_nan=False)
    except BaseException as error:
        receipt.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if fresh_output:
            atomic_write_json(args.output / f"rank{dist.get_rank()}-failed.json", receipt, allow_nan=False)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

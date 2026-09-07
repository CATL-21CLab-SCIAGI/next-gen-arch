"""CPU training contracts and an executable two-GPU clipping/reference check.

Run pytest for CPU contracts, or torchrun this file for the distributed oracle.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from archlab.automodel.checkpointing import read_training_checkpoint, state_digest, write_json
from archlab.automodel.training_config import TrainingConfig


def test_scheduler_warmup_decay_and_fresh_restore():
    pytest.importorskip("nemo_automodel", reason="requires the pinned upstream scheduler")
    config = replace(TrainingConfig(), warmup_steps=2)
    model = torch.nn.Linear(2, 2)
    optimizer = config.build_optimizer(model)
    scheduler = config.build_scheduler(optimizer, total_steps=8)
    assert optimizer.param_groups[0]["lr"] == config.initial_lr
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == pytest.approx((config.initial_lr + config.peak_lr) / 2)
    scheduler.step(1)
    assert optimizer.param_groups[0]["lr"] == config.peak_lr
    saved = scheduler.state_dict()
    restored_optimizer = config.build_optimizer(model)
    restored = config.build_scheduler(restored_optimizer, total_steps=8)
    restored.load_state_dict(saved)
    assert restored.state_dict() == saved
    for _ in range(6):
        scheduler.step(1)
        restored.step(1)
        assert optimizer.param_groups[0]["lr"] == restored_optimizer.param_groups[0]["lr"]
    assert optimizer.param_groups[0]["lr"] == config.minimum_lr


def test_optimizer_excludes_frozen_parameters_and_invalid_config():
    config = TrainingConfig()
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    model[0].requires_grad_(False)
    optimizer = config.build_optimizer(model)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in model[1].parameters()}
    for changes in ({"initial_lr": float("nan")}, {"micro_batch": 2}, {"ep_size": 3},
                    {"warmup_steps": 0}, {"peak_lr": 1e-8}):
        with pytest.raises(ValueError):
            replace(config, **changes)


def test_checkpoint_completion_contract_and_cursor(tmp_path):
    config = asdict(TrainingConfig())
    contract = {"training": config, "total_steps": 20}
    with pytest.raises(FileNotFoundError):
        read_training_checkpoint(tmp_path, contract)
    metadata = {"format": "archlab-simplicial-training-v1", "contract": contract,
                "cursor": 3, "completed_steps": 3, "rank_state_sha256": ["example"] * config["world_size"]}
    write_json(tmp_path / "COMPLETE.json", metadata)
    assert read_training_checkpoint(tmp_path, contract)["cursor"] == 3
    with pytest.raises(ValueError, match="contract"):
        read_training_checkpoint(tmp_path, {**contract, "total_steps": 21})
    for changes in ({"cursor": -1}, {"cursor": 4}, {"rank_state_sha256": []},
                    {"cursor": 21, "completed_steps": 21}):
        write_json(tmp_path / "COMPLETE.json", {**metadata, **changes})
        with pytest.raises(ValueError):
            read_training_checkpoint(tmp_path, contract)


def test_state_digest_handles_scalars_empty_and_bfloat16():
    state = {"step": torch.tensor(2.), "empty": torch.empty(0), "weights": torch.ones(2, 3, dtype=torch.bfloat16)}
    assert state_digest(state) == state_digest({key: value.clone() for key, value in state.items()})
    before = state_digest(state)
    state["weights"][0, 0] = 2
    assert state_digest(state) != before


def test_smoke_windows_resume_targets_are_disjoint():
    pytest.importorskip("nemo_automodel", reason="requires the pinned upstream training entry")
    from archlab.automodel.train import _SmokeWindows

    config = replace(TrainingConfig(), world_size=2, ep_size=2, sequence_length=5)
    data = _SmokeWindows(config, 7)
    targets = []
    for cursor in range(8):
        for rank in range(2):
            args = dict(rank=rank, world_size=2, micro_batch=1, device="cpu")
            batch = data.batch(cursor, **args)
            torch.testing.assert_close(batch["input_ids"], data.batch(cursor, **args)["input_ids"])
            targets.append(batch["labels"].flatten())
    torch.testing.assert_close(torch.cat(targets), data.tokens[1:])
    with pytest.raises(IndexError):
        data.batch(8, rank=0, world_size=2, micro_batch=1, device="cpu")


def distributed_clipping_reference() -> None:
    """Compare FSDP gradients, global clipping norm and Adam update with a CPU oracle."""
    import copy
    import os

    import torch.distributed as dist
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(19)
        reference = torch.nn.Linear(8, 4, bias=False)
        model = copy.deepcopy(reference).cuda()
        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp_shard",))
        fully_shard(model, mesh=mesh)
        source = torch.arange(dist.get_world_size() * 16, dtype=torch.float32).reshape(-1, 8) / 10
        reference(source).square().mean().backward()
        model(source[dist.get_rank() * 2:(dist.get_rank() + 1) * 2].cuda()).square().mean().backward()
        torch.testing.assert_close(model.weight.grad.full_tensor().cpu(), reference.weight.grad, rtol=2e-6, atol=1e-6)
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), .1, foreach=False)
        norm = scale_grads_and_clip_grad_norm(.1, [model], device_mesh=mesh, foreach=False)
        # The upstream sharding-aware norm accumulates in FP64; torch's plain
        # CPU clipping primitive reports FP32. Compare values in a common dtype.
        torch.testing.assert_close(norm.cpu(), expected_norm.double(), rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(model.weight.grad.full_tensor().cpu(), reference.weight.grad, rtol=2e-6, atol=1e-6)
        config = TrainingConfig()
        optimizer, reference_optimizer = config.build_optimizer(model), config.build_optimizer(reference)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.weight.full_tensor().cpu(), reference.weight, rtol=2e-6, atol=1e-7)
        print(json.dumps({"event": "distributed_clipping_reference_pass", "rank": dist.get_rank(),
                          "world_size": dist.get_world_size(), "global_norm": norm.item()}), flush=True)
    finally:
        dist.destroy_process_group()


def compare_resume_checkpoints(before: Path, uninterrupted: Path, resumed: Path) -> None:
    """Compare the tiny resumed update against the same saved-state continuation."""
    import tempfile

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    from archlab.automodel.checkpointing import assert_state_equal

    states = []
    with tempfile.TemporaryDirectory(prefix="archlab-resume-oracle-") as temporary:
        for index, path in enumerate((before, uninterrupted, resumed)):
            metadata = json.loads((path / "COMPLETE.json").read_text())
            if not metadata["contract"]["tiny"]:
                raise ValueError("the materializing comparison is restricted to tiny fixtures")
            output = Path(temporary) / f"state-{index}.pt"
            dcp_to_torch_save(path / "state", output)
            states.append(torch.load(output, weights_only=True, map_location="cpu"))
    first, reference, actual = states
    assert_state_equal(actual["scheduler"], reference["scheduler"])
    assert_state_equal(actual["optimizer"]["param_groups"], reference["optimizer"]["param_groups"])
    for key in reference:
        if key.startswith("rng_rank_"):
            assert_state_equal(actual[key], reference[key])
    error, update_norm, maximum = 0., 0., 0.
    for layer, parameters in reference["adapters"].items():
        for name, expected in parameters.items():
            difference = (actual["adapters"][layer][name] - expected).double()
            delta = (expected - first["adapters"][layer][name]).double()
            error += difference.square().sum().item()
            update_norm += delta.square().sum().item()
            maximum = max(maximum, difference.abs().max().item())
    relative = (error / max(update_norm, 1e-30)) ** .5
    if relative > .01:
        raise AssertionError(f"fresh-process update replay error exceeds 1%: {relative}")
    for index, saved in reference["optimizer"]["state"].items():
        assert_state_equal(actual["optimizer"]["state"][index]["step"], saved["step"])
    print(json.dumps({"event": "fresh_process_update_reference_pass", "update_relative_l2": relative,
                      "update_max_abs": maximum, "scheduler_rng_steps_exact": True}), flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 4:
        compare_resume_checkpoints(*(Path(value) for value in sys.argv[1:]))
    else:
        distributed_clipping_reference()

from __future__ import annotations

import json

import pytest
import torch

from archlab.speedrun.checkpoint import (
    capture_rng_state,
    load_rng_state,
    restore_rng_state,
    save_checkpoint,
    verify_checkpoint_bundle,
)


def test_checkpoint_bundle_is_atomic_and_complete(tmp_path):
    metadata = {"step": 7, "model_config": {"depth": 1}}
    save_checkpoint(
        tmp_path,
        7,
        {"weight": torch.ones(2)},
        {"state": torch.arange(2)},
        metadata,
        rank=0,
        rng_data=capture_rng_state(),
    )
    save_checkpoint(
        tmp_path,
        7,
        None,
        {"state": torch.arange(3)},
        metadata,
        rank=1,
        rng_data=capture_rng_state(),
    )

    bundle = verify_checkpoint_bundle(tmp_path, 7, world_size=2, require_rng=True)
    assert bundle["optimizer_shards"] == 2
    assert bundle["rng_shards"] == 2
    assert {item["path"] for item in bundle["files"]} == {
        "model_000007.pt",
        "meta_000007.json",
        "optim_000007_rank0.pt",
        "optim_000007_rank1.pt",
        "rng_000007_rank0.pt",
        "rng_000007_rank1.pt",
    }
    assert not list(tmp_path.glob("*.tmp*"))


def test_checkpoint_bundle_rejects_missing_rank_or_wrong_step(tmp_path):
    save_checkpoint(
        tmp_path,
        3,
        {"weight": torch.ones(1)},
        {"state": {}},
        {"step": 3},
        rank=0,
    )
    with pytest.raises(RuntimeError, match="missing"):
        verify_checkpoint_bundle(tmp_path, 3, world_size=2)

    meta_path = tmp_path / "meta_000003.json"
    meta_path.write_text(json.dumps({"step": 2}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="step mismatch"):
        verify_checkpoint_bundle(
            tmp_path,
            3,
            world_size=1,
        )


def test_rng_checkpoint_round_trip(tmp_path):
    torch.manual_seed(11)
    state = capture_rng_state()
    expected = torch.rand(4)
    torch.manual_seed(99)
    save_checkpoint(
        tmp_path,
        1,
        {"weight": torch.ones(1)},
        None,
        {"step": 1},
        rng_data=state,
    )

    restored = load_rng_state(tmp_path, 1, torch.device("cpu"))
    assert restored is not None
    restore_rng_state(restored)
    assert torch.equal(torch.rand(4), expected)

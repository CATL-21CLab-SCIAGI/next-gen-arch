from __future__ import annotations

import json
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from archlab.automodel import limite_adapter_rl as rl
from archlab.automodel import limite_native_checkpoint as native
from archlab.automodel.checkpoint_oracle import assert_state_equal
from archlab.automodel.checkpoint_publication import CheckpointPublisher
from archlab.rl.limite_checkpoint import capture_rng, restore_rng


class NativePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.weight = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
        self.model.scale = nn.Parameter(torch.ones((), dtype=torch.float32))
        self.register_buffer("position", torch.arange(2), persistent=True)


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "publisher"
    root.mkdir()
    (root / "model.safetensors").write_bytes(b"CPU fixture, loader is mocked")
    (root / "config.json").write_text("{}")
    receipt = dict(
        repo="paradigma-inc/limite-1b-violetto", revision="fixture-revision",
        files=[dict(path=name, sha256=native.file_hash(root / name))
               for name in ("model.safetensors", "config.json")],
    )
    (root / "DOWNLOAD_VERIFIED.json").write_text(json.dumps(receipt))
    return root


@pytest.fixture
def load_native(monkeypatch):
    calls = []

    def load(snapshot, **kwargs):
        calls.append((snapshot, kwargs))
        return NativePolicy()

    monkeypatch.setattr(native, "load_model", load)
    return calls


def test_native_load_retains_publisher_topology_and_mixed_dtypes(snapshot, load_native):
    model = native.build_native_model(snapshot, "cpu")
    assert load_native == [(snapshot, dict(attn_implementation="sdpa", device_map="cpu"))]
    assert not hasattr(model.model, "adapters")
    assert model.model.trainable_mode == "full"
    assert all(p.requires_grad for p in model.parameters())
    assert model.model.weight.dtype == torch.bfloat16
    assert model.model.scale.dtype == torch.float32
    assert rl.identity_contract(model, {"publisher_identity": native.publisher_identity(snapshot)}) == {
        "model_kind": "native", "trainable_mode": "full",
        "publisher_identity": model.archlab_publisher_identity,
    }


@pytest.mark.parametrize("override,error", [
    ({"warmup": "fake-10B-parent"}, "warmup parent"),
    ({"attention_backend": "tilelang"}, "publisher training attention"),
    ({"normal_kernel": "gqa"}, "publisher training attention"),
    ({"trainable_mode": "adapter"}, "all weights"),
    ({"runtime_sequence_attention": True}, "kernel migration"),
    ({"allow_backend_migration": True}, "kernel migration"),
])
def test_native_launch_rejects_adapter_contract_changes(override, error):
    options = dict(variant="native", warmup=None, attention_backend="native",
                   trainable_mode="full", normal_kernel=None,
                   runtime_sequence_attention=False, allow_backend_migration=False)
    native.check_native_options(**options)
    with pytest.raises(ValueError, match=error):
        native.check_native_options(**(options | override))


def test_native_identity_rejects_frozen_weights_and_foreign_parent(snapshot, load_native):
    model = native.build_native_model(snapshot, "cpu")
    receipt = {"publisher_identity": model.archlab_publisher_identity}
    model.model.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="every model parameter"):
        rl.identity_contract(model, receipt)
    model.model.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="different publisher"):
        rl.identity_contract(model, {"publisher_identity": {"revision": "other"}})
    model.model.adapters = nn.ModuleList()
    with pytest.raises(ValueError, match="inserted adapters"):
        rl.identity_contract(model, receipt)


def test_native_publisher_receipt_checks_weights_and_safe_paths(snapshot):
    (snapshot / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        native.publisher_identity(snapshot)
    receipt = json.loads((snapshot / "DOWNLOAD_VERIFIED.json").read_text())
    receipt["files"] = [dict(path="../outside", sha256="untrusted")]
    (snapshot / "DOWNLOAD_VERIFIED.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unsafe payload path"):
        native.publisher_identity(snapshot)


def test_native_publisher_and_tokenizer_are_bound_to_declared_checkpoint(snapshot, tmp_path):
    identity = native.publisher_identity(snapshot)
    spec = dict(model=dict(repo=identity["repo"], revision=identity["revision"]))
    native.check_native_publisher(snapshot, snapshot, spec, identity)
    alias = tmp_path / "alias"
    alias.symlink_to(snapshot, target_is_directory=True)
    native.check_native_publisher(snapshot, alias, spec, identity)
    for change in (dict(repo="paradigma-inc/limite-1b-base"), dict(revision="other")):
        with pytest.raises(ValueError, match="differs from the experiment"):
            native.check_native_publisher(snapshot, snapshot, spec, identity | change)
    with pytest.raises(ValueError, match="tokenizer must come from"):
        native.check_native_publisher(snapshot, tmp_path / "different-tokenizer", spec, identity)


def test_native_recipe_seals_eight_rank_budget_and_allows_short_correctness_probe():
    spec = yaml.safe_load((Path(__file__).parents[1] / "recipes/limite/violetto_math_rl_async.yaml").read_text())
    native.check_native_recipe(spec, 8, 400)
    native.check_native_recipe(spec, 8, 3, correctness_fixture=True)
    for world, steps in ((16, 400), (8, 128)):
        with pytest.raises(ValueError, match="training budget differ"):
            native.check_native_recipe(spec, world, steps)
    with pytest.raises(ValueError, match="explicit math protocol"):
        native.check_native_recipe(None, 8, 400)
    for change in (dict(overlap_actor_learner=True), dict(rollout_rendezvous="none"), dict(max_policy_lag=2)):
        changed = deepcopy(spec)
        changed["execution"].update(change)
        with pytest.raises(ValueError, match="qualified drain"):
            native.check_native_recipe(changed, 8, 400)
    changed = deepcopy(spec)
    changed["training"]["gradient_accumulation_steps"] = 4
    with pytest.raises(ValueError, match="accumulation budget"):
        native.check_native_recipe(changed, 8, 400)


def test_native_optimizer_selects_fp32_masters_without_casting_weights(snapshot, load_native, monkeypatch):
    model = native.build_native_model(snapshot, "cpu")
    captured = {}

    def optimizer(parameters, **kwargs):
        captured.update(kwargs, parameters=list(parameters))
        return captured

    monkeypatch.setitem(sys.modules, "archlab.optimizers.rl_adam",
                        SimpleNamespace(SignalFusedAdam=optimizer))
    rl.optimizer_for_model(model)
    assert captured["master_weights"] is True
    assert captured["master_weight_dtype"] == captured["exp_avg_dtype"] == captured["exp_avg_sq_dtype"] == torch.float32
    assert model.model.weight.dtype == torch.bfloat16
    assert model.model.scale.dtype == torch.float32


def test_native_checkpoint_exact_model_optimizer_rng_and_resume(snapshot, load_native, tmp_path):
    model = native.build_native_model(snapshot, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    rng = capture_rng()
    payloads = {"trainer_state.json": dict(global_step=10),
                "rl_state.pt": dict(world_size=1, rank_rng=[rng], scheduler={"last_epoch": 10})}
    dest = native.save_native_checkpoint(model, optimizer, tmp_path / "run", 10, 0,
                                        extra={"publisher_snapshot": str(snapshot)}, extra_payloads=payloads)
    receipt = json.loads((dest / "COMPLETE.json").read_text())
    assert receipt["model_kind"] == "native" and "adapter" not in receipt
    assert "warmup_checkpoint" not in receipt and "adapter.pt" not in receipt["files"]
    assert set(receipt["files"]) == {"model.pt", "optimizer.pt", "rng.pt", "trainer_state.json", "rl_state.pt"}
    restored = native.build_native_model(snapshot, "cpu", dest, checkpoint_cache=tmp_path / "cache")
    assert_state_equal(restored.state_dict(), model.state_dict())
    other_optimizer = torch.optim.Adam(restored.parameters(), lr=.1)
    other_optimizer.load_state_dict(torch.load(dest / "optimizer.pt", weights_only=True))
    assert_state_equal(other_optimizer.state_dict(), optimizer.state_dict())
    for p, other in zip(model.parameters(), restored.parameters(), strict=True):
        p.grad = torch.full_like(p, .125)
        other.grad = p.grad.clone()
    optimizer.step()
    other_optimizer.step()
    assert_state_equal(restored.state_dict(), model.state_dict())
    assert_state_equal(other_optimizer.state_dict(), optimizer.state_dict())
    saved = torch.load(dest / "rl_state.pt", weights_only=True)
    restore_rng(saved["rank_rng"][0])
    expected = torch.rand(8)
    restore_rng(rng)
    assert torch.equal(torch.rand(8), expected)


def test_native_checkpoint_rejects_changed_dtype_and_corruption(snapshot, load_native, tmp_path):
    model = native.build_native_model(snapshot, "cpu")
    optimizer = torch.optim.Adam(model.parameters())
    dest = native.save_native_checkpoint(model, optimizer, tmp_path / "run", 1, 0)
    state = torch.load(dest / "model.pt", weights_only=True)
    state["model.scale"] = state["model.scale"].bfloat16()
    torch.save(state, dest / "model.pt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        native.build_native_model(snapshot, "cpu", dest)
    receipt = json.loads((dest / "COMPLETE.json").read_text())
    receipt["files"]["model.pt"] = native.file_hash(dest / "model.pt")
    (dest / "COMPLETE.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="dtypes differ"):
        native.build_native_model(snapshot, "cpu", dest)


def test_native_async_publication_freezes_model_masters_and_prefetch(snapshot, load_native, tmp_path):
    model = native.build_native_model(snapshot, "cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=.1)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    first = optimizer.state[next(iter(model.parameters()))]
    first["master_param"] = model.model.weight.float().detach().clone()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    master = first["master_param"].clone()
    pending = dict(generator=torch.get_rng_state().clone(), pending=dict(completion_ids=[[42, 151645]]))
    release = threading.Event()

    class PausedPublisher(CheckpointPublisher):
        def submit(self, publish, *args):
            def delayed():
                assert release.wait(10)
                return publish(*args)
            return super().submit(delayed)

    publisher = PausedPublisher()
    dest = native.save_native_checkpoint(
        model, optimizer, tmp_path / "run", 10, 0, tmp_path / "oss",
        extra_payloads={"rl_state.pt": dict(rank_rng=[dict(rollout=pending)])}, publisher=publisher,
    )
    with torch.no_grad():
        model.model.weight.add_(1)
        first["master_param"].add_(1)
    pending["pending"]["completion_ids"][0][0] = 99
    release.set()
    publisher.close()
    assert_state_equal(torch.load(dest / "model.pt", weights_only=True), before)
    saved_optimizer = torch.load(dest / "optimizer.pt", weights_only=True)
    assert torch.equal(saved_optimizer["state"][0]["master_param"], master)
    saved = torch.load(dest / "rl_state.pt", weights_only=True)
    assert saved["rank_rng"][0]["rollout"]["pending"]["completion_ids"] == [[42, 151645]]
    assert (tmp_path / "run" / "published" / dest.name).is_symlink()
    assert all((dest / name).is_symlink() for name in json.loads((dest / "COMPLETE.json").read_text())["files"])

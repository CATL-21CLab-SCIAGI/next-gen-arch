"""CPU evidence for independent layer fitting, sealed data, and exact optimizer resume."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch import nn

from archlab.training import latent_match as lm


class Mixer(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.projection = nn.Linear(width, width)

    def forward(self, hidden_states, **kwargs):
        return self.projection(hidden_states).tanh()


class Body(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(13, 4)
        self.mixers = nn.ModuleList([Mixer(), Mixer()])

    def forward(self, input_ids, **kwargs):
        hidden = self.embed(input_ids)
        for mixer in self.mixers:
            hidden = hidden + mixer(hidden_states=hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Body()
        self.head = nn.Linear(4, 13)

    def get_output_embeddings(self):
        return self.head


def setup():
    torch.manual_seed(9)
    model = Teacher().eval().requires_grad_(False)
    mixers = dict(enumerate(model.model.mixers))
    switches = {i: copy.deepcopy(module).requires_grad_(True) for i, module in mixers.items()}
    with torch.no_grad():
        for module in switches.values():
            module.projection.weight.add_(0.1)
    optimizer = torch.optim.AdamW([p for m in switches.values() for p in m.parameters()], lr=0.02, weight_decay=0)
    return model, mixers, switches, optimizer


def build_source(tmp_path, name="source", groups=None, identity=None):
    root = tmp_path / name
    root.mkdir()
    groups = groups or [f"doc-{i}" for i in range(100)]
    documents = [np.array([i % 11] * (4 + i % 4), dtype=np.int32) for i in range(len(groups))]
    np.save(root / "tokens.npy", np.concatenate(documents))
    np.save(root / "offsets.npy", np.array([0, *np.cumsum([len(d) for d in documents])], dtype=np.int64))
    (root / "groups.json").write_text(json.dumps(groups))
    files = {}
    for key, suffix in (("tokens", "npy"), ("offsets", "npy"), ("groups", "json")):
        path = root / f"{key}.{suffix}"
        files[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"format": "archlab-latent-documents-v1", "tokenizer_files": identity or {"tokenizer.json": "a"}, "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return lm.SourceConfig(str(root / "manifest.json")), groups


def test_document_splits_are_disjoint_reproducible_and_windows_never_cross(tmp_path):
    source, groups = build_source(tmp_path)
    data = lm.MixtureWindows([source], 4, tokenizer_files={"tokenizer.json": "a"}, vocab_size=13)
    tokens, offsets, selections = data.parts[0]
    assert set(selections[False]).isdisjoint(selections[True])
    assert {groups[i] for i in selections[False]}.isdisjoint({groups[i] for i in selections[True]})
    for heldout in (False, True):
        first = data.sample(np.random.default_rng(4), 80, heldout=heldout)
        torch.testing.assert_close(first, data.sample(np.random.default_rng(4), 80, heldout=heldout), rtol=0, atol=0)
        assert (first == first[:, :1]).all()  # Each document is constant; joining documents would violate this.
        assert set(first[:, 0].tolist()) <= {i % 11 for i in selections[heldout]}
    # A group appearing in different sources remains on the same side.
    second, _ = build_source(tmp_path, "other", groups=list(reversed(groups)))
    mixed = lm.MixtureWindows([source, second], 4, tokenizer_files={"tokenizer.json": "a"}, vocab_size=13)
    heldout_sets = [{sequence[i] for i in part[2][True]}
                    for sequence, part in zip((groups, list(reversed(groups))), mixed.parts, strict=True)]
    assert heldout_sets[0] == heldout_sets[1]
    assert len(tokens) == offsets[-1]


def test_exact_length_documents_are_sampleable_and_bad_provenance_is_rejected(tmp_path):
    source, _ = build_source(tmp_path)
    with pytest.raises(ValueError, match="tokenizer"):
        lm.MixtureWindows([source], 4, tokenizer_files={"tokenizer.json": "DIFFERENT"}, vocab_size=13)
    with pytest.raises(ValueError, match="vocabulary"):
        lm.MixtureWindows([source], 4, tokenizer_files={"tokenizer.json": "a"}, vocab_size=4)
    manifest_path = tmp_path / "source/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tokens"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        lm.MixtureWindows([source], 4, tokenizer_files={"tokenizer.json": "a"}, vocab_size=13)


def test_exact_tokenizer_fingerprint(tmp_path):
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (tmp_path / name).write_text(name)
    actual = lm.tokenizer_identity(tmp_path)
    assert set(actual) == {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja"}
    (tmp_path / "tokenizer.json").write_text("changed")
    assert actual != lm.tokenizer_identity(tmp_path)


def test_teacher_identity_pins_all_indexed_weight_shards(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors", "c": "part1.safetensors"}}))
    for name in ("part1.safetensors", "part2.safetensors"):
        (tmp_path / name).write_bytes(name.encode())
    before = lm.teacher_identity(tmp_path)
    assert len(before) == 4
    (tmp_path / "part2.safetensors").write_bytes(b"different weights")
    assert before != lm.teacher_identity(tmp_path)


def test_layerwise_update_matches_microbatch_accumulation_and_freezes_teacher():
    model, mixers, switches, optimizer = setup()
    other_model, other_mixers, other_switches, other_optimizer = setup()
    teacher_before = {k: v.clone() for k, v in model.state_dict().items()}
    ids = torch.tensor([[1, 2, 3, 4], [3, 4, 5, 6], [2, 1, 0, 1]])
    result = lm.train_step(model, mixers, switches, optimizer, ids, microbatch_windows=1)
    lm.train_step(other_model, other_mixers, other_switches, other_optimizer, ids, microbatch_windows=3)
    assert result["grad_norm"] > 0
    for i in switches:
        for a, b in zip(switches[i].parameters(), other_switches[i].parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    assert all(not module._forward_hooks for module in mixers.values())


def test_capture_and_patch_restore_even_when_execution_raises():
    model, mixers, switches, _ = setup()
    originals = [module.forward for module in mixers.values()]
    with pytest.raises(RuntimeError, match="injected"):
        with lm.Patched(mixers, switches), lm.capture(mixers):
            raise RuntimeError("injected")
    assert [module.forward for module in mixers.values()] == originals
    assert all(not module._forward_hooks for module in mixers.values())
    with lm.capture(mixers) as store:
        model.model(input_ids=torch.tensor([[1, 2, 3]]))
    assert all(not hidden.requires_grad and not target.requires_grad for hidden, _, target in store.values())


def test_layer_losses_do_not_depend_on_other_student_layers():
    model, mixers, switches, _ = setup()
    with lm.capture(mixers) as store:
        model.model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    losses = dict(lm.layer_losses(switches, store))
    losses[1].backward()
    assert all(p.grad is None for p in switches[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in switches[1].parameters())


def test_composed_evaluation_chunks_match_full_logits_and_restores_teacher():
    model, mixers, switches, _ = setup()
    windows = torch.tensor([[1, 2, 3, 4], [3, 4, 2, 1]])
    chunked = lm.evaluate(model, mixers, switches, windows, torch.device("cpu"), chunk_tokens=1)
    full = lm.evaluate(model, mixers, switches, windows, torch.device("cpu"), chunk_tokens=16)
    for key in ("kl", "ce_teacher", "ce_student", "layer_error_mean", "layer_normalized_error_mean"):
        assert chunked[key] == pytest.approx(full[key], rel=2e-6, abs=1e-7)
    identity = {i: copy.deepcopy(module) for i, module in mixers.items()}
    zero = lm.evaluate(model, mixers, identity, windows, torch.device("cpu"), chunk_tokens=1)
    assert zero["kl"] == pytest.approx(0, abs=1e-8)
    assert zero["layer_error_mean"] == 0
    assert zero["layer_normalized_error_mean"] == 0
    assert zero["ce_teacher"] == zero["ce_student"]
    assert all(not module._forward_hooks for module in mixers.values())


def _distributed_update(rank, rendezvous, output):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model, mixers, switches, optimizer = setup()
        ids = torch.tensor([[1, 2, 3, 4], [4, 1, 3, 2]])
        result = lm.train_step(model, mixers, switches, optimizer, ids[rank:rank + 1])
        torch.save({"metrics": result, "state": {i: m.state_dict() for i, m in switches.items()}}, output / f"{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_update_matches_one_rank_global_batch(tmp_path):
    model, mixers, switches, optimizer = setup()
    expected = lm.train_step(model, mixers, switches, optimizer, torch.tensor([[1, 2, 3, 4], [4, 1, 3, 2]]))
    torch.multiprocessing.spawn(_distributed_update, args=(tmp_path / "rendezvous", tmp_path), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path / f"{rank}.pt", weights_only=True)
        for key, value in expected.items():
            assert actual["metrics"][key] == pytest.approx(value, rel=2e-6, abs=1e-8)
        for index, module in switches.items():
            for name, value in module.state_dict().items():
                torch.testing.assert_close(actual["state"][index][name], value, rtol=1e-6, atol=1e-7)


def test_training_objective_sums_layer_mean_squared_errors():
    model, mixers, switches, optimizer = setup()
    ids = torch.tensor([[1, 2, 3, 4], [4, 1, 3, 2]])
    with torch.no_grad(), lm.capture(mixers) as store:
        model.model(input_ids=ids)
        expected = float(sum(loss for _, loss in lm.layer_losses(switches, store)))
    result = lm.train_step(model, mixers, switches, optimizer, ids)
    assert result["train_loss"] == pytest.approx(expected)
    assert result["train_layer_error_mean"] == pytest.approx(expected / len(switches))


def test_checkpoint_resume_reproduces_next_data_optimizer_update_and_rng(tmp_path):
    model, mixers, switches, optimizer = setup()
    rng = np.random.default_rng(12)
    ids = torch.from_numpy(rng.integers(0, 13, size=(2, 5)))
    lm.train_step(model, mixers, switches, optimizer, ids)
    path = lm.save_checkpoint(tmp_path, 1, switches, optimizer, [lm.rng_state(rng, torch.device("cpu"))], "same")
    next_ids = torch.from_numpy(rng.integers(0, 13, size=(2, 5)))
    expected_random = torch.rand(4)
    expected = lm.train_step(model, mixers, switches, optimizer, next_ids)
    restored_model, restored_mixers, restored_switches, restored_optimizer = setup()
    restored_rng = np.random.default_rng(999)
    assert lm.load_checkpoint(path, restored_switches, restored_optimizer, restored_rng, "same") == 1
    restored_ids = torch.from_numpy(restored_rng.integers(0, 13, size=(2, 5)))
    torch.testing.assert_close(restored_ids, next_ids, rtol=0, atol=0)
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    actual = lm.train_step(restored_model, restored_mixers, restored_switches, restored_optimizer, restored_ids)
    assert actual == expected
    for i in switches:
        for a, b in zip(switches[i].parameters(), restored_switches[i].parameters(), strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match="another experiment"):
        lm.load_checkpoint(path, restored_switches, restored_optimizer, restored_rng, "different")
    with pytest.raises(FileExistsError):
        lm.save_checkpoint(tmp_path, 1, switches, optimizer, [lm.rng_state(rng, torch.device("cpu"))], "same")


def test_run_directory_rejects_mixing_runs_or_old_resume(tmp_path):
    output = tmp_path / "run"
    contract = {"variant": "triadic", "tokens": 64}
    with lm.run_directory(output, contract):
        (output / "checkpoints").mkdir()
        latest = output / "checkpoints/step-00000002.pt"
        latest.touch()
        old = output / "checkpoints/step-00000001.pt"
        old.touch()
    with pytest.raises(FileExistsError), lm.run_directory(output, contract):
        pass
    with pytest.raises(ValueError, match="differs"), lm.run_directory(output, {"variant": "simplicial"}, latest):
        pass
    with pytest.raises(ValueError, match="latest"), lm.run_directory(output, contract, old):
        pass
    with lm.run_directory(output, contract, latest):
        with pytest.raises(BlockingIOError), lm.run_directory(output, contract, latest):
            pass


def test_contract_refuses_rounded_budgets_and_raw_ids():
    values = dict(teacher="teacher", output="new-run", variant="triadic", sources=(lm.SourceConfig("manifest"),),
                  sequence_length=8, batch_tokens=32, tokens=64, eval_windows=4, lr=0.001)
    run = lm.RunConfig(**values)
    assert run.windows_per_rank(2) == 2
    with pytest.raises(ValueError, match="divide"):
        run.windows_per_rank(3)
    with pytest.raises(ValueError, match="whole"):
        lm.RunConfig(**{**values, "tokens": 65})
    with pytest.raises(ValueError, match="raw MSE"):
        lm.RunConfig(**{**values, "loss": "normalized_mse"})
    assert lm.lr_at(99, 100, 1.0, 0.03, 0.1) == pytest.approx(0.1)
    assert lm.lr_at(0, 100, 1.0, 0.03, 0.1) == pytest.approx(1 / 3)

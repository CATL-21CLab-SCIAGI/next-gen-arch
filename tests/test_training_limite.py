"""Exact weights, precision, and native computation at the package boundary."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForCausalLM

from archlab.architectures.limite_adapter import (
    LimiteAdapterConfig,
    install_adapters,
    set_trainable_mode,
)
from archlab.architectures.limite_loader import upstream_classes
from archlab.artifacts import atomic_write_json, sha256_file
from archlab.training import limite

SNAPSHOT = Path("/mnt/oss/models/limite-1b-base-cc612bafcd4a")


@pytest.fixture
def native():
    if not SNAPSHOT.exists():
        pytest.skip("verified publisher fixture unavailable")
    config_class, model_class = upstream_classes(SNAPSHOT)
    config = config_class.from_pretrained(SNAPSHOT, local_files_only=True)
    for name, value in dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        vocab_size=37, padded_vocab_size=37, max_position_embeddings=128,
        global_layers=[1, 3], layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=5, _serialized_sliding_window=4, rope_n_pairs=2, attn_gate_channels=8,
        ve_dim=8, ve_stored_heads=2, ve_layers=[1, 2], ve_gate_channels=4,
        xsa_layers=[0, 1, 2, 3], mudd_layers=[2, 3], mudd_at=[2, 3],
        mudd_tap_idx={"2": [0, 1, 2], "3": [0, 2, 3]}, mudd_inter=4,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
    ).items():
        setattr(config, name, value)
    config._attn_implementation = "sdpa"
    torch.manual_seed(76)
    model = model_class(config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.numel() == 1:
                parameter.fill_(1.)
            else:
                parameter.normal_(std=.1)
            if parameter.ndim == 2 and "mudd" not in name:
                parameter.data = parameter.data.to(torch.bfloat16)
    return model


def policy(native, variant="normal"):
    model = install_adapters(native, LimiteAdapterConfig(variant=variant))
    set_trainable_mode(model, "full")
    with torch.no_grad():
        for adapter in model.model.adapters:
            adapter.native.o_proj.weight.normal_(std=.01)
    return model


def test_serving_weight_refresh_keeps_graph_addresses_and_updates_every_fold(native):
    from archlab.training.serving import _refresh_inference_buffers

    reference = policy(deepcopy(native))
    model = policy(native).eval()
    before = {name: (buffer.data_ptr(), buffer.clone())
              for name, buffer in model.named_buffers(remove_duplicate=False)
              if "._inference_" in name}
    assert before
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.125)
    reference.load_state_dict(model.state_dict())
    reference.train().eval()
    _refresh_inference_buffers(model)
    expected = dict(reference.named_buffers(remove_duplicate=False))
    changed = 0
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if name in before:
            pointer, previous = before[name]
            assert buffer.data_ptr() == pointer, name
            torch.testing.assert_close(buffer, expected[name], rtol=0, atol=0)
            changed += int(not torch.equal(buffer, previous))
    assert changed > 10, "the fixture must exercise projections, gates, residuals and MUDD"


@pytest.mark.parametrize("variant", ["normal", "simplicial"])
def test_lossless_auto_model_roundtrip_and_global_cast_preserves_dtypes(tmp_path, native, variant):
    expected = policy(native, variant)
    root = tmp_path / "export"
    receipt = limite.export_model(expected, SNAPSHOT, root)
    assert not receipt["optimizer_migrated"]
    config = AutoConfig.from_pretrained(root, trust_remote_code=False)
    assert config.model_type == "archlab_limite"
    actual = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=False)
    assert len(actual.policy.model.adapters) == len(expected.model.adapters)
    assert actual.policy.model.adapter_config == expected.model.adapter_config
    actual.to(dtype=torch.float32)
    actual.to(dtype=torch.bfloat16)
    for name, parameter in actual.policy.named_parameters():
        other = dict(expected.named_parameters())[name]
        assert parameter.dtype == other.dtype
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)
    assert actual.get_input_embeddings().weight is actual.get_output_embeddings().weight


def test_uninserted_publisher_export_is_lossless_and_full_weight(tmp_path, native):
    native.train()
    root = tmp_path / "export"
    limite.export_model(native, SNAPSHOT, root)
    config = AutoConfig.from_pretrained(root)
    assert config.adapter == {} and config.conversion == {} and config.trainable_mode == "full"
    actual = AutoModelForCausalLM.from_pretrained(root).train()
    assert not hasattr(actual.policy.model, "adapters")
    ids = torch.tensor([[1, 4, 8, 5, 2], [1, 3, 9, 6, 2]])
    want = native(input_ids=ids, use_cache=False, labels=ids)
    got = actual(input_ids=ids, use_cache=False, labels=ids)
    torch.testing.assert_close(got.logits, want.logits, rtol=0, atol=0)
    got.loss.backward()
    assert all(parameter.grad is not None for parameter in actual.parameters())
    with pytest.raises(ValueError, match="uninserted"):
        limite.LimiteForCausalLM(type(config)(**{**config.to_dict(), "trainable_mode": "adapter"}))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention-2 models construct only with CUDA")
def test_export_records_requested_backbone_attention_without_loading_it(tmp_path, native):
    root = tmp_path / "export"
    receipt = limite.export_model(native, SNAPSHOT, root, backbone_attention="flash_attention_2")
    assert AutoConfig.from_pretrained(root).backbone_attention == "flash_attention_2"
    actual = AutoModelForCausalLM.from_pretrained(root)
    assert actual.policy.config._attn_implementation == "flash_attention_2"
    assert limite._state_digest(actual.policy.state_dict()) == receipt["state_sha256"]


@pytest.mark.parametrize("adapter", [
    LimiteAdapterConfig(variant="gdn", heads=1, attention_backend="reference", compensation=8),
    LimiteAdapterConfig(variant="triadic", heads=1, feature_dim=4, attention_backend="reference"),
])
def test_fresh_delta_insertion_export_starts_at_publisher_with_teacher_template(tmp_path, native, adapter):
    template = Path("/mnt/oss/models/limite-1b-violetto-4cf321846e47")
    if not template.exists():
        pytest.skip("verified teacher publisher unavailable")
    ids = torch.tensor([[1, 4, 8, 5, 2], [1, 3, 9, 6, 2]])
    want = deepcopy(native).eval()(input_ids=ids, use_cache=False).logits
    root = tmp_path / "export"
    receipt = limite.export_model(install_adapters(native, adapter), SNAPSHOT, root, chat_template=template)
    assert receipt["insertion"]["output_state"] == "zero"
    assert receipt["insertion"]["trainable_mode"] == "adapter"
    assert receipt["chat_template"]["repo"] == "paradigma-inc/limite-1b-violetto"
    assert sha256_file(root / "chat_template.jinja") == sha256_file(template / "chat_template.jinja")
    assert sha256_file(root / "tokenizer.json") == sha256_file(SNAPSHOT / "tokenizer.json")
    actual = AutoModelForCausalLM.from_pretrained(root).train()
    got = actual(input_ids=ids, use_cache=False, labels=ids)
    torch.testing.assert_close(got.logits, want, rtol=0, atol=0)
    got.loss.backward()
    trainable = {name: p for name, p in actual.named_parameters() if p.requires_grad}
    assert trainable and all(".adapters." in name and p.grad is not None for name, p in trainable.items())
    assert any(p.grad.count_nonzero() for p in trainable.values())


@pytest.mark.parametrize("checkpoint_layers", [False, True])
def test_native_forward_backward_survives_auto_model_export(tmp_path, native, checkpoint_layers):
    expected = policy(native).train()
    root = tmp_path / "export"
    limite.export_model(expected, SNAPSHOT, root)
    actual = AutoModelForCausalLM.from_pretrained(root).train()
    if checkpoint_layers:
        actual.gradient_checkpointing_enable({"use_reentrant": False})
    ids = torch.tensor([[1, 4, 8, 5, 2], [1, 3, 9, 6, 2]])
    want = expected(input_ids=ids, use_cache=False, labels=ids)
    got = actual(input_ids=ids, use_cache=False, labels=ids)
    torch.testing.assert_close(got.logits, want.logits, rtol=0, atol=0)
    got.loss.backward()
    want.loss.backward()
    for name, parameter in actual.policy.named_parameters():
        other = dict(expected.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)


def test_legacy_export_assets_are_completed_without_changing_model_state(tmp_path, native):
    root = tmp_path / "export"
    receipt = limite.export_model(policy(native), SNAPSHOT, root)
    original_digest = receipt["state_sha256"]
    original_weights = receipt["files"]["model.safetensors"]
    for name in ("tokenizer.json", "generation_config.json"):
        (root / name).unlink()
        del receipt["files"][name]
    atomic_write_json(root / "EXPORT.json", receipt)
    completed = limite.add_publisher_assets(SNAPSHOT, root)
    assert completed["state_sha256"] == original_digest
    assert completed["files"]["model.safetensors"] == original_weights
    assert json.loads((root / "generation_config.json").read_text())["eos_token_id"] == 151645
    assert json.loads((root / "tokenizer_config.json").read_text())["eos_token"] == "<|endoftext|>"


def test_export_checksum_rejects_changed_weights_before_loading(tmp_path, native):
    root = tmp_path / "export"
    limite.export_model(policy(native), SNAPSHOT, root)
    with (root / "model.safetensors").open("ab") as payload:
        payload.write(b"changed")
    with pytest.raises(ValueError, match="export checksum mismatch"):
        AutoModelForCausalLM.from_pretrained(root)


def test_trained_insertion_export_changes_only_preludes_and_keeps_parent_assets(tmp_path, native):
    from safetensors.torch import save_model

    template = Path("/mnt/oss/models/limite-1b-violetto-4cf321846e47")
    if not template.exists():
        pytest.skip("verified teacher publisher unavailable")
    parent = tmp_path / "parent"
    limite.export_model(install_adapters(native, LimiteAdapterConfig(variant="normal")), SNAPSHOT, parent,
                        chat_template=template)

    def checkpoint(name, change):
        model = AutoModelForCausalLM.from_pretrained(parent)
        with torch.no_grad():
            change(model.policy)
        model.train().eval()
        root = tmp_path / name
        root.mkdir()
        save_model(model, root / "model.safetensors")
        model.config.save_pretrained(root)
        atomic_write_json(root / "CHECKPOINT.json", dict(files={p.name: sha256_file(p) for p in root.iterdir()}))
        return model, root / "CHECKPOINT.json"

    def train_preludes(policy):
        for adapter in policy.model.adapters:
            adapter.native.o_proj.weight.normal_(std=.01)

    trained, receipt_path = checkpoint("trained", train_preludes)
    receipt = limite.export_trained_insertion(parent, receipt_path, tmp_path / "export", chat_template=template)
    assert receipt["insertion"]["output_state"] == "trained"
    assert receipt["insertion"]["trainable_mode"] == "adapter"
    assert receipt["source_checkpoint"] == dict(path=str(receipt_path.parent.resolve()), receipt="CHECKPOINT.json",
                                                receipt_sha256=sha256_file(receipt_path))
    assert receipt["state_sha256"] == limite._state_digest(trained.policy.state_dict())
    ids = torch.tensor([[1, 4, 8, 5, 2], [1, 3, 9, 6, 2]])
    actual = AutoModelForCausalLM.from_pretrained(tmp_path / "export")
    want = trained(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual(input_ids=ids, use_cache=False).logits, want, rtol=0, atol=0)
    original = AutoModelForCausalLM.from_pretrained(parent)(input_ids=ids, use_cache=False).logits
    assert not torch.equal(want, original)
    full = limite.export_trained_insertion(parent, receipt_path, tmp_path / "full", chat_template=template,
                                           trainable_mode="full")
    assert full["insertion"] == dict(receipt["insertion"], trainable_mode="full")
    assert full["state_sha256"] == receipt["state_sha256"]
    assert full["parent_export"] == receipt["parent_export"] == dict(
        path=str(parent.resolve()), receipt_sha256=sha256_file(parent / "EXPORT.json"),
        state_sha256=json.loads((parent / "EXPORT.json").read_text())["state_sha256"])
    assert all(p.requires_grad for p in AutoModelForCausalLM.from_pretrained(tmp_path / "full").parameters())

    def change_backbone(policy):
        train_preludes(policy)
        next(p for name, p in policy.named_parameters() if ".adapters." not in name).add_(.5)

    _, backbone = checkpoint("backbone", change_backbone)
    with pytest.raises(ValueError, match="changed backbone tensors"):
        limite.export_trained_insertion(parent, backbone, tmp_path / "backbone-export", chat_template=template)
    _, unchanged = checkpoint("unchanged", lambda policy: None)
    with pytest.raises(ValueError, match="unchanged"):
        limite.export_trained_insertion(parent, unchanged, tmp_path / "unchanged-export", chat_template=template)
    with (receipt_path.parent / "model.safetensors").open("ab") as payload:
        payload.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        limite.export_trained_insertion(parent, receipt_path, tmp_path / "corrupt-export", chat_template=template)
    from archlab.training.limite_artifacts import verify_trained_insertion

    atomic_write_json(receipt_path, dict(json.loads(receipt_path.read_text()), note="rewritten"))
    with pytest.raises(ValueError, match="checkpoint receipt changed"):
        verify_trained_insertion(tmp_path / "export")
    atomic_write_json(parent / "EXPORT.json", dict(json.loads((parent / "EXPORT.json").read_text()), note="x"))
    with pytest.raises(ValueError, match="parent receipt changed"):
        verify_trained_insertion(tmp_path / "full")


@pytest.mark.parametrize("source_mode,target_backend", [
    ("full", None), ("adapter", None), ("adapter", "tilelang"),
])
def test_legacy_conversion_validates_source_payloads_and_preserves_backbone(
        tmp_path, native, monkeypatch, source_mode, target_backend):
    import hashlib

    pristine = deepcopy(native)
    expected = install_adapters(native, LimiteAdapterConfig(variant="normal"))
    digest = hashlib.sha256()
    for name, parameter in expected.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().reshape(-1)
                          .view(torch.uint8).numpy().tobytes())
    set_trainable_mode(expected, source_mode)
    with torch.no_grad():
        next(parameter for parameter in expected.parameters() if parameter.requires_grad).add_(.25)
    root = tmp_path / "legacy"
    root.mkdir()
    state = expected.state_dict()
    if source_mode == "full":
        torch.save({k: v for k, v in state.items() if not k.startswith("model.adapters.")},
                   root / "backbone.pt")
    torch.save(expected.model.adapters.state_dict(), root / "adapter.pt")
    receipt = dict(adapter=dict(expected.model.adapter_config),
                   files={p.name: sha256_file(p) for p in root.iterdir()})
    if source_mode == "full":
        receipt.update(trainable_mode="full", base_snapshot_sha256=digest.hexdigest())
    else:
        receipt["adapter"].pop("attention_backend")  # Actual early warmup format.
        receipt["frozen_sha256"] = digest.hexdigest()
    atomic_write_json(root / "COMPLETE.json", receipt)
    monkeypatch.setattr(limite, "load_model", lambda *a, **kw: deepcopy(pristine))
    actual, loaded_receipt = limite.load_legacy_checkpoint(
        SNAPSHOT, root, target_attention_backend=target_backend)
    assert loaded_receipt == receipt
    assert actual.model.adapter_config["attention_backend"] == (target_backend or "native")
    assert all(parameter.requires_grad for parameter in actual.parameters())
    if target_backend == "tilelang":
        attention = actual.model.adapters[0].native
        assert attention.forward.__func__ is not type(attention).forward
        assert attention.config._attn_implementation == "archlab_adapter"
    for name, tensor in actual.state_dict().items():
        torch.testing.assert_close(tensor, state[name], rtol=0, atol=0)
        assert tensor.dtype == state[name].dtype
    hash_field = "base_snapshot_sha256" if source_mode == "full" else "frozen_sha256"
    broken = dict(receipt, **{hash_field: "incorrect"})
    atomic_write_json(root / "COMPLETE.json", broken)
    with pytest.raises(ValueError, match="another publisher snapshot"):
        limite.load_legacy_checkpoint(SNAPSHOT, root)
    atomic_write_json(root / "COMPLETE.json", receipt)
    with (root / "adapter.pt").open("ab") as payload:
        payload.write(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint checksum mismatch"):
        limite.load_legacy_checkpoint(SNAPSHOT, root)


@pytest.mark.parametrize("inserted", [True, False])
def test_normal_serving_has_distinct_cache_slots_and_preserves_prefill_decode_and_logits(native, inserted):
    from types import SimpleNamespace

    from torch import nn
    from torch.nn import functional as F

    from archlab.training.serving import VLLMLimiteForCausalLM, bind_normal_attention

    native_copy = deepcopy(native)
    expected = (policy(native) if inserted else native).eval()
    actual = policy(native_copy) if inserted else native_copy
    actual.load_state_dict(expected.state_dict(), strict=True)
    actual.eval()
    slots = {}

    class CacheAttention(nn.Module):
        def __init__(self, attention):
            super().__init__()
            self.spec = SimpleNamespace(heads=attention.num_heads, kv=attention.num_kv_heads,
                                        dim=attention.head_dim, scale=attention.scaling,
                                        window=attention.window_span)
            self.keys = self.values = None

        def forward(self, q, k, v):
            spec = self.spec
            count = q.shape[0]
            q = q.view(1, count, spec.heads, spec.dim).transpose(1, 2)
            k, v = (x.view(1, count, spec.kv, spec.dim).transpose(1, 2) for x in (k, v))
            past = 0 if self.keys is None else self.keys.shape[2]
            self.keys = k if self.keys is None else torch.cat((self.keys, k), dim=2)
            self.values = v if self.values is None else torch.cat((self.values, v), dim=2)
            age = torch.arange(past, past + count)[:, None] - torch.arange(past + count)[None]
            mask = (age >= 0) & (age < (spec.window or past + count))
            out = F.scaled_dot_product_attention(q, self.keys, self.values,
                                                attn_mask=mask, scale=spec.scale, enable_gqa=True)
            return out.transpose(1, 2).reshape(count, -1)

    def factory(attention, *, prefix):
        assert prefix not in slots
        slots[prefix] = CacheAttention(attention)
        return slots[prefix]

    attached = bind_normal_attention(actual, factory)
    assert len(attached) == len(slots) == (8 if inserted else 4)
    serving = VLLMLimiteForCausalLM.__new__(VLLMLimiteForCausalLM)
    nn.Module.__init__(serving)
    serving.policy = actual
    assert serving.model is actual.model
    assert list(serving._modules) == ["policy"]
    assert all(name.startswith("policy.") for name, _ in serving.named_parameters(remove_duplicate=False))
    ids = torch.tensor([1, 4, 8, 5, 6, 10, 13, 3, 9, 7])
    cache = None
    for start, stop in ((0, 6), (6, 7), (7, 8), (8, 9), (9, 10)):
        want = expected(input_ids=ids[None, start:stop], use_cache=True, past_key_values=cache)
        cache = want.past_key_values
        hidden = serving(ids[start:stop], torch.arange(start, stop))
        got = serving.compute_logits(hidden)
        torch.testing.assert_close(got, want.logits.squeeze(0), rtol=0, atol=0)
    assert all(slot.keys.shape[2] == len(ids) for slot in slots.values())


@pytest.mark.parametrize("length", [3, 5, 6, 12])
def test_shared_training_mask_matches_native_dispatch(native, length):
    model = policy(native)
    hidden = torch.empty((2, length, 0), dtype=torch.bfloat16)
    positions = torch.arange(length)[None].expand(2, -1)
    expected = model.model._masks(hidden, positions, None, None)
    model.model._share_training_masks = True
    actual = model.model._masks(hidden, positions, None, None)
    for kind in expected:
        if expected[kind] is None:
            assert actual[kind] is None
        else:
            torch.testing.assert_close(actual[kind], expected[kind], rtol=0, atol=0)
            # All rows share one physical mask, independent of batch size.
            assert actual[kind].untyped_storage().nbytes() == length * length


def test_replacement_export_preserves_native_geometry_and_trains_short_axis(tmp_path, native):
    from archlab.architectures.limite_conversion import (
        LimiteConversionConfig,
        replace_limite_attention,
    )

    expected = replace_limite_attention(native, LimiteConversionConfig(backend="reference"))
    root = tmp_path / "replacement"
    receipt = limite.export_model(expected, SNAPSHOT, root)
    assert receipt["conversion"]["intervention"] == "replace_attention"
    actual = AutoModelForCausalLM.from_pretrained(root).train()
    assert actual.config.adapter == {}
    assert not hasattr(actual.policy.model, "adapters")
    assert len(actual.policy.model.layers) == native.config.num_hidden_layers
    actual.gradient_checkpointing_enable({"use_reentrant": False})
    ids = torch.tensor([[1, 4, 8, 5, 2]])
    result = actual(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, labels=ids)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    gradients = [layer.self_attn.short_kv_delta.grad for layer in actual.policy.model.layers]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.float().square().sum()) for value in gradients) > 0
    with pytest.raises(ValueError, match="padding"):
        actual(input_ids=ids, attention_mask=torch.tensor([[1, 1, 1, 1, 0]]), use_cache=False)
    with pytest.raises(ValueError, match="contiguous"):
        actual(input_ids=ids, position_ids=torch.tensor([[0, 1, 0, 1, 2]]), use_cache=False)

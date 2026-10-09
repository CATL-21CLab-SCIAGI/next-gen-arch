"""Full-weight continuation preserves native arithmetic and historical adapters."""

import copy
import json
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from archlab.architectures.limite_adapter import backbone_state, set_trainable_mode
from archlab.automodel import limite_adapter_common as common
from archlab.automodel.limite_adapter_communication import check_warmup_resume
from archlab.optimizers.limite_warmup import (
    FullWeightWarmupAdamW,
    build_warmup_optimizer,
    warmup_learning_rates,
)


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.scale = nn.Parameter(torch.ones(4, dtype=torch.float32))
        self.adapters = nn.Linear(4, 4, bias=False)
        self.trainable_mode = "adapter"
        self.adapter_config = {"variant": "normal", "attention_backend": "native"}

    def forward(self, x):
        return self.base(x.to(torch.bfloat16)).float() * self.scale + self.adapters(x)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBackbone()
        self.lm_head = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
        self.requires_grad_(False)
        self.model.adapters.requires_grad_(True)

    def forward(self, x):
        return self.lm_head(self.model(x).to(torch.bfloat16)).float()


@pytest.fixture
def factory(monkeypatch):
    torch.manual_seed(73)
    original = TinyPolicy()
    monkeypatch.setattr(common, "load_model", lambda *args, **kwargs: copy.deepcopy(original))
    monkeypatch.setattr(common, "LimiteAdapterConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    def install(model, config):
        model.model.adapter_config = {
            "variant": config.variant,
            "attention_backend": config.attention_backend,
        }
        return model

    monkeypatch.setattr(common, "install_adapters", install)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(
        common,
        "set_normal_attention_backward",
        lambda model, backward: setattr(model.model, "normal_backward", backward),
    )
    return original


def test_full_scope_changes_backbone_and_adapter_without_changing_native_dtypes(factory):
    model = set_trainable_mode(copy.deepcopy(factory), "full")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.randn(8, 4)).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    for name, parameter in model.named_parameters():
        assert parameter.dtype == before[name].dtype
        assert not torch.equal(parameter, before[name])
    set_trainable_mode(model, "adapter")
    assert all(
        parameter.requires_grad == name.startswith("model.adapters.")
        for name, parameter in model.named_parameters()
    )


def test_legacy_optimizer_continuation_and_full_roundtrip_are_exact():
    parameter = nn.Parameter(torch.tensor([0.4, -0.7]))
    old = torch.optim.AdamW([parameter], lr=1e-4, betas=(0.9, 0.95), weight_decay=0)
    for _ in range(3):
        parameter.grad = torch.tensor([0.13, -0.21])
        old.step()
    resumed = nn.Parameter(parameter.detach().clone())
    backbone = nn.Parameter(torch.tensor([0.3]))
    composed = FullWeightWarmupAdamW(
        torch.optim.AdamW([resumed], lr=1e-4, betas=(0.9, 0.95), weight_decay=0),
        torch.optim.AdamW([backbone], lr=1e-5, betas=(0.9, 0.95), weight_decay=0),
    )
    composed.load_state_dict(copy.deepcopy(old.state_dict()))
    assert not composed.backbone.state
    parameter.grad = resumed.grad = torch.tensor([0.15, -0.19])
    backbone.grad = torch.tensor([0.11])
    old.step()
    composed.step()
    torch.testing.assert_close(parameter, resumed, atol=0, rtol=0)
    restored = FullWeightWarmupAdamW(
        torch.optim.AdamW([resumed], lr=1e-4), torch.optim.AdamW([backbone], lr=1e-5)
    )
    restored.load_state_dict(copy.deepcopy(composed.state_dict()))
    assert (
        restored.adapter.state_dict()["param_groups"]
        == composed.adapter.state_dict()["param_groups"]
    )
    for optimizer, reloaded in (
        (composed.adapter, restored.adapter),
        (composed.backbone, restored.backbone),
    ):
        for key, value in next(iter(optimizer.state.values())).items():
            torch.testing.assert_close(
                value, next(iter(reloaded.state.values()))[key], atol=0, rtol=0
            )


def test_full_payload_roundtrip_and_verified_oss_aliases(tmp_path, factory):
    model = common.build_model("snapshot", "normal", "cpu", trainable_mode="full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(8, 4)).square().mean().backward()
    optimizer.step()
    expected = copy.deepcopy(model.state_dict())
    checkpoint = common.save_adapter(
        model, optimizer, tmp_path / "run", 7631, 2_000_420_864, tmp_path / "oss"
    )
    manifest = json.loads((checkpoint / "COMPLETE.json").read_text())
    assert manifest["trainable_mode"] == "full"
    assert "frozen_sha256" not in manifest
    assert all((checkpoint / name).is_symlink() for name in manifest["files"])
    assert set(backbone_state(model)).isdisjoint(
        {"model.adapters." + name for name in model.model.adapters.state_dict()}
    )
    loaded = common.build_model("snapshot", "normal", "cpu", checkpoint)
    assert loaded.model.trainable_mode == "full"
    for name, value in expected.items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    with (checkpoint / "backbone.pt").open("ab") as payload:
        payload.write(b"invalid")
    with pytest.raises(ValueError, match="checksum mismatch"):
        common.build_model("snapshot", "normal", "cpu", checkpoint)


@pytest.mark.parametrize("backward", ["tilelang", "fa4"])
def test_execution_kernel_survives_full_checkpoint_without_changing_geometry(
    tmp_path, factory, monkeypatch, backward
):
    def select(model, kernel):
        model.model.normal_kernel = kernel

    monkeypatch.setattr(common, "set_normal_attention_kernel", select)
    model = common.build_model(
        "snapshot",
        "normal",
        "cpu",
        attention_backend="tilelang",
        normal_kernel="gqa",
        normal_backward=backward,
        trainable_mode="full",
    )
    geometry = copy.deepcopy(model.model.adapter_config)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.randn(8, 4)).square().mean().backward()
    optimizer.step()
    saved = common.save_adapter(model, optimizer, tmp_path, 9, 128)
    manifest = json.loads((saved / "COMPLETE.json").read_text())
    assert manifest["normal_kernel"] == "gqa"
    assert manifest["normal_backward"] == backward
    assert "normal_kernel" not in manifest["adapter"]
    assert "normal_backward" not in manifest["adapter"]
    restored = common.build_model("snapshot", "normal", "cpu", saved)
    assert restored.model.normal_kernel == "gqa"
    assert restored.model.normal_backward == backward
    assert restored.model.adapter_config == geometry
    assert list(restored.state_dict()) == list(model.state_dict())
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    assert common.attention_kernel_name("normal", "tilelang", normal_kernel="gqa") == (
        "tilelang-gqa-key-owned-bf16-v1"
    )


def test_native_to_tilelang_migration_is_explicit_and_preserves_weights(tmp_path, factory):
    model = common.build_model("snapshot", "normal", "cpu")
    checkpoint = common.save_adapter(
        model,
        torch.optim.AdamW(model.model.adapters.parameters()),
        tmp_path,
        7630,
        2_000_158_720,
        extra={"frozen_sha256": common.frozen_fingerprint(model)},
    )
    with pytest.raises(ValueError, match="geometry"):
        common.build_model(
            "snapshot",
            "normal",
            "cpu",
            checkpoint,
            attention_backend="tilelang",
            trainable_mode="full",
        )
    full = common.build_model(
        "snapshot",
        "normal",
        "cpu",
        checkpoint,
        attention_backend="tilelang",
        trainable_mode="full",
        allow_backend_migration=True,
    )
    assert full.model.adapter_config["attention_backend"] == "tilelang"
    for name, parameter in full.state_dict().items():
        torch.testing.assert_close(parameter, model.state_dict()[name], atol=0, rtol=0)


def test_full_resume_retains_phase_origin_and_refuses_budget_or_lr_changes():
    schedule = dict(
        target_tokens=10_000_000_000,
        adapter_peak_lr=1e-4,
        adapter_warmup_steps=100,
        backbone_peak_lr=1e-5,
        backbone_warmup_steps=100,
        full_weight_start_step=7630,
        full_weight_start_tokens=2_000_158_720,
        source_2b_checkpoint="matched-2B",
    )
    prior = dict(
        trainable_mode="full",
        global_batch=128,
        context=2048,
        data_contract="sealed",
        warmup_schedule=schedule,
    )
    check_warmup_resume(
        prior,
        global_batch=128,
        context=2048,
        data_contract="sealed",
        warmup_schedule=copy.deepcopy(schedule),
    )
    first = warmup_learning_rates(7630, 2_000_158_720, schedule)[1]
    resumed = warmup_learning_rates(15260, 4_000_317_440, json.loads(json.dumps(schedule)))[1]
    assert first == pytest.approx(1e-7)
    assert resumed > first * 50
    for step, tokens in ((7630, 2_000_158_720), (15260, 4_000_317_440), (30520, 8_000_634_880)):
        original_adapter_lr = (
            1e-4
            * min(1.0, (step + 1) / 100)
            * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * tokens / 10_000_000_000)))
        )
        assert warmup_learning_rates(step, tokens, schedule)[0] == original_adapter_lr
    for key, value in (
        ("target_tokens", 12_000_000_000),
        ("backbone_peak_lr", 1e-4),
        ("full_weight_start_step", 15260),
    ):
        changed = dict(schedule, **{key: value})
        with pytest.raises(ValueError, match="budget or learning-rate schedule"):
            check_warmup_resume(
                prior,
                global_batch=128,
                context=2048,
                data_contract="sealed",
                warmup_schedule=changed,
            )


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="container TE requires CUDA")
def test_container_te_master_and_moments_roundtrip_next_update_is_exact():
    pytest.importorskip("transformer_engine.pytorch")
    torch.manual_seed(73)
    model = set_trainable_mode(TinyPolicy().cuda(), "full")
    optimizer = build_warmup_optimizer(model)
    inputs = torch.randn(8, 4, device="cuda")
    for _ in range(3):
        optimizer.zero_grad()
        model(inputs).square().mean().backward()
        optimizer.step()
    checkpoint_model = copy.deepcopy(model.state_dict())
    checkpoint_optimizer = copy.deepcopy(optimizer.state_dict())
    restored = set_trainable_mode(TinyPolicy().cuda(), "full")
    restored.load_state_dict(checkpoint_model)
    restored_optimizer = build_warmup_optimizer(restored)
    restored_optimizer.load_state_dict(checkpoint_optimizer)
    for original_parameter, parameter in zip(
        optimizer.backbone.state, restored_optimizer.backbone.state, strict=True
    ):
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            value = restored_optimizer.backbone.get_unscaled_state(parameter, key)
            assert value.dtype == torch.float32
            original = optimizer.backbone.get_unscaled_state(original_parameter, key)
            torch.testing.assert_close(value, original, atol=0, rtol=0)
    for policy, adam in ((model, optimizer), (restored, restored_optimizer)):
        adam.zero_grad()
        policy(inputs).square().mean().backward()
        adam.step()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    for original_parameter, restored_parameter in zip(
        optimizer.backbone.state, restored_optimizer.backbone.state, strict=True
    ):
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.backbone.get_unscaled_state(original_parameter, key),
                restored_optimizer.backbone.get_unscaled_state(restored_parameter, key),
                atol=0,
                rtol=0,
            )

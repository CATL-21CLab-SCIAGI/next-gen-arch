import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

import archlab.megatron.train as megatron_train
from archlab.megatron.train import (
    SpeedrunSchedule,
    _current_training_iteration,
    _external_batch_loader,
    _global_rank,
    _invoke_megatron_pretrain,
    _loss_func,
    _megatron_arguments,
    _restore_megatron_checkpoint,
    _source_provenance,
    get_megatron_backend_profile,
    resolve_fineweb_variant,
)
from archlab.optimizers.recipes import get_optimization_recipe
from archlab.speedrun.campaigns import (
    TEN_M_BATCH_TOKENS,
    get_campaign_variant,
    get_ten_m_variant,
)


def test_current_training_iteration_prefers_live_megatron_counter():
    args = SimpleNamespace(iteration=0, curr_iteration=43)

    assert _current_training_iteration(args) == 43


def test_current_training_iteration_falls_back_before_train_loop():
    args = SimpleNamespace(iteration=7)

    assert _current_training_iteration(args) == 7


def test_global_rank_defaults_to_single_process_and_honors_torchrun(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    assert _global_rank() == 0
    monkeypatch.setenv("RANK", "3")
    assert _global_rank() == 3


def test_climbmix_resume_derives_exact_microbatch_cursor(monkeypatch):
    captured = {}

    def fake_loader(*_args, **kwargs):
        captured.update(kwargs)
        yield torch.zeros(1), torch.ones(1), {"batch_index": 36}

    monkeypatch.setattr(
        megatron_train,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        fake_loader,
    )
    monkeypatch.setattr(megatron_train, "_megatron_resume_iteration", lambda: 3)
    monkeypatch.setattr(megatron_train, "_world_size", lambda: 1)

    batch = next(_external_batch_loader(object(), "train"))

    assert captured["resume_state_dict"] == {"batch_index": 36}
    assert batch["tokens"].item() == 0
    assert batch["labels"].item() == 1


def test_exact_replay_resume_uses_one_packed_batch_per_step(monkeypatch):
    captured = {}

    def fake_loader(*_args, **kwargs):
        captured.update(kwargs)
        yield torch.zeros(1), torch.ones(1), {"batch_index": 7}, 192

    monkeypatch.setattr(
        megatron_train,
        "tokenizing_replicated_global_batch_loader_with_state_bos_bestfit",
        fake_loader,
    )
    monkeypatch.setattr(megatron_train, "_megatron_resume_iteration", lambda: 7)

    batch = next(_external_batch_loader(object(), "train", exact_global_batch_replay=True))

    assert captured["resume_state_dict"] == {"batch_index": 7}
    assert batch["tokens"].item() == 0
    assert batch["labels"].item() == 1


class _FakeTimer:
    def __init__(self):
        self.events = []

    def start(self, **kwargs):
        self.events.append(("start", kwargs))

    def stop(self, **kwargs):
        self.events.append(("stop", kwargs))


def test_megatron_checkpoint_restore_precedes_external_loader():
    timer = _FakeTimer()
    args = SimpleNamespace(
        load="/checkpoints",
        pretrained_checkpoint=None,
        iteration=0,
        num_floating_point_operations_so_far=0,
    )
    schedule = SimpleNamespace(iteration=0)
    calls = []

    def load_checkpoint(model, optimizer, restored_schedule, *, checkpointing_context):
        calls.append((model, optimizer, restored_schedule, checkpointing_context))
        restored_schedule.iteration = 10
        return 10, 1234

    def timers(*_args, **_kwargs):
        return timer

    timers.log = lambda names: calls.append(("timers", names))
    training_module = SimpleNamespace(
        get_args=lambda: args,
        get_timers=lambda: timers,
        load_checkpoint=load_checkpoint,
    )

    restored = _restore_megatron_checkpoint(
        training_module,
        "model",
        "optimizer",
        schedule,
        {"local": "context"},
    )

    assert restored == 10
    assert args.iteration == 10
    assert args.num_floating_point_operations_so_far == 1234
    assert calls[0] == ("model", "optimizer", schedule, {"local": "context"})
    assert timer.events == [("start", {"barrier": True}), ("stop", {"barrier": True})]


def test_megatron_checkpoint_restore_rejects_silent_restart():
    timer = _FakeTimer()
    args = SimpleNamespace(
        load="/checkpoints",
        pretrained_checkpoint=None,
        iteration=0,
        num_floating_point_operations_so_far=0,
    )
    schedule = SimpleNamespace(iteration=0)

    def timers(*_args, **_kwargs):
        return timer

    timers.log = lambda _names: None
    training_module = SimpleNamespace(
        get_args=lambda: args,
        get_timers=lambda: timers,
        load_checkpoint=lambda *_args, **_kwargs: (0, 0),
    )

    with pytest.raises(RuntimeError, match="did not restore a positive iteration"):
        _restore_megatron_checkpoint(
            training_module,
            "model",
            "optimizer",
            schedule,
            {},
        )


def test_pretrain_adapter_supports_config_container_api(monkeypatch):
    calls = []
    argument_utils = ModuleType("megatron.training.argument_utils")
    argument_utils.pretrain_cfg_container_from_args = lambda args: ("config", args)
    arguments = ModuleType("megatron.training.arguments")
    arguments.parse_and_validate_args = lambda **kwargs: ("args", kwargs)
    monkeypatch.setitem(sys.modules, "megatron.training.argument_utils", argument_utils)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments)

    def pretrain(cfg_container, datasets, model, model_type, forward):
        calls.append((cfg_container, datasets, model, model_type, forward))

    _invoke_megatron_pretrain(SimpleNamespace(pretrain=pretrain), "data", "model", "type")

    assert calls[0][:4] == (
        ("config", ("args", {"args_defaults": {"tokenizer_type": "NullTokenizer"}})),
        "data",
        "model",
        "type",
    )


def test_pretrain_adapter_supports_legacy_cli_api():
    calls = []

    def pretrain(datasets, model, model_type, forward, *, args_defaults):
        calls.append((datasets, model, model_type, forward, args_defaults))

    _invoke_megatron_pretrain(SimpleNamespace(pretrain=pretrain), "data", "model", "type")

    assert calls[0][:3] == ("data", "model", "type")
    assert calls[0][-1] == {"tokenizer_type": "NullTokenizer"}


@pytest.mark.parametrize(
    (
        "name",
        "compiled",
        "mcore_master",
        "finite",
        "compile_mode",
        "overlap_grad_reduce",
        "ddp_num_buckets",
        "average_in_collective",
    ),
    (
        ("legacy", False, True, True, None, False, None, False),
        ("compile", True, True, True, None, False, None, False),
        ("compile-reduce-overhead", True, True, True, "reduce-overhead", False, None, False),
        ("compile-max-autotune", True, True, True, "max-autotune", False, None, False),
        ("compile-safe-autotune", True, True, True, "max-autotune", False, None, False),
        ("compile-dp-overlap", True, True, True, None, True, 4, False),
        ("compile-dp-overlap-average", True, True, True, None, True, 4, True),
        ("native-master", False, False, True, None, False, None, False),
        ("speedrun", True, False, True, None, False, None, False),
    ),
)
def test_backend_profiles_are_explicit_factorial_ablation(
    name,
    compiled,
    mcore_master,
    finite,
    compile_mode,
    overlap_grad_reduce,
    ddp_num_buckets,
    average_in_collective,
):
    profile = get_megatron_backend_profile(name)

    assert profile.compile_architecture is compiled
    assert profile.use_mcore_bf16_master is mcore_master
    assert profile.finite_checks is finite
    assert profile.compile_mode == compile_mode
    assert profile.overlap_grad_reduce is overlap_grad_reduce
    assert profile.ddp_num_buckets == ddp_num_buckets
    assert profile.ddp_average_in_collective is average_in_collective


def test_safe_autotune_profile_falls_back_for_numerically_sensitive_mixers():
    profile = get_megatron_backend_profile("compile-safe-autotune")

    assert profile.resolved_compile_mode("baseline") == "max-autotune"
    assert profile.resolved_compile_mode("engram") == "max-autotune"
    assert profile.resolved_compile_mode("kda") is None
    assert profile.resolved_compile_mode("kimi-k3-kda-update") is None
    assert profile.resolved_compile_mode("qwen-gdn") is None


def test_megatron_arguments_follow_profile_precision_and_finite_policy():
    variant = get_ten_m_variant("baseline")
    recipe = get_optimization_recipe("baseline")
    legacy = _megatron_arguments(variant, get_megatron_backend_profile("legacy"), recipe)
    speedrun = _megatron_arguments(variant, get_megatron_backend_profile("speedrun"), recipe)

    assert "--bf16" in legacy
    assert "--no-check-for-nan-in-loss-and-grad" not in legacy
    assert "--bf16" not in speedrun
    assert "--no-check-for-nan-in-loss-and-grad" not in speedrun


def test_100m_exact_replay_keeps_192_active_sequences_on_world_15(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "15")
    variant = get_campaign_variant("100m", "baseline")
    arguments = _megatron_arguments(
        variant,
        get_megatron_backend_profile("speedrun"),
        get_optimization_recipe("baseline"),
        scale="100m",
        exact_global_batch_replay=True,
    )

    assert arguments[arguments.index("--num-layers") + 1] == "10"
    assert arguments[arguments.index("--hidden-size") + 1] == "384"
    assert arguments[arguments.index("--num-attention-heads") + 1] == "6"
    assert arguments[arguments.index("--micro-batch-size") + 1] == "13"
    assert arguments[arguments.index("--global-batch-size") + 1] == "195"
    assert arguments[arguments.index("--eval-interval") + 1] == "250"
    assert "--calculate-per-token-loss" in arguments


def test_fineweb_100m_contract_uses_24_way_dp_and_gpt2_vocabulary(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "24")
    variant = resolve_fineweb_variant("100m", "baseline")
    arguments = _megatron_arguments(
        variant,
        get_megatron_backend_profile("compile-safe-autotune"),
        get_optimization_recipe("baseline"),
        dataset="fineweb10b",
        scale="100m",
    )

    assert variant.parameter_count == 98_009_374
    assert variant.steps == 2_991
    assert arguments[arguments.index("--micro-batch-size") + 1] == "8"
    assert arguments[arguments.index("--global-batch-size") + 1] == "192"
    assert arguments[arguments.index("--vocab-size") + 1] == "50304"
    assert arguments[arguments.index("--tensor-model-parallel-size") + 1] == "1"
    assert arguments[arguments.index("--pipeline-model-parallel-size") + 1] == "1"
    assert arguments[arguments.index("--context-parallel-size") + 1] == "1"


def test_fineweb_7b_contract_uses_100b_tokens_and_32_way_dp(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "32")
    variant = resolve_fineweb_variant("7b", "baseline")
    arguments = _megatron_arguments(
        variant,
        get_megatron_backend_profile("compile-dp-overlap"),
        get_optimization_recipe("baseline"),
        dataset="fineweb100b",
        scale="7b",
        checkpoint_dir=Path("/checkpoints"),
        save_interval=10_000,
    )

    assert variant.parameter_count == 6_829_675_290
    assert variant.steps == 254_313
    assert variant.training_tokens == 99_999_940_608
    assert arguments[arguments.index("--num-layers") + 1] == "32"
    assert arguments[arguments.index("--hidden-size") + 1] == "3200"
    assert arguments[arguments.index("--num-attention-heads") + 1] == "25"
    assert arguments[arguments.index("--micro-batch-size") + 1] == "6"
    assert arguments[arguments.index("--global-batch-size") + 1] == "192"
    assert arguments[arguments.index("--save-interval") + 1] == "10000"
    assert arguments[arguments.index("--ckpt-format") + 1] == "torch"


def test_fineweb_single_gpu_microbatch_override_accumulates_to_global_batch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    variant = resolve_fineweb_variant("1m", "baseline")
    arguments = _megatron_arguments(
        variant,
        get_megatron_backend_profile("compile"),
        get_optimization_recipe("baseline"),
        dataset="fineweb10b",
        scale="1m",
        micro_batch_size_override=16,
    )

    assert arguments[arguments.index("--micro-batch-size") + 1] == "16"
    assert arguments[arguments.index("--global-batch-size") + 1] == "192"


def test_fineweb_microbatch_override_must_divide_global_batch(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    variant = resolve_fineweb_variant("1m", "baseline")

    with pytest.raises(ValueError, match="global batch must be divisible"):
        _megatron_arguments(
            variant,
            get_megatron_backend_profile("compile"),
            get_optimization_recipe("baseline"),
            dataset="fineweb10b",
            scale="1m",
            micro_batch_size_override=17,
        )


def test_masked_padding_is_excluded_from_megatron_loss_and_bpb(monkeypatch):
    monkeypatch.setattr(megatron_train, "_TOKEN_BYTES", torch.tensor([0, 1, 2, 3]))
    monkeypatch.setattr(megatron_train, "_TRAIN_METRICS_ENABLED", False)
    labels = torch.tensor([[1, 2], [-1, -1]])
    losses = torch.tensor([[2.0, 3.0], [0.0, 0.0]])

    loss_sum, token_count, report = _loss_func(labels, True, losses)

    assert loss_sum.item() == 5.0
    assert token_count.item() == 2
    assert report["lm loss"].tolist() == [5.0, 2.0]
    assert report["bpb"][0].item() == 5.0


def test_recipe_controls_reach_megatron_arguments():
    variant = get_ten_m_variant("baseline")
    recipe = get_optimization_recipe("grad-clip-01")
    arguments = _megatron_arguments(variant, get_megatron_backend_profile("compile"), recipe)

    assert arguments[arguments.index("--clip-grad") + 1] == "0.1"


def test_dp_overlap_controls_reach_megatron_arguments():
    variant = get_ten_m_variant("baseline")
    recipe = get_optimization_recipe("baseline")
    overlap = _megatron_arguments(
        variant, get_megatron_backend_profile("compile-dp-overlap"), recipe
    )
    averaged = _megatron_arguments(
        variant, get_megatron_backend_profile("compile-dp-overlap-average"), recipe
    )

    assert "--overlap-grad-reduce" in overlap
    assert overlap[overlap.index("--ddp-num-buckets") + 1] == "4"
    assert "--ddp-average-in-collective" not in overlap
    assert "--ddp-average-in-collective" in averaged


def test_compound_recipe_is_explicit_and_portable():
    recipe = get_optimization_recipe("marin-compound")

    assert recipe.model_overrides["rope_fraction"] == 0.5
    assert recipe.model_overrides["partial_key_offset"] == "all"
    assert recipe.model_overrides["cached_attention_layers"] == 3


def test_source_provenance_records_dirty_patch_without_exposing_it(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    clean = _source_provenance(tmp_path)
    tracked.write_text("after\n", encoding="utf-8")
    tracked_dirty = _source_provenance(tmp_path)
    (tmp_path / "untracked.txt").write_text("not in the diff\n", encoding="utf-8")
    untracked_dirty = _source_provenance(tmp_path)

    assert clean["source_dirty"] is False
    assert tracked_dirty["source_dirty"] is True
    assert clean["source_diff_sha256"] != tracked_dirty["source_diff_sha256"]
    assert tracked_dirty["source_diff_sha256"] == untracked_dirty["source_diff_sha256"]
    assert tracked_dirty["source_worktree_sha256"] != untracked_dirty["source_worktree_sha256"]
    assert untracked_dirty["source_untracked_files"] == ["untracked.txt"]
    assert set(untracked_dirty) == {
        "source_commit",
        "source_dirty",
        "source_diff_sha256",
        "source_untracked_files",
        "source_untracked_sha256",
        "source_worktree_sha256",
    }


def test_throughput_summary_separates_compile_pause_from_steady_steps() -> None:
    optimizer = type("Optimizer", (), {"param_groups": []})()
    schedule = SpeedrunSchedule(optimizer, get_ten_m_variant("baseline"))
    schedule.step_timestamps = [*map(float, range(10)), 109.0, *map(float, range(110, 119))]

    summary = schedule.throughput_summary()

    assert summary["measured_training_seconds"] == 9.0
    assert summary["tokens_per_second"] == TEN_M_BATCH_TOKENS
    assert summary["median_step_seconds"] == 1.0
    assert summary["steady_state_tokens_per_second"] == TEN_M_BATCH_TOKENS
    assert summary["throughput_sample_intervals"] == 9
    assert summary["throughput_protocol"]["warmup_steps"] == 10

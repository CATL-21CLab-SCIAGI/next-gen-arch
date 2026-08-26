from types import SimpleNamespace

import pytest
import torch

import next_gen_arch.training.megatron_train as megatron_train
from next_gen_arch.training.campaigns import (
    TEN_M_BATCH_TOKENS,
    get_campaign_variant,
    get_ten_m_variant,
)
from next_gen_arch.training.megatron_train import (
    SpeedrunSchedule,
    _current_training_iteration,
    _global_rank,
    _loss_func,
    _megatron_arguments,
    _source_provenance,
    get_megatron_backend_profile,
)
from next_gen_arch.training.optimization_recipes import get_optimization_recipe


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

    assert summary["measured_training_seconds"] == 109.0
    assert summary["tokens_per_second"] == 10 * TEN_M_BATCH_TOKENS / 109.0
    assert summary["median_step_seconds"] == 1.0
    assert summary["steady_state_tokens_per_second"] == TEN_M_BATCH_TOKENS

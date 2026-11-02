"""Migration preserves the live contract; resolution never substitutes for GPU admission."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from archlab.training.config import compose_verl, read_experiment


@pytest.fixture
def deployment(monkeypatch):
    for name in ("PARENT_CHECKPOINT", "MODEL_MANIFEST", "CODE_DATA", "OUTPUT", "HARNESS_SPEC",
                 "E2B_KEY_FILE", "E2B_TEMPLATE_MAP", "DLC_SANDBOX_POOL"):
        monkeypatch.setenv("ARCHLAB_" + name, "/fixture/" + name.lower())
    monkeypatch.setenv("E2B_DOMAIN", "sandbox.example.invalid")
    monkeypatch.delenv("ARCHLAB_RESUME_MODE", raising=False)
    monkeypatch.delenv("ARCHLAB_RESUME_CHECKPOINT", raising=False)
    monkeypatch.delenv("ARCHLAB_RAY_ADDRESS", raising=False)


def resolve(overlay="local", overrides=()):
    experiment = read_experiment(Path("recipes/rl/qwen35_code.yaml"), overrides,
                                 overlays=[Path(f"recipes/rl/qwen35_code_{overlay}.yaml")])
    return experiment, compose_verl(experiment, Path("third_party/verl"))


@pytest.mark.parametrize("overlay,batch,cpus,address", [("local", 8, "0.03125", "local"),
                                                       ("efficiency", 16, "1", "auto")])
def test_native_qwen_overlays_preserve_science_and_placement(deployment, overlay, batch, cpus, address):
    experiment, config = resolve(overlay)
    assert experiment.family == "qwen3_5" and experiment.intervention == "none"
    assert config.model_engine == "dp" and config.actor_rollout_ref.actor.strategy == "fsdp2"
    rollout = config.actor_rollout_ref.rollout
    assert rollout.max_model_len == 262144
    assert rollout.prompt_length == 8192 and rollout.response_length == 253952
    assert (rollout.n, rollout.top_p, rollout.top_k) == (8, .95, 20)
    assert rollout.enable_prefix_caching and not rollout.enforce_eager
    assert config.algorithm.filter_groups.enable
    assert not config.algorithm.norm_adv_by_std_in_grpo
    assert config.actor_rollout_ref.actor.loss_agg_mode == "prompt-mean"
    assert config.trainer.v1.sampler.max_off_policy_threshold == experiment.maximum_policy_lag == 2
    assert config.trainer.resume_mode == "disable" and config.trainer.resume_from_path is None
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == batch
    assert config.ray_kwargs.ray_init.address == address
    assert config.ray_kwargs.ray_init.runtime_env.env_vars.UNI_AGENT_RUNNER_TASK_NUM_CPUS == cpus
    assert ("num_cpus" in config.ray_kwargs.ray_init) == (overlay == "local")
    # Preserve resolved historical values even where upstream interpolation has changed.
    assert config.actor_rollout_ref.actor.rollout_n == config.critic.rollout_n == 4
    assert config.actor_rollout_ref.model.use_fused_kernels
    assert not config.actor_rollout_ref.actor.use_fused_kernels


@pytest.mark.parametrize("overrides,reason", [
    (["settings.trainer.resume_mode=auto"], "fresh RL"),
    (["resume_checkpoint=/checkpoint", "settings.trainer.resume_mode=resume_path",
      "settings.trainer.resume_from_path=/other"], "lineage"),
    (["settings.trainer.v1.sampler.max_off_policy_threshold=3"], "policy lag"),
    (["settings.actor_rollout_ref.rollout.response_length=8192"], "context"),
    (["settings.actor_rollout_ref.rollout.calculate_log_probs=false"], "probabilities"),
    (["settings.algorithm.rollout_correction.bypass_mode=true"], "importance correction"),
    (["settings.actor_rollout_ref.model.override_config={attn_implementation: eager}"], "Qwen native"),
])
def test_qwen_shared_invariants_reject_invalid_deployments(deployment, overrides, reason):
    with pytest.raises(ValueError, match=reason):
        resolve(overrides=overrides)


def test_resume_lineage_is_explicit(deployment, monkeypatch):
    monkeypatch.setenv("ARCHLAB_RESUME_CHECKPOINT", "/checkpoint")
    monkeypatch.setenv("ARCHLAB_RESUME_MODE", "resume_path")
    experiment, config = resolve()
    assert experiment.resume_checkpoint == config.trainer.resume_from_path == "/checkpoint"


def test_configuration_parity_does_not_admit_unqualified_qwen(deployment, tmp_path, monkeypatch):
    from archlab.training import qualification

    experiment, config = resolve()
    monkeypatch.setattr(qualification, "source_manifest", lambda: {})
    monkeypatch.setattr(qualification, "check_source_unchanged", lambda _: {})
    result = qualification.qualify(experiment, config, Path("third_party/verl"), tmp_path)
    assert not result["passed"]
    assert result["blockers"] == ["native_qwen_deployment"]
    assert (tmp_path / "qualification.json").exists()
    assert OmegaConf.load(tmp_path / "resolved.yaml") == config

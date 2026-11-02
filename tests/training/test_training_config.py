from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from archlab.training.config import read_experiment


def test_unknown_scientific_setting_is_rejected(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("name: demo\nparent_checkpoint: parent\ndata_manifest: data\nengien: dp\n")
    with pytest.raises(ConfigKeyError):
        read_experiment(path)


def test_missing_parent_does_not_silently_start_from_scratch(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("name: demo\ndata_manifest: data\n")
    with pytest.raises(MissingMandatoryValue):
        read_experiment(path)


def test_machine_paths_are_explicit_deployment_inputs(monkeypatch):
    for name, value in {"ARCHLAB_PARENT_CHECKPOINT": "/parent", "ARCHLAB_CODE_DATA": "/data",
                        "ARCHLAB_POLICY": "/policy", "ARCHLAB_OUTPUT": "/output",
                        "ARCHLAB_HARNESS_SPEC": "/harness/single_bash.yaml",
                        "E2B_DOMAIN": "sandbox.example.test",
                        "ARCHLAB_E2B_KEY_FILE": "/private/key",
                        "ARCHLAB_E2B_TEMPLATE_MAP": "/data/templates.json"}.items():
        monkeypatch.setenv(name, value)
    experiment = read_experiment(Path("recipes/rl/limite_code.yaml"))
    assert experiment.attention == "native"
    assert experiment.parent_checkpoint == "/parent"
    assert experiment.parent_stage == "adapter_sft" and experiment.parent_tokens == 2_000_158_720
    assert experiment.name == "limite-normal-sft2b-mimo-code-fresh-v1"
    assert experiment.data_manifest == "/data/manifest.json"
    assert experiment.settings["data.train_files"] == ["/data/train.parquet"]
    assert experiment.settings["trainer.resume_mode"] == "disable"
    assert experiment.settings["data.seed"] == 42
    assert experiment.settings["ray_kwargs.ray_init"]["runtime_env"]["env_vars"]["MIXED_HARNESS_SPEC"] == "/harness/single_bash.yaml"
    assert experiment.settings["ray_kwargs.ray_init"]["address"] == "local"
    assert experiment.settings["transfer_queue.backend.SimpleStorage.num_data_storage_units"] == 2
    assert experiment.settings["ray_kwargs.ray_init"]["runtime_env"]["worker_process_setup_hook"] == (
        "archlab.architectures.fa4_attention.fa4_runtime_contract"
    )


def test_invalid_execution_mode_fails_before_model_load(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("name: demo\nparent_checkpoint: parent\ndata_manifest: data\nexecution: overlap_maybe\n")
    with pytest.raises(ValueError, match="execution"):
        read_experiment(path)


def test_upstream_cli_override_replaces_literal_recipe_setting(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("name: demo\nparent_checkpoint: parent\ndata_manifest: data\n"
                    "settings:\n  actor_rollout_ref.actor.optim.lr: 0.001\n")
    experiment = read_experiment(path, ["settings.actor_rollout_ref.actor.optim.lr=0.00001"])
    assert experiment.settings == {"actor_rollout_ref.actor.optim.lr": 0.00001}


def test_invalid_override_does_not_silently_disappear(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text("name: demo\nparent_checkpoint: parent\ndata_manifest: data\n")
    with pytest.raises(ValueError, match="key=value"):
        read_experiment(path, ["engine"])


@pytest.mark.parametrize("override", [
    "settings.actor_rollout_ref.actor.strategy=fsdp2",
    "settings.algorithm.adv_estimator=gae",
    "settings.actor_rollout_ref.actor.megatron.dtype=float32",
    "settings.actor_rollout_ref.actor.megatron.patch_recomputation_backward=true",
    "settings.actor_rollout_ref.rollout.checkpoint_engine.backend=delta_sharded",
    "settings.trainer.metrics_extension=null",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.flash_attn_version=3",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm.attention_config.backend=TRITON_ATTN",
    "variant=simplicial",
    "settings.actor_rollout_ref.rollout.custom.agent_framework.trajectory_validator=null",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.mode=3",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=[1,2]",
    "settings.actor_rollout_ref.rollout.max_num_seqs=2",
])
def test_resolved_upstream_overrides_cannot_change_the_scientific_contract(monkeypatch, override):
    from archlab.training.config import compose_verl

    upstream = Path("third_party/verl")
    if not (upstream / "recipes/code/config/train.yaml").exists():
        pytest.skip("pinned upstream recipe is not initialized")
    for name in ("PARENT_CHECKPOINT", "CODE_DATA", "POLICY", "OUTPUT", "HARNESS_SPEC"):
        monkeypatch.setenv("ARCHLAB_" + name, "/fixture/" + name.lower())
    for name in ("E2B_DOMAIN", "ARCHLAB_E2B_KEY_FILE", "ARCHLAB_E2B_TEMPLATE_MAP"):
        monkeypatch.setenv(name, "/fixture/" + name.lower())
    experiment = read_experiment(Path("recipes/rl/limite_code.yaml"), [override])
    with pytest.raises(ValueError, match="strategy|GRPO|precision|patches|contracts|FlashAttention-2|decode graphs"):
        compose_verl(experiment, upstream)


@pytest.fixture
def conversion_deployment(monkeypatch):
    for name in ("PARENT_CHECKPOINT", "CODE_DATA", "POLICY", "OUTPUT"):
        monkeypatch.setenv("ARCHLAB_" + name, "/fixture/" + name.lower())
    monkeypatch.setenv("ARCHLAB_HARNESS_SPEC", str(Path("recipes/rl/harness/conversion_single_bash.yaml").resolve()))
    for name in ("E2B_DOMAIN", "ARCHLAB_E2B_KEY_FILE", "ARCHLAB_E2B_TEMPLATE_MAP"):
        monkeypatch.setenv(name, "/fixture/" + name.lower())


def test_replacement_recipe_uses_actual_package_sft_and_explicit_eager_curriculum(conversion_deployment):
    from archlab.training.config import compose_verl
    from archlab.training.qualification import check_curriculum

    experiment = read_experiment(Path("recipes/rl/limite_conversion_code.yaml"))
    config = compose_verl(experiment, Path("third_party/verl"))
    rollout = config.actor_rollout_ref.rollout
    assert experiment.intervention == "replace_attention" and experiment.variant == "simplicial"
    assert experiment.parent_stage == "package_sft" and experiment.parent_tokens is None
    assert rollout.enforce_eager and rollout.max_num_seqs == 1
    assert not rollout.enable_chunked_prefill and not rollout.enable_prefix_caching
    assert rollout.max_model_len == rollout.max_num_batched_tokens == 131072
    assert config.data.train_batch_size == config.actor_rollout_ref.actor.ppo_mini_batch_size == 2
    assert rollout.n == 4 and config.trainer.v1.sampler.max_off_policy_threshold == 1
    assert config.algorithm.rollout_correction.rollout_is_threshold == 2
    assert config.trainer.resume_mode == "disable"
    assert config.actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler
    assert dict(config.actor_rollout_ref.actor.optim.override_optimizer_config) == {
        "adam_beta1": .9, "adam_beta2": .95}
    assert rollout.custom.agent_framework.mask_unfinished_episode
    assert rollout.temperature == rollout.val_kwargs.temperature == .6
    assert rollout.top_p == rollout.val_kwargs.top_p == 1.
    assert rollout.top_k == rollout.val_kwargs.top_k == -1
    assert rollout.logprobs_mode == "processed_logprobs"
    result = check_curriculum(experiment, config)
    assert result["thresholds"]["max_turn_tokens"] == 8192
    assert result["thresholds"]["maximum_truncated_trajectories"] == 0
    assert result["tool_observation_max_characters"] == 12000
    assert result["sampling"] == {"temperature": .6, "top_p": 1., "top_k": -1}


@pytest.mark.parametrize("observation_limit", [None, 0])
def test_replacement_curriculum_requires_explicit_positive_tool_observation_budget(
    conversion_deployment, tmp_path, observation_limit,
):
    import yaml

    from archlab.training.config import compose_verl
    from archlab.training.qualification import check_curriculum

    experiment = read_experiment(Path("recipes/rl/limite_conversion_code.yaml"))
    config = compose_verl(experiment, Path("third_party/verl"))
    harness = yaml.safe_load(Path("recipes/rl/harness/conversion_code_bash.yaml").read_text())
    if observation_limit is None:
        harness["agent"].pop("max_observation_length")
    else:
        harness["agent"]["max_observation_length"] = observation_limit
    (tmp_path / "harness.yaml").write_text(yaml.safe_dump(harness))
    spec = tmp_path / "spec.yaml"
    spec.write_text(yaml.safe_dump({"harnesses": [{"config": "harness.yaml"}]}))
    config.ray_kwargs.ray_init.runtime_env.env_vars.MIXED_HARNESS_SPEC = str(spec)
    with pytest.raises(ValueError, match="curriculum"):
        check_curriculum(experiment, config)


@pytest.mark.parametrize("changed", [
    {"temperature": 1.}, {"top_p": .95}, {"top_k": 20},
    {"extra_body": {"temperature": 1.}}, {"presence_penalty": 1.},
])
def test_replacement_rejects_harness_overrides_of_learner_distribution(
    conversion_deployment, tmp_path, changed,
):
    import yaml

    from archlab.training.config import compose_verl
    from archlab.training.qualification import check_curriculum

    experiment = read_experiment(Path("recipes/rl/limite_conversion_code.yaml"))
    config = compose_verl(experiment, Path("third_party/verl"))
    harness = yaml.safe_load(Path("recipes/rl/harness/conversion_code_bash.yaml").read_text())
    harness["model"]["model_kwargs"].update(changed)
    (tmp_path / "harness.yaml").write_text(yaml.safe_dump(harness))
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump({"harnesses": [{"config": "harness.yaml"}]}))
    config.ray_kwargs.ray_init.runtime_env.env_vars.MIXED_HARNESS_SPEC = str(tmp_path / "spec.yaml")
    with pytest.raises(ValueError, match="sampling"):
        check_curriculum(experiment, config)


@pytest.mark.parametrize("override", [
    "parent_tokens=2000158720", "parent_stage=adapter_sft", "intervention=insert_attention",
    "settings.actor_rollout_ref.rollout.enforce_eager=false",
    "settings.actor_rollout_ref.rollout.enable_chunked_prefill=true",
    "settings.actor_rollout_ref.rollout.enable_prefix_caching=true",
    "settings.actor_rollout_ref.rollout.max_num_seqs=2",
    "settings.actor_rollout_ref.rollout.max_num_batched_tokens=16384",
    "settings.actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2",
    "settings.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2",
    "settings.actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=false",
    "settings.actor_rollout_ref.actor.optim.override_optimizer_config.adam_beta2=0.999",
    "settings.actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=false",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm.enforce_eager=false",
    "pilot.maximum_truncated_trajectories=1", "pilot.minimum_reward_contrast_groups=0",
    "settings.actor_rollout_ref.rollout.temperature=0",
    "settings.actor_rollout_ref.rollout.temperature=nan",
    "settings.actor_rollout_ref.rollout.temperature=.nan",
    "settings.actor_rollout_ref.rollout.top_p=0.95",
    "settings.actor_rollout_ref.rollout.top_k=20",
    "settings.actor_rollout_ref.rollout.do_sample=false",
    "settings.actor_rollout_ref.rollout.logprobs_mode=raw_logprobs",
    "settings.actor_rollout_ref.rollout.val_kwargs.temperature=1",
    "settings.actor_rollout_ref.rollout.val_kwargs.do_sample=false",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm.logprobs_mode=raw_logprobs",
])
def test_replacement_overrides_cannot_bypass_cache_lineage_or_health_contract(conversion_deployment, override):
    from omegaconf.errors import OmegaConfBaseException

    from archlab.training.config import compose_verl

    with pytest.raises((ValueError, OmegaConfBaseException)):
        compose_verl(read_experiment(Path("recipes/rl/limite_conversion_code.yaml"), [override]),
                     Path("third_party/verl"))


@pytest.fixture
def muse_deployment(monkeypatch):
    for name in ("PARENT_CHECKPOINT", "CODE_DATA", "OUTPUT", "MODEL_MANIFEST"):
        monkeypatch.setenv("ARCHLAB_" + name, "/fixture/" + name.lower())
    monkeypatch.setenv("ARCHLAB_HARNESS_SPEC", str(Path("recipes/rl/harness/muse_single_bash.yaml").resolve()))
    for name in ("E2B_DOMAIN", "ARCHLAB_E2B_KEY_FILE", "ARCHLAB_E2B_TEMPLATE_MAP"):
        monkeypatch.setenv(name, "/fixture/" + name.lower())


def test_muse_native_control_composes_standard_fsdp_and_explicit_mimo_tasks(muse_deployment):
    import yaml

    from archlab.training.config import compose_verl

    experiment = read_experiment(Path("recipes/rl/muse_code.yaml"))
    config = compose_verl(experiment, Path("third_party/verl"))
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    assert experiment.parent_stage == "published_checkpoint" and experiment.parent_tokens is None
    assert experiment.publisher_manifest == "/fixture/model_manifest"
    assert experiment.intervention == "none" and experiment.variant == "normal"
    assert config.actor_rollout_ref.model.path == experiment.parent_checkpoint
    assert config.model_engine == "dp" and actor.strategy == actor.fsdp_config.strategy == "fsdp2"
    assert "megatron" not in actor
    assert actor.fsdp_config.model_dtype == "fp32" and actor.fsdp_config.dtype == "bfloat16"
    assert actor.optim.optimizer == "Adam" and actor.optim.lr == 1e-6
    assert list(actor.optim.betas) == [.9, .95]
    assert not config.actor_rollout_ref.model.external_lib
    assert dict(config.actor_rollout_ref.model.override_config) == {"attn_implementation": "flash_attention_4"}
    assert not config.actor_rollout_ref.model.use_remove_padding
    assert not config.actor_rollout_ref.model.use_fused_kernels
    assert actor.ppo_micro_batch_size_per_gpu == 1
    assert config.data.custom_cls.name == "MimoAgentSWEDataset"
    assert config.reward.custom_reward_function.path == "pkg://recipes.code.reward"
    assert rollout.agent.agent_loop_manager_class == "uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    assert rollout.multi_turn.format == "muse_glimmer"
    assert dict(config.data.apply_chat_template_kwargs) == {"reasoning_strength": "high"}
    assert rollout.max_model_len == rollout.prompt_length + rollout.response_length == 131072
    assert dict(rollout.engine_kwargs.vllm) == {
        "model_impl": "vllm", "limit_mm_per_prompt": {"image": 0, "video": 0}}
    assert rollout.enforce_eager and rollout.quantization is None
    assert rollout.custom.agent_framework.mask_unfinished_episode
    assert config.data.train_batch_size == actor.ppo_mini_batch_size == 2 and rollout.n == 4
    assert config.trainer.v1.sampler.max_off_policy_threshold == 1
    assert config.algorithm.rollout_correction.rollout_is_threshold == 2.
    assert config.trainer.resume_mode == "disable"
    assert config.trainer.total_training_steps == 20
    assert config.trainer.save_freq == config.trainer.test_freq == 5
    assert config.trainer.max_actor_ckpt_to_keep == 2
    assert config.trainer.project_name == "Muse — RL"
    spec = yaml.safe_load(Path("recipes/rl/harness/muse_single_bash.yaml").read_text())
    harness = yaml.safe_load((Path("recipes/rl/harness") / spec["harnesses"][0]["config"]).read_text())
    assert harness["agent"]["type"] == "bashonly-agent"
    assert harness["agent"]["step_limit"] == experiment.pilot.step_limit == 500
    assert harness["model"]["model_kwargs"]["max_tokens"] == experiment.pilot.max_turn_tokens == 8192
    assert {key: harness["model"]["model_kwargs"][key] for key in ("temperature", "top_p", "top_k")} == {
        key: rollout[key] for key in ("temperature", "top_p", "top_k")}


@pytest.mark.parametrize("override", [
    "engine=megatron", "attention=tilelang", "variant=simplicial", "intervention=replace_attention",
    "parent_stage=adapter_sft", "parent_tokens=2000158720", "publisher_manifest=null",
    "settings.actor_rollout_ref.model.path=/different-checkpoint",
    "settings.actor_rollout_ref.model.custom_chat_template=non-native",
    "settings.actor_rollout_ref.model.override_config={}",
    "settings.actor_rollout_ref.model.override_config={attn_implementation: sdpa}",
    "settings.actor_rollout_ref.model.override_config={attn_implementation: flash_attention_4, hidden_size: 128}",
    "settings.actor_rollout_ref.actor.strategy=fsdp",
    "settings.actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
    "settings.actor_rollout_ref.actor.optim.betas=[0.9,0.999]",
    "settings.actor_rollout_ref.model.use_remove_padding=true",
    "settings.actor_rollout_ref.model.lora.rank=8",
    "settings.actor_rollout_ref.model.use_fused_kernels=true",
    "settings.actor_rollout_ref.rollout.enforce_eager=false",
    "settings.actor_rollout_ref.rollout.ignore_eos=true",
    "settings.actor_rollout_ref.rollout.quantization=fp8",
    "settings.actor_rollout_ref.rollout.engine_kwargs.vllm={model_impl: transformers}",
    "settings.actor_rollout_ref.rollout.multi_turn.format=hermes",
    "settings.data.apply_chat_template_kwargs={reasoning_strength: low}",
    "settings.actor_rollout_ref.rollout.top_p=0.95",
    "settings.actor_rollout_ref.rollout.val_kwargs.do_sample=false",
    "settings.actor_rollout_ref.rollout.n=2",
    "settings.actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=false",
    "settings.trainer.v1.sampler.max_off_policy_threshold=2",
    "settings.trainer.resume_mode=auto",
    "pilot.max_turn_tokens=131072", "pilot.minimum_reward_contrast_groups=0",
])
def test_muse_control_rejects_unqualified_lineage_runtime_or_algorithm_changes(muse_deployment, override):
    from omegaconf.errors import OmegaConfBaseException

    from archlab.training.config import compose_verl

    with pytest.raises((ValueError, OmegaConfBaseException)):
        compose_verl(read_experiment(Path("recipes/rl/muse_code.yaml"), [override]),
                     Path("third_party/verl"))

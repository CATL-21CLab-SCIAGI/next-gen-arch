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
    with pytest.raises(ValueError, match="strategy|GRPO|precision|patches|contracts|FlashAttention-2"):
        compose_verl(experiment, upstream)

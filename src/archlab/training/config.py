"""Compose the pinned MiMo recipe with a small, typed experiment contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import MISSING, DictConfig, OmegaConf

from archlab.training.performance import ThroughputContract


@dataclass
class Experiment:
    name: str = MISSING
    family: str = "limite"
    variant: str = "normal"
    engine: str = "megatron"
    attention: str = "tilelang"
    execution: str = "colocate_async"
    native_context: int = 131072
    parent_checkpoint: str = MISSING
    parent_stage: str = "adapter_sft"
    parent_tokens: int = 2_000_158_720
    data_manifest: str = MISSING
    throughput: ThroughputContract = field(default_factory=ThroughputContract)
    # Overrides use upstream names; there is no second set of optimizer defaults.
    settings: dict[str, Any] = field(default_factory=dict)


def read_experiment(path: Path, overrides: list[str] = ()) -> Experiment:
    config = OmegaConf.merge(OmegaConf.structured(Experiment), OmegaConf.load(path))
    for override in overrides:
        key, separator, raw = override.partition("=")
        if not separator or not key:
            raise ValueError("overrides must use key=value")
        value = OmegaConf.from_dotlist([f"value={raw}"]).value
        if key.startswith("settings."):
            # Recipe settings are literal upstream paths, not a nested shadow
            # configuration. Upstream's structured schema checks them below.
            config.settings[key.removeprefix("settings.")] = value
        else:
            OmegaConf.update(config, key, value, merge=False)
    experiment = OmegaConf.to_object(config)
    experiment.throughput.validate()
    if experiment.engine != "megatron":
        raise ValueError("the primary execution engine is megatron")
    if experiment.attention not in {"native", "tilelang", "te"}:
        raise ValueError("attention must be native, tilelang or te")
    if experiment.execution not in {"sync", "colocate_async"}:
        raise ValueError("execution must be sync or colocate_async")
    if experiment.variant not in {"normal", "simplicial", "linsimp", "triadic"}:
        raise ValueError("unknown architecture variant")
    if experiment.native_context <= 0:
        raise ValueError("native_context must be positive")
    if experiment.parent_stage not in {"adapter_sft", "full_sft"} or experiment.parent_tokens <= 0:
        raise ValueError("this fresh RL pilot requires an explicit SFT token endpoint")
    return experiment


def compose_verl(experiment: Experiment, upstream: Path) -> DictConfig:
    """Reuse MiMo's agent, dataset, reward, trainer and monitoring configuration.

    Megatron and attention kernels are separate axes. Selecting an axis does not
    establish model support; numerical admission is required before execution.
    """
    with initialize_config_dir(
        config_dir=str((upstream / "recipes/code/config").resolve()), version_base=None
    ):
        searchpath = json.dumps("file://" + str((upstream / "verl/trainer/config").resolve()))
        config = compose(config_name="train", overrides=[
            f"hydra.searchpath=[{searchpath}]", f"model_engine={experiment.engine}",
        ])
    OmegaConf.set_struct(config, True)
    for key, value in experiment.settings.items():
        OmegaConf.update(config, key, value, merge=False)
    config.trainer.experiment_name = experiment.name
    config.trainer.use_v1 = True
    config.trainer.v1.trainer_mode = experiment.execution
    if not config.reward.reward_model.enable:
        config.reward.reward_model.rollout.name = None
    validate_science(experiment, config)
    # Missing values and interpolations fail before Ray or any model is loaded.
    OmegaConf.resolve(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config


def validate_science(experiment: Experiment, config: DictConfig) -> None:
    actor = config.actor_rollout_ref.actor
    if config.model_engine != experiment.engine or actor.strategy != "megatron":
        raise ValueError("resolved learner strategy differs from the experiment engine")
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("the agentic code pilot requires GRPO")
    if actor.megatron.dtype != "bfloat16":
        raise ValueError("native mixed precision requires the bfloat16 Megatron contract")
    if actor.megatron.patch_recomputation_backward:
        raise ValueError("native Limite owns checkpointing; runtime Megatron patches must be disabled")
    if config.actor_rollout_ref.model.override_config:
        raise ValueError("model geometry overrides require a new verified export")
    rollout = config.actor_rollout_ref.rollout
    if not rollout.enforce_eager:
        compilation = rollout.engine_kwargs.vllm.get("compilation_config", {})
        if (rollout.max_num_seqs != 1 or compilation.get("mode") != 0
                or compilation.get("cudagraph_mode") != "FULL_DECODE_ONLY"
                or compilation.get("cudagraph_capture_sizes") != [1]
                or compilation.get("max_cudagraph_capture_size") != 1):
            raise ValueError("native serving requires eager execution or explicit single-sequence decode graphs")
    if experiment.attention == "native":
        attention = rollout.engine_kwargs.vllm.attention_config
        if (experiment.variant != "normal" or attention.backend != "FLASH_ATTN"
                or attention.flash_attn_version != 2):
            raise ValueError("native normal RL requires explicit FlashAttention-2 serving")
    if rollout.checkpoint_engine.backend != "naive":
        raise ValueError("native mixed precision requires qualified full-weight synchronization")
    if rollout.max_model_len != experiment.native_context:
        raise ValueError("rollout must retain the model's native context")
    if rollout.prompt_length + rollout.response_length != experiment.native_context:
        raise ValueError("prompt and response allocation must cover native context")
    if (config.data.max_prompt_length != rollout.prompt_length
            or config.data.max_response_length != rollout.response_length):
        raise ValueError("dataset and agent context geometry differ")
    if config.data.truncation != "error":
        raise ValueError("silently truncating agentic task prompts is not allowed")
    if not rollout.multi_turn.enable or not rollout.calculate_log_probs:
        raise ValueError("agentic RL requires multi-turn rollout and behavior log probabilities")
    if rollout.n < 2:
        raise ValueError("GRPO requires multiple responses and recorded behavior probabilities")
    correction = config.algorithm.rollout_correction
    if (correction.bypass_mode or correction.rollout_is != "token"
            or correction.rollout_is_threshold != 2.0 or correction.rollout_is_batch_normalize):
        raise ValueError("the pilot requires one upstream token importance correction, capped at 2")
    if config.actor_rollout_ref.actor.use_kl_loss or config.algorithm.use_kl_in_reward:
        raise ValueError("this controlled code pilot does not include a reference-policy KL")
    if rollout.custom.agent_framework.mask_unfinished_episode:
        raise ValueError("unfinished trajectories must remain visible in this pilot")
    if not 0 <= config.trainer.v1.sampler.max_off_policy_threshold <= 1:
        raise ValueError("initial agentic pilot permits at most one policy version of lag")
    if config.trainer.resume_mode != "disable":
        raise ValueError("fresh RL must reset the optimizer, RNG and training counters")
    if (rollout.custom.agent_framework.get("trajectory_validator") != "archlab.training.rollout.RolloutContract"
            or config.trainer.get("metrics_extension") != "archlab.training.metrics.step_metrics"):
        raise ValueError("the primary pilot requires validated rollout and performance contracts")
    if config.ray_kwargs.ray_init.runtime_env.env_vars.get("VERL_MFU_DEVICE_NAME") != "B300":
        raise ValueError("MFU hardware must reflect the actual B300 allocation")


def fingerprint(config: DictConfig) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(config, resolve=True, sort_keys=True).encode()).hexdigest()

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
class AgenticPilot:
    max_turn_tokens: int = 8192
    step_limit: int = 500
    minimum_prompt_groups: int = 2
    minimum_completed_verifiers: int = 8
    minimum_tool_calls: int = 1
    minimum_observation_tokens: int = 1
    minimum_reward_contrast_groups: int = 1
    maximum_truncated_trajectories: int = 0


@dataclass
class Experiment:
    name: str = MISSING
    family: str = "limite"
    variant: str = "normal"
    intervention: str = "insert_attention"
    engine: str = "megatron"
    attention: str = "tilelang"
    execution: str = "colocate_async"
    native_context: int = 131072
    parent_checkpoint: str = MISSING
    parent_stage: str = "adapter_sft"
    parent_tokens: int | None = 2_000_158_720
    data_manifest: str = MISSING
    pilot: AgenticPilot | None = None
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
    if experiment.intervention not in {"insert_attention", "replace_attention"}:
        raise ValueError("unknown architecture intervention")
    if experiment.native_context <= 0:
        raise ValueError("native_context must be positive")
    if experiment.parent_stage == "package_sft":
        if (experiment.intervention != "replace_attention" or experiment.variant != "simplicial"
                or experiment.attention != "tilelang" or experiment.parent_tokens is not None):
            raise ValueError("package SFT replacement uses simplicial TileLang and receipt-derived token budgets")
    elif (experiment.parent_stage not in {"adapter_sft", "full_sft"}
          or experiment.parent_tokens is None or experiment.parent_tokens <= 0):
        raise ValueError("this fresh RL pilot requires an explicit SFT token endpoint")
    if experiment.intervention == "replace_attention" and experiment.parent_stage != "package_sft":
        raise ValueError("attention replacement RL requires a trained package SFT endpoint")
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
    if experiment.intervention == "replace_attention":
        _validate_conversion_science(experiment, config)
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
    if (experiment.intervention != "replace_attention"
            and rollout.custom.agent_framework.mask_unfinished_episode):
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


def _validate_conversion_science(experiment: Experiment, config: DictConfig) -> None:
    """The replacement cache bridge supports only its explicit eager geometry."""
    actor = config.actor_rollout_ref.actor
    model = config.actor_rollout_ref.model
    rollout = config.actor_rollout_ref.rollout
    if (experiment.parent_stage != "package_sft" or experiment.parent_tokens is not None
            or experiment.variant != "simplicial" or experiment.attention != "tilelang"
            or experiment.execution != "colocate_async"):
        raise ValueError("replacement RL requires package SFT lineage and bounded async TileLang execution")
    if (model.external_lib != "archlab.training.megatron" or model.use_remove_padding
            or model.use_fused_kernels or model.lora_rank or model.lora.rank
            or actor.megatron.use_remove_padding or actor.megatron.pad_bshd_to_minibatch_max
            or actor.ppo_micro_batch_size_per_gpu != 1 or actor.use_dynamic_bsz):
        raise ValueError("replacement RL requires full-weight native BSHD microbatches of one")
    if (any(actor.megatron[name] != 1 for name in (
        "tensor_model_parallel_size", "pipeline_model_parallel_size",
        "context_parallel_size", "expert_model_parallel_size",
    )) or actor.megatron.sequence_parallel or actor.megatron.dynamic_context_parallel
            or actor.megatron.virtual_pipeline_model_parallel_size is not None
            or config.trainer.nnodes != 1 or config.trainer.n_gpus_per_node != 8):
        raise ValueError("replacement learner requires DP8 with TP=PP=CP=EP=1")
    attention = rollout.engine_kwargs.vllm.attention_config
    if (rollout.name != "vllm" or not rollout.enforce_eager or rollout.max_num_seqs != 1
            or rollout.enable_prefix_caching or rollout.enable_chunked_prefill
            or rollout.max_num_batched_tokens < experiment.native_context
            or rollout.tensor_model_parallel_size != 1 or rollout.pipeline_model_parallel_size != 1
            or rollout.data_parallel_size != 1
            or attention.backend != "FLASH_ATTN" or attention.flash_attn_version != 2):
        raise ValueError("replacement serving requires eager FA2, one request, TP=PP=1 and no prefix/chunked prefill")
    conflicting = {
        "enforce_eager", "max_num_seqs", "enable_prefix_caching", "enable_chunked_prefill",
        "tensor_parallel_size", "pipeline_parallel_size", "decode_context_parallel_size",
        "prefill_context_parallel_size", "quantization", "kv_cache_dtype", "speculative_config",
        "hf_overrides", "model",
    } & set(rollout.engine_kwargs.vllm)
    if conflicting:
        raise ValueError("replacement engine kwargs override the serving contract: " + ", ".join(sorted(conflicting)))
    if actor.optim.use_checkpoint_opt_param_scheduler:
        raise ValueError("replacement RL requires a fresh optimizer and parameter scheduler")
    pilot = experiment.pilot
    if (pilot is None or not 0 < pilot.max_turn_tokens <= 8192 or pilot.step_limit != 500
            or pilot.minimum_prompt_groups < 2
            or pilot.minimum_completed_verifiers < pilot.minimum_prompt_groups * rollout.n
            or min(pilot.minimum_tool_calls, pilot.minimum_observation_tokens,
                   pilot.minimum_reward_contrast_groups) < 1
            or pilot.maximum_truncated_trajectories != 0
            or config.data.train_batch_size < pilot.minimum_prompt_groups
            or actor.ppo_mini_batch_size * rollout.n < 8
            or not rollout.custom.agent_framework.mask_unfinished_episode):
        raise ValueError("replacement pilot requires explicit short turns, usable traces and stock truncation masking")


def fingerprint(config: DictConfig) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(config, resolve=True, sort_keys=True).encode()).hexdigest()

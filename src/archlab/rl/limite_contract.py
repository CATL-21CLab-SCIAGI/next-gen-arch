"""Bind matched Limite RL recipes to the effective shared trainer settings."""

from __future__ import annotations

from archlab.rl.limite_protocol import MathRolloutProtocol


def check_matched_recipe(spec, effective, *, variant, world_size, phase_start,
                         checkpoint_steps, split_sha256, heldout_count,
                         correctness_fixture=False):
    """Reject ignored recipe settings before starting a new matched RL phase.

    Small distributed correctness fixtures may reduce the process count and
    number of updates/checkpoints. Their objective, native-context protocol and
    source data still match the production experiment.
    """
    model, training, execution = spec["model"], spec["training"], spec["execution"]
    if model.get("variants") != ["normal", "simplicial"] or variant not in model["variants"]:
        raise ValueError("matched RL requires the declared normal/simplicial comparison")
    if model.get("trainable_mode") != "full" or model.get("attention_backend") != "tilelang":
        raise ValueError("matched RL requires full-weight TileLang training")
    if MathRolloutProtocol(**spec["rollout"]).budget_mode != "native_context":
        raise ValueError("matched RL requires the native context budget")
    expected = {
        "learning_rate": training["learning_rate"],
        "epsilon": training["epsilon"],
        "epsilon_high": training["epsilon_high"],
        "seed": training["seed"],
        "data_seed": training["seed"],
        "num_generations": training["num_generations"],
        "generation_batch_size": training["responses_per_update"],
        "max_completion_length": spec["rollout"]["max_tokens"],
        "loss_type": "dapo", "scale_rewards": "group", "beta": 0.0,
        "num_iterations": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0,
        "mask_truncated_completions": False,
    }
    # Fixtures replace generation with short known responses, but must never
    # rewrite the experiment's declared full-context budget.
    if correctness_fixture:
        expected.pop("max_completion_length")
    else:
        expected["max_steps"] = training["max_steps"]
    for key, value in expected.items():
        if effective.get(key) != value:
            raise ValueError(f"effective matched RL {key} differs from the recipe")
    if (not correctness_fixture and (world_size != training["world_size_per_variant"]
                                    or checkpoint_steps != training["checkpoint_steps"])):
        raise ValueError("matched RL world size or checkpoint interval differs from the recipe")
    if phase_start != training["phase_start"]:
        raise ValueError("matched RL phase clock differs from the recipe")
    responses = training["responses_per_update"]
    if (responses != execution["responses_per_update"] or responses % world_size
            or responses % training["num_generations"]
            or effective["gradient_accumulation_steps"] * world_size != responses):
        raise ValueError("matched RL response and accumulation budgets differ")
    if (split_sha256 != spec["data"]["split_sha256"]
            or heldout_count != spec["data"]["heldout"]):
        raise ValueError("matched RL data split or heldout coverage differs from the recipe")
    if not execution.get("require_pinned_verl") or not execution.get("replay_head_chunk_size"):
        raise ValueError("matched native-context RL requires pinned verl and chunked policy scoring")
    if execution.get("async_rollouts") and (
        execution.get("overlap_actor_learner") is not False
        or execution.get("rollout_rendezvous") != "gloo_after_drain"
        or execution.get("max_policy_lag") != 1
    ):
        raise ValueError("matched asynchronous RL requires qualified drain, rendezvous and policy lag")
    return dict(variant=variant, world_size=world_size, phase_start=phase_start,
                checkpoint_steps=checkpoint_steps, correctness_fixture=correctness_fixture,
                effective_training={key: effective[key] for key in expected})

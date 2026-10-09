"""Audited protocol migrations retain optimizer/RNG but discard old rollouts."""

from copy import deepcopy


def protocol_transition(receipt, protocol, spec, *, phase_start, curriculum_sha256, new_tracking_phase):
    previous = receipt.get("math_protocol")
    current = protocol.contract() if protocol is not None else None
    if previous == current:
        if previous and (receipt.get("phase_start") != phase_start
                         or receipt.get("curriculum_sha256") != curriculum_sha256):
            raise ValueError("resume must retain the math curriculum and phase clock")
        return None
    transition = (spec or {}).get("protocol_transition", {})
    if not (
        previous and protocol and protocol.budget_mode == "native_context" and new_tracking_phase
        and previous == transition.get("previous_protocol")
        and phase_start == receipt["step"] == transition.get("checkpoint_step")
        and receipt.get("curriculum_sha256") == curriculum_sha256
        and transition.get("discard_prefetched_rollouts") is True
        and transition.get("retain_optimizer_scheduler_rng") is True
    ):
        raise ValueError("resume must retain the saved math RL protocol unless its explicit phase migration is declared")
    return dict(previous_protocol=previous, math_protocol=current, phase_start=phase_start,
                checkpoint_step=receipt["step"], optimizer_scheduler_retained=True,
                learner_rank_rng_retained=True, actor_rng="rewound before discarded old-protocol prefetch",
                same_experiment_phase=False)


def discard_prefetched_rollouts(state):
    """Discard old-protocol tokens and restore the actor draw clock before them."""
    result = dict(state, rank_rng=deepcopy(state["rank_rng"]), evidence=deepcopy(state["evidence"]))
    count = 0
    for rank in result["rank_rng"]:
        rollout = rank.get("rollout")
        if rollout and rollout.get("pending") is not None:
            rollout["generator"] = rollout["pending"]["rng_before"].clone()
            rollout["pending"] = None
            count += 1
    result["evidence"]["discarded_old_protocol_prefetched_batches"] = count
    result["evidence"]["flat_batches"] = 0
    result["evidence"]["pending"] = False
    return result

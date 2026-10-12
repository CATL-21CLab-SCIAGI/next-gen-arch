"""Validate upstream trajectories; do not copy their token transport or scheduler."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol


class Trajectory(Protocol):
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    generation_spans: list[tuple[int, int]]
    extra_fields: dict[str, Any]
    reward_info: dict[str, Any]


@dataclass(frozen=True)
class RolloutMetadata:
    question_id: str
    group_id: str
    response_id: str
    session_id: int
    policy_version_first: int
    policy_version_last: int
    termination_reason: Literal["completed", "truncated"]
    rng_identity: str
    # Backend scheduling affects RNG consumption: this is identity, not a claim
    # that the engine seed alone can replay an asynchronous request bit-exactly.
    rng_replayable: bool
    generated_tokens: int
    observation_tokens: int
    native_eos_tokens: int
    terminal_native_eos: bool
    context_limit_reached: bool


def validate_trajectory(trajectory: Trajectory, *, question_id: str, group_id: str,
                        response_id: str, session_id: int, context: int, eos_ids: set[int],
                        rng_identity: str) -> RolloutMetadata:
    if not all(isinstance(x, str) and x for x in (question_id, group_id, response_id, rng_identity)):
        raise ValueError("rollout identity is incomplete")
    if type(session_id) is not int or session_id < 0:
        raise ValueError("invalid rollout session")
    prompt, response, mask = trajectory.prompt_ids, trajectory.response_ids, trajectory.response_mask
    if not prompt or not response or len(prompt) + len(response) > context:
        raise ValueError("empty or over-context trajectory")
    if any(type(t) is not int or t < 0 for t in (*prompt, *response)):
        raise ValueError("invalid token IDs")
    logprobs = trajectory.response_logprobs
    if logprobs is None or len(response) != len(mask) or len(mask) != len(logprobs):
        raise ValueError("response tokens, masks and behavior probabilities must align")
    expected = [0] * len(response)
    previous = 0
    for start, end in trajectory.generation_spans:
        if not 0 <= previous <= start < end <= len(response):
            raise ValueError("generation spans overlap or are out of order")
        expected[start:end] = [1] * (end - start)
        previous = end
    if not any(expected) or mask != expected:
        raise ValueError("loss mask must select exactly the generated spans")
    if any(not math.isfinite(p) or (not m and p != 0) for p, m in zip(logprobs, mask, strict=True)):
        raise ValueError("non-finite behavior probabilities or unmasked tool observations")
    versions = [trajectory.extra_fields.get(k) for k in ("min_global_steps", "max_global_steps")]
    if any(type(v) is not int or v < 0 for v in versions) or versions[0] > versions[1]:
        raise ValueError("backend must report the actual policy version span")
    reason = trajectory.reward_info.get("termination_kind")
    if reason not in {"completed", "truncated"}:
        raise ValueError("missing or inadmissible termination reason")
    generated = [t for t, m in zip(response, mask, strict=True) if m]
    return RolloutMetadata(
        question_id, group_id, response_id, session_id, *versions, reason, rng_identity, False,
        len(generated), len(response) - len(generated), sum(t in eos_ids for t in generated),
        generated[-1] in eos_ids, len(prompt) + len(response) == context,
    )


class RolloutContract:
    """Optional Uni-Agent boundary callback, shared by all rollout backends."""

    def __init__(self, config):
        root = Path(config.actor_rollout_ref.model.path)
        generation = json.loads((root / "generation_config.json").read_text())
        eos = generation["eos_token_id"]
        self.eos_ids = {eos} if isinstance(eos, int) else set(eos)
        if not self.eos_ids or any(type(x) is not int or x < 0 for x in self.eos_ids):
            raise ValueError("invalid native generation EOS contract")
        rollout = config.actor_rollout_ref.rollout
        self.context = int(rollout.max_model_len)
        self.rng_scope = f"{rollout.name}:engine-seed={rollout.seed}:backend-managed"
        self.mask_unfinished = bool(rollout.get("custom", {}).get(
            "agent_framework", {}).get("mask_unfinished_episode", False))

    def __call__(self, *, trajectory, sample_fields, group_id, response_id, session_id):
        metadata = validate_trajectory(
            trajectory, question_id=sample_fields["extra_info"]["instance_id"],
            group_id=group_id, response_id=response_id, session_id=session_id, context=self.context,
            eos_ids=self.eos_ids, rng_identity=f"{self.rng_scope}:{response_id}",
        )
        if self.mask_unfinished and trajectory.reward_info.get("finished") is not (
            metadata.termination_reason == "completed"
        ):
            raise ValueError("agent termination must set finished before stock unfinished-episode masking")
        return asdict(metadata)

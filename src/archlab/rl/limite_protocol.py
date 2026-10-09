# Adapted detector/formula: Copyright 2025–2026 Bytedance Ltd. and/or affiliates.
# Licensed under Apache License 2.0; see docs/licenses/verl-Apache-2.0.txt.
"""Explicit finish reasons and MiMo/verl-informed math reward shaping."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from archlab.rl.limite_data import math_reward, reasoning_complete


@dataclass(frozen=True)
class MathRolloutProtocol:
    max_tokens: int = 16384
    overlong_buffer: int = 4096
    overlong_penalty: float = 0.5
    unfinished_penalty: float = 0.5
    repetition_penalty: float = 0.5
    budget_mode: str = "fixed_response"

    def __post_init__(self):
        if self.budget_mode not in ("fixed_response", "native_context"):
            raise ValueError("unknown response budget mode")
        if self.budget_mode == "native_context" and (
            self.max_tokens != 131072 or self.overlong_penalty or self.unfinished_penalty
        ):
            raise ValueError("native context requires 131072 and no length/censor reward penalties")
        if not 0 < self.overlong_buffer <= self.max_tokens:
            raise ValueError("overlong buffer must fit the response budget")
        if any(not 0 <= x <= 1 for x in (
            self.overlong_penalty, self.unfinished_penalty, self.repetition_penalty
        )):
            raise ValueError("math reward penalties must be between zero and one")

    def contract(self):
        values = asdict(self)
        if self.budget_mode == "fixed_response":
            # Retain the exact historical receipt for unchanged experiments.
            values.pop("budget_mode")
        return dict(
            version=("native-context-math-v3" if self.budget_mode == "native_context" else "mimo-math-v2"), **values,
            accuracy="symbolic_math_only_after_natural_eos_and_closed_reasoning",
            train_unfinished=True, forced_eos=False,
            upstream="XiaomiMiMo/verl@a2ad9f6160b03ff2d47e59832bfb6b289f37c917",
        )


_VERL_REPETITION = None


def configure_verl_repetition(root):
    """Bind the pinned upstream implementation before starting actor threads."""
    global _VERL_REPETITION
    from archlab.rl.verl_components import repetition_component

    _VERL_REPETITION, receipt = repetition_component(root)
    return receipt


def degenerate_repetition(text, *, min_words=500, uniq4_threshold=0.15):
    """MiMo web-dev's word 4-gram detector; provenance in docs/NOTICE.md."""
    if _VERL_REPETITION is not None:
        return _VERL_REPETITION(text, min_words=min_words, uniq4_threshold=uniq4_threshold)
    words = text.split()
    if len(words) < min_words:
        return False
    grams = [" ".join(words[i : i + 4]) for i in range(len(words) - 3)]
    return bool(grams) and len(set(grams)) / len(grams) < uniq4_threshold


def response_budget(config, prompt_tokens, context_limit):
    """Apply the native total-context limit after actual chat tokenization."""
    maximum = config.max_new_tokens
    if not 0 < prompt_tokens < context_limit or not isinstance(maximum, int) or maximum < 1:
        raise ValueError("invalid prompt or response budget")
    if getattr(config, "archlab_budget_mode", "fixed_response") == "native_context":
        if maximum != context_limit:
            raise ValueError("native response budget must match the model context")
        return context_limit - prompt_tokens
    if prompt_tokens + maximum > context_limit:
        raise ValueError("rollout budget exceeds the native model context")
    return maximum


def score_math_rollout(text, answer, token_ids, finish_reason, protocol, *, training, completion_budget=None):
    # A safety stop, a user pause, or a token cap must never become a successful
    # answer simply because an intermediate calculation contains a boxed value.
    if finish_reason not in ("eos", "length", "stop", "repetition"):
        raise ValueError("unknown rollout finish reason")
    natural_eos = bool(token_ids) and token_ids[-1] in (151643, 151645)
    if (finish_reason == "eos") != natural_eos or any(token in (151643, 151645) for token in token_ids[:-1]):
        raise ValueError("EOS receipt does not match the actually sampled token")
    content = text[-1]["content"] if isinstance(text, list) else text
    closed = reasoning_complete(content)
    repeated = degenerate_repetition(content)
    accuracy = math_reward(text, answer) if finish_reason == "eos" and closed and not repeated else 0.0
    if protocol.budget_mode == "native_context" and completion_budget is None:
        raise ValueError("native-context scoring requires the actual per-prompt completion budget")
    budget = protocol.max_tokens if completion_budget is None else completion_budget
    if not isinstance(budget, int) or not 0 < budget <= protocol.max_tokens or len(token_ids) > budget:
        raise ValueError("completion exceeds its recorded budget")
    if protocol.budget_mode == "native_context" and finish_reason == "length" and len(token_ids) != budget:
        raise ValueError("native context exhaustion receipt has fewer tokens than its budget")
    expected_length = budget - min(protocol.overlong_buffer, budget)
    # Same linear soft-overlong ramp as verl's DAPO reward manager. A separate
    # unfinished penalty distinguishes real answers from shorter censored text.
    overlong = min(
        -(len(token_ids) - expected_length) / protocol.overlong_buffer * protocol.overlong_penalty,
        0.0,
    )
    unfinished = -protocol.unfinished_penalty if finish_reason != "eos" or not closed else 0.0
    repetition = -protocol.repetition_penalty if repeated else 0.0
    shaped = accuracy + overlong + unfinished + repetition
    return dict(
        accuracy=accuracy, reward=shaped if training else accuracy,
        overlong_penalty=overlong, unfinished_penalty=unfinished,
        repetition_penalty=repetition, finish_reason=finish_reason,
        reasoning_closed=closed, repeated=repeated, tokens=len(token_ids),
        completion_budget=budget, native_context_exhausted=(
            protocol.budget_mode == "native_context" and finish_reason == "length"
        ),
    )

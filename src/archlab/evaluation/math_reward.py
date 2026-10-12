"""Symbolic final-answer grading shared by retained mathematical evaluations."""

from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=8192)
def gold_expression(answer):
    from math_verify import parse

    return parse(r"\boxed{" + answer + "}", fallback_mode="no_fallback", parsing_timeout=2)


def reasoning_complete(text):
    """An earlier surplus closing tag cannot close a later reasoning block."""
    depth = 0
    for tag in re.finditer(r"</?think>", text):
        if tag.group() == "<think>":
            depth += 1
        else:
            depth = max(0, depth - 1)
    return depth == 0


def math_reward(completion, answer):
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    if isinstance(completion, list):
        completion = completion[-1]["content"]
    if not reasoning_complete(completion):
        return 0.0
    completion = completion.rsplit("</think>", 1)[-1]
    prediction = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(try_extract_without_anchor=False, boxed_match_priority=0),
            ExprExtractionConfig(try_extract_without_anchor=False),
        ],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
        parsing_timeout=2,
    )
    if not prediction:
        from archlab.rl.rewards import canonical_math_answer

        scalar = canonical_math_answer(completion.strip().rstrip("."))
        if scalar is not None:
            prediction = gold_expression(scalar)
        else:
            # Base-model answers need not be boxed. Restrict upstream expression
            # extraction to an explicit concluding paragraph ending in math;
            # never mine numbers from an intermediate calculation in the body.
            lines = completion.strip().splitlines()
            for index in range(len(lines) - 1, -1, -1):
                conclusion = lines[index].strip().rstrip(".")
                suffix = "\n".join(lines[index + 1 :])
                if re.search(r"[0-9]|\\boxed", suffix):
                    break
                if (
                    re.search(
                        r"\b(therefore|thus|hence|answer|result|sum|option|equals)\b",
                        conclusion,
                        re.I,
                    )
                    and conclusion
                    and conclusion[-1] in "0123456789})$]"
                ):
                    prediction = parse(
                        conclusion,
                        extraction_config=[ExprExtractionConfig()],
                        fallback_mode="no_fallback",
                        extraction_mode="first_match",
                        parsing_timeout=2,
                    )
                    break
    return float(
        bool(prediction) and verify(gold_expression(answer), prediction, timeout_seconds=2)
    )

import pytest

from archlab.tracking.deepseek_experiments import (
    FROM_SCRATCH,
    FULL_FINE_TUNING,
    RL,
    canonical_experiment,
)


@pytest.mark.parametrize(
    "name",
    [
        "DeepSeek V4.1 — Scratch w640 d20",
        "DeepSeek V4.1 — Width scaling 10B",
        "DeepSeek V4.1 — Figure 5 repeated-data sweep",
        "DeepSeek V4.1 — Aligned scaling 3.0",
    ],
)
def test_scratch_aliases(name):
    assert canonical_experiment(name) == FROM_SCRATCH


@pytest.mark.parametrize(
    "name", ["deepseek-v41-nemotron-rloo-20260922", "DeepSeek-V4.1 Miles GRPO"]
)
def test_rl_aliases(name):
    assert canonical_experiment(name) == RL


@pytest.mark.parametrize("name", [FROM_SCRATCH, FULL_FINE_TUNING, RL, "CritPt", "qwen35_9b_sft"])
def test_canonical_and_unrelated_names_preserved(name):
    assert canonical_experiment(name) == name

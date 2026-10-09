"""Stable top-level groups for DeepSeek V4.1 training runs."""

FROM_SCRATCH = "DeepSeek V4.1 — From scratch"
FULL_FINE_TUNING = "DeepSeek V4.1 — Full fine-tuning"
RL = "DeepSeek V4.1 — RL"

_ALIASES = {
    "DeepSeek V4.1 — Scratch w640 d20": FROM_SCRATCH,
    "DeepSeek V4.1 — Width scaling 10B": FROM_SCRATCH,
    "DeepSeek V4.1 — Figure 5 repeated-data sweep": FROM_SCRATCH,
    "DeepSeek V4.1 — Aligned scaling 3.0": FROM_SCRATCH,
    "deepseek-v41-nemotron-rloo-20260922": RL,
    "DeepSeek-V4.1 Miles GRPO": RL,
}


def canonical_experiment(name: str) -> str:
    """Keep legacy launch configurations from recreating variant-level groups."""
    return _ALIASES.get(name, name)

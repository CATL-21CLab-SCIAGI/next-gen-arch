"""Pinned eval_pipeline integration contracts, with no model/runtime imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from archlab.artifacts import sha256_file


def completion_budget(prompt_tokens: int, requested: int, context: int) -> int:
    """Spend the full *remaining* native context without truncating the prompt."""
    if not 0 < prompt_tokens < context or requested < 1:
        raise ValueError("prompt does not fit the model's native context")
    return min(requested, context - prompt_tokens)


def validate_pipeline_contract(plan: dict, cases: list[dict]) -> None:
    if len(cases) != 30 or {case["task"] for case in cases} != {"aime26"}:
        raise ValueError("eval_pipeline requires the complete sealed 30-question AIME26 exam")
    sampling = plan["sampling"]
    if sampling["context_limit"] != 131072 or sampling["max_new_tokens"] != 131072:
        raise ValueError("this contract uses Limite's full native 131072-token context")
    if sampling.get("budget_policy") != "native_context_minus_prompt":
        raise ValueError("completion budget must explicitly account for prompt tokens")
    if sampling["top_k"] != 0 or sampling.get("repetition_watchdog") is not False:
        raise ValueError("unregistered sampling or early termination policy")
    dependency = plan["eval_pipeline"]
    root = Path(dependency["source"])
    actual = {str(p.relative_to(root)): sha256_file(p) for p in sorted((root / "custom_eval").rglob("*.py"))}
    if actual != dependency["files_sha256"]:
        raise ValueError("pinned eval_pipeline source changed")
    if not dependency.get("upstream_revision") or not dependency.get("integration_patch_sha256"):
        raise ValueError("eval_pipeline provenance is missing")


def load_pipeline(plan: dict):
    root = Path(plan["eval_pipeline"]["source"]).resolve()
    sys.path.insert(0, str(root))
    module = importlib.import_module("custom_eval.eval_aime26")
    if Path(module.__file__).resolve() != root / "custom_eval" / "eval_aime26.py":
        raise ValueError("a different eval_pipeline package was already imported")
    return module


class SealedDataset(list):
    """The tiny dataset interface used by upstream's public evaluate function."""

    def select(self, indices):
        return type(self)(self[index] for index in indices)

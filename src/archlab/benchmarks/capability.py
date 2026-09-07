"""Dataset, scoring, and paired-statistics contracts for capability evaluation.

No model or training imports. Math grading reuses the existing pinned lm-eval
0.4.13 extraction/AIME code without importing unrelated BLEU dependencies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int = 20260907
    max_context: int = 2048
    max_new_tokens: int = 1024
    reasoning_effort: str = "medium"

    def __post_init__(self) -> None:
        if not 1 <= self.max_new_tokens < self.max_context <= 2048:
            raise ValueError("this pilot is qualified only up to 2048 total tokens; no silent context truncation")
        if self.reasoning_effort != "medium":
            raise ValueError("the registered pilot uses medium reasoning")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(output: Path, *, gsm8k: Path, mmlu: Path) -> dict:
    """Materialize public, pinned held-out datasets once; never overwrite a bundle."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    output.mkdir(parents=True, exist_ok=False)
    sources = {
        "gsm8k": {"path": str(gsm8k), "repo": "openai/gsm8k", "revision": "local-cache-recorded-sha256", "split": "test"},
        "mmlu": {"path": str(mmlu), "repo": "cais/mmlu", "revision": "local-cache-recorded-sha256", "split": "test"},
    }
    downloads = {
        "aime24": ("Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792", "aime_2024_problems.parquet", "train (published exam set)"),
        "aime25": ("math-ai/aime25", "563bb8404243c5f09de6ec262f2db674fe5bce9b", "test.jsonl", "test"),
        "arc_challenge": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge/test-00000-of-00001.parquet", "test"),
    }
    for task, (repo, revision, filename, split) in downloads.items():
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                               local_dir=output / "downloads" / task, token=False)
        sources[task] = {"path": path, "repo": repo, "revision": revision, "split": split}
    rows = []
    expected_counts = {"gsm8k": 1319, "mmlu": 14042, "arc_challenge": 1172, "aime24": 30, "aime25": 30}
    for task, source in sources.items():
        path = Path(source["path"])
        source["sha256"] = file_sha256(path)
        raw = ([json.loads(line) for line in path.read_text().split("\n") if line]
               if path.suffix == ".jsonl" else pq.read_table(path).to_pylist())
        if len(raw) != expected_counts[task]:
            raise ValueError(f"unexpected held-out row count for {task}: {len(raw)}")
        source["rows"] = len(raw)
        for index, doc in enumerate(raw):
            row = {"task": task, "index": index, "id": f"{task}:{index}", "subject": doc.get("subject")}
            if task == "gsm8k":
                row.update(question=doc["question"], answer=doc["answer"].split("####")[-1].strip())
            elif task == "mmlu":
                row.update(question=doc["question"], choices=doc["choices"], answer=int(doc["answer"]))
            elif task == "arc_challenge":
                labels, choices = doc["choices"]["label"], doc["choices"]["text"]
                row.update(question=doc["question"], choices=choices, answer=labels.index(doc["answerKey"]))
            else:
                keys = {k.lower(): k for k in doc}
                row.update(question=doc[keys["problem"]], answer=str(doc[keys["answer"]]))
            if not isinstance(row["question"], str) or not row["question"].strip():
                raise ValueError(f"empty question: {row['id']}")
            if "choices" in row and (not 0 <= row["answer"] < len(row["choices"]) or any(not c for c in row["choices"])):
                raise ValueError(f"invalid choices: {row['id']}")
            rows.append(row)
    path = output / "cases.jsonl"
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"schema_version": 1, "sources": sources, "cases_sha256": file_sha256(path),
                "note": "GSM8K/MMLU reuse local cached test sets; their source file SHA256 is recorded, not an inferred HF revision."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def select_cases(rows: list[dict], *, limits: dict[str, int], seed: int) -> list[dict]:
    """Fix the same examples for both modes; stratify the MMLU pilot by subject."""
    selected = []
    for task, limit in limits.items():
        pool = [row for row in rows if row["task"] == task]
        if type(limit) is not int or not 1 <= limit <= len(pool):
            raise ValueError(f"invalid sample count for {task}: {limit}")
        rng = random.Random(f"{seed}:{task}")
        if task == "mmlu" and limit == len({row["subject"] for row in pool}):
            sample = [rng.choice([row for row in pool if row["subject"] == subject])
                      for subject in sorted({row["subject"] for row in pool})]
        else:
            sample = rng.sample(pool, limit)
        selected.extend(sorted(sample, key=lambda row: row["index"]))
    return selected


class MathGrader:
    """Use pinned upstream math extraction; score final answers, not thinking text."""

    def __init__(self, harness: Path) -> None:
        from lm_eval.filters.extraction import RegexFilter

        path = harness / "lm_eval/tasks/aime/utils.py"
        spec = importlib.util.spec_from_file_location("archlab_aime_reference", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot read upstream AIME grader: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.aime = module
        self.strict = RegexFilter(regex_pattern=r"The answer is (\-?[0-9\.\,]+).")
        self.flexible = RegexFilter(regex_pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)
        self.provenance = {str(p.relative_to(harness)): file_sha256(p) for p in (
            path, harness / "lm_eval/filters/extraction.py", harness / "lm_eval/api/metrics.py",
            harness / "lm_eval/tasks/gsm8k/gsm8k-cot-zeroshot.yaml")}

    def score(self, row: dict, completion: str) -> dict:
        """Require the native medium-reasoning response to finish its think block."""
        final = completion.rsplit("</think>", 1)[-1] if "</think>" in completion else ""
        if row["task"] == "gsm8k":
            extracted = {"strict_match": list(self.strict.apply([[final]], [{}]))[0][0],
                         "flexible_extract": list(self.flexible.apply([[final]], [{}]))[0][0]}

            def normalize(value: str) -> str:
                # The pinned GSM8K YAML's exact_match regexes/ignore_case policy.
                for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
                    value = re.sub(pattern, "", value)
                return value.lower()

            metrics = {name: int(normalize(value) == normalize(str(row["answer"])))
                       for name, value in extracted.items()}
            return {"metrics": metrics, "extracted": extracted, "final_answer_present": bool(final.strip())}
        score = self.aime.process_results({"Answer": row["answer"]}, [final])
        return {"metrics": score, "boxed_answer": self.aime.last_boxed_only_string(final),
                "final_answer_present": bool(final.strip())}


def paired_statistics(base: list[int], adapted: list[int]) -> dict:
    """Report paired transitions and a conservative, non-degenerate delta interval.

    Combine 97.5% Wilson intervals for the gain/loss fractions using Bonferroni.
    Unlike a naive bootstrap, zero observed discordances do not imply certainty
    of zero degradation. The interval is approximate and intentionally wide.
    """
    if not base or len(base) != len(adapted) or any(x not in (0, 1) for x in base + adapted):
        raise ValueError("paired binary scores must be nonempty and have equal lengths")
    n = len(base)
    wins = sum(a > b for b, a in zip(base, adapted, strict=True))
    losses = sum(a < b for b, a in zip(base, adapted, strict=True))
    z = NormalDist().inv_cdf(0.9875)

    def interval(count: int) -> tuple[float, float]:
        p, denominator = count / n, 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return max(0.0, center - half), min(1.0, center + half)

    wlo, whi = interval(wins)
    llo, lhi = interval(losses)
    discordant = wins + losses
    p_value = (min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant)
               if discordant else 1.0)
    return {"n": n, "pretrained_accuracy": sum(base) / n, "adapted_accuracy": sum(adapted) / n,
            "delta_percentage_points": 100 * (wins - losses) / n,
            "gains": wins, "regressions": losses, "both_correct": sum(b and a for b, a in zip(base, adapted, strict=True)),
            "both_wrong": sum(not b and not a for b, a in zip(base, adapted, strict=True)),
            "delta_95pct_interval_pp": [100 * (wlo - lhi), 100 * (whi - llo)],
            "interval_method": "Bonferroni-combined 97.5% Wilson gain/loss intervals; approximate",
            "mcnemar_exact_two_sided_p": p_value}

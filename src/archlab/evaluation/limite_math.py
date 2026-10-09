"""Public AIME data, symbolic grading, and problem-level statistics for Limite.

This module owns benchmark contracts only. Models, generation, and machine
paths live in the execution adapter. It reuses the existing pinned verifier
and paired-score statistics instead of introducing a second math reward.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.evaluation.capability import paired_statistics
from archlab.rl.limite_data import math_reward, reasoning_complete

AIME_SOURCES = {
    "aime24": {
        "repo": "Maxwell-Jia/AIME_2024",
        "revision": "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        "filename": "aime_2024_problems.parquet",
        "split": "train (published exam set)",
    },
    "aime25": {
        "repo": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "filename": "test.jsonl",
        "split": "test",
    },
    "aime26": {
        "repo": "MathArena/aime_2026",
        "revision": "d2de22f3c656b4f56cf8981212186377d1e23bc3",
        "filename": "data/train-00000-of-00001.parquet",
        "split": "train (published exam set)",
    },
}


def prepare_aime(output: Path, existing_bundle: Path | None = None) -> dict:
    """Seal all 30 questions/year, with references never entering prompts."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    output.mkdir(parents=True, exist_ok=False)
    sources, cases = {}, []
    for task, specification in AIME_SOURCES.items():
        path = (existing_bundle / "downloads" / task / specification["filename"]
                if existing_bundle is not None else None)
        if path is None or not path.is_file():
            path = Path(hf_hub_download(
                specification["repo"], specification["filename"],
                repo_type="dataset", revision=specification["revision"],
                local_dir=output / "downloads" / task, token=False,
            ))
        documents = ([json.loads(line) for line in path.read_text().splitlines() if line]
                     if path.suffix == ".jsonl" else pq.read_table(path).to_pylist())
        if len(documents) != 30:
            raise ValueError(f"{task} needs the complete 30-question exam")
        sources[task] = dict(specification, sha256=sha256_file(path), count=len(documents))
        for index, document in enumerate(documents):
            names = {key.lower(): key for key in document}
            problem, answer = document[names["problem"]], str(document[names["answer"]])
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError("benchmark contains an empty problem")
            if not answer.strip().isdigit() or not 0 <= int(answer) <= 999:
                raise ValueError("AIME gold must be an integer in [0,999]")
            cases.append(dict(id=f"{task}:{index}", task=task, index=index,
                              question=problem, answer=str(int(answer))))
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases))
    manifest = dict(format="archlab-limite-aime-bundle-v1", sources=sources,
                    count=len(cases), cases_sha256=sha256_file(cases_path),
                    order="aime24, aime25, aime26; source row order retained",
                    references_in_prompt=False)
    atomic_write_json(output / "MANIFEST.json", manifest)
    return manifest


def benchmark_seed(seed: int, problem_id: str, sample_index: int) -> int:
    """Give each sample an independent stable seed, unaffected by sharding."""
    if sample_index < 0:
        raise ValueError("sample index must be nonnegative")
    digest = hashlib.sha256(f"{seed}:{problem_id}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def score_aime(completion: str, answer: str, finish_reason: str) -> dict:
    """Use the shared symbolic final-answer grader; do not mine thinking text.

    Unlike training's natural-EOS reward requirement, benchmark final answers
    may score at the token limit. That fact is recorded separately so censoring
    cannot disappear from the benchmark report.
    """
    if finish_reason not in ("eos", "length"):
        raise ValueError("benchmark samples must end naturally or at the exact token budget")
    if not answer.isdigit() or not 0 <= int(answer) <= 999:
        raise ValueError("invalid AIME reference")
    closed = reasoning_complete(completion)
    return dict(correct=bool(math_reward(completion, answer)),
                reasoning_complete=closed, natural_eos=finish_reason == "eos",
                truncated=finish_reason == "length", grader="math-verify 0.8.0 symbolic; shared math_reward")


def first_eos_prefix(token_ids: list[int], eos_token_ids: list[int]) -> tuple[list[int], str]:
    """Recover the identical prefix under an explicitly corrected EOS contract.

    Stopping never changes logits or RNG draws before EOS. This permits a
    deterministic correction without resampling or selecting by correctness.
    """
    if not token_ids or not eos_token_ids:
        raise ValueError('token and EOS lists must be nonempty')
    for index, token in enumerate(token_ids):
        if token in eos_token_ids:
            return token_ids[:index + 1], 'eos'
    return token_ids, 'length'


def pass_at_k(correct: int, samples: int, k: int) -> float:
    """Unbiased sampling estimator; never report pass@k with fewer than k draws."""
    if not 0 <= correct <= samples or not 1 <= k <= samples:
        raise ValueError("pass@k requires 0 <= correct <= samples and 1 <= k <= samples")
    return 1.0 - (math.comb(samples - correct, k) / math.comb(samples, k)
                  if samples - correct >= k else 0.0)


def wilson_interval(correct: int, total: int) -> list[float]:
    if not 0 <= correct <= total or total < 1:
        raise ValueError("Wilson interval needs nonempty binary observations")
    z = NormalDist().inv_cdf(.975)
    p, denominator = correct / total, 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_aime(records: list[dict], expected_ids: list[str], samples: int,
                   *, bootstrap_draws: int = 10000, seed: int = 20261002) -> dict:
    """Only complete, balanced question samples form a scored exam summary."""
    if not expected_ids or len(set(expected_ids)) != len(expected_ids) or samples < 1:
        raise ValueError("benchmark requires unique nonempty expected problems and positive samples")
    groups = defaultdict(dict)
    for row in records:
        key, index = row["problem_id"], row["sample_index"]
        if key not in expected_ids or type(index) is not int or not 0 <= index < samples:
            raise ValueError("record is outside the registered exam sample contract")
        if index in groups[key]:
            raise ValueError("duplicate benchmark response")
        if type(row["correct"]) is not bool:
            raise ValueError("benchmark correctness must be Boolean")
        groups[key][index] = row
    completed = [key for key in expected_ids if len(groups[key]) == samples]
    if len(completed) != len(expected_ids):
        return dict(complete=False, completed_problems=len(completed), expected_problems=len(expected_ids),
                    recorded_samples=len(records), expected_samples=len(expected_ids) * samples)
    rng = random.Random(seed)
    counts = [sum(row["correct"] for row in groups[key].values()) for key in expected_ids]
    first = [groups[key][0]["correct"] for key in expected_ids]
    metrics = {}
    for k in (1, 2, 4):
        if k > samples:
            continue
        values = [pass_at_k(count, samples, k) for count in counts]
        draws = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                       for _ in range(bootstrap_draws))
        metrics[f"pass_at_{k}"] = sum(values) / len(values)
        metrics[f"pass_at_{k}_cluster_bootstrap_95pct"] = [
            draws[int(.025 * len(draws))], draws[min(len(draws) - 1, int(.975 * len(draws)))]]
    token_counts = sorted(row["generated_tokens"] for row in records)
    return dict(complete=True, problems=len(expected_ids), samples_per_problem=samples,
                independent_responses=len(records), correct_responses=sum(counts), **metrics,
                first_sample_correct=sum(first), first_sample_pass_at_1=sum(first) / len(first),
                first_sample_wilson_95pct=wilson_interval(sum(first), len(first)),
                uncertainty_unit="problem; repeats are not treated as independent exam questions",
                natural_eos_rate=sum(row["finish_reason"] == "eos" for row in records) / len(records),
                truncation_rate=sum(row["finish_reason"] == "length" for row in records) / len(records),
                mean_completion_tokens=sum(token_counts) / len(token_counts),
                p50_completion_tokens=token_counts[len(token_counts) // 2],
                p95_completion_tokens=token_counts[min(len(token_counts) - 1, int(.95 * len(token_counts)))])


def compare_aime(first: list[dict], second: list[dict], expected_ids: list[str]) -> dict:
    """Reuse the existing conservative paired test on the first registered draw."""
    def scores(rows):
        selected = {row["problem_id"]: int(row["correct"]) for row in rows if row["sample_index"] == 0}
        if set(selected) != set(expected_ids):
            raise ValueError("paired comparison requires the same complete benchmark")
        return [selected[key] for key in expected_ids]

    return paired_statistics(scores(first), scores(second))

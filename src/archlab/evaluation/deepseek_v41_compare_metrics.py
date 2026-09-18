"""Paired metrics for the two full-weight attention variants."""

from __future__ import annotations

import math

import numpy as np

from archlab.benchmarks.capability import paired_statistics

VARIANTS = ("simplicial", "normal")


def score_choices(case, scores):
    lengths = (
        [1] * len(case["choices"]) if case["task"] == "mmlu" else [len(x) for x in case["choices"]]
    )
    if any(n <= 0 for n in lengths):
        raise ValueError("empty choice text")
    result = {}
    for variant in VARIANTS:
        values = scores[variant]
        if len(values) != len(lengths) or any(v is None or not math.isfinite(v) for v in values):
            raise ValueError(f"incomplete/nonfinite choice scores: {case['id']}")
        normalized = [v / n for v, n in zip(values, lengths, strict=True)]
        pred = max(range(len(values)), key=values.__getitem__)
        pred_norm = max(range(len(values)), key=normalized.__getitem__)
        result[variant] = {
            "prediction": pred,
            "normalized_prediction": pred_norm,
            "accuracy": int(pred == case["answer"]),
            "accuracy_norm": int(pred_norm == case["answer"]),
        }
    return result


def paired_binary(first, second):
    result = paired_statistics(first, second)
    result["simplicial_accuracy"] = result.pop("pretrained_accuracy")
    result["normal_accuracy"] = result.pop("adapted_accuracy")
    result["normal_only_correct"] = result.pop("gains")
    result["simplicial_only_correct"] = result.pop("regressions")
    result["delta_definition"] = "normal minus simplicial"
    return result


def mc_summary(records):
    result = {}
    for task in sorted({r["task"] for r in records}):
        selected = [r for r in records if r["task"] == task]
        result[task] = {
            metric: paired_binary(
                [r["result"]["simplicial"][metric] for r in selected],
                [r["result"]["normal"][metric] for r in selected],
            )
            for metric in ("accuracy", "accuracy_norm")
        }
        result[task]["samples"] = len(selected)
        result[task]["prediction_agreement"] = sum(
            r["result"]["simplicial"]["prediction"] == r["result"]["normal"]["prediction"]
            for r in selected
        ) / len(selected)
        result[task]["exact_consumed_training_overlaps"] = [
            r["id"] for r in selected if r.get("exact_overlap_consumed_training")
        ]
    return result


def aggregate_math(records):
    total = sum(r["targets"] for r in records)
    if not total:
        raise ValueError("empty math evaluation")
    result = {"windows": len(records), "supervised_targets": total, "variants": {}}
    for v in VARIANTS:
        nll = sum(r[v]["nll"] for r in records) / total
        result["variants"][v] = {
            "cross_entropy": nll,
            "perplexity": math.exp(nll),
            "top1_token_accuracy": sum(r[v]["top1"] for r in records) / total,
            "top5_token_accuracy": sum(r[v]["top5"] for r in records) / total,
            "mean_predictive_entropy": sum(r[v]["entropy"] for r in records) / total,
        }
    for field in ("simplicial_to_normal_kl", "normal_to_simplicial_kl"):
        result[field] = sum(r[field + "_sum"] for r in records) / total
    result["token_argmax_agreement"] = sum(r["argmax_agreement_count"] for r in records) / total
    result["normal_minus_simplicial_ce"] = (
        result["variants"]["normal"]["cross_entropy"]
        - result["variants"]["simplicial"]["cross_entropy"]
    )
    return result


def math_summary(records, *, seed=20260915, replicates=10000):
    result = {"overall": aggregate_math(records), "by_effort": {}, "by_tools": {}}
    for mode in sorted({r["mode"] for r in records}):
        result["by_effort"][mode] = aggregate_math([r for r in records if r["mode"] == mode])
    for tools in (False, True):
        subset = [r for r in records if bool(r["has_tools"]) == tools]
        if subset:
            result["by_tools"]["tools" if tools else "no_tools"] = aggregate_math(subset)
    groups = {}
    for r in records:
        totals = groups.setdefault(r["problem_sha256"], np.zeros(4, dtype=np.float64))
        totals += np.array(
            [
                r["targets"],
                r["normal"]["nll"] - r["simplicial"]["nll"],
                r["normal"]["top1"] - r["simplicial"]["top1"],
                r["normal"]["top5"] - r["simplicial"]["top5"],
            ]
        )
    values = np.stack(list(groups.values()))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        sample = values[rng.integers(0, len(values), len(values))].sum(0)
        samples.append(sample[1:] / sample[0])
    bounds = np.quantile(np.array(samples), [0.025, 0.975], axis=0)
    result["paired_uncertainty"] = {
        "method": "paired percentile bootstrap clustered by exact problem hash",
        "problem_clusters": len(groups),
        "replicates": replicates,
        "seed": seed,
        "delta_definition": "normal minus simplicial",
        "cross_entropy_delta_95pct_interval": bounds[:, 0].tolist(),
        "top1_accuracy_delta_95pct_interval_pp": (bounds[:, 1] * 100).tolist(),
        "top5_accuracy_delta_95pct_interval_pp": (bounds[:, 2] * 100).tolist(),
    }
    return result

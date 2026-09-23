"""Render the completed paired full-checkpoint evaluation with scoped claims."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def pct(value):
    return f"{100 * value:.2f}%"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    results = json.loads((run / "COMPLETE.json").read_text())
    if not results["passed"] or not results["complete"]:
        raise ValueError("evaluation is not complete")
    provenance = json.loads((run / "RUN.json").read_text())
    math = results["heldout_math"]
    overall = math["overall"]
    s = overall["variants"]["simplicial"]
    n = overall["variants"]["normal"]
    interval = math["paired_uncertainty"]["cross_entropy_delta_95pct_interval"]
    delta = overall["normal_minus_simplicial_ce"]
    if interval[1] < 0:
        conclusion = (
            "The paired held-out token-loss comparison favors normal attention on this pilot."
        )
    elif interval[0] > 0:
        conclusion = (
            "The paired held-out token-loss comparison favors2-simplicial attention on this pilot."
        )
    else:
        conclusion = (
            "The paired held-out token-loss comparison does not resolve a winner on this pilot."
        )
    lines = [
        conclusion.replace("favors2", "favors 2"),
        "",
        f"Both evaluated models are full-weight checkpoints at {results['checkpoint_tokens']:,} supervised training tokens. The evaluation used the released16 B300 GPUs, with both model states on GPU and no CPU weight offload.",
        "",
        f"Held-out math: {overall['supervised_targets']:,} assistant targets across {overall['windows']} windows and {math['paired_uncertainty']['problem_clusters']} unique problem hashes.",
        "",
        "| Metric | 2-simplicial | Normal attention |",
        "|---|---:|---:|",
        f"| Cross-entropy, lower is better | {s['cross_entropy']:.6f} | {n['cross_entropy']:.6f} |",
        f"| Perplexity, lower is better | {s['perplexity']:.6f} | {n['perplexity']:.6f} |",
        f"| Top1 next-token accuracy | {pct(s['top1_token_accuracy'])} | {pct(n['top1_token_accuracy'])} |",
        f"| Top5 next-token accuracy | {pct(s['top5_token_accuracy'])} | {pct(n['top5_token_accuracy'])} |",
        f"| Mean predictive entropy, nats | {s['mean_predictive_entropy']:.6f} | {n['mean_predictive_entropy']:.6f} |",
        "",
        f"CE difference (normal minus2-simplicial): {delta:+.6f}, with paired problem-cluster bootstrap95% interval [{interval[0]:+.6f}, {interval[1]:+.6f}]. "
        f"Argmax agreement is {pct(overall['token_argmax_agreement'])}. KL(S || N) is {overall['simplicial_to_normal_kl']:.6f} nats and KL(N || S) is {overall['normal_to_simplicial_kl']:.6f} nats.",
        "",
        "| Effort stratum | Targets | 2-simplicial CE | Normal CE |",
        "|---|---:|---:|---:|",
    ]
    for mode, row in math["by_effort"].items():
        lines.append(
            f"| {mode} | {row['supervised_targets']:,} | {row['variants']['simplicial']['cross_entropy']:.6f} | {row['variants']['normal']['cross_entropy']:.6f} |"
        )
    lines += [
        "",
        "Multiple-choice samples use shared zero-shot continuation-likelihood prompts and identical examples. These are deterministic subsets, not full public benchmark results.",
        "",
        "| Task / metric | Samples | 2-simplicial | Normal | Normal − S, pp | Paired95% interval, pp |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for task, task_result in results["multiple_choice"].items():
        for metric, label in [
            ("accuracy", "raw accuracy"),
            ("accuracy_norm", "character-normalized accuracy"),
        ]:
            r = task_result[metric]
            lo, hi = r["delta_95pct_interval_pp"]
            lines.append(
                f"| {task} / {label} | {r['n']} | {pct(r['simplicial_accuracy'])} | {pct(r['normal_accuracy'])} | {r['delta_percentage_points']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |"
            )
    lines += [
        "",
        "Multiple-choice intervals use the existing conservative paired Wilson construction; exact McNemar p-values and disagreements are in MULTIPLE_CHOICE.json. These comparisons are exploratory and are not adjusted for multiple testing.",
        "",
        "All full-checkpoint weight payloads were checksum-verified while loading. Both production inference paths passed the recorded next-training-batch CE check; qualification errors are listed below.",
        "",
        "| Model | Training-reference CE | Evaluation CE | Absolute error |",
        "|---|---:|---:|---:|",
    ]
    for variant, r in results["qualification"].items():
        lines.append(
            f"| {variant} | {r['training_reference_ce']:.8f} | {r['eval_ce']:.8f} | {r['absolute_ce_error']:.3g} |"
        )
    lines += [
        "",
        "The held-out math problem hashes are disjoint from the planned fine-tuning problem hashes. No exact benchmark-question overlap was found by the recorded normalized-hash audit. This does not audit paraphrases or the base model’s pretraining corpus.",
        "",
        "Math metrics are teacher-forced next-token metrics, not generative problem-solving accuracy.",
        "",
        f"Evaluation source: {provenance['project_commit']}. Report generated at {datetime.now(timezone.utc).isoformat()}.",
        "",
        "- [Full results](COMPLETE.json)",
        "- [Per-window paired math metrics](math-pairs.jsonl)",
        "- [Per-question multiple-choice scores](multiple-choice-pairs.jsonl)",
        "- [Run and data provenance](RUN.json)",
    ]
    text = "\n".join(lines) + "\n"
    for old, new in [
        ("step552", "step 552"),
        ("released16", "released 16"),
        ("Top1", "Top 1"),
        ("Top5", "Top 5"),
        ("minus2", "minus 2"),
        ("bootstrap95", "bootstrap 95"),
        ("Paired95", "Paired 95"),
        ("step711 /118", "step 711 / 118"),
        ("separate2", "separate 2"),
    ]:
        text = text.replace(old, new)
    (run / "REPORT.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

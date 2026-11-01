"""Held-out RL-split pass@1 of Limite exports sampled through vLLM.

``limite_teacher_traces generate --record-logprobs`` samples each export. Inserted
preludes are served by archlab's own vLLM integration, so ``admit`` replays a fixed
subset of those samples through the native model the export was trained as and
applies the serving contract's absolute bounds to the per-token log-ratio. ``summarize``
scores only admitted exports and compares them on the same problems with a paired,
problem-clustered bootstrap of per-problem pass@1.

usage: python -m archlab.evaluation.limite_heldout admit|summarize ...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.evaluation.limite_math import paired_cluster_bootstrap, pass_at_k, summarize_aime
from archlab.preprocessing.limite_teacher_traces import SERVING, completed

# Absolute bounds of the Limite serving-versus-learner contract (docs/TRAINING.md).
MAXIMUM_ABSOLUTE_LOGRATIO = 0.5
MAXIMUM_K3_KL = 0.005
ADMISSION_PROBLEMS = 8
SPLIT = "heldout"


def split_order(split: Path) -> list[str]:
    return [json.loads(line)["uuid"] for line in split.read_text().splitlines()]


def logratio_stats(ratios: list[float]) -> dict:
    """``ratios`` are native minus serving log-probabilities of the sampled tokens."""
    if not ratios or not all(map(math.isfinite, ratios)):
        raise ValueError("log-ratio statistics need finite values")
    absolute = sorted(abs(value) for value in ratios)
    return dict(tokens=len(ratios), mean=sum(ratios) / len(ratios),
                mean_absolute=sum(absolute) / len(absolute),
                p95_absolute=absolute[int(.95 * (len(absolute) - 1))],
                maximum_absolute=absolute[-1],
                k3_kl_estimate=sum(math.exp(value) - value - 1 for value in ratios) / len(ratios))


def admission_records(records: list[dict], order: list[str], problems: int) -> list[dict]:
    """The first draw of the first ``problems`` split problems: fixed before any result is seen."""
    first = {record["uuid"]: record for record in records if record["sample"] == 0}
    selected = [first[uuid] for uuid in order[:problems] if uuid in first]
    if len(selected) != problems:
        raise ValueError("admission problems lack their first draw")
    return selected


def native_logprobs(model, prompt_ids: list[int], response_ids: list[int], *, chunk: int = 4096) -> list[float]:
    import torch

    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids + response_ids], device=device)
    targets = torch.tensor(response_ids, device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[0, len(prompt_ids) - 1:-1]
        values = [logits[start:start + chunk].float().log_softmax(-1)
                  .gather(-1, targets[start:start + chunk, None])[:, 0]
                  for start in range(0, len(response_ids), chunk)]
    return torch.cat(values).double().tolist()


def admit(args) -> dict:
    from archlab.training.limite import LimiteForCausalLM

    receipts, records = completed(args.traces, SPLIT)
    export_receipt = sha256_file(args.export / "EXPORT.json")
    for path in receipts:
        receipt = json.loads(path.read_text())
        if (receipt["export_receipt_sha256"] != export_receipt or receipt.get("response_logprobs") != "raw"
                or receipt.get("serving") != SERVING):
            raise ValueError("traces were not sampled from this export under the serving contract "
                             "with raw log-probabilities")
    model = LimiteForCausalLM.from_pretrained(args.export, device_map=args.device)
    rows, ratios = [], []
    for record in admission_records(records, split_order(args.split), args.problems):
        native = native_logprobs(model, record["prompt_ids"], record["response_ids"])
        serving = record["response_logprobs"]
        if len(native) != len(serving):
            raise ValueError("recorded serving log-probabilities do not cover the response")
        values = [a - b for a, b in zip(native, serving, strict=True)]
        ratios.extend(values)
        rows.append(dict(uuid=record["uuid"], sample=record["sample"], finish_reason=record["finish_reason"],
                         **logratio_stats(values)))
    aggregate = logratio_stats(ratios)
    result = dict(
        format="archlab-limite-serving-admission-v1", export=str(args.export),
        export_receipt_sha256=export_receipt, split=str(args.split), split_sha256=sha256_file(args.split),
        traces={path.name: sha256_file(path) for path in receipts},
        native="LimiteForCausalLM teacher-forced replay, use_cache=False, FP32 log-softmax",
        serving="vLLM raw_logprobs of the sampled tokens",
        bounds=dict(maximum_absolute=MAXIMUM_ABSOLUTE_LOGRATIO, k3_kl_estimate=MAXIMUM_K3_KL),
        records=rows, aggregate=aggregate,
        passed=(aggregate["maximum_absolute"] <= MAXIMUM_ABSOLUTE_LOGRATIO
                and aggregate["k3_kl_estimate"] <= MAXIMUM_K3_KL),
    )
    atomic_write_json(args.output, result)
    return result


def scored(records: list[dict]) -> list[dict]:
    """The benchmark summary schema; serving stops only at EOS or the token budget."""
    rows = []
    for record in records:
        if record["finish_reason"] not in ("stop", "length"):
            raise ValueError("unexpected serving finish reason: " + str(record["finish_reason"]))
        rows.append(dict(problem_id=record["uuid"], sample_index=record["sample"], correct=record["correct"],
                         generated_tokens=len(record["response_ids"]),
                         finish_reason="eos" if record["finish_reason"] == "stop" else "length"))
    return rows


def summarize(args) -> dict:
    order = split_order(args.split)
    models, scores = {}, {}
    for name, traces in args.model:
        admission = json.loads((traces / "ADMISSION.json").read_text())
        receipts, records = completed(traces, SPLIT)
        if (admission["traces"] != {path.name: sha256_file(path) for path in receipts}
                or admission["split_sha256"] != sha256_file(args.split)):
            raise ValueError(f"{name}: admission covers other traces or another split")
        entry = models[name] = dict(traces=str(traces), export=admission["export"],
                                    admission=dict(passed=admission["passed"], **admission["aggregate"]))
        if not admission["passed"]:
            continue
        rows = scored(records)
        entry["heldout"] = summarize_aime(rows, order, args.samples)
        if not entry["heldout"]["complete"]:
            raise ValueError(f"{name}: incomplete held-out samples")
        counts = {uuid: 0 for uuid in order}
        for row in rows:
            counts[row["problem_id"]] += row["correct"]
        scores[name] = [pass_at_k(counts[uuid], args.samples, 1) for uuid in order]
    comparisons = {
        f"{name}-minus-{reference}": paired_cluster_bootstrap(scores[reference], scores[name])
        for reference in args.reference if reference in scores
        for name in scores if name != reference
    }
    result = dict(format="archlab-limite-heldout-pass-at-1-v1", split=str(args.split),
                  split_sha256=sha256_file(args.split), samples_per_problem=args.samples,
                  scored_only_if_admitted=True, models=models, paired_pass_at_1=comparisons)
    atomic_write_json(args.output, result)
    return result


def _model(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=TRACES_DIRECTORY")
    return name, Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("admit")
    check.add_argument("--export", type=Path, required=True)
    check.add_argument("--traces", type=Path, required=True)
    check.add_argument("--split", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--problems", type=int, default=ADMISSION_PROBLEMS)
    check.add_argument("--device", default="cuda")
    report = commands.add_parser("summarize")
    report.add_argument("--split", type=Path, required=True)
    report.add_argument("--samples", type=int, required=True)
    report.add_argument("--model", type=_model, action="append", required=True)
    report.add_argument("--reference", action="append", default=[])
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = admit(args) if args.command == "admit" else summarize(args)
    print(json.dumps({k: v for k, v in result.items() if k not in ("records", "traces")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

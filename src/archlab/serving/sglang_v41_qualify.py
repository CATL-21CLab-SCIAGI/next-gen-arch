"""Compare SGLang teacher-forced probabilities with stored resident-model evals."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from pathlib import Path


def prepare_cases(pilot, reference_dir, *, target_budget=65536):
    reference_dir = Path(reference_dir)
    run = json.loads((reference_dir / "RUN.json").read_text())
    records = sorted((json.loads(line) for line in (reference_dir / "math-pairs.jsonl").read_text().splitlines()),
                     key=lambda row: row["pilot_index"])
    cases, count = [], 0
    for row in records:
        index = row["pilot_index"]
        inputs, labels, targets = pilot.batch(index, device="cpu")
        if targets != row["targets"] or pilot.windows[index]["problem_sha256"] != row["problem_sha256"]:
            raise ValueError("sealed pilot and resident-model reference differ")
        positions = (labels[0] != -100).nonzero().flatten().tolist()
        last = positions[-1]
        tokens = inputs[0, :last + 1].tolist() + [int(labels[0, last])]
        if any(tokens[position + 1] != int(labels[0, position]) for position in positions):
            raise ValueError("teacher-forced labels are not shifted exactly once")
        cases.append(dict(pilot_index=index, input_ids=tokens, target_positions=[p + 1 for p in positions],
                          targets=targets, problem_sha256=row["problem_sha256"],
                          reference={v: row[v] for v in ("normal", "simplicial")}))
        count += targets
        if count >= target_budget:
            break
    if count < target_budget:
        raise ValueError("reference has too few held-out targets")
    return dict(format="archlab-sglang-heldout-qualification-v1", reference_cursors=run["cursors"],
                reference_dir=str(reference_dir), pilot_manifest=pilot.manifest,
                targets=count, cases=cases)


def score_case(case, response):
    tokens = case["input_ids"]
    metadata = response["meta_info"]
    probabilities = metadata["input_token_logprobs"]
    top = metadata["input_top_logprobs"]
    # Engines may omit the first unconditioned token rather than emit None.
    if len(probabilities) == len(tokens) - 1:
        probabilities = [[None, tokens[0]]] + probabilities
    if len(top) == len(tokens) - 1:
        top = [None] + top
    if len(probabilities) != len(tokens) or len(top) != len(tokens):
        raise ValueError("engine did not return probabilities for the entire prompt")
    if any(int(entry[1]) != token for entry, token in zip(probabilities, tokens, strict=True)):
        raise ValueError("engine logprob token alignment differs from the sealed prompt")
    terms, top1, top5 = [], 0, 0
    for position in case["target_positions"]:
        if not 1 <= position < len(tokens):
            raise ValueError("invalid next-token target position")
        value = probabilities[position][0]
        if value is None or not math.isfinite(value) or value > 1e-6:
            raise ValueError("nonfinite or invalid conditional log probability")
        terms.append(-value)
        candidates = top[position]
        if not candidates or len(candidates) < 5:
            raise ValueError("engine did not return five candidate tokens")
        if any(not math.isfinite(entry[0]) for entry in candidates):
            raise ValueError("nonfinite candidate probability")
        target = tokens[position]
        top1 += int(max(candidates, key=lambda entry: entry[0])[1] == target)
        top5 += int(any(entry[1] == target for entry in candidates))
    if len(terms) != case["targets"] or len(set(case["target_positions"])) != len(terms):
        raise ValueError("validation targets were lost or duplicated")
    return dict(targets=len(terms), nll=math.fsum(terms), top1=top1, top5=top5)


def qualify(origin, token, variant, cases, *, timeout=1800):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    for case in cases:
        payload = dict(input_ids=case["input_ids"], return_logprob=True, logprob_start_len=0,
                       top_logprobs_num=5, sampling_params=dict(max_new_tokens=1, temperature=0))
        request = urllib.request.Request(origin.rstrip("/") + "/generate",
                                         data=json.dumps(payload).encode(), headers={
                                             "Authorization": "Bearer " + token,
                                             "Content-Type": "application/json"})
        began = time.monotonic()
        with opener.open(request, timeout=timeout) as response:
            observed = score_case(case, json.load(response))
        expected = case["reference"][variant]
        error = observed["nll"] / observed["targets"] - expected["nll"] / case["targets"]
        row = dict(pilot_index=case["pilot_index"], observed=observed, expected=expected,
                   ce_difference=error, seconds=time.monotonic() - began)
        rows.append(row)
        print(json.dumps(dict(event="sglang_heldout_case", variant=variant, **row)), flush=True)
    count = sum(row["observed"]["targets"] for row in rows)
    if not count:
        raise ValueError("empty qualification")
    actual = math.fsum(row["observed"]["nll"] for row in rows) / count
    expected = math.fsum(row["expected"]["nll"] for row in rows) / count
    top1_difference = sum(row["observed"]["top1"] - row["expected"]["top1"] for row in rows) / count
    top5_difference = sum(row["observed"]["top5"] - row["expected"]["top5"] for row in rows) / count
    max_error = max(abs(row["ce_difference"]) for row in rows)
    passed = (abs(actual - expected) <= 1e-4 and max_error <= 1e-3
              and abs(top1_difference) <= 1e-3 and abs(top5_difference) <= 1e-3)
    return dict(passed=passed, variant=variant, targets=count, engine_ce=actual,
                reference_ce=expected, absolute_ce_error=abs(actual - expected),
                max_window_ce_error=max_error, top1_accuracy_difference=top1_difference,
                top5_accuracy_difference=top5_difference,
                thresholds=dict(absolute_ce=1e-4, max_window_ce=1e-3, accuracy_difference=1e-3),
                full_logit_vectors_compared=False, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--variant", choices=("normal", "simplicial"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = json.loads(args.cases.read_text())
    config = json.loads((args.model / "config.json").read_text())["archlab"]
    if (config["variant"] != args.variant
            or config["checkpoint_cursor"] != packet["reference_cursors"][args.variant]):
        raise ValueError("qualification reference and candidate checkpoint differ")
    token = args.token_file.read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(args.origin.rstrip("/") + "/get_model_info",
                                     headers={"Authorization": "Bearer " + token})
    with opener.open(request, timeout=30) as response:
        info = json.load(response)
    if Path(info["model_path"]).resolve() != args.model.resolve():
        raise ValueError("candidate backend is serving a different model path")
    report = qualify(args.origin, token, args.variant, packet["cases"])
    report["reference_cursors"] = packet["reference_cursors"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("SGLang has not passed numerical qualification; do not switch endpoints")


if __name__ == "__main__":
    main()

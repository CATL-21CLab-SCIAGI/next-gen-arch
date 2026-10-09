"""Sealed, oracle-labelled counterfactual reasoning probes and paired reports.

These are text adaptations inspired by the literature, not a paper replication
or an OOD generalization claim. The model receives only public prompts/tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

from archlab.evaluation.capability import paired_statistics
from archlab.evaluation.pair_search import TEMPLATE_PATH, make_twins, witnesses


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def format_pairs(values):
    return " ".join(f"{x},{y}" for x, y in values)


def format_pair(value):
    return f"({value[0]},{value[1]})"


def generate(config):
    templates = yaml.safe_load(
        (Path(__file__).parents[1] / "prompts/deepseek_v41_reasoning_v1.yaml").read_text()
    )["templates"]
    pair_template = yaml.safe_load(TEMPLATE_PATH.read_text())["templates"]["modular_pair"]
    rows = []
    for split in ("calibration", "test"):
        rng = random.Random(config[f"{split}_seed"])
        for task in config["tasks"]:
            for size in config["levels"]:
                for number in range(config[f"{split}_pairs_per_level"]):
                    pair_id = f"{split}/{task}/{size}/{number}"
                    twins = []
                    if task == "modular_pair_search":
                        a, negative, positive, target = make_twins(rng, earlier_count=size)
                        fmt = format_pairs
                        for b in (negative, positive):
                            text = pair_template.format(
                                modulus=97,
                                long_entries=fmt(b),
                                recent_entries=fmt(a),
                                target=fmt([target]),
                            )
                            twins.append(
                                dict(
                                    text=text,
                                    expected="YES" if witnesses(a, b, target, 97) else "NO",
                                    facts="B: " + fmt(b),
                                    query="A: " + fmt(a) + "\nT: " + fmt([target]),
                                    oracle=dict(a=a, b=b, target=target, modulus=97),
                                )
                            )
                    elif task in ("composition", "join", "retrieval"):
                        depth = (
                            (2 if size == 8 else 3)
                            if task == "composition"
                            else (2 if task == "join" else 1)
                        )
                        functions = [rng.sample(range(size), size) for _ in range(depth)]
                        x = rng.randrange(size)
                        result = x
                        for f in functions:
                            result = f[result]
                        wrong = rng.choice([v for v in range(size) if v != result])
                        tables = "\n".join(
                            f"{chr(102 + k)}: " + " ".join(f"{i}->{v}" for i, v in enumerate(f))
                            for k, f in enumerate(functions)
                        )
                        if task == "join":
                            tables = "\n".join(
                                " ".join(f"{name}({i},{v})" for i, v in enumerate(f))
                                for name, f in zip(("R", "S"), functions, strict=True)
                            )
                        for y in (result, wrong):
                            text = templates[task].format(
                                tables=tables,
                                facts=tables,
                                order=",".join(chr(102 + k) for k in range(depth)),
                                x=x,
                                y=y,
                            )
                            twins.append(
                                dict(
                                    text=text,
                                    expected="YES" if y == result else "NO",
                                    facts=tables,
                                    query=f"a={x}; c={y}" if task == "join" else f"x={x}; y={y}",
                                    oracle=dict(functions=functions, x=x, y=y, result=result),
                                )
                            )
                    elif task == "arithmetic":
                        a, b = [[rng.randrange(97) for _ in range(2)] for _ in range(2)]
                        target = [(x + y) % 97 for x, y in zip(a, b, strict=True)]
                        negative = [target[0], (target[1] + rng.randrange(1, 97)) % 97]
                        for t in (target, negative):
                            pair = format_pair
                            facts = f"a={pair(a)}; b={pair(b)}; T={pair(t)}"
                            text = templates[task].format(
                                modulus=97, a=pair(a), b=pair(b), target=pair(t)
                            )
                            twins.append(
                                dict(
                                    text=text,
                                    expected="YES" if t == target else "NO",
                                    facts=facts,
                                    query="T=" + pair(t),
                                    oracle=dict(a=a, b=b, target=t, modulus=97),
                                )
                            )
                    else:
                        raise ValueError(f"unknown task {task}")
                    rng.shuffle(twins)
                    for index, row in enumerate(twins):
                        rows.append(
                            dict(
                                id=f"{pair_id}/{index}",
                                pair_id=pair_id,
                                task=task,
                                level=size,
                                split=split,
                                **row,
                            )
                        )
    if len({r["text"] for r in rows}) != len(rows):
        raise ValueError("duplicate prompts across calibration/test")
    return rows


def seal(config_path, output, assets):
    from transformers import AutoTokenizer

    from archlab.preprocessing.deepseek_v41 import DeepSeekV41Renderer

    config = yaml.safe_load(config_path.read_text())
    rows = generate(config)
    tokenizer = AutoTokenizer.from_pretrained(
        assets, local_files_only=True, trust_remote_code=False
    )
    renderer = DeepSeekV41Renderer(assets)
    eos = tokenizer.encode(renderer.encoder.eos_token, add_special_tokens=False)
    if len(eos) != 1:
        raise ValueError("expected single native EOS")
    public, labels = [], []
    for row in rows:
        prompt = renderer.encoder.encode_messages(
            [dict(role="user", content=row["text"])], thinking_mode="chat"
        )
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        if len(ids) + config["max_new_tokens"] > config["max_context"]:
            raise ValueError(f"context overflow: {row['id']}")
        distances = {}
        for name in ("facts", "query"):
            start = prompt.index(row[name])
            stop = start + len(row[name])
            positions = [i for i, (a, b) in enumerate(offsets) if b > start and a < stop]
            if not positions:
                raise ValueError("native token offsets missing")
            distances[name] = dict(
                first=len(ids) - 1 - min(positions), last=len(ids) - 1 - max(positions)
            )
        audit = dict(
            prompt_tokens=len(ids),
            distances_from_first_prediction=distances,
            all_facts_in_long_window=distances["facts"]["first"] < config["long_window"],
            query_in_short_window=distances["query"]["first"] < config["short_window"],
        )
        metadata = {k: row[k] for k in ("id", "pair_id", "task", "level", "split")}
        public.append(dict(**metadata, text=row["text"], input_ids=ids, token_audit=audit))
        labels.append(dict(**metadata, expected=row["expected"], oracle=row["oracle"]))
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("cases.json", public), ("labels.json", labels)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    manifest = dict(
        format="archlab-matched-reasoning-v1",
        config=config,
        config_sha256=sha(config_path),
        cases_sha256=sha(output / "cases.json"),
        labels_sha256=sha(output / "labels.json"),
        eos_id=eos[0],
        rows=len(public),
        tokenizer_sha256=sha(assets / "tokenizer.json"),
        encoder_sha256=sha(assets / "encoding/encoding.py"),
        caveat="Frozen zero-shot text adaptation; window audit is not a proof of a one-layer computation.",
    )
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(bundle, normal, simplicial):
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    for key in ("cases", "labels"):
        if sha(bundle / f"{key}.json") != manifest[f"{key}_sha256"]:
            raise ValueError("sealed suite changed")
    labels = json.loads((bundle / "labels.json").read_text())
    cases = {r["id"]: r for r in json.loads((bundle / "cases.json").read_text())}
    predictions, markers = [], []
    for variant, directory in (("normal", normal), ("simplicial", simplicial)):
        marker = json.loads((directory / "COMPLETE.json").read_text())
        if marker["variant"] != variant or marker.get("tiny"):
            raise ValueError("wrong variant or qualification-only predictions")
        markers.append(marker)
        if marker["cases_sha256"] != manifest["cases_sha256"]:
            raise ValueError("wrong suite")
        rows = [
            json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()
        ]
        indexed = {r["id"]: r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(cases):
            raise ValueError("missing/duplicate/extra predictions")
        predictions.append(indexed)
    if any(markers[0][key] != markers[1][key] for key in ("matched_training_tokens", "kind")):
        raise ValueError("checkpoints are not matched at the same phase/token count")
    groups = defaultdict(list)
    for label in labels:
        groups[(label["split"], label["task"], label["level"])].append(label)
    result = {}
    for (split, task, level), rows in groups.items():
        scores, pairs = [[], []], defaultdict(lambda: [[], []])
        invalid = [0, 0]
        for row in rows:
            for arm, pred in enumerate(predictions):
                answer = pred[row["id"]]["text"].strip()
                value = int(answer == row["expected"])
                invalid[arm] += int(answer not in ("YES", "NO"))
                scores[arm].append(value)
                pairs[row["pair_id"]][arm].append(value)
        both = [[int(all(pair[arm])) for pair in pairs.values()] for arm in (0, 1)]
        stats = paired_statistics(*both)
        stats["normal_both_twins_correct"] = stats.pop("pretrained_accuracy")
        stats["simplicial_both_twins_correct"] = stats.pop("adapted_accuracy")
        rng = random.Random(20260929)
        values = [(sum(v[1]) - sum(v[0])) / 2 for v in pairs.values()]
        deltas = sorted(
            100 * sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000)
        )
        result[f"{split}/{task}/{level}"] = dict(
            n=len(rows),
            normal_accuracy=sum(scores[0]) / len(rows),
            simplicial_accuracy=sum(scores[1]) / len(rows),
            invalid_outputs=invalid,
            delta_pp=100 * (sum(scores[1]) - sum(scores[0])) / len(rows),
            twin_cluster_bootstrap_95pct_pp=[deltas[250], deltas[9749]],
            both_twins_paired_statistics=stats,
            window_eligible=sum(
                cases[r["id"]]["token_audit"]["all_facts_in_long_window"]
                and cases[r["id"]]["token_audit"]["query_in_short_window"]
                for r in rows
            ),
        )
    controls = [
        v
        for k, v in result.items()
        if k.startswith("calibration/") and any(t in k for t in ("retrieval", "arithmetic"))
    ]
    control_ok = all(
        min(v["normal_accuracy"], v["simplicial_accuracy"])
        >= manifest["config"]["control_min_accuracy"]
        for v in controls
    )
    return dict(
        cases_sha256=manifest["cases_sha256"],
        groups=result,
        calibration_controls_pass=control_ok,
        interpretation="Exploratory frozen zero-shot probes. Do not attribute a search deficit to attention if arithmetic/retrieval controls fail. Task comparisons are descriptive; no multiple-comparison-adjusted claim.",
        bootstrap_note="Resample whole twins. Percentile intervals can degenerate at zero discordance; conservative Wilson intervals for both-twins success are also reported.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--recipe", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.add_argument("--assets", type=Path, required=True)
    report = sub.add_parser("report")
    for name in ("bundle", "normal", "simplicial", "output"):
        report.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        print(json.dumps(seal(args.recipe, args.output, args.assets)))
    else:
        args.output.write_text(
            json.dumps(summarize(args.bundle, args.normal, args.simplicial), indent=2) + "\n"
        )


if __name__ == "__main__":
    main()

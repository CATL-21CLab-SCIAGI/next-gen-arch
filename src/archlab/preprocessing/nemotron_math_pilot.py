"""Prepare an exact supervised-token pilot from the completed native corpus.

This is a new experiment data-order contract, not a change to frozen readers.
Shuffle parts and then conversations deterministically; traverse each selected
conversation's windows in order. Each window resets context. Adjacent windows
overlap by one input/target boundary token, so assistant targets are not lost
or counted twice. No cross-document packing or invented text is permitted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def shifted_window_spans(spans, start, length, *, limit=None):
    """Half-open spans in next-token labels for an input window [start,start+L)."""
    result, remaining = [], limit
    previous = 0
    for lo, hi in spans:
        if type(lo) is not int or type(hi) is not int or lo < previous or hi <= lo:
            raise ValueError("assistant spans must be sorted, nonoverlapping positive ranges")
        previous = hi
        begin, end = max(lo, start + 1), min(hi, start + length + 1)
        if begin >= end:
            continue
        if remaining is not None:
            end = min(end, begin + remaining)
            remaining -= end - begin
        if end > begin:
            result.append([begin - start - 1, end - start - 1])
        if remaining == 0:
            break
    return result


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(source: Path, output: Path, *, target_tokens=1_000_000_000, context=16384,
            seed=1234, split="train"):
    if type(target_tokens) is not int or target_tokens < 1 or type(context) is not int or context < 2:
        raise ValueError("positive supervised budget and context >= 2 are required")
    if split not in ("train", "validation"):
        raise ValueError("unknown corpus split")
    source = source.resolve(strict=True)
    if source == output.resolve() or source in output.resolve().parents:
        raise ValueError("pilot output must be independent of the tokenized corpus")
    ready_bytes = (source / "DATA_READY.json").read_bytes()
    ready = json.loads(ready_bytes)
    if ready["completed_documents"] != ready["expected_documents"] or ready["completed_parts"] != len(ready["parts"]):
        raise ValueError("source corpus is incomplete")
    parts = list(ready["parts"])
    random.Random(seed).shuffle(parts)
    output.mkdir(parents=True, exist_ok=False)
    targets, windows, documents, processed = 0, 0, 0, 0
    zero_target_windows, context_resets = 0, 0
    modes, variants, flags = Counter(), Counter(), Counter()
    verified = {}
    with (output / "windows.jsonl").open("x") as stream:
        for part in parts:
            if Path(part["id"]).name != part["id"]:
                raise ValueError("invalid source part path")
            root = source / "parts" / part["id"]
            declared = {f["name"]: f for f in part["files"]}
            for partition, info in sorted(part["partitions"].items()):
                if info["split"] != split:
                    continue
                prefix = info["prefix"]
                if Path(prefix).name != prefix:
                    raise ValueError("invalid indexed data prefix")
                for suffix in (".bin", ".idx", ".metadata.jsonl.gz"):
                    name = prefix + suffix
                    path = root / name
                    spec = declared[name]
                    if path.stat().st_size != spec["bytes"] or _sha(path) != spec["sha256"]:
                        raise ValueError(f"source payload changed: {path}")
                    verified[str(path.relative_to(source))] = spec
                with gzip.open(root / (prefix + ".metadata.jsonl.gz"), "rt") as metadata:
                    records = [json.loads(line) for line in metadata]
                if len(records) != info["documents"] or sum(r["tokens"] for r in records) != info["tokens"]:
                    raise ValueError("metadata count mismatch")
                if any(r["sequence"] != i or r["split"] != split for i, r in enumerate(records)):
                    raise ValueError("metadata sequence/split mismatch")
                rng = random.Random(f"{seed}:{part['id']}:{partition}")
                rng.shuffle(records)
                for record in records:
                    spans = record["assistant_token_spans"]
                    if any(hi > record["tokens"] for _, hi in spans):
                        raise ValueError("supervision outside the source conversation")
                    used = False
                    for start in range(0, record["tokens"] - 1, context):
                        length = min(context, record["tokens"] - 1 - start)
                        selected = shifted_window_spans(spans, start, length, limit=target_tokens - targets)
                        count = sum(hi - lo for lo, hi in selected)
                        if not count:
                            zero_target_windows += 1
                            continue
                        item = {"prefix": str((root / prefix).relative_to(source)),
                                "sequence": record["sequence"], "start": start, "length": length,
                                "assistant_label_spans": selected, "targets": count,
                                "problem_sha256": record["problem_sha256"], "mode": record["mode"],
                                "has_tools": record["has_tools"], "context_reset": start > 0}
                        stream.write(json.dumps(item, sort_keys=True) + "\n")
                        targets += count
                        windows += 1
                        processed += context
                        context_resets += int(start > 0)
                        modes[record["mode"]] += count
                        variants["tools" if record["has_tools"] else "no-tools"] += count
                        if not used:
                            documents += 1
                            for flag in record.get("message_repairs", []):
                                flags[flag["policy"]] += 1
                            used = True
                        if targets == target_tokens:
                            break
                    if targets == target_tokens:
                        break
                print(json.dumps({"event": "pilot_part", "part": part["id"], "supervised_tokens": targets,
                                  "windows": windows, "documents": documents}), flush=True)
                if targets == target_tokens:
                    break
            if targets == target_tokens:
                break
    if targets != target_tokens:
        raise ValueError("requested budget exceeds the available supervised targets")
    manifest = {"format": "archlab-native-math-window-pilot-v1", "source": str(source),
                "source_ready_sha256": hashlib.sha256(ready_bytes).hexdigest(),
                "source_contract_sha256": ready["contract_sha256"], "split": split, "seed": seed,
                "context": context, "supervised_tokens": targets, "windows": windows,
                "selected_conversations": documents, "processed_input_tokens_including_padding": processed,
                "context_reset_windows": context_resets, "zero_target_windows_skipped": zero_target_windows,
                "supervised_tokens_by_mode": dict(modes), "supervised_tokens_by_tools": dict(variants),
                "retained_source_flags": dict(flags), "verified_source_files": verified,
                "order": "seeded-part-shuffle-then-conversation-shuffle-then-ordered-windows",
                "context_policy": "independent-contiguous-windows-no-cross-document-packing",
                "source_selection": "retain-all-source-variants-and-trajectory-flags",
                "end_policy": "last-window-target-mask-clipped-to-exact-budget",
                "windows_sha256": _sha(output / "windows.jsonl")}
    with (output / "PILOT_READY.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, target_tokens=args.target_tokens,
                       context=args.context, seed=args.seed, split=args.split)
    print(json.dumps({k: v for k, v in manifest.items() if k != "verified_source_files"}), flush=True)


if __name__ == "__main__":
    main()

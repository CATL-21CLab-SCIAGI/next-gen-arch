"""Prepare unseen-problem RL candidates without reading solution trajectories.

This excludes every problem in every prepared row group referenced by a sealed
train/evaluation pilot, not just the selected windows. Entire original Parquet
files are NOT unused: the manifest states this distinction explicitly. Only
problem/answer/identity columns are read from the remaining row groups.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import struct
import unicodedata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def problem_key(problem: str) -> str:
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("A nonempty problem is required")
    text = " ".join(unicodedata.normalize("NFC", problem).split())
    return hashlib.sha256(text.encode()).hexdigest()


def _json_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _verified_json(path: Path, expected: str | None = None):
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Changed provenance artifact: {path}")
    return json.loads(path.read_text()), {"path": str(path.resolve()), "sha256": actual}


def _part_groups(part_id: str) -> tuple[str, range]:
    match = re.fullmatch(r"(.+)-rg(\d+)-(\d+)", part_id)
    if not match or int(match[2]) >= int(match[3]):
        raise ValueError(f"Invalid part ID: {part_id}")
    return match[1] + ".parquet", range(int(match[2]), int(match[3]))


def parquet_identity(path: Path) -> dict:
    """Hash the footer, not the potentially enormous trajectory payload."""
    import pyarrow.parquet as pq

    stat = path.stat()
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        length, magic = struct.unpack("<I4s", stream.read(8))
        if magic != b"PAR1" or length > stat.st_size - 8:
            raise ValueError(f"Invalid Parquet footer: {path}")
        stream.seek(-8 - length, 2)
        footer_sha = hashlib.sha256(stream.read(length + 8)).hexdigest()
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()), "bytes": stat.st_size,
        "footer_sha256": footer_sha, "full_file_sha256": None,
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "row_group_rows": [parquet.metadata.row_group(i).num_rows
                           for i in range(parquet.metadata.num_row_groups)],
        "schema_sha256": hashlib.sha256(str(parquet.schema_arrow).encode()).hexdigest(),
    }


def audit_exposure(pilot_paths: list[Path], extra_part_paths: list[Path] = ()) -> dict:
    """Include both train and validation sidecars of every referenced part.

    Extra part markers can conservatively exclude early smoke/probe data. A
    missing or changed sidecar fails closed; no model-visible trace is read.
    """
    if not pilot_paths:
        raise ValueError("At least one sealed historical train/evaluation pilot is required")
    parts, artifacts, pilots, sources = {}, [], [], {}
    selected_window_hashes = set()
    for pilot_path in sorted(map(Path, pilot_paths)):
        pilot, artifact = _verified_json(pilot_path)
        artifacts.append(artifact)
        source = Path(pilot["source"])
        ready_path = source / "DATA_READY.json"
        ready, ready_artifact = _verified_json(ready_path, pilot["source_ready_sha256"])
        if ready["status"] != "complete" or ready["contract_sha256"] != pilot["source_contract_sha256"]:
            raise ValueError("Historical source is incomplete or has changed contract")
        artifacts.append(ready_artifact)
        source_manifest, source_artifact = _verified_json(source / "manifest.json")
        artifacts.append(source_artifact)
        for spec in source_manifest["sources"]:
            name = Path(spec["path"]).name
            if name in sources and (sources[name]["bytes"], sources[name]["rows"]) != (spec["bytes"], spec["rows"]):
                raise ValueError(f"Conflicting historical source identities: {name}")
            sources[name] = spec
        ready_parts = {part["id"]: part for part in ready["parts"]}
        touched = set()
        for relative, spec in pilot["verified_source_files"].items():
            rel = Path(relative)
            if len(rel.parts) != 3 or rel.parts[0] != "parts" or ".." in rel.parts:
                raise ValueError(f"Invalid historical source path: {relative}")
            if not relative.endswith(".metadata.jsonl.gz"):
                continue
            part_id = rel.parts[1]
            part = ready_parts[part_id]
            published = {entry["name"]: entry for entry in part["files"]}[rel.name]
            if published != spec:
                raise ValueError(f"Pilot metadata differs from source manifest: {relative}")
            parts[(str(source), part_id)] = (source / "parts" / part_id, part)
            touched.add(part_id)
        # Sealed windows identify all selected problem hashes, including a
        # partially consumed last conversation. Check the windows themselves.
        window_path = pilot_path.with_name("windows.jsonl")
        if sha256_file(window_path) != pilot["windows_sha256"]:
            raise ValueError(f"Changed historical windows: {window_path}")
        with window_path.open() as stream:
            for line in stream:
                window = json.loads(line)
                if Path(window["prefix"]).parts[1] not in touched:
                    raise ValueError("Historical window is absent from verified pilot parts")
                selected_window_hashes.add(window["problem_sha256"])
        artifacts.append({"path": str(window_path.resolve()), "sha256": pilot["windows_sha256"]})
        pilots.append({"path": str(pilot_path.resolve()), "split": pilot["split"],
                       "supervised_tokens": pilot["supervised_tokens"],
                       "selected_conversations": pilot["selected_conversations"],
                       "referenced_parts": sorted(touched)})
    for path in sorted(map(Path, extra_part_paths)):
        part, artifact = _verified_json(path)
        artifacts.append(artifact)
        parts[(str(path.parent.parent.parent), part["id"])] = (path.parent, part)
    hashes, uuids, groups, sidecar_rows = set(), set(), defaultdict(set), 0
    sidecars = []
    for (_, part_id), (directory, part) in sorted(parts.items()):
        source_name, group_range = _part_groups(part_id)
        groups[source_name].update(group_range)
        count = 0
        for spec in sorted(part["files"], key=lambda item: item["name"]):
            if not spec["name"].endswith(".metadata.jsonl.gz"):
                continue
            path = directory / spec["name"]
            if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
                raise ValueError(f"Changed historical sidecar: {path}")
            sidecars.append({"path": str(path.resolve()), "sha256": spec["sha256"]})
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    key = row["problem_sha256"]
                    if not re.fullmatch(r"[0-9a-f]{64}", key):
                        raise ValueError("Invalid historical problem hash")
                    hashes.add(key)
                    if row.get("uuid"):
                        uuids.add(str(row["uuid"]))
                    count += 1
        if count != part["documents"]:
            raise ValueError(f"Missing historical sidecar rows: {directory}")
        sidecar_rows += count
    if not selected_window_hashes.issubset(hashes):
        raise ValueError("Historical selected problems are missing from sidecar exclusions")
    return {
        "problem_hashes": hashes, "uuids": uuids, "row_groups": dict(groups),
        "source_specs": sources, "pilots": pilots, "sidecars": sidecars,
        "artifacts": list({item["path"]: item for item in artifacts}.values()),
        "sidecar_rows": sidecar_rows,
    }


def select_records(rows, exposure, *, train_size: int, heldout_size: int, seed: int,
                   canonicalizer, max_problem_chars: int = 6000):
    """Deduplicate before splitting; discard conflicting golds and UUID aliases."""
    if min(train_size, heldout_size) < 1:
        raise ValueError("Train and heldout sizes must be positive")
    records, answers, uuid_keys = {}, defaultdict(set), defaultdict(set)
    counts = Counter()
    for row in rows:
        counts["scanned_rows"] += 1
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip() or len(problem) > max_problem_chars:
            counts["invalid_or_long_problem"] += 1
            continue
        key, uuid = problem_key(problem), str(row.get("uuid") or "")
        if key in exposure["problem_hashes"] or uuid and uuid in exposure["uuids"]:
            counts["previously_exposed_problem_or_uuid"] += 1
            continue
        if uuid:
            uuid_keys[uuid].add(key)
        expected = row.get("expected_answer")
        canonical = canonicalizer(expected) if isinstance(expected, str) else None
        # An unsupported duplicate answer poisons that problem as well: silently
        # retaining one of two disagreeing annotations would select its label.
        answers[key].add(canonical)
        if canonical is None:
            counts["unsupported_gold"] += 1
            continue
        record = {"id": key, "problem_sha256": key, "uuid": uuid,
                  "prompt": [{"role": "user", "content": problem}],
                  "expected_answer": expected, "canonical_answer": canonical,
                  "source": row["_source"]}
        if key not in records or (record["source"]["parquet"], record["source"]["row"]) < (
                records[key]["source"]["parquet"], records[key]["source"]["row"]):
            records[key] = record
    ambiguous = set().union(*(keys for keys in uuid_keys.values() if len(keys) > 1)) if uuid_keys else set()
    # Duplicate texts can also carry several UUIDs; discard if any of those
    # UUIDs appeared with a second text, not just the chosen representative.
    clean = [record for key, record in records.items()
             if len(answers[key]) == 1 and key not in ambiguous]
    counts["conflicting_or_aliased_problems"] = len(records) - len(clean)
    counts["eligible_unique_problems"] = len(clean)
    clean.sort(key=lambda row: hashlib.sha256(f"rl-nemotron:{seed}:{row['id']}".encode()).digest())
    if len(clean) < train_size + heldout_size:
        raise ValueError(f"Only {len(clean)} eligible unique problems; need {train_size + heldout_size}; {dict(counts)}")
    heldout, train = clean[:heldout_size], clean[heldout_size:heldout_size + train_size]
    return train, heldout, dict(counts)


def prepare(source_root: Path, output: Path, pilot_paths: list[Path], *,
            extra_part_paths: list[Path] = (), train_size: int = 8192,
            heldout_size: int = 512, max_rows: int = 200000, seed: int = 20260922):
    import pyarrow.parquet as pq
    from archlab.rl.rewards import canonical_math_answer

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite preparation: {output}")
    reward_sha = sha256_file(Path(__file__).with_name("rewards.py"))
    exposure = audit_exposure(pilot_paths, extra_part_paths)
    identities, tasks = [], []
    parquet_paths = sorted((source_root / "data").glob("*.parquet"))
    if not parquet_paths:
        raise ValueError("No Parquet source shards")
    for path in parquet_paths:
        identity = parquet_identity(path)
        if path.name in exposure["source_specs"]:
            old = exposure["source_specs"][path.name]
            if identity["bytes"] != old["bytes"] or identity["rows"] != old["rows"]:
                raise ValueError(f"Source identity differs from historical corpus: {path}")
        identity["excluded_row_groups"] = sorted(exposure["row_groups"].get(path.name, ()))
        identities.append(identity)
        offset = 0
        for group, length in enumerate(identity["row_group_rows"]):
            if group not in exposure["row_groups"].get(path.name, ()):
                tasks.append((path, group, offset, length))
            offset += length
    tasks.sort(key=lambda task: hashlib.sha256(f"rl-nemotron-groups:{seed}:{task[0].name}:{task[1]}".encode()).digest())
    if max_rows < train_size + heldout_size:
        raise ValueError("Scan budget cannot fill requested split")
    scan_evidence, scanned = [], 0

    def rows():
        nonlocal scanned
        for path, group, offset, length in tasks:
            if scanned + length > max_rows:
                break
            # No messages, solution, metadata traces, or tool outputs are read.
            selected = pq.ParquetFile(path).read_row_group(
                group, columns=["problem", "expected_answer", "uuid"], use_threads=False).to_pylist()
            if len(selected) != length:
                raise ValueError("Source row count changed")
            scan_evidence.append({"parquet": path.name, "row_group": group,
                                  "rows": length, "selected_columns_sha256": _json_digest(selected)})
            for i, row in enumerate(selected):
                row["_source"] = {"parquet": path.name, "row_group": group, "row": offset + i}
                yield row
            scanned += length

    train, heldout, counts = select_records(rows(), exposure, train_size=train_size,
                                           heldout_size=heldout_size, seed=seed,
                                           canonicalizer=canonical_math_answer)
    # Refuse source changes during preparation; footer hashes are explicitly
    # distinguished from full-file hashes and projected-column content hashes.
    for identity, path in zip(identities, parquet_paths, strict=True):
        current = parquet_identity(path)
        if any(current[key] != identity[key] for key in ("bytes", "footer_sha256", "rows")):
            raise ValueError(f"Source changed during preparation: {path}")
    if sha256_file(Path(__file__).with_name("rewards.py")) != reward_sha:
        raise ValueError("Reward implementation changed during preparation")
    output.mkdir(parents=True)
    files = {}
    for split, records in (("train", train), ("heldout", heldout)):
        path = output / f"{split}.jsonl"
        with path.open("x") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        files[split] = {"path": str(path.resolve()), "rows": len(records), "sha256": sha256_file(path)}
    exclusions_path = output / "excluded_problem_ids.json"
    exclusions_path.write_text(json.dumps({"problem_sha256": sorted(exposure["problem_hashes"]),
                                           "uuid": sorted(exposure["uuids"])}, sort_keys=True) + "\n")
    manifest = {
        "format": "archlab-nemotron-unseen-problem-rl-v1", "seed": seed,
        "status": "candidate_pending_source_shard_interpretation", "training_authorized": False,
        "scope": "Unused prepared row groups and unexposed normalized problems, NOT untouched original Parquet files",
        "exclusion_policy": "All problems/UUIDs in both partitions of every sealed train/eval pilot part, plus explicit smoke parts",
        "normalization": "NFC then whitespace collapse; exact SHA256, no paraphrase deduplication",
        "policy_input": "Only prompt; expected_answer and canonical_answer are reward-only fields",
        "solution_columns_read": [], "columns_read": ["problem", "expected_answer", "uuid"],
        "source_root": str(source_root.resolve()), "source_files": identities,
        "scan_budget_rows": max_rows, "scanned_groups": scan_evidence,
        "statistics": counts, "files": files, "prior_pilots": exposure["pilots"],
        "exclusion_sidecars": exposure["sidecars"], "provenance_artifacts": exposure["artifacts"],
        "excluded_problem_hashes": len(exposure["problem_hashes"]),
        "excluded_uuids": len(exposure["uuids"]), "excluded_sidecar_rows": exposure["sidecar_rows"],
        "exclusion_index": {"path": str(exclusions_path.resolve()), "sha256": sha256_file(exclusions_path)},
        "implementation_sha256": sha256_file(Path(__file__)),
        "reward_implementation_sha256": reward_sha,
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prior-pilot", type=Path, action="append", required=True)
    parser.add_argument("--extra-excluded-part", type=Path, action="append", default=[])
    parser.add_argument("--train-size", type=int, default=8192)
    parser.add_argument("--heldout-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    result = prepare(args.source_root, args.output, args.prior_pilot,
                     extra_part_paths=args.extra_excluded_part, train_size=args.train_size,
                     heldout_size=args.heldout_size, max_rows=args.max_rows, seed=args.seed)
    print(json.dumps({"statistics": result["statistics"], "files": result["files"],
                      "excluded_problem_hashes": result["excluded_problem_hashes"]}, indent=2))


if __name__ == "__main__":
    main()

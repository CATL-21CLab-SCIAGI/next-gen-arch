"""Restore NVIDIA's pinned RL questions and prepare native Limite prompt tokens.

No solution tokens are appended to policy inputs. All original training rows and
their reward metadata are preserved; this does not invent a validation split.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

from archlab.rl.nemotron_data import problem_key, sha256_file


def policy_messages(row):
    question, answer = row.get("question"), row.get("expected_answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("unrestored or empty question")
    if not isinstance(answer, str) or not answer.strip() or answer == "None":
        raise ValueError("missing reward answer")
    messages = (row.get("responses_create_params") or {}).get("input")
    if messages != [{"role": "user", "content": question}]:
        raise ValueError("unexpected policy messages; refuse answer leakage or silent rewriting")
    if row.get("_hf_question_placeholder"):
        raise ValueError("unresolved source placeholder")
    return messages


def tokenize_row(row, tokenizer):
    messages = policy_messages(row)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    native = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    if ids != native or not ids or len(ids) >= tokenizer.model_max_length:
        raise ValueError("native token mismatch, empty input, or prompt exhausts model context")
    if not text.endswith("<|im_start|>assistant\n"):
        raise ValueError("unexpected Limite generation prefix")
    return dict(
        uuid=row["uuid"],
        problem_sha256=problem_key(row["question"]),
        prompt=messages,
        prompt_text=text,
        input_ids=ids,
        attention_mask=[1] * len(ids),
        prompt_tokens=len(ids),
    )


def prepare(source_receipt, tokenizer_path, stage):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    sources = json.loads(source_receipt.read_text())
    by_name = {s["repo"]: s for s in sources}
    source = by_name["nvidia/Nemotron-RL-Math-v2"]
    for item in sources:
        for spec in item["files"]:
            path = Path(item["local_path"]) / spec["path"]
            if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
                raise ValueError(f"source changed: {path}")
    script = Path(source["local_path"]) / "fill_placeholders.py"
    module_spec = importlib.util.spec_from_file_location("nvidia_fill_placeholders", script)
    upstream = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(upstream)
    hf = {}
    for repo, split in upstream.HF_SOURCES:
        item = by_name[repo]
        paths = [
            Path(item["local_path"]) / f["path"]
            for f in item["files"]
            if f["path"].endswith(".parquet")
        ]
        if len(paths) != 1:
            raise ValueError("source row indexing requires exactly the pinned single math shard")
        # The upstream restore_row function consumes only prompt and reward_model.
        hf[(repo, split)] = pq.read_table(paths[0], columns=["prompt", "reward_model"]).to_pylist()
    model_receipt = json.loads((tokenizer_path / "DOWNLOAD_VERIFIED.json").read_text())
    if model_receipt["repo"] != "paradigma-inc/limite-1b-base":
        raise ValueError("requires the requested Limite base tokenizer")
    tokenizer_files = []
    for spec in model_receipt["files"]:
        if spec["path"] in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "merges.txt",
            "vocab.json",
        ):
            if sha256_file(tokenizer_path / spec["path"]) != spec["sha256"]:
                raise ValueError("verified tokenizer snapshot changed")
            tokenizer_files.append(spec)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )
    raw = [
        json.loads(line)
        for line in (Path(source["local_path"]) / "data/train.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(raw) != 7732 or len({r["uuid"] for r in raw}) != len(raw):
        raise ValueError("unexpected NVIDIA row count or duplicate UUID")
    stage.mkdir(parents=True, exist_ok=False)
    records, lengths, keys, restored_counts = [], [], [], Counter()
    with (
        (stage / "restored.jsonl").open("x") as restored,
        (stage / "prompts.jsonl").open("x") as prompts,
        (stage / "rewards.jsonl").open("x") as rewards,
        (stage / "reconstruction.jsonl").open("x") as audit,
    ):
        for index, original in enumerate(raw):
            placeholder = original.get("_hf_question_placeholder")
            if placeholder and placeholder.get("mode", "exact") not in ("exact", "canonical"):
                raise ValueError("unsupported reconstruction mode")
            row = upstream.restore_row(copy.deepcopy(original), hf)
            encoded = tokenize_row(row, tokenizer)
            reward = dict(
                uuid=row["uuid"],
                expected_answer=row["expected_answer"],
                verifier_type=row["verifier_type"],
                agent_ref=row["agent_ref"],
                responses_create_params=row["responses_create_params"],
            )
            restored.write(json.dumps(row, ensure_ascii=False) + "\n")
            prompts.write(json.dumps(encoded, ensure_ascii=False) + "\n")
            rewards.write(json.dumps(reward, ensure_ascii=False) + "\n")
            audit.write(
                json.dumps(
                    dict(
                        uuid=row["uuid"],
                        source_row=index,
                        placeholder=placeholder,
                        restored=bool(placeholder),
                        problem_sha256=encoded["problem_sha256"],
                    )
                )
                + "\n"
            )
            records.append(
                dict(
                    **encoded,
                    expected_answer=row["expected_answer"],
                    reward_model=dict(ground_truth=row["expected_answer"]),
                    verifier_type=row["verifier_type"],
                    agent_ref_json=json.dumps(row["agent_ref"]),
                    data_source=source["repo"],
                    source_row=index,
                )
            )
            lengths.append(len(encoded["input_ids"]))
            keys.append(encoded["problem_sha256"])
            if placeholder:
                restored_counts[placeholder["dataset"]] += 1
    table = pa.Table.from_pylist(records)
    for name, dtype in (
        ("input_ids", pa.list_(pa.int32())),
        ("attention_mask", pa.list_(pa.int8())),
    ):
        idx = table.schema.get_field_index(name)
        table = table.set_column(idx, name, table.column(name).cast(dtype))
    pq.write_table(table, stage / "train.parquet", compression="zstd", row_group_size=512)
    # Full readback, including both token paths and reward alignment.
    readback = pq.read_table(stage / "train.parquet").to_pylist()
    if readback != records:
        raise ValueError("Parquet round-trip changed records")
    for item in sources:
        dest = stage / "raw" / item["repo"].replace("/", "--")
        for spec in item["files"]:
            target = dest / spec["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(item["local_path"]) / spec["path"], target)
    tok_dest = stage / "tokenizer"
    tok_dest.mkdir()
    for spec in tokenizer_files:
        shutil.copyfile(tokenizer_path / spec["path"], tok_dest / spec["path"])
    manifest = dict(
        format="archlab-nemotron-rl-math-limite-v1",
        rows=len(records),
        split="original train; no automatic holdout",
        dataset_revision=source["revision"],
        model_id=model_receipt["repo"],
        model_revision=model_receipt["revision"],
        sources=sources,
        tokenizer_files=tokenizer_files,
        reconstruction_counts=dict(restored_counts),
        unresolved_placeholders=0,
        excluded_rows=0,
        duplicate_prompt_rows=len(keys) - len(set(keys)),
        total_prompt_tokens=sum(lengths),
        min_prompt_tokens=min(lengths),
        max_prompt_tokens=max(lengths),
        mean_prompt_tokens=sum(lengths) / len(lengths),
        p50_prompt_tokens=sorted(lengths)[len(lengths) // 2],
        p95_prompt_tokens=sorted(lengths)[int(len(lengths) * 0.95)],
        policy_input_fields=["input_ids", "attention_mask"],
        answer_tokens_in_policy_input=False,
        template="native checkpoint chat_template.jinja; add_generation_prompt=True",
        padding=False,
        truncation=False,
        appended_eos=False,
        model_context=tokenizer.model_max_length,
        verifier="Original math_with_judge metadata retained; no judge or RL training launched",
        implementation_sha256=sha256_file(Path(__file__)),
        upstream_script_sha256=sha256_file(script),
        files=[
            dict(path=str(p.relative_to(stage)), bytes=p.stat().st_size, sha256=sha256_file(p))
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        ],
    )
    (stage / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def publish(stage, output):
    manifest = json.loads((stage / "MANIFEST.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    for spec in manifest["files"] + [
        dict(
            path="MANIFEST.json",
            bytes=(stage / "MANIFEST.json").stat().st_size,
            sha256=sha256_file(stage / "MANIFEST.json"),
        )
    ]:
        source, target = stage / spec["path"], output / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if target.stat().st_size != spec["bytes"] or sha256_file(target) != spec["sha256"]:
            raise ValueError(f"published checksum mismatch: {target}")
    # Consumers must require this final marker, never infer completion from files.
    ready = dict(
        status="complete",
        rows=manifest["rows"],
        total_prompt_tokens=manifest["total_prompt_tokens"],
        manifest_sha256=sha256_file(output / "MANIFEST.json"),
        all_files_readback_verified=True,
    )
    (output / "DATA_READY.json").write_text(json.dumps(ready, indent=2) + "\n")
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sources", "tokenizer", "stage", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.sources, args.tokenizer, args.stage)
    print(json.dumps(publish(args.stage, args.output), indent=2))


if __name__ == "__main__":
    main()

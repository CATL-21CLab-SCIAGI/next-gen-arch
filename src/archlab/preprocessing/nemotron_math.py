"""Lossless, resumable Nemotron-Math-v2 -> Qwen chat indexed documents.

Uses the checkpoint's native Transformers chat template and the existing
NeMo AutoModel indexed-dataset writer. No GPU, model weights, package installs,
training imports, truncation, trajectory deduplication, or tool execution.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

EFFORT = {"high": "xhigh", "medium": "medium", "low": "low"}
TOKENIZER_FILES = (
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
    "vocab.json", "merges.txt", "config.json", "special_tokens_map.json",
    "added_tokens.json",
)
_STATE = {}


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def problem_key(problem):
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("Missing problem: cannot assign a leakage-safe split")
    normalized = " ".join(unicodedata.normalize("NFC", problem).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def choose_split(key, validation_basis_points=100):
    bucket = int(hashlib.sha256(("nemotron-math-v2:v1:" + key).encode()).hexdigest(), 16)
    return "validation" if bucket % 10000 < validation_basis_points else "train"


def normalize_row(row):
    """Decode structured arguments for the native template; never run tools."""
    messages = copy.deepcopy(row["messages"])
    if not messages or not any(m["role"] == "assistant" for m in messages):
        raise ValueError("Conversation must contain a completed assistant trajectory")
    has_tools = bool(row.get("tools"))
    for message in messages:
        if message.get("role") == "tool" or message.get("tool_calls"):
            has_tools = True
        for call in message.get("tool_calls") or []:
            arguments = call["function"].get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be a JSON object")
            call["function"]["arguments"] = arguments
    return messages, has_tools


def render_row(tokenizer, row, mode):
    messages, has_tools = normalize_row(row)
    kwargs = dict(
        tools=row.get("tools") or None,
        reasoning_effort=EFFORT[mode],
        preserve_thinking=True,
        enable_thinking=True,
        add_generation_prompt=False,
    )
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    if not text or not text.rstrip().endswith(tokenizer.eos_token):
        raise ValueError("Native template did not produce a completed conversation")
    return text, has_tools, messages, kwargs


def worker_init(tokenizer_root, tokenizer_format="qwen", reasoning_effort=75):
    import numpy as np
    from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import (
        IndexedDataset,
        IndexedDatasetBuilder,
    )
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False,
    )
    fast = Tokenizer.from_file(str(Path(tokenizer_root) / "tokenizer.json"))
    fast.no_truncation()
    fast.no_padding()
    _STATE.update(
        tokenizer=tokenizer, fast=fast, np=np, reader=IndexedDataset,
        builder=IndexedDatasetBuilder, parity_checked=set(), tokenizer_format=tokenizer_format,
    )
    if tokenizer_format == "deepseek-v41":
        from archlab.preprocessing.deepseek_v41 import DeepSeekV41Renderer

        _STATE["renderer"] = DeepSeekV41Renderer(tokenizer_root, reasoning_effort)


def verify_part(directory, manifest):
    for item in manifest["files"]:
        path = Path(directory) / item["name"]
        if path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
            raise ValueError(f"Part checksum mismatch: {path}")


def publish_part(local, destination, manifest):
    destination.mkdir(parents=True, exist_ok=True)
    ready = destination / "READY.json"
    if ready.exists():
        existing = json.loads(ready.read_text())
        if existing != manifest:
            raise ValueError(f"Refusing to overwrite a different completed part: {destination}")
        verify_part(destination, manifest)
        return
    for item in manifest["files"]:
        shutil.copyfile(local / item["name"], destination / item["name"])
    verify_part(destination, manifest)
    # Publish readiness only after every destination file has been read back.
    write_json(ready, manifest)


def process_part(task, settings):
    import pyarrow.parquet as pq

    started = time.monotonic()
    local = Path(settings["stage"]) / "parts" / task["id"]
    destination = Path(settings["output"]) / "parts" / task["id"]
    local.mkdir(parents=True, exist_ok=True)
    task_hash = json_sha(task)
    marker = local / "READY.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["contract_sha256"] != settings["contract_sha256"] or manifest["task_sha256"] != task_hash:
            raise ValueError(f"Resume contract mismatch: {local}")
        verify_part(local, manifest)
        publish_part(local, destination, manifest)
        return manifest
    if task["id"] in settings.get("reuse_completed", {}).get("parts", {}):
        from archlab.preprocessing.reuse import reuse_part

        return reuse_part(task, settings, local, destination)

    writers, sidecars, summaries = {}, {}, {}
    done = 0
    tokens = 0
    parquet = pq.ParquetFile(task["source"])
    mode = task["mode"]
    try:
        for batch in parquet.iter_batches(
            batch_size=settings["batch_size"],
            row_groups=list(range(task["rg_start"], task["rg_stop"])),
            use_threads=False,
        ):
            rows = batch.to_pylist()
            if task.get("row_limit"):
                rows = rows[:task["row_limit"] - done]
            rendered = []
            for offset, row in enumerate(rows):
                try:
                    if _STATE["tokenizer_format"] == "deepseek-v41":
                        rendered.append(_STATE["renderer"].render(row))
                    else:
                        rendered.append(render_row(_STATE["tokenizer"], row, mode))
                except Exception as error:
                    raise ValueError(
                        f"Render failed: part={task['id']} source_row={task['row_start'] + done + offset}: {error}"
                    ) from error
            encodings = _STATE["fast"].encode_batch(
                [item[0] for item in rendered], add_special_tokens=False,
            )
            for row, rendering, encoding in zip(rows, rendered, encodings, strict=True):
                text, has_tools, messages, kwargs = rendering
                ids = encoding.ids
                if not ids or len(ids) >= 2**31 or max(ids) >= len(_STATE["tokenizer"]):
                    raise ValueError("Invalid token sequence or int32 length overflow")
                parity_key = (mode, has_tools)
                if parity_key not in _STATE["parity_checked"]:
                    if _STATE["tokenizer_format"] == "deepseek-v41":
                        native = _STATE["tokenizer"](text, add_special_tokens=False)["input_ids"]
                    else:
                        native = _STATE["tokenizer"].apply_chat_template(messages, tokenize=True, **kwargs)
                    if hasattr(native, "keys"):
                        native = native["input_ids"]
                    if native != ids:
                        raise ValueError("Fast tokenizer differs from native chat tokenization")
                    _STATE["parity_checked"].add(parity_key)
                key = problem_key(row["problem"])
                split = choose_split(key, settings["validation_basis_points"])
                partition = f"{split}-{'tools' if has_tools else 'no-tools'}"
                prefix = partition + "_text_document"
                if partition not in writers:
                    writers[partition] = _STATE["builder"](
                        str(local / (prefix + ".bin")), dtype=_STATE["np"].int32,
                    )
                    sidecars[partition] = gzip.open(local / (prefix + ".metadata.jsonl.gz"), "wt")
                    summaries[partition] = dict(
                        prefix=prefix, mode=mode, split=split, has_tools=has_tools,
                        documents=0, tokens=0, min_length=len(ids), max_length=0,
                        length_bins={str(n): 0 for n in (4096, 8192, 16384, 32768, 65536, 131072, 262144)},
                        over_262144=0,
                    )
                writers[partition].add_document(_STATE["np"].asarray(ids, dtype="int32"), [len(ids)])
                summary = summaries[partition]
                record = {
                    field: row.get(field)
                    for field in (
                        "uuid", "data_source", "license", "url", "user_name", "user_url",
                        "expected_answer", "original_expected_answer", "changed_answer_to_majority", "used_in",
                    )
                }
                record.update(
                    source=Path(task["source"]).name, source_row=task["row_start"] + done,
                    problem_sha256=key, sequence=summary["documents"], tokens=len(ids),
                    mode=mode, split=split, has_tools=has_tools,
                )
                if _STATE["tokenizer_format"] == "deepseek-v41":
                    from archlab.preprocessing.deepseek_v41 import token_spans

                    record["assistant_token_spans"] = token_spans(
                        encoding.offsets, kwargs["assistant_character_spans"],
                    )
                    record["reasoning_effort"] = _STATE["renderer"].reasoning_effort
                    if kwargs.get("message_repairs"):
                        record["message_repairs"] = kwargs["message_repairs"]
                        summary["repaired_documents"] = summary.get("repaired_documents", 0) + 1
                sidecars[partition].write(json.dumps(record, ensure_ascii=False) + "\n")
                summary["documents"] += 1
                summary["tokens"] += len(ids)
                summary["min_length"] = min(summary["min_length"], len(ids))
                summary["max_length"] = max(summary["max_length"], len(ids))
                for upper in summary["length_bins"]:
                    if len(ids) <= int(upper):
                        summary["length_bins"][upper] += 1
                        break
                else:
                    summary["over_262144"] += 1
                done += 1
                tokens += len(ids)
            write_json(local / "progress.json", dict(
                pid=os.getpid(), documents=done, tokens=tokens, seconds=time.monotonic() - started,
            ))
            if task.get("row_limit") and done >= task["row_limit"]:
                break
        if done != task["rows"]:
            raise ValueError(f"Source row coverage mismatch: {done} != {task['rows']}")
        for partition, writer in writers.items():
            prefix = summaries[partition]["prefix"]
            writer.finalize(str(local / (prefix + ".idx")))
            sidecars[partition].close()
            dataset = _STATE["reader"](str(local / prefix), mmap=False)
            summary = summaries[partition]
            if len(dataset) != summary["documents"] or len(dataset.document_indices) != len(dataset) + 1:
                raise ValueError("Indexed dataset lost document boundaries")
            if int(dataset.sequence_lengths.sum()) != summary["tokens"]:
                raise ValueError("Indexed token count mismatch")
            if (local / (prefix + ".bin")).stat().st_size != 4 * summary["tokens"]:
                raise ValueError("Binary length mismatch")
            for index in {0, len(dataset) - 1}:
                if len(dataset[index]) != int(dataset.sequence_lengths[index]):
                    raise ValueError("Indexed reader failed round trip")
        files = []
        for partition in sorted(summaries):
            prefix = summaries[partition]["prefix"]
            for suffix in (".bin", ".idx", ".metadata.jsonl.gz"):
                path = local / (prefix + suffix)
                files.append(dict(name=path.name, bytes=path.stat().st_size, sha256=file_sha(path)))
        manifest = dict(
            id=task["id"], task_sha256=task_hash, contract_sha256=settings["contract_sha256"],
            source=task["source"], row_start=task["row_start"], documents=done, tokens=tokens,
            partitions=summaries, files=files, seconds=time.monotonic() - started,
        )
        write_json(marker, manifest)
        publish_part(local, destination, manifest)
        return manifest
    finally:
        for handle in sidecars.values():
            handle.close()
        for writer in writers.values():
            writer.data_file.close()


def inventory(source, row_groups_per_part, smoke_rows):
    import pyarrow.parquet as pq

    directory = Path(source) / "data"
    paths = sorted(directory.glob("high_part*.parquet"))
    paths += [directory / "medium.parquet", directory / "low.parquet"]
    if not paths or not any(p.name.startswith("high_part") for p in paths):
        raise ValueError("Missing high-effort Parquet shards")
    sources, tasks = [], []
    for path in paths:
        metadata = pq.ParquetFile(path).metadata
        mode = path.stem.split("_")[0]
        counts = [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)]
        sources.append(dict(
            path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
            rows=metadata.num_rows, row_group_rows=counts,
            schema_sha256=hashlib.sha256(metadata.schema.to_arrow_schema().serialize().to_pybytes()).hexdigest(),
        ))
        row_start = 0
        for start in range(0, len(counts), row_groups_per_part):
            stop = min(start + row_groups_per_part, len(counts))
            count = sum(counts[start:stop])
            tasks.append(dict(
                id=f"{path.stem}-rg{start:05d}-{stop:05d}", source=str(path), mode=mode,
                rg_start=start, rg_stop=stop, row_start=row_start,
                rows=min(count, smoke_rows) if smoke_rows else count, row_limit=smoke_rows,
            ))
            row_start += count
            if smoke_rows:
                break
    return sources, tasks


def prepare_contract(args):
    import nemo_automodel.components.datasets.llm.megatron.indexed_dataset as indexed

    sources, tasks = inventory(args.source, args.row_groups_per_part, args.smoke_rows)
    asset_names = TOKENIZER_FILES
    tokenizer_format = getattr(args, "tokenizer_format", "qwen")
    if tokenizer_format == "deepseek-v41":
        asset_names += ("encoding/encoding.py", "LICENSE")
        if not (Path(args.tokenizer) / "encoding/encoding.py").is_file():
            raise ValueError("DeepSeek V4.1 requires its official encoding/encoding.py")
    tokenizer_files = {
        name: file_sha(Path(args.tokenizer) / name)
        for name in asset_names if (Path(args.tokenizer) / name).is_file()
    }
    versions = {}
    for name in ("numpy", "pyarrow", "tokenizers", "transformers", "torch"):
        versions[name] = importlib.metadata.version(name)
    contract = dict(
        schema_version=1, dataset="nv-community/Nemotron-Math-v2", sources=sources,
        representation="Parquet only; JSONL mirror is not ingested", tasks=tasks,
        expected_documents=sum(task["rows"] for task in tasks), smoke_only=bool(args.smoke_rows),
        tokenizer_source=str(Path(args.tokenizer).resolve()), tokenizer_files=tokenizer_files,
        template="checkpoint-native apply_chat_template", reasoning_effort_mapping=EFFORT,
        preserve_thinking=True, enable_thinking=True, add_generation_prompt=False,
        tokenization_add_special_tokens=False, truncation=False, append_eod=False,
        ending="Exact native chat ending: <|im_end|> plus template whitespace; no additional EOD",
        dtype="int32", document_unit="one complete source conversation",
        loss_mask="not applied; token corpus plus source metadata, not an SFT trainer policy",
        split=dict(
            validation_basis_points=args.validation_basis_points, algorithm="sha256-seeded-modulo-10000",
            seed="nemotron-math-v2:v1:", key="sha256(NFC(problem), collapsed whitespace)",
            grouping="same normalized problem across every reasoning/tool variant",
        ),
        deduplication=False, filtered_rows=False,
        implementation_sha256=file_sha(__file__), indexed_writer_sha256=file_sha(indexed.__file__),
        indexed_writer_path=indexed.__file__, versions=versions, python=sys.version,
        host=socket.gethostname(), platform=platform.platform(),
        container_versions={key: os.environ[key] for key in (
            "NVIDIA_PYTORCH_VERSION", "NVIDIA_BUILD_ID", "CUDA_VERSION", "NEMO_VERSION",
        ) if key in os.environ},
    )
    if tokenizer_format == "deepseek-v41":
        from archlab.preprocessing import deepseek_v41

        contract.update(
            schema_version=3, tokenizer_format=tokenizer_format,
            model_id=deepseek_v41.MODEL_ID, model_revision=deepseek_v41.REVISION,
            template="checkpoint official encoding.encode_messages; no Jinja template",
            reasoning_effort_mapping=None, reasoning_effort=args.reasoning_effort,
            source_effort_policy="retain original mode in metadata; use one explicit native numeric budget",
            preserve_thinking=True, enable_thinking=True, drop_thinking=False,
            ending="One native BOS; native assistant EOS; no extra EOD",
            loss_mask="assistant_token_spans: half-open token indices including reasoning, answers, calls, EOS; excluding prompt headers and tool results",
            renderer_sha256=file_sha(deepseek_v41.__file__),
            official_encoder_sha256=deepseek_v41.ENCODER_SHA256,
            assistant_message_policy=deepseek_v41.ASSISTANT_MESSAGE_POLICY,
        )
    if getattr(args, "reuse_completed_from", None):
        from archlab.preprocessing.reuse import import_contract

        origin = Path(args.reuse_completed_from).resolve()
        for destination in (Path(args.stage).resolve(), Path(args.output).resolve()):
            if origin == destination or origin in destination.parents or destination in origin.parents:
                raise ValueError("Legacy dataset and new destinations must not overlap")
        contract["reuse_completed"] = import_contract(contract, origin)
    return contract


def run(args):
    stage, output = Path(args.stage).resolve(), Path(args.output).resolve()
    if stage == output or stage in output.parents or output in stage.parents:
        raise ValueError("Staging and output must be separate directories")
    stage.mkdir(parents=True, exist_ok=True)
    lock = open(stage / ".lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = prepare_contract(args)
    contract_hash = json_sha(contract)
    output.mkdir(parents=True, exist_ok=True)
    for root in (stage, output):
        path = root / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != contract:
                raise ValueError(f"Existing dataset uses a different contract: {root}")
        else:
            allowed = {".lock", "job.log", "launcher.json"} if root == stage else set()
            if any(p.name not in allowed for p in root.iterdir()):
                raise ValueError(f"Refusing an unrecognized nonempty directory: {root}")
            write_json(path, contract)
    # Archive only tokenizer assets (never the pretrained model weights).
    for root in (stage, output):
        snapshot = root / "tokenizer"
        snapshot.mkdir(exist_ok=True)
        for name, expected in contract["tokenizer_files"].items():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(Path(args.tokenizer) / name, destination)
            if file_sha(destination) != expected:
                raise ValueError(f"Tokenizer snapshot mismatch: {destination}")
        readme = root / "SOURCE_README.md"
        if not readme.exists():
            shutil.copyfile(Path(args.source) / "README.md", readme)
    settings = dict(
        stage=str(stage), output=str(output), contract_sha256=contract_hash,
        validation_basis_points=args.validation_basis_points, batch_size=args.batch_size,
    )
    if "reuse_completed" in contract:
        settings["reuse_completed"] = contract["reuse_completed"]
    started = time.monotonic()
    completed = []
    tasks = iter(contract["tasks"])
    context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context,
        initializer=worker_init, initargs=(str(stage / "tokenizer"), args.tokenizer_format, args.reasoning_effort),
    )
    pending = {}

    def progress(status):
        report = dict(
            status=status, pid=os.getpid(), updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            contract_sha256=contract_hash, workers=args.workers, seconds=time.monotonic() - started,
            completed_parts=len(completed), total_parts=len(contract["tasks"]),
            completed_documents=sum(item["documents"] for item in completed),
            completed_tokens=sum(item["tokens"] for item in completed),
            expected_documents=contract["expected_documents"],
            active_parts=[task["id"] for task in pending.values()],
        )
        if "reuse_completed" in contract:
            report.update(
                reused_parts=sum("reused_from" in item for item in completed),
                reused_documents=sum(item["documents"] for item in completed if "reused_from" in item),
                newly_tokenized_documents=sum(item["documents"] for item in completed if "reused_from" not in item),
                repaired_documents=sum(p.get("repaired_documents", 0) for item in completed for p in item["partitions"].values()),
            )
        write_json(stage / "progress.json", report)
        write_json(output / "progress.json", report)
        print(json.dumps(report), flush=True)
        return report

    try:
        for _ in range(args.workers):
            task = next(tasks, None)
            if task is not None:
                pending[pool.submit(process_part, task, settings)] = task
        progress("running")
        while pending:
            finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                completed.append(future.result())
                del pending[future]
                task = next(tasks, None)
                if task is not None:
                    pending[pool.submit(process_part, task, settings)] = task
            progress("running")
        if sum(item["documents"] for item in completed) != contract["expected_documents"]:
            raise ValueError("Final document coverage mismatch")
        report = progress("complete")
        report["parts"] = sorted(completed, key=lambda item: item["id"])
        report["prefixes"] = {}
        for part in report["parts"]:
            for partition, summary in sorted(part["partitions"].items()):
                group = summary["mode"] + "/" + partition
                report["prefixes"].setdefault(group, []).append(
                    "parts/" + part["id"] + "/" + summary["prefix"],
                )
        marker = "SMOKE_READY.json" if args.smoke_rows else "DATA_READY.json"
        write_json(stage / marker, report)
        write_json(output / marker, report)
    except BaseException:
        # Report the offending row immediately, not after all workers drain.
        traceback.print_exc()
        progress("failed")
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-format", choices=("qwen", "deepseek-v41"), default="qwen")
    parser.add_argument("--reasoning-effort", type=int, default=75,
                        help="DeepSeek V4.1 native numeric budget, fixed across source effort variants")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--row-groups-per-part", type=int, default=2)
    parser.add_argument("--validation-basis-points", type=int, default=100)
    parser.add_argument("--smoke-rows", type=int, default=0)
    parser.add_argument("--reuse-completed-from", help="Import verified READY parts from the audited DeepSeek v1 dataset into a new version")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.row_groups_per_part) < 1 or args.smoke_rows < 0:
        parser.error("Workers, batch size and row groups must be positive")
    if not 0 <= args.validation_basis_points <= 10000:
        parser.error("Validation basis points must be between 0 and 10000")
    if not 1 <= args.reasoning_effort <= 100:
        parser.error("DeepSeek reasoning effort must be within 1..100")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ.update(
        CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    if args.detach:
        stage = Path(args.stage).resolve()
        stage.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "archlab.preprocessing.nemotron_math"]
        command.extend(arg for arg in sys.argv[1:] if arg != "--detach")
        with open(stage / "job.log", "ab", buffering=0) as log:
            process = subprocess.Popen(
                ["nice", "-n", "10", *command], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        report = dict(pid=process.pid, command=command, stage=str(stage), output=args.output)
        write_json(stage / "launcher.json", report)
        print(json.dumps(report), flush=True)
    else:
        run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate repository-owned Qwen3.8 checkpoints on PIQA and plot the curve.

The evaluator reads the cached PIQA Arrow file directly and reproduces the
zero-shot ``lm-eval==0.4.13`` prompt, causal token boundary, and character
length normalization.  It loads Megatron distributed checkpoints in-place, so
an intermediate Hugging Face export is not required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import torch
import torch.nn as nn
from tokenizers import Tokenizer
from torch.distributed.checkpoint import load as load_checkpoint

from archlab.architectures.qwen38_27b import Qwen38Dense, Qwen38DenseConfig

PROMPT_TEMPLATE = "Question: {goal}\nAnswer:"
TARGET_DELIMITER = " "
LM_EVAL_VERSION = "0.4.13"
EVALUATION_IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Request:
    example_index: int
    choice_index: int
    input_ids: tuple[int, ...]
    continuation_ids: tuple[int, ...]
    continuation_text: str


class CheckpointModel(nn.Module):
    """Match the ``architecture.*`` keys written by the Megatron adapter."""

    def __init__(self, *, sequence_len: int, gdn_kernel: str) -> None:
        super().__init__()
        config = Qwen38DenseConfig(sequence_len=sequence_len)
        self.architecture = Qwen38Dense(
            config,
            runtime_backend="te_bf16",
            gdn_kernel=gdn_kernel,
        )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_identity(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        raise RuntimeError(f"checkpoint directory is missing: {checkpoint}")
    metadata = checkpoint / ".metadata"
    if not metadata.is_file():
        raise RuntimeError(f"distributed checkpoint metadata is missing: {metadata}")
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"checkpoint directory is empty: {checkpoint}")
    return {
        "path": str(checkpoint),
        "metadata_sha256": _sha256(metadata),
        "files": [
            {
                "path": str(path.relative_to(checkpoint)),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ],
    }


def _evaluation_identity(args: argparse.Namespace, checkpoint: Path) -> dict[str, Any]:
    tokenizer = args.tokenizer.expanduser().resolve()
    piqa_arrow = args.piqa_arrow.expanduser().resolve()
    evaluator = Path(__file__).resolve()
    architecture = Path(inspect.getfile(Qwen38Dense)).resolve()
    for label, path in (("tokenizer", tokenizer), ("PIQA Arrow", piqa_arrow)):
        if not path.is_file():
            raise RuntimeError(f"{label} artifact is missing: {path}")
    return {
        "schema_version": EVALUATION_IDENTITY_SCHEMA_VERSION,
        "checkpoint": _checkpoint_identity(checkpoint),
        "tokenizer": {"path": str(tokenizer), "sha256": _sha256(tokenizer)},
        "piqa": {"path": str(piqa_arrow), "sha256": _sha256(piqa_arrow)},
        "evaluator": {"path": str(evaluator), "sha256": _sha256(evaluator)},
        "architecture": {"path": str(architecture), "sha256": _sha256(architecture)},
        "protocol": {
            "lm_eval_version": LM_EVAL_VERSION,
            "prompt_template": PROMPT_TEMPLATE,
            "target_delimiter": TARGET_DELIMITER,
            "tokens_per_iteration": args.tokens_per_iteration,
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "gdn_kernel": args.gdn_kernel,
            "pad_token_id": args.pad_token_id,
        },
        "runtime": {
            "torch": torch.__version__,
            "pyarrow": pa.__version__,
            "tokenizers": importlib.metadata.version("tokenizers"),
        },
    }


def _validate_cached_result(
    result: dict[str, Any],
    expected_identity: dict[str, Any],
) -> None:
    stored_identity = result.get("evaluation_identity")
    stored_sha256 = result.get("evaluation_identity_sha256")
    expected_sha256 = _canonical_sha256(expected_identity)
    if (
        not isinstance(stored_identity, dict)
        or stored_sha256 != _canonical_sha256(stored_identity)
        or stored_sha256 != expected_sha256
        or stored_identity != expected_identity
    ):
        raise RuntimeError(
            "cached PIQA result identity does not match the requested evaluation; "
            "pass --overwrite to recompute it"
        )


def _read_piqa(path: Path) -> list[dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    required = {"goal", "sol1", "sol2", "label"}
    if not required.issubset(table.column_names):
        raise RuntimeError(f"PIQA Arrow schema lacks {sorted(required - set(table.column_names))}")
    return table.to_pylist()


def _encode_requests(
    examples: list[dict[str, Any]], tokenizer: Tokenizer
) -> list[Request]:
    requests = []
    for example_index, example in enumerate(examples):
        context = PROMPT_TEMPLATE.format(goal=example["goal"])
        context_ids = tokenizer.encode(context, add_special_tokens=True).ids
        for choice_index, solution in enumerate((example["sol1"], example["sol2"])):
            continuation = TARGET_DELIMITER + solution
            whole_ids = tokenizer.encode(
                context + continuation,
                add_special_tokens=True,
            ).ids
            continuation_ids = whole_ids[len(context_ids) :]
            if not context_ids or not continuation_ids:
                raise RuntimeError(
                    f"empty token span at example {example_index}, choice {choice_index}"
                )
            requests.append(
                Request(
                    example_index=example_index,
                    choice_index=choice_index,
                    input_ids=tuple((context_ids + continuation_ids)[:-1]),
                    continuation_ids=tuple(continuation_ids),
                    continuation_text=solution,
                )
            )
    return requests


def _load_model_state(model: CheckpointModel, checkpoint: Path) -> dict[str, Any]:
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.endswith("._extra_state")
    }
    started = time.monotonic()
    load_checkpoint(state, checkpoint_id=str(checkpoint))
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(
        key for key in incompatible.missing_keys if not key.endswith("._extra_state")
    )
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint state mismatch: missing={missing}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return {
        "seconds": time.monotonic() - started,
        "loaded_tensors": len(state),
        "ignored_extra_state_tensors": len(incompatible.missing_keys),
    }


@torch.inference_mode()
def _score_requests(
    model: CheckpointModel,
    requests: list[Request],
    *,
    batch_size: int,
    pad_token_id: int,
) -> list[float]:
    architecture = model.architecture
    device = architecture.get_device()
    scores = [math.nan] * len(requests)
    # Preserve dataset order.  Fused recurrent kernels can accumulate at
    # slightly different BF16 roundoff when the surrounding batch shape
    # changes, which matters for a handful of nearly tied PIQA choices.
    indexed = list(enumerate(requests))
    for offset in range(0, len(indexed), batch_size):
        chunk = indexed[offset : offset + batch_size]
        max_length = max(len(request.input_ids) for _, request in chunk)
        input_ids = torch.full(
            (len(chunk), max_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        labels = torch.full_like(input_ids, -1)
        for row, (_, request) in enumerate(chunk):
            input_length = len(request.input_ids)
            continuation_length = len(request.continuation_ids)
            input_ids[row, :input_length] = torch.tensor(
                request.input_ids,
                dtype=torch.long,
                device=device,
            )
            labels[
                row,
                input_length - continuation_length : input_length,
            ] = torch.tensor(
                request.continuation_ids,
                dtype=torch.long,
                device=device,
            )
        logits = architecture(input_ids)
        losses = architecture._token_cross_entropy(logits, labels)
        loglikelihoods = -losses.sum(dim=-1)
        for row, (original_index, _) in enumerate(chunk):
            scores[original_index] = float(loglikelihoods[row])
        del input_ids, labels, logits, losses, loglikelihoods
    if any(not math.isfinite(score) for score in scores):
        raise RuntimeError("PIQA scoring produced a non-finite log likelihood")
    return scores


def _standard_error(accuracy: float, examples: int) -> float:
    return math.sqrt(accuracy * (1.0 - accuracy) / examples)


def _result_payload(
    examples: list[dict[str, Any]],
    requests: list[Request],
    scores: list[float],
    *,
    checkpoint: Path,
    iteration: int,
    tokens_per_iteration: int,
    elapsed_seconds: float,
    load_stats: dict[str, Any],
    source_commit: str | None,
    evaluation_identity: dict[str, Any],
) -> dict[str, Any]:
    pairs = [[math.nan, math.nan] for _ in examples]
    for request, score in zip(requests, scores, strict=True):
        pairs[request.example_index][request.choice_index] = score
    raw_correct = 0
    normalized_correct = 0
    sample_records = []
    for index, (example, pair) in enumerate(zip(examples, pairs, strict=True)):
        solutions = [example["sol1"], example["sol2"]]
        prediction = max(range(2), key=lambda choice: pair[choice])
        normalized_prediction = max(
            range(2),
            key=lambda choice: pair[choice] / len(solutions[choice]),
        )
        gold = int(example["label"])
        raw_correct += prediction == gold
        normalized_correct += normalized_prediction == gold
        if index < 20:
            sample_records.append(
                {
                    "goal": example["goal"],
                    "solutions": solutions,
                    "gold": gold,
                    "loglikelihood": pair,
                    "prediction": prediction,
                    "normalized_prediction": normalized_prediction,
                }
            )
    count = len(examples)
    accuracy = raw_correct / count
    normalized_accuracy = normalized_correct / count
    return {
        "task": "piqa",
        "task_version": 1.0,
        "lm_eval_version": LM_EVAL_VERSION,
        "num_fewshot": 0,
        "validation_examples": count,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": iteration,
        "training_tokens_at_checkpoint": iteration * tokens_per_iteration,
        "acc": accuracy,
        "acc_stderr": _standard_error(accuracy, count),
        "acc_norm": normalized_accuracy,
        "acc_norm_stderr": _standard_error(normalized_accuracy, count),
        "raw_correct": raw_correct,
        "normalized_correct": normalized_correct,
        "prompt_template": PROMPT_TEMPLATE.replace("{goal}", "{{goal}}"),
        "target_delimiter": TARGET_DELIMITER,
        "normalization": "lm_eval 0.4.13 character length",
        "elapsed_seconds": elapsed_seconds,
        "checkpoint_load": load_stats,
        "architecture_source_commit": source_commit,
        "evaluation_identity": evaluation_identity,
        "evaluation_identity_sha256": _canonical_sha256(evaluation_identity),
        "sample_records": sample_records,
    }


def _validate_result(result: dict[str, Any], known: dict[str, Any]) -> dict[str, Any]:
    if result.get("validation_examples") != known.get("validation_examples"):
        raise RuntimeError("lm-eval validation example count differs from the reference")
    count_differences = {
        field: abs(int(result[field]) - int(known[field]))
        for field in ("raw_correct", "normalized_correct")
    }
    paired_samples = zip(
        result.get("sample_records", []),
        known.get("sample_records", []),
        strict=True,
    )
    likelihood_differences = [
        abs(float(current_score) - float(known_score))
        for current, reference in paired_samples
        for current_score, known_score in zip(
            current["loglikelihood"],
            reference["loglikelihood"],
            strict=True,
        )
    ]
    max_likelihood_difference = max(likelihood_differences, default=math.inf)
    # Fused GDN kernels accumulate in BF16. Changing only the surrounding batch
    # shape can move a near-tied choice by a few ulps, so allow one example of
    # count drift while requiring stored sample likelihoods to agree closely.
    if max(count_differences.values()) > 1 or max_likelihood_difference > 1e-3:
        raise RuntimeError(
            "lm-eval equivalence check failed: "
            f"count_differences={count_differences}, "
            f"max_sample_loglikelihood_difference={max_likelihood_difference}"
        )
    return {
        "count_differences": count_differences,
        "max_sample_loglikelihood_difference": max_likelihood_difference,
        "tolerance": {
            "maximum_count_difference": 1,
            "maximum_sample_loglikelihood_difference": 1e-3,
        },
    }


def _aggregate(output_dir: Path, iterations: list[int]) -> list[dict[str, Any]]:
    rows = []
    for iteration in sorted(iterations):
        path = output_dir / f"checkpoint-{iteration:07d}.json"
        if not path.is_file():
            continue
        result = json.loads(path.read_text())
        rows.append(
            {
                key: result[key]
                for key in (
                    "checkpoint_iteration",
                    "training_tokens_at_checkpoint",
                    "acc",
                    "acc_stderr",
                    "acc_norm",
                    "acc_norm_stderr",
                    "raw_correct",
                    "normalized_correct",
                )
            }
        )
    _atomic_json(
        output_dir / "piqa-curve.json",
        {
            "task": "piqa",
            "lm_eval_version": LM_EVAL_VERSION,
            "checkpoints": rows,
        },
    )
    csv_path = output_dir / "piqa-curve.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8") as handle:
        if fields:
            handle.write(",".join(fields) + "\n")
            for row in rows:
                handle.write(",".join(str(row[field]) for field in fields) + "\n")
    os.replace(temporary, csv_path)
    return rows


def _plot(rows: list[dict[str, Any]], output: Path) -> None:
    if not rows:
        return
    import matplotlib.pyplot as plt

    tokens = [row["training_tokens_at_checkpoint"] / 1e9 for row in rows]
    figure, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    for metric, error, label, color in (
        ("acc", "acc_stderr", "PIQA accuracy", "#3178c6"),
        ("acc_norm", "acc_norm_stderr", "PIQA normalized accuracy", "#d97706"),
    ):
        values = [100.0 * row[metric] for row in rows]
        errors = [100.0 * row[error] for row in rows]
        axis.errorbar(
            tokens,
            values,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
            color=color,
        )
    axis.axhline(50.0, color="#6b7280", linestyle="--", linewidth=1, label="chance")
    axis.set_title("Quartered Qwen3.8-27B: PIQA over FineWeb-Edu pretraining")
    axis.set_xlabel("Training tokens (billions)")
    axis.set_ylabel("Validation accuracy (%)")
    axis.set_xticks(tokens)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _iterations(root: Path, requested: list[int] | None) -> list[int]:
    if requested:
        return sorted(set(requested))
    found = []
    for path in root.glob("iter_*"):
        if path.is_dir():
            found.append(int(path.name.removeprefix("iter_")))
    if not found:
        raise RuntimeError(f"no iter_* checkpoints found under {root}")
    return sorted(found)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--piqa-arrow", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", nargs="+", type=int)
    parser.add_argument("--validation-iteration", type=int)
    parser.add_argument("--validation-artifact", type=Path)
    parser.add_argument("--tokens-per-iteration", type=int, default=1_048_576)
    parser.add_argument("--sequence-length", type=int, default=2_048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gdn-kernel", default="fla", choices=("fla",))
    parser.add_argument("--pad-token-id", type=int, default=248_044)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if (args.validation_iteration is None) != (args.validation_artifact is None):
        raise ValueError("validation iteration and artifact must be specified together")
    iterations = _iterations(args.checkpoint_root, args.iterations)
    if args.validation_iteration is not None:
        if args.validation_iteration not in iterations:
            raise ValueError("validation iteration is absent from the requested checkpoints")
        iterations.remove(args.validation_iteration)
        iterations.insert(0, args.validation_iteration)
    source_commit = None
    try:
        architecture_root = Path(inspect.getfile(Qwen38Dense)).resolve().parents[3]
        source_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=architecture_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    examples = _read_piqa(args.piqa_arrow)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    requests = _encode_requests(examples, tokenizer)
    torch.cuda.set_device(0)
    with torch.device("cuda:0"):
        model = CheckpointModel(
            sequence_len=args.sequence_length,
            gdn_kernel=args.gdn_kernel,
        )
    # Megatron stores these model parameters as BF16.  Matching the load target
    # dtype avoids a silent BF16-to-FP32 upcast that can flip near-tied choices.
    model.bfloat16()
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for iteration in iterations:
        checkpoint = args.checkpoint_root / f"iter_{iteration:07d}"
        output = args.output_dir / f"checkpoint-{iteration:07d}.json"
        evaluation_identity = _evaluation_identity(args, checkpoint)
        if output.is_file() and not args.overwrite:
            result = json.loads(output.read_text())
            _validate_cached_result(result, evaluation_identity)
        else:
            started = time.monotonic()
            load_stats = _load_model_state(model, checkpoint)
            scores = _score_requests(
                model,
                requests,
                batch_size=args.batch_size,
                pad_token_id=args.pad_token_id,
            )
            result = _result_payload(
                examples,
                requests,
                scores,
                checkpoint=checkpoint,
                iteration=iteration,
                tokens_per_iteration=args.tokens_per_iteration,
                elapsed_seconds=time.monotonic() - started,
                load_stats=load_stats,
                source_commit=source_commit,
                evaluation_identity=evaluation_identity,
            )
            _atomic_json(output, result)
        if iteration == args.validation_iteration:
            known = json.loads(args.validation_artifact.read_text())
            validation = _validate_result(result, known)
            _atomic_json(
                args.output_dir / "LM_EVAL_PROTOCOL_VALIDATED.json",
                {
                    "iteration": iteration,
                    "reference": str(args.validation_artifact),
                    "raw_correct": result["raw_correct"],
                    "normalized_correct": result["normalized_correct"],
                    "reference_raw_correct": known["raw_correct"],
                    "reference_normalized_correct": known["normalized_correct"],
                    **validation,
                    "validated_at_unix": time.time(),
                },
            )
        completed.append(iteration)
        rows = _aggregate(args.output_dir, completed)
        _plot(rows, args.output_dir / "piqa-curve.png")
        print(
            f"iteration={iteration} acc={result['acc']:.6f} "
            f"acc_norm={result['acc_norm']:.6f} elapsed={result['elapsed_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Validate full-state checkpoints, export the final Qwen model, and run easy evals."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def expected_checkpoint_iterations(train_iters: int, save_interval: int) -> list[int]:
    if train_iters < 1 or save_interval < 1:
        raise ValueError("train-iters and save-interval must be positive")
    iterations = list(range(save_interval, train_iters + 1, save_interval))
    if not iterations or iterations[-1] != train_iters:
        iterations.append(train_iters)
    return iterations


def _checkpoint_dir(root: Path, iteration: int) -> Path:
    return root / f"iter_{iteration:07d}"


def validate_checkpoints(args: argparse.Namespace) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader

    root = args.checkpoint_root.expanduser().resolve()
    latest_path = root / "latest_checkpointed_iteration.txt"
    if not latest_path.is_file():
        raise FileNotFoundError(latest_path)
    latest = int(latest_path.read_text(encoding="utf-8").strip())
    if latest != args.train_iters:
        raise RuntimeError(f"latest checkpoint is {latest}, expected {args.train_iters}")

    checkpoints = []
    for iteration in expected_checkpoint_iterations(args.train_iters, args.save_interval):
        path = _checkpoint_dir(root, iteration)
        for required in (path / ".metadata", path / "common.pt"):
            if not required.is_file() or required.stat().st_size <= 0:
                raise RuntimeError(f"incomplete checkpoint {iteration}: {required}")
        metadata = FileSystemReader(path).read_metadata()
        keys = tuple(metadata.state_dict_metadata)
        required_state = {
            "model": any(
                key.startswith("embedding.") or key.startswith("decoder.") for key in keys
            ),
            "optimizer_param": any("optimizer." in key and key.endswith(".param") for key in keys),
            "optimizer_exp_avg": any("optimizer." in key and "exp_avg" in key for key in keys),
            "optimizer_exp_avg_sq": any(
                "optimizer." in key and "exp_avg_sq" in key for key in keys
            ),
            "rng": any(key.startswith("rng_state") for key in keys),
        }
        missing = [name for name, present in required_state.items() if not present]
        if missing:
            raise RuntimeError(f"checkpoint {iteration} lacks full resume state: {missing}")
        checkpoints.append(
            {
                "iteration": iteration,
                "trained_tokens": iteration * args.tokens_per_iteration,
                "path": str(path),
                "distcp_files": len(list(path.glob("*.distcp"))),
                "state_entries": len(keys),
                "full_resume_state": required_state,
            }
        )
    payload = {
        "checkpoint_root": str(root),
        "train_iters": args.train_iters,
        "save_interval": args.save_interval,
        "tokens_per_iteration": args.tokens_per_iteration,
        "checkpoints": checkpoints,
        "validated_at_unix": time.time(),
    }
    _write_json(args.output, payload)
    return payload


def _latest_checkpoint(root: Path) -> Path:
    latest = int((root / "latest_checkpointed_iteration.txt").read_text().strip())
    path = _checkpoint_dir(root, latest)
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _valid_hf_export(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        and (path / "tokenizer.json").is_file()
        and (any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin")))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_oss_path(path: Path) -> bool:
    try:
        path.relative_to("/mnt/oss")
    except ValueError:
        return False
    return True


def _copy_export_to_oss(source: Path, output: Path) -> None:
    """Copy a complete NAS export to OSS without unsupported rename/copystat calls."""
    marker = output / "EXPORT_COPY_COMPLETE.json"
    if _valid_hf_export(output) and marker.is_file():
        return
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative = source_file.relative_to(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        source_hash = _sha256(source_file)
        target_hash = _sha256(target)
        if source_hash != target_hash:
            raise RuntimeError(f"HF export copy hash mismatch: {relative}")
        manifest.append(
            {
                "path": str(relative),
                "bytes": source_file.stat().st_size,
                "sha256": source_hash,
            }
        )
    if not _valid_hf_export(output):
        raise RuntimeError(f"copied HF export is incomplete: {output}")
    marker.write_text(json.dumps({"source": str(source), "files": manifest}, sort_keys=True) + "\n")


def _export_hf(checkpoint: Path, reference: Path, output: Path) -> None:
    oss_output = _is_oss_path(output)
    copy_marker = output / "EXPORT_COPY_COMPLETE.json"
    if _valid_hf_export(output) and (not oss_output or copy_marker.is_file()):
        return
    export_output = (
        checkpoint.parent.parent / "hf-export-staging" / checkpoint.name / output.name
        if oss_output
        else output
    )
    if not _valid_hf_export(export_output) and export_output.exists():
        raise RuntimeError(f"incomplete HF export already occupies staging path: {export_output}")
    temporary = export_output.with_name(f".{export_output.name}.incomplete-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary export path already exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)

    from megatron.bridge import AutoBridge
    from megatron.bridge.training.model_load_save import (
        load_megatron_model,
        temporary_distributed_context,
    )

    if not _valid_hf_export(export_output):
        bridge = AutoBridge.from_hf_pretrained(str(reference))
        # ``pretrain_gpt.py`` checkpoints carry Megatron-LM args rather than a
        # Megatron-Bridge run_config. Keep the Gloo process group alive across both
        # loading and conversion, and identify the raw model family explicitly.
        with temporary_distributed_context(backend="gloo"):
            model = load_megatron_model(
                str(checkpoint),
                model_type="gpt",
                use_cpu_init=True,
                skip_temp_dist_context=True,
            )
            bridge.save_hf_pretrained(
                model,
                str(temporary),
                show_progress=True,
                strict=True,
                source_path=str(reference),
            )
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
        ):
            source = reference / name
            target = temporary / name
            if source.is_file() and not target.exists():
                shutil.copy2(source, target)
        if not _valid_hf_export(temporary):
            raise RuntimeError(f"HF export is incomplete: {temporary}")
        os.replace(temporary, export_output)
    if oss_output:
        _copy_export_to_oss(export_output, output)


EASY_TASKS = (
    ("arc_easy", 25),
    ("hellaswag", 10),
    ("piqa", 0),
    ("winogrande", 5),
    ("gsm8k", 5),
)


def _lm_eval_model_args(model: Path) -> str:
    if "," in str(model):
        raise ValueError("lm-eval model paths cannot contain commas")
    return f"pretrained={model},dtype=bfloat16"


def _lm_eval_environment(site: Path) -> dict[str, str]:
    site = site.expanduser().resolve()
    if not (site / "lm_eval" / "__init__.py").is_file():
        raise RuntimeError(f"pinned lm-eval runtime is unavailable: {site}")
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = f"{site}{os.pathsep}{existing}" if existing else str(site)
    return environment


def _run_evaluations(
    model: Path,
    output: Path,
    lm_eval_site: Path,
    *,
    eval_limit: int | None = None,
) -> list[dict[str, Any]]:
    base_environment = _lm_eval_environment(lm_eval_site)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as m; "
                "import lm_eval.models.huggingface; "
                "assert m.version('lm_eval') == '0.4.13'"
            ),
        ],
        capture_output=True,
        text=True,
        env=base_environment,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"pinned lm-eval 0.4.13 Hugging Face adapter failed to import: {probe.stderr.strip()}"
        )
    output.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[str, int, subprocess.Popen[str], Any, Path]] = []
    model_args = _lm_eval_model_args(model)
    for gpu, (task, fewshot) in enumerate(EASY_TASKS):
        task_output = output / task
        task_output.mkdir(parents=True, exist_ok=True)
        log_path = output / f"{task}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        environment = dict(base_environment)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        environment["TOKENIZERS_PARALLELISM"] = "false"
        command = [
            sys.executable,
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            task,
            "--num_fewshot",
            str(fewshot),
            "--batch_size",
            "auto:4",
            "--max_batch_size",
            "64",
            "--device",
            "cuda:0",
            "--output_path",
            str(task_output),
            "--seed",
            "42",
            "--show_config",
        ]
        if eval_limit is not None:
            command.extend(("--limit", str(eval_limit)))
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
        processes.append((task, fewshot, process, log_handle, log_path))

    results = []
    failures = []
    for task, fewshot, process, log_handle, log_path in processes:
        returncode = process.wait()
        log_handle.close()
        if returncode != 0:
            failures.append(f"{task} exited {returncode}; see {log_path}")
            continue
        candidates = sorted((output / task).rglob("*.json"), key=lambda path: path.stat().st_mtime)
        result_path = None
        metrics = None
        for candidate in reversed(candidates):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "results" in payload:
                result_path = candidate
                metrics = payload["results"].get(task, payload["results"])
                break
        if result_path is None:
            failures.append(f"{task} produced no result JSON; see {log_path}")
            continue
        results.append(
            {
                "task": task,
                "fewshot": fewshot,
                "metrics": metrics,
                "result_path": str(result_path),
                "log_path": str(log_path),
            }
        )
    if failures:
        raise RuntimeError("; ".join(failures))
    _write_json(output / "SUMMARY.json", {"benchmarks": results})
    return results


def _generate_samples(model_path: Path, output: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda:5")
    torch.manual_seed(42)
    prompts = [
        "The most important idea in algebra is",
        "A farmer has 24 apples and gives one third to a neighbor. The number left is",
        "In a quiet town beside the sea,",
        "Photosynthesis is the process by which",
        "To solve a difficult problem, first",
    ]
    samples = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        samples.append(
            {
                "prompt": prompt,
                "text": tokenizer.decode(generated[0], skip_special_tokens=True),
            }
        )
    _write_json(output, {"samples": samples})


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    reference = args.hf_reference.expanduser().resolve()
    hf_output = args.hf_output.expanduser().resolve()
    eval_output = args.eval_output.expanduser().resolve()
    checkpoint = _latest_checkpoint(checkpoint_root)
    _export_hf(checkpoint, reference, hf_output)
    results = _run_evaluations(
        hf_output,
        eval_output,
        args.lm_eval_site,
        eval_limit=args.eval_limit,
    )
    _generate_samples(hf_output, eval_output / "samples.json")
    payload = {
        "checkpoint": str(checkpoint),
        "hf_export": str(hf_output),
        "benchmarks": results,
        "samples": str(eval_output / "samples.json"),
        "completed_at_unix": time.time(),
    }
    _write_json(args.completion_marker, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    validate = sub.add_parser("validate-checkpoints")
    validate.add_argument("--checkpoint-root", required=True, type=Path)
    validate.add_argument("--train-iters", required=True, type=int)
    validate.add_argument("--save-interval", required=True, type=int)
    validate.add_argument("--tokens-per-iteration", required=True, type=int)
    validate.add_argument("--output", required=True, type=Path)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--checkpoint-root", required=True, type=Path)
    evaluate.add_argument("--hf-reference", required=True, type=Path)
    evaluate.add_argument("--hf-output", required=True, type=Path)
    evaluate.add_argument("--eval-output", required=True, type=Path)
    evaluate.add_argument("--completion-marker", required=True, type=Path)
    evaluate.add_argument("--lm-eval-site", required=True, type=Path)
    evaluate.add_argument("--eval-limit", type=int)
    skipped = sub.add_parser("mark-skipped")
    skipped.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "validate-checkpoints":
        validate_checkpoints(args)
    elif args.action == "run":
        run(args)
    else:
        _write_json(args.output, {"skipped": True, "completed_at_unix": time.time()})


if __name__ == "__main__":
    main()

"""Paired capability pilot: unchanged pretrained parent versus trained additions.

Run by filename with the immutable training source snapshot on PYTHONPATH.
New evaluation helpers are loaded by their fixed repository paths, so importing
them does not replace the checkpoint's architecture/integration implementation.
No training process, installed package, or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import yaml


def _load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_sampling = _load_local("archlab_eval_sampling", Path(__file__).with_name("sample.py"))
_capability = _load_local("archlab_eval_capability", Path(__file__).parent.parent / "benchmarks/capability.py")


@contextmanager
def added_modules(model: torch.nn.Module, *, enabled: bool) -> Iterator[None]:
    """Toggle only branch execution, preserving every pretrained weight/module."""
    from archlab.automodel.simplicial import AdditiveMoERead

    modules = [m for m in model.modules() if isinstance(m, AdditiveMoERead)]
    if not modules:
        raise ValueError("no installed simplicial branches")
    original = [m.adapter_enabled for m in modules]
    try:
        for module in modules:
            module.adapter_enabled = enabled
        yield
    finally:
        for module, value in zip(modules, original, strict=True):
            module.adapter_enabled = value


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Follow lm-eval's continuation boundary; refuse ambiguous token merging."""
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    prefix = tokenizer.encode(context, add_special_tokens=False)
    combined = tokenizer.encode(context + continuation, add_special_tokens=False)
    if not prefix or combined[:len(prefix)] != prefix or len(combined) == len(prefix):
        raise ValueError("unsupported context/continuation token boundary")
    return prefix, combined[len(prefix):]


def _sync_integer(value: int, *, minimum: bool, synchronize: bool, device: str) -> int:
    """Keep EP/FSDP forward counts identical, including finished/dummy ranks."""
    if not synchronize:
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN if minimum else dist.ReduceOp.MAX)
    return int(tensor.item())


def _check_logits(logits: torch.Tensor, *, synchronize: bool, device: str) -> None:
    finite = int(torch.isfinite(logits).all())
    if not _sync_integer(finite, minimum=True, synchronize=synchronize, device=device):
        raise FloatingPointError("nonfinite benchmark logits on an evaluation rank")


@torch.inference_mode()
def continuation_scores(model: torch.nn.Module, tokenizer, context: str,
                        continuations: list[str], *, max_context: int, device: str = "cuda",
                        synchronize: bool = False) -> dict:
    """Score each candidate's causal token log-probabilities without padding."""
    pairs = [encode_pair(tokenizer, context, choice) for choice in continuations]
    if any(len(prefix) + len(target) - 1 > max_context for prefix, target in pairs):
        raise ValueError("multiple-choice prompt exceeds the registered context budget")
    scores = []
    # MMLU's four answer labels share one prefix and are single tokens.
    fast = all(len(target) == 1 and prefix == pairs[0][0] for prefix, target in pairs)
    if _sync_integer(int(fast), minimum=True, synchronize=synchronize, device=device):
        ids = torch.tensor([pairs[0][0]], device=device, dtype=torch.long)
        logits = model(input_ids=ids, logits_to_keep=1, use_cache=False, output_hidden_states=False).logits[0, -1].float()
        _check_logits(logits, synchronize=synchronize, device=device)
        probabilities = logits.log_softmax(-1)
        scores = [probabilities[target[0]].item() for _, target in pairs]
    else:
        count = _sync_integer(len(pairs), minimum=False, synchronize=synchronize, device=device)
        for index in range(count):
            prefix, target = pairs[index % len(pairs)]
            ids = torch.tensor([prefix + target[:-1]], device=device, dtype=torch.long)
            logits = model(input_ids=ids, logits_to_keep=len(target), use_cache=False,
                           output_hidden_states=False).logits[0].float()
            _check_logits(logits, synchronize=synchronize, device=device)
            labels = torch.tensor(target, device=device, dtype=torch.long)
            score = logits.log_softmax(-1).gather(1, labels[:, None]).sum().item()
            if index < len(pairs):
                scores.append(score)
    # lm-eval normalizes by the choice text length, excluding target_delimiter.
    normalized = [score / len(choice.removeprefix(" ")) for score, choice in zip(scores, continuations, strict=True)]
    return {"loglikelihoods": scores, "character_normalized_loglikelihoods": normalized,
            "prediction": max(range(len(scores)), key=scores.__getitem__),
            "normalized_prediction": max(range(len(scores)), key=normalized.__getitem__),
            "prefix_tokens": [len(p) for p, _ in pairs], "continuation_tokens": [len(t) for _, t in pairs]}


@torch.inference_mode()
def math_completion(model: torch.nn.Module, tokenizer, prompt: str, *, config,
                    eos_ids: set[int], device: str = "cuda", synchronize: bool = False) -> dict:
    """Greedy native-chat decoding; persist the cap and unfinished-reasoning state."""
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    original = len(tokens)
    if not tokens or original >= config.max_context:
        raise ValueError("math prompt exceeds the registered context budget")
    budget = min(config.max_new_tokens, config.max_context - original)
    started = time.monotonic()
    stop_reason = "token_cap"
    finished = False
    loop_budget = _sync_integer(budget, minimum=False, synchronize=synchronize, device=device)
    for index in range(loop_budget):
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        logits = model(input_ids=ids, logits_to_keep=1, use_cache=False, output_hidden_states=False).logits[0, -1].float()
        _check_logits(logits, synchronize=synchronize, device=device)
        if not finished:
            token = logits.argmax(-1).item()
            tokens.append(token)
            if token in eos_ids:
                stop_reason = "eos"
                finished = True
            elif len(tokens) - original >= budget:
                finished = True
        if not _sync_integer(int(not finished), minimum=False, synchronize=synchronize, device=device):
            break
        if index % 64 == 0:
            _sampling.emit("generation_progress", new_tokens=index + 1, seconds=time.monotonic() - started)
    generated = tokens[original:]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    # Remove only the actual terminal EOS token; retain </think> for grading.
    if stop_reason == "eos":
        text = tokenizer.decode(generated[:-1], skip_special_tokens=False)
    return {"completion": text, "generated_token_ids": generated, "prompt_tokens": original,
            "new_tokens": len(generated), "allowed_new_tokens": budget, "stop_reason": stop_reason,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "seconds": time.monotonic() - started}


def _summary(records: list[dict], selected: list[dict], *, complete: bool) -> dict:
    tasks = {}
    for task in dict.fromkeys(row["task"] for row in selected):
        rows = [row for row in records if row["task"] == task]
        metrics = rows[0]["pretrained"]["metrics"] if rows else {}
        tasks[task] = {"completed_pairs": len(rows), "planned_pairs": sum(row["task"] == task for row in selected),
                       "metrics": {metric: _capability.paired_statistics(
                           [row["pretrained"]["metrics"][metric] for row in rows],
                           [row["adapted"]["metrics"][metric] for row in rows]) for metric in metrics}}
        for mode in ("pretrained", "adapted"):
            tasks[task][mode + "_token_cap_count"] = sum(row[mode].get("stop_reason") == "token_cap" for row in rows)
            tasks[task][mode + "_unfinished_thinking_count"] = sum(
                not row[mode].get("final_answer_present", True) for row in rows)
    return {"complete": complete, "pilot_not_full_benchmark": True, "tasks": tasks,
            "completed_pairs": len(records), "planned_pairs": len(selected),
            "interpretation": "Small subsets and capped reasoning cannot establish preservation of full benchmark ability."}


def _provenance(checkpoint: Path, harness: Path, *, environment: str =
                "existing approved DSW venv plus existing pinned lm-eval source; no installs") -> dict:
    import nemo_automodel

    import archlab
    from archlab.automodel.simplicial import UPSTREAM_COMMIT

    metadata = json.loads((checkpoint / "COMPLETE.json").read_text())
    source = Path(archlab.__file__).resolve().parent
    for relative, expected in metadata["contract"]["runtime"]["project_source_sha256"].items():
        if _capability.file_sha256(source / relative) != expected:
            raise ValueError(f"use the checkpoint's immutable training source: {relative}")
    upstream = Path(nemo_automodel.__file__).resolve().parent.parent
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != UPSTREAM_COMMIT or dirty or importlib.metadata.version("lm_eval") != "0.4.13":
        raise RuntimeError("the qualified upstream model and evaluation source pins are required")
    return {"hostname": socket.gethostname(), "python": sys.executable, "gpu": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda, "upstream_commit": revision, "training_source": str(source),
            "harness": str(harness), "checkpoint": str(checkpoint.resolve()), "checkpoint_step": metadata["cursor"],
            "checkpoint_complete_sha256": _capability.file_sha256(checkpoint / "COMPLETE.json"),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "fla-core", "triton", "lm_eval")},
            "environment": environment,
            "entry_source_sha256": {str(p): _capability.file_sha256(p) for p in (
                Path(__file__), Path(_sampling.__file__), Path(_capability.__file__))}}


def prepare_benchmark(args) -> tuple:
    """Share exact selection, prompts and fail-closed budgets across executors."""
    from jinja2 import Environment, StrictUndefined
    from transformers import AutoTokenizer

    recipe = yaml.safe_load(args.recipe.read_text())
    config = _capability.EvaluationConfig(**recipe["evaluation"])
    prompts = yaml.safe_load(args.prompts.read_text())
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    if _capability.file_sha256(args.data / "cases.jsonl") != data_manifest["cases_sha256"]:
        raise ValueError("benchmark bundle hash mismatch")
    with (args.data / "cases.jsonl").open() as stream:
        cases = [json.loads(line) for line in stream]
    selected = _capability.select_cases(cases, limits=recipe["limits"], seed=config.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    template = Environment(undefined=StrictUndefined, autoescape=False)
    rendered = {}
    for row in selected:
        task = row["task"]
        if task in prompts["multiple_choice"]:
            context = template.from_string(prompts["multiple_choice"][task]).render(**row)
            choices = [" " + c for c in (list("ABCD") if task == "mmlu" else row["choices"])]
            if any(len(p) + len(t) - 1 > config.max_context for p, t in
                   [encode_pair(tokenizer, context, choice) for choice in choices]):
                raise ValueError(f"selected question {row['id']} exceeds context; do not resample silently")
            rendered[row["id"]] = (context, choices)
        else:
            text = template.from_string(prompts["math"]["gsm8k" if task == "gsm8k" else "aime"]).render(**row)
            context = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=True,
                                                    reasoning_effort=config.reasoning_effort)
            if len(tokenizer.encode(context, add_special_tokens=False)) + config.max_new_tokens > config.max_context:
                raise ValueError(f"selected math question {row['id']} would receive a smaller token budget")
            rendered[row["id"]] = (context, None)
    grader = _capability.MathGrader(args.harness)
    eos_ids = set(json.loads((args.base / "generation_config.json").read_text())["eos_token_id"])
    return recipe, config, data_manifest, selected, tokenizer, rendered, grader, eos_ids


def main() -> None:
    from archlab.automodel.checkpointing import write_json
    from archlab.automodel.runtime import configure_frozen_gdn_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "checkpoint", "data", "recipe", "prompts", "harness", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true", help="validate all selected prompts without allocating the model")
    args = parser.parse_args()
    recipe, config, data_manifest, selected, tokenizer, rendered, grader, eos_ids = prepare_benchmark(args)
    if args.preflight_only:
        _sampling.emit("benchmark_preflight_passed", selected=len(selected),
                       tasks=recipe["limits"], eos_ids=sorted(eos_ids))
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / "writer.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(16)
        torch.cuda.set_device(0)
        sampling_config = _sampling.SamplingConfig()
        torch.cuda.set_per_process_memory_fraction(sampling_config.gpu_memory_fraction)
        provenance = _provenance(args.checkpoint, args.harness)
        contract = {"config": asdict(config), "recipe": recipe, "provenance": provenance,
                    "data_manifest": data_manifest, "selected_ids": [row["id"] for row in selected],
                    "prompt_file_sha256": _capability.file_sha256(args.prompts), "grader_source_sha256": grader.provenance,
                    "tokenizer_chat_template_sha256": _capability.file_sha256(args.base / "chat_template.jinja"),
                    "eos_ids": sorted(eos_ids)}
        if args.resume:
            if json.loads((args.output / "contract.json").read_text()) != contract:
                raise ValueError("resume contract changed")
        else:
            write_json(args.output / "contract.json", contract)
        path = args.output / "pairs.jsonl"
        if path.exists():
            with path.open() as stream:
                records = [json.loads(line) for line in stream]
        else:
            records = []
        done = {row["id"] for row in records}
        if len(done) != len(records) or not done <= set(contract["selected_ids"]):
            raise ValueError("invalid/duplicate persisted pairs")
        _sampling.emit("benchmark_start", pid=os.getpid(), selected=len(selected), completed=len(records),
                       gdn_runtime=configure_frozen_gdn_runtime())
        write_json(args.output / "summary.json", _summary(records, selected, complete=False))
        model, model_report = _sampling.build_model(args.base, args.checkpoint, sampling_config)
        write_json(args.output / "model.json", model_report)
        versions = {name: (id(p), p._version) for name, p in model.named_parameters()}
        with path.open("a") as stream:
            for index, row in enumerate(selected):
                if row["id"] in done:
                    continue
                context, choices = rendered[row["id"]]
                _sampling.emit("pair_begin", id=row["id"], completed=len(records), total=len(selected))
                result = {"id": row["id"], "task": row["task"], "subject": row["subject"], "target": row["answer"]}
                modes = ("pretrained", "adapted") if index % 2 == 0 else ("adapted", "pretrained")
                for mode in modes:
                    started = time.monotonic()
                    with added_modules(model, enabled=mode == "adapted"):
                        if choices is not None:
                            output = continuation_scores(model, tokenizer, context, choices, max_context=config.max_context)
                            output["metrics"] = {"acc": int(output["prediction"] == row["answer"])}
                            if row["task"] == "arc_challenge":
                                output["metrics"]["acc_norm"] = int(output["normalized_prediction"] == row["answer"])
                        else:
                            output = math_completion(model, tokenizer, context, config=config, eos_ids=eos_ids)
                            output.update(grader.score(row, output["completion"]))
                    output["seconds"] = time.monotonic() - started
                    result[mode] = output
                    _sampling.emit("mode_complete", id=row["id"], mode=mode, metrics=output["metrics"],
                                   seconds=output["seconds"], stop_reason=output.get("stop_reason"))
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                records.append(result)
                write_json(args.output / "summary.json", _summary(records, selected, complete=False))
            if versions != {name: (id(p), p._version) for name, p in model.named_parameters()}:
                raise RuntimeError("evaluation changed a model parameter")
        write_json(args.output / "summary.json", _summary(records, selected, complete=True))
        _sampling.emit("benchmark_complete", output=str(args.output), pairs=len(records))


if __name__ == "__main__":
    main()

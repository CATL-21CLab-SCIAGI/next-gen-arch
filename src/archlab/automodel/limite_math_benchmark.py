"""Bounded native Limite benchmark inference, separate from RL sampling.

The publisher's sampling warpers and native cache are used unchanged. The
qualified architecture-owned decode graph is admitted against eager inference
before the real public benchmark; references never enter model inputs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
from archlab.architectures.limite_decode import GraphDecoderPool
from archlab.architectures.limite_gqa import set_native_decode_gqa
from archlab.architectures.limite_loader import load_model
from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.limite_adapter_common import build_model, runtime_contract
from archlab.evaluation.limite_math import benchmark_seed, score_aime


def implementation_manifest() -> dict:
    root = Path(__file__).parents[1]
    files = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(files=files, aggregate_sha256=digest)


@torch.no_grad()
def qualify_decode(model, prompt_ids, token, *, capacity: int, steps: int = 32) -> dict:
    """Compare complete native eager and graph probability distributions."""
    from archlab.automodel.limite_decode_qualification import distribution_error

    output = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    source = output.past_key_values
    del output
    pool = GraphDecoderPool(model, max_entries=1)
    pool.synchronize()
    decoder = pool.get(source, token, capacity)
    errors = []
    for index in range(steps):
        graph = decoder(token, prompt_ids.shape[1] + index).clone()
        eager = model(input_ids=token, past_key_values=source, use_cache=True,
                      logits_to_keep=1).logits[:, -1]
        errors.append(distribution_error(graph, eager))
    result = dict(max_weighted_error=max(row["weighted_error"] for row in errors),
                  max_kl=max(abs(row["kl"]) for row in errors), steps=steps,
                  prompt_tokens=prompt_ids.shape[1], capacity=capacity,
                  runtime_sliding_window=int(model.config.sliding_window),
                  serialized_sliding_window=getattr(model.config, "_serialized_sliding_window", None),
                  oracle="unmodified publisher eager attention and native DynamicCache")
    result["passed"] = result["max_weighted_error"] < .02 and result["max_kl"] < .001
    return result


def prepare_decode(model, tokenizer, question, plan):
    """Freeze an existing eager protocol when redistributing benchmark work."""
    mode = plan.get("decode_mode", "qualify_graph")
    if mode == "native_eager":
        return None, dict(passed=False, skipped=True, reason="sealed native eager execution")
    if mode != "qualify_graph":
        raise ValueError("unknown benchmark decode mode")
    admission_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        return_dict=False, add_generation_prompt=True)
    admission_ids = torch.tensor([admission_ids], device="cuda")
    probe = admission_ids.repeat(1, 1027 // admission_ids.shape[1] + 1)[:, :1027]
    token = torch.tensor([[tokenizer.encode("42", add_special_tokens=False)[0]]], device="cuda")
    admission = qualify_decode(model, probe, token, capacity=plan.get("qualification_capacity", 33792))
    del admission_ids, probe, token
    gc.collect()
    torch.cuda.empty_cache()
    pool = (GraphDecoderPool(model, max_entries=plan["graph_pool_entries"])
            if admission["passed"] else None)
    return pool, admission


@torch.no_grad()
def sample_math(model, prompt_ids, *, tokenizer, pool, temperature: float, top_p: float,
                max_new_tokens: int, seed: int, eos_token_ids: tuple[int, ...], progress=None) -> dict:
    """One independent native response; no forced EOS or repetition termination."""
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

    if prompt_ids.shape[0] != 1 or max_new_tokens < 1:
        raise ValueError("registered benchmark uses one unpadded independent sample")
    if prompt_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("benchmark prompt exceeds the published context; never truncate")
    warpers = [TemperatureLogitsWarper(temperature), TopPLogitsWarper(top_p)]
    generator = torch.Generator(device=prompt_ids.device).manual_seed(seed)
    if pool is not None:
        pool.synchronize()
    captures_before = pool.capture_seconds if pool is not None else 0.0
    hits_before = pool.hits if pool is not None else 0
    output = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    logits = output.logits[:, -1].float()
    eager_cache = output.past_key_values if pool is None else None
    tokens = prompt_ids.new_empty((1, max_new_tokens))
    decoder = None
    torch.cuda.synchronize()
    started = time.perf_counter()
    reason = "length"
    for index in range(max_new_tokens):
        scores = logits
        for warper in warpers:
            scores = warper(None, scores)
        token = torch.multinomial(scores.softmax(-1), num_samples=1, generator=generator)
        tokens[:, index:index + 1].copy_(token)
        if progress is not None and (index + 1) % 1024 == 0:
            progress(dict(generated_tokens=index + 1, seconds=time.perf_counter() - started))
        if int(token[0, 0]) in eos_token_ids:
            reason = "eos"
            break
        if index + 1 == max_new_tokens:
            break
        if pool is None:
            logits = model(input_ids=token, past_key_values=eager_cache, use_cache=True,
                           logits_to_keep=1).logits[:, -1].float()
            continue
        if decoder is None:
            decoder = pool.get(output.past_key_values, token, prompt_ids.shape[1] + max_new_tokens)
            del output
        logits = decoder(token, prompt_ids.shape[1] + index).float()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    generated = tokens[0, :index + 1].tolist()
    text_tokens = generated[:-1] if reason == "eos" else generated
    return dict(generated_ids=generated, completion=tokenizer.decode(text_tokens, skip_special_tokens=False),
                generated_tokens=len(generated), finish_reason=reason, seconds=seconds,
                graph_capture_seconds=pool.capture_seconds - captures_before if pool is not None else 0.0,
                graph_pool_hits=pool.hits - hits_before if pool is not None else 0,
                seed=seed, decoding="qualified-native-graph" if pool is not None else "publisher-eager-native-cache")


def read_records(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--checkpoint-cache", type=Path, required=True)
    parser.add_argument("--only-models", nargs="+")
    parser.add_argument("--case-limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        raise ValueError("invalid benchmark shard")
    plan = json.loads(args.plan.read_text())
    if plan["format"] != "archlab-limite-math-evaluation-v1":
        raise ValueError("unsupported evaluation plan")
    bundle = Path(plan["bundle"])
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    if (sha256_file(bundle / "MANIFEST.json") != plan["manifest_sha256"]
            or sha256_file(bundle / "cases.jsonl") != manifest["cases_sha256"]):
        raise ValueError("sealed public benchmark cases changed")
    all_cases = read_records(bundle / "cases.jsonl")
    cases = [row for index, row in enumerate(all_cases) if index % args.shards == args.shard]
    if args.case_limit is not None:
        if args.case_limit < 1:
            raise ValueError("case limit must be positive")
        cases = cases[:args.case_limit]
    implementation = implementation_manifest()
    if implementation["aggregate_sha256"] != plan["implementation_sha256"]:
        raise ValueError("benchmark/model source differs from the sealed plan")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(plan["gpu_memory_limit_gib"] * 2**30 / total)
    torch.manual_seed(plan["sampling"]["seed"])
    tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"], local_files_only=True,
                                             trust_remote_code=False)
    generation_defaults = json.loads((Path(plan["tokenizer"]) / "generation_config.json").read_text())
    endings = tuple(plan['sampling']['eos_token_ids'])
    if (set(endings) != {tokenizer.eos_token_id, generation_defaults['eos_token_id']}
            or sha256_file(Path(plan["tokenizer"]) / "chat_template.jinja") != plan["chat_template_sha256"]):
        raise ValueError("publisher tokenizer/template changed")
    output_root = Path(plan["output"])
    selected_models = [row for row in plan["models"]
                       if args.only_models is None or row["name"] in args.only_models]
    if not selected_models:
        raise ValueError("no registered models selected")
    for sample_index in range(plan["sampling"]["samples_per_problem"]):
        for phase in selected_models:
            destination = output_root / phase["name"] / f"shard-{args.shard}"
            destination.mkdir(parents=True, exist_ok=True)
            records_path = destination / "records.jsonl"
            records = read_records(records_path)
            completed = {(row["problem_id"], row["sample_index"]) for row in records}
            pending = [row for row in cases if (row["id"], sample_index) not in completed]
            if not pending:
                continue
            receipt = (json.loads((Path(phase["checkpoint"]) / "COMPLETE.json").read_text())
                       if phase.get("checkpoint") else None)
            if receipt is not None and sha256_file(Path(phase["checkpoint"]) / "COMPLETE.json") != phase["checkpoint_receipt_sha256"]:
                raise ValueError("registered model checkpoint changed")
            atomic_write_json(destination / "CURRENT.json", dict(model=phase["name"], sample_index=sample_index,
                              checkpoint=phase.get("checkpoint"), loading=True, pid=os.getpid()))
            loaded = time.perf_counter()
            if phase["variant"] == "base":
                model = load_model(plan["model"], attn_implementation="sdpa", device_map="cuda")
            else:
                model = build_model(plan["model"], phase["variant"], "cuda", phase["checkpoint"],
                                    checkpoint_cache=args.checkpoint_cache)
                enable_runtime_sequence_attention(model)
            model.requires_grad_(False)
            model.eval()
            set_native_decode_gqa(model)
            if model.config.max_position_embeddings != plan["sampling"]["context_limit"]:
                raise ValueError("model's native context differs from the registered benchmark")
            pool, admission = prepare_decode(model, tokenizer, cases[0]["question"], plan)
            run = dict(format="archlab-limite-math-benchmark-run-v1", phase=phase,
                       checkpoint_receipt=receipt, sampling=plan["sampling"],
                       shard=args.shard, shards=args.shards, problems=len(cases),
                       case_limit=args.case_limit, training_updates_enabled=False,
                       optimizer_created=False, source_implementation_sha256=implementation["aggregate_sha256"],
                       runtime=runtime_contract(), admission=admission,
                       decoding="qualified-native-graph" if pool is not None else "publisher-eager-native-cache",
                       loaded_seconds=time.perf_counter() - loaded, gpu_allocator_limit_gib=plan["gpu_memory_limit_gib"])
            atomic_write_json(destination / "RUN.json", run)
            print(json.dumps(dict(phase=phase["name"], shard=args.shard, admission=admission,
                                  loaded_seconds=run["loaded_seconds"])), flush=True)
            for case in pending:
                if (output_root / "STOP").exists():
                    raise RuntimeError("benchmark stop requested; completed samples retained")
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": case["question"]}], tokenize=True,
                    return_dict=False, add_generation_prompt=True)
                ids = torch.tensor([prompt], device="cuda")
                result = sample_math(
                    model, ids, tokenizer=tokenizer, pool=pool,
                    temperature=plan["sampling"]["temperature"], top_p=plan["sampling"]["top_p"],
                    max_new_tokens=plan["sampling"]["max_new_tokens"],
                    seed=benchmark_seed(plan["sampling"]["seed"], case["id"], sample_index),
                    eos_token_ids=endings)
                result.update(score_aime(result["completion"], case["answer"], result["finish_reason"]))
                result.update(problem_id=case["id"], task=case["task"], sample_index=sample_index,
                              prompt_tokens=len(prompt), prompt_ids_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                              source_implementation_sha256=implementation["aggregate_sha256"],
                              expected_answer=case["answer"], model=phase["name"])
                with records_path.open("a") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                atomic_write_json(destination / "CURRENT.json", dict(
                    model=phase["name"], problem_id=case["id"], sample_index=sample_index,
                    finished_reason=result["finish_reason"], generated_tokens=result["generated_tokens"],
                    correct=result["correct"], seconds=result["seconds"],
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, pid=os.getpid()))
                print(json.dumps({key: result[key] for key in (
                    "model", "problem_id", "sample_index", "correct", "generated_tokens", "finish_reason", "seconds")}), flush=True)
            del pool, model, ids
            gc.collect()
            torch.cuda.empty_cache()
    atomic_write_json(output_root / f"SHARD-{args.shard}-COMPLETE.json", dict(
        shard=args.shard, models=[phase["name"] for phase in selected_models], problems=len(cases),
        samples_per_problem=plan["sampling"]["samples_per_problem"], case_limit=args.case_limit,
        implementation_sha256=implementation["aggregate_sha256"]))


if __name__ == "__main__":
    main()

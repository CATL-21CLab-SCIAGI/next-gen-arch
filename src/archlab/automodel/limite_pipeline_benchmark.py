"""Run eval_pipeline AIME26 with the exact native Limite adapter checkpoints.

Upstream owns prompts, answer extraction, and pass@1/pass@n aggregation. This
adapter owns checkpoint identity, full-context native inference, stable seeds,
and durable full responses. Strict math verification is reported alongside the
upstream extractor, which also accepts guesses inside unfinished reasoning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import torch

from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
from archlab.architectures.limite_gqa import set_native_decode_gqa
from archlab.architectures.limite_loader import load_model
from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.limite_adapter_common import build_model, runtime_contract
from archlab.automodel.limite_math_benchmark import (
    implementation_manifest,
    prepare_decode,
    read_records,
    sample_math,
)
from archlab.automodel.limite_math_queue import validate_plan, verify_shard
from archlab.evaluation.limite_math import benchmark_seed, score_aime
from archlab.evaluation.limite_pipeline import SealedDataset, completion_budget, load_pipeline


class NativeRunner:
    """Injected runner for upstream's HF/custom-model evaluation route."""

    def __init__(self, model, tokenizer, pool, pipeline, plan, phase, cases, destination):
        self.model, self.tokenizer, self.pool = model, tokenizer, pool
        self.pipeline, self.plan, self.phase = pipeline, plan, phase
        self.destination = destination
        self.path = destination / "records.jsonl"
        self.cases = {row["question"]: row for row in cases}
        if len(self.cases) != len(cases):
            raise ValueError("duplicate question text")
        self.calls = Counter()
        self.existing = {}
        for row in read_records(self.path):
            key = (row["problem_id"], row["sample_index"])
            if key in self.existing or row["source_implementation_sha256"] != plan["implementation_sha256"]:
                raise ValueError("duplicate response or changed implementation")
            if row["seed"] != benchmark_seed(plan["sampling"]["seed"], *key):
                raise ValueError("saved response seed changed")
            self.existing[key] = row

    @torch.no_grad()
    def chat_generate(self, messages_list, *, max_new_tokens, temperature, enable_thinking,
                      top_p, top_k=None, presence_penalty=None, repetition_penalty=None):
        sampling = self.plan["sampling"]
        if (temperature != sampling["temperature"] or top_p != sampling["top_p"]
                or max_new_tokens != sampling["max_new_tokens"] or not enable_thinking
                or top_k is not None or presence_penalty is not None or repetition_penalty is not None):
            raise ValueError("eval_pipeline changed the registered inference contract")
        answers = []
        for messages in messages_list:
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("expected a single unmodified benchmark question")
            case = self.cases[messages[0]["content"]]
            sample = self.calls[case["id"]]
            self.calls[case["id"]] += 1
            if sample >= sampling["samples_per_problem"]:
                raise ValueError("upstream requested extra samples")
            key = (case["id"], sample)
            if key in self.existing:
                answers.append(self.existing[key]["completion"])
                continue
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                       add_generation_prompt=True, enable_thinking=True)
            budget = completion_budget(len(prompt), max_new_tokens, sampling["context_limit"])
            ids = torch.tensor([prompt], device="cuda")
            current = dict(model=self.phase["name"], problem_id=case["id"], sample_index=sample,
                           prompt_tokens=len(prompt), completion_budget=budget, started_at=time.time(),
                           generated_tokens=0, pid=os.getpid(), status="generating")

            def progress(values, current=current):
                atomic_write_json(self.destination / "CURRENT.json", dict(current, **values, time=time.time()))

            progress({})
            result = sample_math(self.model, ids, tokenizer=self.tokenizer, pool=self.pool,
                                 temperature=temperature, top_p=top_p, max_new_tokens=budget,
                                 seed=benchmark_seed(sampling["seed"], *key),
                                 eos_token_ids=tuple(sampling["eos_token_ids"]), progress=progress)
            result.update(score_aime(result["completion"], case["answer"], result["finish_reason"]))
            predicted = self.pipeline.extract_final_answer_robust(result["completion"])
            result.update(pipeline_answer=predicted,
                          pipeline_correct=bool(predicted is not None and self.pipeline.is_math_correct(predicted, case["answer"])),
                          problem_id=case["id"], task=case["task"], sample_index=sample,
                          expected_answer=case["answer"], model=self.phase["name"],
                          prompt_tokens=len(prompt), completion_budget=budget,
                          budget_policy=sampling["budget_policy"],
                          prompt_ids_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                          source_implementation_sha256=self.plan["implementation_sha256"],
                          eval_pipeline_revision=self.plan["eval_pipeline"]["upstream_revision"])
            with self.path.open("a") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.existing[key] = result
            atomic_write_json(self.destination / "CURRENT.json", dict(current, status="recorded", time=time.time(),
                              generated_tokens=result["generated_tokens"], seconds=result["seconds"],
                              finish_reason=result["finish_reason"], correct=result["correct"],
                              peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30))
            print(json.dumps({k: result[k] for k in ("model", "problem_id", "sample_index", "correct",
                             "pipeline_correct", "generated_tokens", "finish_reason", "seconds")}), flush=True)
            answers.append(result["completion"])
        return answers


def load_phase_model(plan, phase, *, checkpoint_cache, device="cuda"):
    """Load the registered weights without silently evaluating the publisher parent."""
    variant = phase["variant"]
    checkpoint = phase.get("checkpoint")
    if checkpoint is not None:
        receipt = Path(checkpoint) / "COMPLETE.json"
        if sha256_file(receipt) != phase["checkpoint_receipt_sha256"]:
            raise ValueError("registered model checkpoint changed")
    if variant == "base":
        if checkpoint is not None:
            raise ValueError("publisher base evaluation cannot ignore a checkpoint")
        return load_model(plan["model"], attn_implementation="sdpa", device_map=device)
    if checkpoint is None:
        raise ValueError("trained model evaluation requires its registered checkpoint")
    if variant == "native":
        from archlab.automodel.limite_native_checkpoint import build_native_model

        return build_native_model(plan["model"], device, checkpoint,
                                  checkpoint_cache=checkpoint_cache)
    model = build_model(plan["model"], variant, device, checkpoint,
                        checkpoint_cache=checkpoint_cache)
    enable_runtime_sequence_attention(model)
    return model


def main():
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--shard", required=True, type=int)
    parser.add_argument("--shards", default=16, type=int)
    parser.add_argument("--checkpoint-cache", required=True, type=Path)
    args = parser.parse_args()
    if args.shards != 16 or not 0 <= args.shard < args.shards:
        raise ValueError("registered evaluation uses sixteen independent GPU workers")
    plan = json.loads(args.plan.read_text())
    all_cases = validate_plan(plan)
    if implementation_manifest()["aggregate_sha256"] != plan["implementation_sha256"]:
        raise ValueError("running executor differs from the sealed source")
    cases = all_cases[args.shard::args.shards]
    phase = plan["models"][0]
    destination = Path(plan["output"]) / phase["name"] / f"shard-{args.shard}"
    destination.mkdir(parents=True, exist_ok=True)
    pipeline = load_pipeline(plan)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(plan["gpu_memory_limit_gib"] * 2**30 / total)
    tokenizer = AutoTokenizer.from_pretrained(plan["tokenizer"], local_files_only=True, trust_remote_code=False)
    if sha256_file(Path(plan["tokenizer"]) / "chat_template.jinja") != plan["chat_template_sha256"]:
        raise ValueError("publisher chat template changed")
    defaults = json.loads((Path(plan["tokenizer"]) / "generation_config.json").read_text())
    if set(plan["sampling"]["eos_token_ids"]) != {tokenizer.eos_token_id, defaults["eos_token_id"]}:
        raise ValueError("publisher EOS contract changed")
    model = load_phase_model(plan, phase, checkpoint_cache=args.checkpoint_cache)
    model.requires_grad_(False)
    model.eval()
    if model.config.max_position_embeddings != plan["sampling"]["context_limit"]:
        raise ValueError("model context does not match the registered limit")
    set_native_decode_gqa(model, backend=plan.get("native_gqa_backend", "sdpa"))
    pool, admission = prepare_decode(model, tokenizer, cases[0]["question"], plan)
    atomic_write_json(destination / "RUN.json", dict(
        format="archlab-eval-pipeline-run-v1", phase=phase, sampling=plan["sampling"],
        eval_pipeline=plan["eval_pipeline"], runtime=runtime_contract(), admission=admission,
        inference_engine=("native full-weight Limite" if phase["variant"] == "native"
                          else "native Limite with architecture-owned inserted attention"),
        training_updates_enabled=False, shard=args.shard, shards=args.shards,
        implementation_sha256=plan["implementation_sha256"]))
    runner = NativeRunner(model, tokenizer, pool, pipeline, plan, phase, cases, destination)
    result = pipeline.evaluate(
        model_path=phase.get("checkpoint", plan["model"]), output_path=str(destination / "PIPELINE_RESULT.json"),
        runner=runner, dataset=SealedDataset(cases), model_backend="hf", batch_size=1,
        enable_thinking=True, use_boxed_hint=False, temperature=plan["sampling"]["temperature"],
        top_p=plan["sampling"]["top_p"], top_k=-1, max_tokens=plan["sampling"]["max_new_tokens"],
        num_samples=plan["sampling"]["samples_per_problem"])
    rows = read_records(runner.path)
    if len(rows) != len(cases) * plan["sampling"]["samples_per_problem"]:
        raise ValueError("upstream did not produce every expected sample")
    measured = 100 * sum(row["pipeline_correct"] for row in rows) / len(rows)
    if abs(measured - result["pass_at_1"]) > 1e-9:
        raise ValueError("pipeline aggregation disagrees with the full recorded responses")
    atomic_write_json(Path(plan["output"]) / f"SHARD-{args.shard}-COMPLETE.json", dict(
        shard=args.shard, models=[phase["name"]], problems=len(cases),
        samples_per_problem=plan["sampling"]["samples_per_problem"], case_limit=None,
        implementation_sha256=plan["implementation_sha256"]))
    verify_shard(plan, all_cases, args.shard)


if __name__ == "__main__":
    main()

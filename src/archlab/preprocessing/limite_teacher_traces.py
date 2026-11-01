"""Verified teacher solutions for matched Limite math distillation.

``generate`` samples a Limite export through the archlab vLLM boundary. Prompts are
the question turns of a sealed RL split, rendered by the teacher's own chat template;
each (problem, sample) has a shard-independent seed, and a shard resumes from its own
records. Every sample is kept and graded by the shared symbolic ``math_reward``;
``--record-logprobs`` also keeps each sampled token's raw serving log-probability.
``prepare`` selects correct samples with closed reasoning that end with the teacher's
own EOS and writes assistant-only SFT rows (unshifted target masks, supervised EOS).

usage: python -m archlab.preprocessing.limite_teacher_traces generate|prepare ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.evaluation.limite_math import benchmark_seed
from archlab.evaluation.math_reward import math_reward, reasoning_complete

GENERATION_EOS = 151645
TOKENIZER_EOS = 151643
# RL142 ends its responses with <|endoftext|>; neither its RL rollouts nor its sampled
# traces emit <|im_end|>. Evaluation stops on both tokens.
TEACHER_EOS = TOKENIZER_EOS
# The Limite rollout serving contract: inserted-prelude caches read FlashAttention-2 metadata,
# and every sampled request runs its own prefill.
SERVING = dict(attention_config=dict(backend="FLASH_ATTN", flash_attn_version=2),
               enable_prefix_caching=False)
_USER, _ASSISTANT = "<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n"


def question(prompt: str) -> str:
    """Recover the single user turn of a natively rendered RL prompt."""
    if prompt.count(_USER) != 1 or not prompt.endswith(_ASSISTANT):
        raise ValueError("expected one user turn followed by the assistant generation prompt")
    return prompt.split(_USER, 1)[1].removesuffix(_ASSISTANT)


def problems(split: Path, tokenizer) -> list[dict]:
    rows = []
    for line in split.read_text().splitlines():
        row = json.loads(line)
        ids = tokenizer.apply_chat_template([{"role": "user", "content": question(row["prompt"])}],
                                            tokenize=True, add_generation_prompt=True, return_dict=False)
        rows.append(dict(uuid=row["uuid"], problem_sha256=row["problem_sha256"],
                         expected_answer=row["expected_answer"], prompt_ids=list(ids)))
    if len({row["uuid"] for row in rows}) != len(rows):
        raise ValueError("split problem UUIDs must be unique")
    return rows


def grade(text: str, answer: str, finish_reason: str, last_token: int | None) -> dict:
    natural = finish_reason == "stop" and last_token in (GENERATION_EOS, TOKENIZER_EOS)
    return dict(correct=bool(math_reward(text, answer)), reasoning_complete=reasoning_complete(text),
                natural_eos=natural, stop_token=last_token if natural else None)


def _records(path: Path) -> dict:
    if not path.exists():
        return {}
    records = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        records[(row["uuid"], row["sample"])] = row
    return records


def completed(traces: Path, split: str) -> tuple[list[Path], list[dict]]:
    """Records of every completed shard of a split, after checking each shard's checksum."""
    receipts = sorted(traces.glob(f"{split}-shard-*.complete.json"))
    if not receipts:
        raise ValueError("no completed teacher shards for split " + split)
    records = []
    for path in receipts:
        receipt = json.loads(path.read_text())
        source = traces / receipt["file"]
        if sha256_file(source) != receipt["file_sha256"]:
            raise ValueError("teacher shard changed after completion: " + source.name)
        records.extend(_records(source).values())
    return receipts, records


def generate(args) -> dict:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from archlab.training.serving import register

    tokenizer = AutoTokenizer.from_pretrained(args.export)
    rows = problems(args.split, tokenizer)[args.shard::args.shards]
    output = args.output / f"{args.name}-shard-{args.shard:03d}-of-{args.shards:03d}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _records(output)
    pending = [(row, sample) for row in rows for sample in range(args.samples)
               if (row["uuid"], sample) not in done]
    register()
    llm = LLM(model=str(args.export), tokenizer=str(args.export), dtype="bfloat16",
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=True,
              seed=args.seed, logprobs_mode="raw_logprobs", **SERVING)
    with output.open("a") as stream:
        for start in range(0, len(pending), args.chunk):
            chunk = pending[start:start + args.chunk]
            params = [SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=-1,
                                     max_tokens=args.max_tokens,
                                     seed=benchmark_seed(args.seed, row["uuid"], sample),
                                     stop_token_ids=[GENERATION_EOS, TOKENIZER_EOS],
                                     skip_special_tokens=True,
                                     logprobs=0 if args.record_logprobs else None)
                      for row, sample in chunk]
            outputs = llm.generate([{"prompt_token_ids": row["prompt_ids"]} for row, _ in chunk],
                                   params, use_tqdm=False)
            for (row, sample), result in zip(chunk, outputs, strict=True):
                completion = result.outputs[0]
                ids = list(completion.token_ids)
                record = dict(split=args.name, uuid=row["uuid"], problem_sha256=row["problem_sha256"],
                              sample=sample, seed=benchmark_seed(args.seed, row["uuid"], sample),
                              prompt_ids=row["prompt_ids"], response_ids=ids,
                              finish_reason=completion.finish_reason, text=completion.text,
                              **grade(completion.text, row["expected_answer"], completion.finish_reason,
                                      ids[-1] if ids else None))
                if args.record_logprobs:
                    record["response_logprobs"] = [
                        step[token].logprob for token, step in zip(ids, completion.logprobs, strict=True)]
                stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    records = _records(output)
    expected = {(row["uuid"], sample) for row in rows for sample in range(args.samples)}
    if set(records) != expected:
        raise ValueError("shard records differ from its problem/sample assignment")
    summary = dict(format="archlab-limite-teacher-traces-v1", export=str(args.export),
                   export_receipt_sha256=sha256_file(args.export / "EXPORT.json"),
                   split=str(args.split), split_sha256=sha256_file(args.split), name=args.name,
                   shard=args.shard, shards=args.shards, samples=args.samples, seed=args.seed,
                   sampling=dict(temperature=args.temperature, top_p=args.top_p, top_k=-1,
                                 max_tokens=args.max_tokens, stop_token_ids=[GENERATION_EOS, TOKENIZER_EOS]),
                   serving=SERVING, response_logprobs="raw" if args.record_logprobs else None,
                   records=len(records), correct=sum(r["correct"] for r in records.values()),
                   natural_eos=sum(r["natural_eos"] for r in records.values()),
                   response_tokens=sum(len(r["response_ids"]) for r in records.values()),
                   file=output.name, file_sha256=sha256_file(output))
    atomic_write_json(output.with_suffix(".complete.json"), summary)
    return summary


def supervised(record: dict, max_length: int) -> dict | None:
    """One assistant-only row, or None if the sample is not usable distillation data."""
    ids = record["prompt_ids"] + record["response_ids"]
    if not (record["correct"] and record["reasoning_complete"] and record["natural_eos"]
            and record["stop_token"] == TEACHER_EOS and len(ids) <= max_length):
        return None
    mask = [0] * len(record["prompt_ids"]) + [1] * len(record["response_ids"])
    return dict(uuid=record["uuid"], problem_sha256=record["problem_sha256"], sample=record["sample"],
                input_ids=ids, loss_mask=mask, attention_mask=[1] * len(ids),
                position_ids=list(range(len(ids))))


def prepare(args) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.mkdir(parents=True, exist_ok=False)
    splits, index = {}, []
    for name, split in (("train", "train"), ("validation", "heldout")):
        receipts, records = completed(args.traces, split)
        rows = []
        for record in records:
            row = supervised(record, args.max_length)
            index.append(dict(split=split, uuid=record["uuid"], sample=record["sample"],
                              selected=row is not None, correct=record["correct"],
                              natural_eos=record["natural_eos"],
                              response_tokens=len(record["response_ids"])))
            if row is not None:
                rows.append(row)
        rows.sort(key=lambda row: (row["uuid"], row["sample"]))
        # Small row groups let every training rank stream ~36k-token rows.
        pq.write_table(pa.Table.from_pylist(rows), args.output / f"{name}.parquet", row_group_size=256)
        splits[name] = dict(name=f"{name}.parquet", rows=len(rows),
                            sha256=sha256_file(args.output / f"{name}.parquet"),
                            supervised_tokens=sum(sum(row["loss_mask"]) for row in rows),
                            source_receipts={p.name: sha256_file(p) for p in receipts})
    (args.output / "index.jsonl").write_text("".join(json.dumps(row) + "\n" for row in index))
    splits["index"] = dict(name="index.jsonl", sha256=sha256_file(args.output / "index.jsonl"))
    train, held = ({*pq.read_table(args.output / f"{name}.parquet", columns=["problem_sha256"])
                    .column(0).to_pylist()} for name in ("train", "validation"))
    if train & held:
        raise ValueError("teacher train and validation problems overlap")
    template = args.tokenizer / "chat_template.jinja"
    files = {name: sha256_file(args.tokenizer / name)
             for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    manifest = dict(
        format="archlab.prepared-agentic-sft.v1", files=splits,
        tokenization=dict(generation_eos_token_id=TEACHER_EOS, tokenizer_eos_token_id=TOKENIZER_EOS,
                          loss_mask_alignment="unshifted-target-token", max_length=args.max_length,
                          files=files, chat_template_sha256=sha256_file(template),
                          loaded_chat_template_sha256=sha256_file(template)),
        supervision=dict(kind="teacher_sequence_distillation",
                         selection="symbolically correct; closed reasoning; natural teacher EOS",
                         objective="assistant_ce"),
        selection_sha256=hashlib.sha256((args.output / "index.jsonl").read_bytes()).hexdigest(),
    )
    atomic_write_json(args.output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--export", type=Path, required=True)
    gen.add_argument("--split", type=Path, required=True)
    gen.add_argument("--name", choices=("train", "heldout"), required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--shard", type=int, required=True)
    gen.add_argument("--shards", type=int, required=True)
    gen.add_argument("--samples", type=int, default=8)
    gen.add_argument("--seed", type=int, default=20261101)
    gen.add_argument("--temperature", type=float, default=0.6)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--max-tokens", type=int, default=32768)
    gen.add_argument("--max-model-len", type=int, default=36864)
    gen.add_argument("--max-num-seqs", type=int, default=256)
    gen.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    gen.add_argument("--chunk", type=int, default=1024)
    gen.add_argument("--record-logprobs", action="store_true",
                     help="keep each sampled token's raw vLLM log-probability for native replay admission")
    prep = commands.add_parser("prepare")
    prep.add_argument("--traces", type=Path, required=True)
    prep.add_argument("--tokenizer", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--max-length", type=int, default=36864)
    args = parser.parse_args(argv)
    if args.command == "generate" and not 0 <= args.shard < args.shards:
        raise ValueError("shard must be in [0, shards)")
    result = generate(args) if args.command == "generate" else prepare(args)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fixed-grid evaluation through pinned Uni-Agent/MiMo and one vLLM engine.

Upstream owns agent execution, sandbox lifecycle, transcript capture and grading.
This module supplies token transport and an auditable, resumable measurement grid.
It can run as a file against an immutable upstream snapshot or as a module.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

SCHEMA = "archlab-agentic-evaluation-v1"
SEEDS = (101, 211, 307, 401, 503, 601, 701, 809)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_seed(task, seed):
    """Independent, deterministic random stream per task/sample, shared by checkpoints."""
    return int(hashlib.sha256(f"{task}\0{seed}".encode()).hexdigest()[:8], 16)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def completed_records(records, tasks, seeds):
    expected = {(task, seed) for task in tasks for seed in seeds}
    completed = {}
    for row in records:
        key = (row["task"], row["seed"])
        if key not in expected:
            raise ValueError(f"Episode outside the declared grid: {key}")
        if row.get("status") == "scored":
            if key in completed:
                raise ValueError(f"Duplicate scored episode: {key}")
            if row.get("reward") not in (0.0, 1.0):
                raise ValueError(f"Expected binary executable reward: {key}")
            completed[key] = row
    return expected, completed


def summarize(records, tasks, seeds, *, bootstrap=2000):
    """Missing infrastructure results never silently become zero or shrink the denominator."""
    expected, completed = completed_records(records, tasks, seeds)
    per_task = {task: [completed[(task, seed)]["reward"] for seed in seeds
                       if (task, seed) in completed] for task in tasks}
    full_tasks = {task: sum(values) / len(seeds) for task, values in per_task.items()
                  if len(values) == len(seeds)}
    complete = len(completed) == len(expected)
    result = dict(schema=SCHEMA, complete=complete, preliminary=len(tasks) < 48 or len(seeds) < 8,
                  expected_samples=len(expected),
                  scored_samples=len(completed), missing_samples=len(expected) - len(completed),
                  task_count=len(tasks), complete_task_count=len(full_tasks), samples_per_task=len(seeds),
                  pass_at_1=None, success_at_k=None, task_bootstrap_95ci=None,
                  provisional_complete_task_pass_at_1=(sum(full_tasks.values()) / len(full_tasks)
                                                       if full_tasks else None),
                  per_task={key: dict(scored=len(value), successes=sum(value), expected=len(seeds))
                            for key, value in per_task.items()},
                  infrastructure_attempts=sum(row.get("status") == "missing" for row in records),
                  terminations=dict(Counter(row.get("termination", "unknown") for row in completed.values())),
                  cap_reasons=dict(Counter(row.get("cap_reason", "none") for row in completed.values())),
                  length_limited_calls=sum(row.get("length_limited_calls", 0) for row in completed.values()),
                  generation_calls=sum(row.get("generation_calls", 0) for row in completed.values()),
                  generated_tokens=sum(row.get("generated_tokens", 0) for row in completed.values()),
                  episode_seconds=sum(row.get("seconds", 0) for row in completed.values()))
    if complete:
        import numpy as np
        values = np.asarray(list(full_tasks.values()), dtype=float)
        rng = np.random.default_rng(20261017)
        draws = rng.choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1)
        result.update(pass_at_1=float(values.mean()), success_at_k=sum(any(x) for x in per_task.values()) / len(tasks),
                      task_bootstrap_95ci=[float(x) for x in np.quantile(draws, [0.025, 0.975])])
    return result


def bind_invocation(path, invocation):
    """Permit deterministic sample expansion only; model/data/runtime remain fixed."""
    path = Path(path)
    normalized = json.loads(json.dumps(invocation))
    if path.exists():
        old = json.loads(path.read_text())
        old_seeds, new_seeds = old.get("seeds", []), normalized.get("seeds", [])
        old_fixed = {key: value for key, value in old.items() if key not in {"seeds", "scope"}}
        new_fixed = {key: value for key, value in normalized.items() if key not in {"seeds", "scope"}}
        if old_fixed != new_fixed or new_seeds[:len(old_seeds)] != old_seeds:
            raise ValueError("Output contains a different evaluation contract; use a new directory")
        if old != normalized:
            atomic_json(path.with_name(f"invocation-{sha(path)}.json"), old)
    atomic_json(path, normalized)


def paired_difference(left, right, tasks, seeds, *, bootstrap=2000):
    """Task-paired uncertainty; compare only identical, complete evaluation grids."""
    import numpy as np
    expected, a = completed_records(left, tasks, seeds)
    _, b = completed_records(right, tasks, seeds)
    if set(a) != expected or set(b) != expected:
        raise ValueError("Paired comparison requires both complete grids")
    values = np.asarray([sum(b[(task, seed)]["reward"] - a[(task, seed)]["reward"]
                              for seed in seeds) / len(seeds) for task in tasks])
    draws = np.random.default_rng(20261017).choice(values, size=(bootstrap, len(values)), replace=True).mean(axis=1)
    return dict(delta_pass_at_1=float(values.mean()), task_paired_95ci=[float(x) for x in np.quantile(draws, [.025, .975])])


def error_outcome(error):
    # The upstream wall timer also covers resource waits; without proof of active
    # policy time, a wall timeout cannot be classified as a model failure.
    wall_limit = isinstance(error, (TimeoutError, asyncio.TimeoutError)) and "session exceeded wall timeout" in str(error)
    return {"status": "missing", "reward": None,
            "termination": "wall_budget_unclassified" if wall_limit else "infrastructure_or_unclassified",
            "error_type": type(error).__name__}


def policy_cap_reason(reward, turns):
    """A verifier result cannot turn a transport failure into a policy outcome."""
    detail = str(reward.get("result", "")).lower()
    context_limit = any(marker in detail for marker in (
        "maximum context length", "max_model_len", "trajectory capacity", "context length exceeded"))
    if str(reward.get("agent_status")) == "ModelQueryError" and not context_limit:
        raise RuntimeError("Ungradable model backend failure")
    if reward.get("agent_completed"):
        return "none"
    if turns >= 300:
        return "turn_limit"
    return "context_limit" if context_limit else "unfinished_other"


class VLLMBackend:
    """Token transport only; seed receipts demonstrate that the full grid is sampled."""

    def __init__(self, args, output):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        self.output, self.turns, self.stats, self.sequence = output, {}, {}, 0
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(args.model), skip_tokenizer_init=True, dtype="bfloat16", tensor_parallel_size=1,
            language_model_only=True, max_model_len=262144, max_num_seqs=max(8, args.concurrency),
            max_num_batched_tokens=32768, enable_chunked_prefill=True,
            gpu_memory_utilization=args.gpu_memory, enable_prefix_caching=True, seed=0))

    def reset(self, seed):
        self.turns[seed] = 0
        self.stats[seed] = dict(generated_tokens=0, maximum_prompt_tokens=0, generation_calls=0,
                               length_limited_calls=0)

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, video_data=None):
        from verl.workers.rollout.replica import TokenOutput
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        if image_data or video_data:
            raise ValueError("The held-out code grid is text-only")
        base_seed = sampling_params.get("seed")
        if base_seed not in self.turns:
            raise ValueError("Gateway lost the fixed sample seed")
        turn = self.turns[base_seed]
        self.turns[base_seed] += 1
        effective_seed = sample_seed(str(base_seed), turn)
        params = dict(sampling_params, seed=effective_seed, logprobs=1)
        self.sequence += 1
        unique_id = f"eval-{self.sequence}"
        output = None
        async for result in self.engine.generate(TokensPrompt(prompt_token_ids=prompt_ids),
                                                 SamplingParams(**params), request_id=unique_id):
            output = result
        if output is None or len(output.outputs) != 1:
            raise RuntimeError("vLLM returned no unique generation")
        generated = output.outputs[0]
        logprobs = [float(entries[token].logprob) for token, entries in
                    zip(generated.token_ids, generated.logprobs, strict=True)]
        stats = self.stats[base_seed]
        stats["generated_tokens"] += len(generated.token_ids)
        stats["maximum_prompt_tokens"] = max(stats["maximum_prompt_tokens"], len(prompt_ids))
        stats["generation_calls"] += 1
        stats["length_limited_calls"] += int(generated.finish_reason == "length")
        with (self.output / "generation-receipts.jsonl").open("a") as handle:
            handle.write(json.dumps(dict(sample_seed=base_seed, turn=turn, effective_seed=effective_seed,
                                         prompt_tokens=len(prompt_ids), generated_tokens=len(generated.token_ids),
                                         finish_reason=generated.finish_reason)) + "\n")
        return TokenOutput(token_ids=list(generated.token_ids), log_probs=logprobs,
                           stop_reason=generated.finish_reason,
                           extra_fields={"min_global_steps": 0, "max_global_steps": 0})


async def evaluate(args):
    import pyarrow.parquet as pq
    import ray
    from omegaconf import OmegaConf
    from recipes.code.mimoagent_runner import _require_code_verdict
    from transformers import AutoTokenizer
    from uni_agent.framework.framework import OpenAICompatibleAgentFramework
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    data_path = args.manifest.parent / manifest["parquet"]
    if sha(data_path) != manifest["parquet_sha256"] or not manifest.get("runtime_qualified"):
        raise ValueError("Dataset payload is unqualified or differs from its immutable manifest")
    if not manifest.get("native_prompt_audited"):
        raise ValueError("Native prompt budget audit is required")
    rows = pq.read_table(data_path).to_pylist()
    by_task = {row["extra_info"]["instance_id"]: row for row in rows}
    tasks = manifest["task_ids"]
    if len(tasks) != len(set(tasks)) or set(tasks) != set(by_task):
        raise ValueError("Dataset and complete task grid disagree")
    seeds = tuple(int(x) for x in args.seeds.split(","))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Sample seeds must be nonempty and unique")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    eos = tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    bindings = {str(args.harness): sha(args.harness)}
    for relative in ("src/archlab/training/dlc_sandbox.py", "third_party/verl/recipes/code/mimoagent_runner.py",
                     "third_party/verl/third_party/uni_agent/uni_agent/framework/framework.py",
                     "third_party/verl/third_party/uni_agent/uni_agent/gateway/gateway.py",
                     "third_party/verl/third_party/mimoagent-osr/src/mimoagent/agents/bashonly/bashonly_agent.py"):
        path = args.source / relative
        bindings[str(path)] = sha(path)
    import yaml
    for spec in yaml.safe_load(args.harness.read_text())["harnesses"]:
        path = args.harness.parent / spec["config"]
        bindings[str(path)] = sha(path)
        harness = yaml.safe_load(path.read_text())
        if harness["agent"]["step_limit"] != 300 or harness["model"]["model_kwargs"]["max_tokens"] != 8192:
            raise ValueError("Evaluation must retain the training agent's 300 turns and8192-token turn budget")
    weight_files = sorted(args.model.glob("*.safetensors")) or sorted(args.model.glob("pytorch_model*.bin"))
    if not weight_files:
        raise ValueError("Evaluation requires an exported Hugging Face weight checkpoint")
    assets = [path for pattern in ("*.index.json", "tokenizer*", "chat_template*", "generation_config.json", "*RECEIPT*.json")
              for path in args.model.glob(pattern) if path.is_file()]
    model_bindings = {path.name: {"sha256": sha(path), "bytes": path.stat().st_size} for path in weight_files + assets}
    invocation = dict(schema=SCHEMA, label=args.label, model=str(args.model.resolve()), model_bindings=model_bindings,
                      model_config_sha256=sha(args.model / "config.json"), manifest_sha256=sha(args.manifest),
                      tasks=tasks, seeds=seeds, context_tokens=262144, initial_prompt_tokens=8192,
                      session_seconds=7200, harness_bindings=bindings, source_snapshot=str(args.source.resolve()),
                      evaluator_sha256=sha(__file__), temperature=1.0, top_p=1.0, top_k=-1,
                      enable_thinking=False, seed_algorithm="sha256(task NUL sample) then sha256(sample_seed NUL turn), first32bits",
                      scope="preliminary-subset" if len(tasks) < 48 or len(seeds) < 8 else "heldout-fixed-grid",
                      training_axes=json.loads(args.training_axes.read_text()) if args.training_axes else {})
    invocation_path = output / "invocation.json"
    bind_invocation(invocation_path, invocation)
    env = {"MIXED_HARNESS_SPEC": str(args.harness.resolve()), "MIXED_HARNESS_MODE": "prompt",
           "UNI_AGENT_LOG_DIR": str(output / "sessions"), "ARCHLAB_DLC_SANDBOX_POOL": str(args.pool),
           "UNI_AGENT_RUNNER_TASK_NUM_CPUS": "0.25", "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
           "NO_PROXY": "*", "no_proxy": "*"}
    for name in ("PATH", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH", "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH", "VLLM_PLUGINS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if name in os.environ:
            env[name] = os.environ[name]
    os.environ.update(env)
    backend = VLLMBackend(args, output)
    gateway = _GatewayActor(GatewayActorConfig(
        tokenizer=tokenizer, generation_eos_token_id=eos, tool_parser_name="qwen3_coder", rollout_backend="vllm",
        apply_chat_template_kwargs={"enable_thinking": False}, prompt_length=8192, response_length=253952), backend)
    ray.init(address="local", num_cpus=max(4, math.ceil(args.concurrency / 4) + 2), num_gpus=0,
             include_dashboard=False, runtime_env={"env_vars": dict(env, CUDA_VISIBLE_DEVICES="")})
    await gateway.start()
    runner = dict(runner_fqn="recipes.code.mimoagent_runner.mimoagent_runner", dispatch_mode="ray_task",
                  max_concurrent_sessions=args.concurrency, trajectory_selection="longest",
                  session_timeout_seconds=7200, agent_request_idle_timeout_seconds=720)
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {
        "temperature": 1.0, "top_p": 1.0, "top_k": -1, "calculate_log_probs": True,
        "custom": {"agent_framework": {"agent_runners": {"policy": runner}, "log_dir": str(output / "sessions"),
                                        "mask_unfinished_episode": False, "use_reward_loop_worker": False}}}}})
    framework = OpenAICompatibleAgentFramework.from_config(config=config, gateway_manager=gateway, tokenizer=tokenizer)
    episodes_path = output / "episodes.jsonl"
    records = [json.loads(line) for line in episodes_path.read_text().splitlines()] if episodes_path.exists() else []
    expected, completed = completed_records(records, tasks, seeds)
    semaphore, lock = asyncio.Semaphore(args.concurrency), asyncio.Lock()
    started = time.monotonic()

    async def episode(position, task, seed):
        async with semaphore:
            row = by_task[task]
            instance = json.loads(row["extra_info"]["instance_json"])
            encoded_seed = sample_seed(task, seed)
            backend.reset(encoded_seed)
            began = time.monotonic()
            record = dict(schema=SCHEMA, task=task, seed=seed, sample_seed=encoded_seed, label=args.label)
            try:
                sample = dict(raw_prompt=row["prompt"], uid=f"{task}-s{seed}", data_source=row.get("data_source"),
                              extra_info=row["extra_info"], tools_kwargs={"instance": instance, "dataset_index": position})
                sampling = framework._build_session_sampling_params(partition_id="train", sample_fields=sample)
                sampling.update(seed=encoded_seed, max_tokens=8192, stop_token_ids=eos, temperature=1.0, top_p=1.0, top_k=-1)
                trajectories, _ = await framework._run_session(
                    sample_fields=sample, sample_index=position, session_index=seed, global_steps=0,
                    runner_name="policy", runner_config=framework.runner_registry["policy"], sampling_params=sampling)
                if not trajectories:
                    raise RuntimeError("No materialized trajectory")
                trajectory = trajectories[-1]
                reward = trajectory.reward_info or {}
                _require_code_verdict(instance, reward.get("reward"), reward)
                agent_status = str(reward.get("agent_status", "unknown"))
                cap_reason = policy_cap_reason(reward, trajectory.num_turns)
                record.update(status="scored", reward=float(reward["reward"]), cap_reason=cap_reason,
                              termination=reward.get("termination_kind", "unknown"), agent_status=agent_status,
                              verifier_completed=True, verifier_returncode=reward["verifier_returncode"],
                              prompt_tokens=len(trajectory.prompt_ids), replay_tokens=len(trajectory.response_ids),
                              assistant_turns=trajectory.num_turns, agent_completed=reward.get("agent_completed"),
                              tool_call_error_count=reward.get("tool_call_error_count", 0))
            except Exception as error:
                record.update(error_outcome(error))
            record.update(backend.stats[encoded_seed], seconds=round(time.monotonic() - began, 3))
            async with lock:
                with episodes_path.open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                records.append(record)
                summary = summarize(records, tasks, seeds)
                summary.update(label=args.label, manifest_sha256=sha(args.manifest),
                               invocation_sha256=sha(invocation_path), wall_seconds=time.monotonic() - started,
                               gpu_count=1, training_axes=invocation["training_axes"])
                atomic_json(output / "summary.json", summary)
            print(json.dumps({key: record.get(key) for key in ("task", "seed", "status", "reward", "termination", "seconds")}), flush=True)

    try:
        for attempt in range(args.attempts):
            _, done = completed_records(records, tasks, seeds)
            pending = [(task, seed) for seed in seeds for task in tasks if (task, seed) not in done]
            if not pending:
                break
            await asyncio.gather(*(episode(i, task, seed) for i, (task, seed) in enumerate(pending)))
            if attempt + 1 < args.attempts:
                await asyncio.sleep(30)
    finally:
        await gateway.shutdown()
        ray.shutdown()
        backend.engine.shutdown()
    return 0 if summarize(records, tasks, seeds)["complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "manifest", "source", "harness", "pool", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--training-axes", type=Path)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 32 or not 0 < args.gpu_memory < 1 or args.attempts < 1:
        parser.error("Invalid resource/retry bound")
    paths = [str(args.source / part) for part in ("src", "third_party/verl", "third_party/verl/third_party/uni_agent",
                                                 "third_party/verl/third_party/mimoagent-osr/src")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = ":".join(paths + [os.environ.get("PYTHONPATH", "")])
    raise SystemExit(asyncio.run(evaluate(args)))


if __name__ == "__main__":
    main()

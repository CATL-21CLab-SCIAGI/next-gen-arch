"""Run paired evaluation and bounded chat between baseline optimizer updates.

The baseline weights/optimizer stay resident. Its gradients are released, the
other checkpoint is loaded on the same 16-rank mesh, and all RNG states/modes are
restored before the next update. No training or inference weights are offloaded.
"""

from __future__ import annotations

import gc
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from queue import Empty


@contextmanager
def inference_head(head):
    import torch
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    if torch.is_grad_enabled():
        raise ValueError("inference head requires no_grad")
    sharded = isinstance(head, FSDPModule)
    if sharded:
        head.unshard()
    try:
        if isinstance(head.weight, DTensor) or head.weight.dtype != torch.float32:
            raise ValueError("expected the native unsharded FP32 vocabulary head")
        yield head
    finally:
        if sharded:
            head.reshard()


def latest_checkpoint(root, variant):
    candidates = []
    for marker in Path(root).glob("step-*/COMPLETE.json"):
        value = json.loads(marker.read_text())
        if value["contract"]["variant"] != variant:
            raise ValueError("wrong checkpoint variant")
        candidates.append((value["cursor"]["step"], marker.parent))
    if not candidates:
        raise ValueError(f"no complete {variant} checkpoint")
    return max(candidates)[1]


def construct_secondary(checkpoint, variant, *, weights, assets, registry, tiny=False):
    from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
    from archlab.automodel.deepseek_v41_full_boundaries import install_full_training_boundaries
    from archlab.automodel.deepseek_v41_full_eval_checkpoint import marker_contract, restore_weights
    from archlab.automodel.deepseek_v41_full_eval_construct import build_eval_shell
    from archlab.automodel.deepseek_v41_full_indexer import install_trainable_indexers
    from archlab.automodel.deepseek_v41_official_adapter import install_official_adapters

    marker = json.loads((Path(checkpoint) / "COMPLETE.json").read_text())
    trained_source = Path(registry[marker["contract"]["project_commit"]])
    marker_contract(checkpoint, variant=variant, trained_source=trained_source)
    model, _setup, loading = build_eval_shell(weights=weights, assets=assets, tiny=tiny)
    for key in ("container_image", "packages", "cuda", "nccl", "reproducibility"):
        if loading[key] != marker["contract"]["runtime"][key]:
            raise ValueError(f"wrong trained runtime: {key}")
    install_official_adapters(
        model,
        V41AdapterConfig(width=256) if tiny else V41AdapterConfig(),
        layer_indices=(1, 3, 5) if tiny else (4, 9, 14, 19, 24, 29, 34, 39),
        device="cuda",
        variant=variant,
        backend="deterministic" if variant == "simplicial" else "flash-attn-deterministic",
    )
    install_full_training_boundaries(model)
    install_trainable_indexers(model)
    receipt = restore_weights(model, checkpoint, variant=variant, trained_source=trained_source)
    return model, _setup, receipt


def next_token_logits(model, token_ids, context):
    import torch

    if not 0 < len(token_ids) <= context:
        raise ValueError("chat context limit exceeded")
    inputs = torch.zeros((1, context), device="cuda", dtype=torch.long)
    inputs[0, : len(token_ids)] = torch.tensor(token_ids, device="cuda", dtype=torch.long)
    hidden = model(input_ids=inputs, return_hidden_states=True).hidden_states
    with inference_head(model.lm_head) as head:
        logits = torch.nn.functional.linear(hidden[0, len(token_ids) - 1].float(), head.weight)
    if not bool(logits.isfinite().all()):
        raise FloatingPointError("nonfinite chat logits")
    return logits


def generate_chat(
    model, tokenizer, renderer, payload, *, max_context, ticket=None, fingerprint=None
):
    import torch
    import torch.distributed as dist

    rank = dist.get_rank()
    message = [None]
    if rank == 0:
        try:
            prompt = renderer.encoder.encode_messages(payload["messages"], thinking_mode="chat")
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            if not ids or len(ids) + payload["max_tokens"] > max_context:
                raise ValueError(f"prompt plus generation must fit {max_context} tokens")
            eos = tokenizer.encode(renderer.encoder.eos_token, add_special_tokens=False)
            if len(eos) != 1:
                raise ValueError("native EOS must be one token")
            message[0] = {"ids": ids, "eos": eos[0]}
        except Exception as error:
            message[0] = {"error": str(error)}
    dist.broadcast_object_list(message, src=0)
    if "error" in message[0]:
        if rank == 0 and ticket:
            ticket.events.put({"type": "error", "message": message[0]["error"]})
        return None
    ids = list(message[0]["ids"])
    initial = len(ids)
    generated = []
    previous = ""
    reason = "length"
    context = max(128, 1 << (initial + payload["max_tokens"] - 1).bit_length())
    if context > max_context:
        raise ValueError("chat context bucket exceeds configured limit")
    generator = torch.Generator(device="cpu").manual_seed(payload["seed"])
    for _ in range(payload["max_tokens"]):
        logits = next_token_logits(model, ids, context)
        selected = torch.zeros(2, device="cuda", dtype=torch.long)
        if rank == 0:
            if ticket and ticket.cancelled.is_set():
                selected[1] = 1
            if payload["temperature"] == 0:
                selected[0] = logits.argmax()
            else:
                probabilities = (logits.double() / payload["temperature"]).softmax(-1).cpu()
                if payload["top_p"] < 1:
                    values, indices = probabilities.sort(descending=True)
                    remove = values.cumsum(0) - values >= payload["top_p"]
                    probabilities[indices[remove]] = 0
                    probabilities /= probabilities.sum()
                selected[0] = torch.multinomial(probabilities, 1, generator=generator).item()
        dist.broadcast(selected, src=0)
        token, cancelled = map(int, selected.tolist())
        if cancelled:
            reason = "stop"
            break
        generated.append(token)
        if token == message[0]["eos"]:
            reason = "stop"
            break
        ids.append(token)
        if rank == 0 and ticket:
            text = tokenizer.decode(generated, skip_special_tokens=True)
            # Hold incomplete UTF-8 sequences until the decoder has full bytes.
            if "\ufffd" not in text and text.startswith(previous):
                delta = text[len(previous) :]
                if delta:
                    ticket.events.put({"type": "text", "text": delta})
                previous = text
    if rank == 0 and ticket:
        final = tokenizer.decode(generated, skip_special_tokens=True)
        if final.startswith(previous) and final != previous:
            ticket.events.put({"type": "text", "text": final[len(previous) :]})
        ticket.events.put(
            {
                "type": "done",
                "finish_reason": reason,
                "system_fingerprint": fingerprint,
                "usage": {
                    "prompt_tokens": initial,
                    "completion_tokens": len(generated),
                    "total_tokens": initial + len(generated),
                },
            }
        )
    return generated


class Activity:
    def __init__(self, output, cursor):
        self.path = Path(output) / "ACTIVITY.json"
        self.cursor = dict(cursor)
        self.phase = "evaluating"
        self.stage = "loading"
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        from archlab.artifacts import atomic_write_json

        while not self.stop.is_set():
            atomic_write_json(
                self.path,
                {
                    "phase": self.phase,
                    "stage": self.stage,
                    "heartbeat_unix": time.time(),
                    "cursor": self.cursor,
                },
            )
            self.stop.wait(10)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        from archlab.artifacts import atomic_write_json

        self.stop.set()
        self.thread.join()
        atomic_write_json(
            self.path, {"phase": "running", "heartbeat_unix": time.time(), "cursor": self.cursor}
        )


def run_window(
    model,
    optimizer,
    cursor,
    checkpoint,
    policy,
    *,
    assets,
    weights,
    front,
    output,
    verification_batches,
    final=False,
):
    import torch
    import torch.distributed as dist
    import yaml
    from transformers import PreTrainedTokenizerFast

    from archlab.artifacts import atomic_write_json
    from archlab.automodel.deepseek_v41_data import MathPilot
    from archlab.automodel.deepseek_v41_full_evaluate import math_evaluate, mc_evaluate
    from archlab.automodel.deepseek_v41_full_validation import evaluate_batches, evaluation_state
    from archlab.evaluation.deepseek_v41_compare_data import (
        build_jobs,
        jobs_digest,
        read_jsonl,
        sha,
    )
    from archlab.preprocessing.deepseek_v41 import DeepSeekV41Renderer

    rank = dist.get_rank()
    choice = [None]
    if rank == 0:
        from archlab.automodel.deepseek_v41_control import pending_evaluation

        pending = pending_evaluation(policy, "simplicial")
        choice[0] = (
            pending["checkpoint"]
            if pending
            else str(latest_checkpoint(policy["simplicial_checkpoints"], "simplicial"))
        )
    dist.broadcast_object_list(choice, src=0)
    secondary_path = Path(choice[0])
    markers = {
        v: json.loads((Path(p) / "COMPLETE.json").read_text())
        for v, p in {"normal": checkpoint, "simplicial": secondary_path}.items()
    }
    name = [None]
    if rank == 0:
        name[0] = (
            f"normal-{cursor['step']:06d}-simplicial-{markers['simplicial']['cursor']['step']:06d}-{uuid.uuid4().hex[:8]}"
        )
    dist.broadcast_object_list(name, src=0)
    window = Path(policy["evaluation_output"]) / name[0]
    lease = (
        secondary_path / ".checkpoint-readers" / f"eval-window-{uuid.uuid4().hex}.json"
        if rank == 0
        else None
    )
    if rank == 0:
        window.mkdir(parents=True, exist_ok=False)
        atomic_write_json(lease, {"owner": "baseline-evaluation-window", "window": str(window)})
    dist.barrier()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    before_reference = evaluate_batches(model, verification_batches)
    activity = Activity(output, cursor) if rank == 0 else None
    if activity:
        activity.__enter__()
    secondary = _setup = models = None
    try:
        with evaluation_state(model):
            registry = json.loads(Path(policy["source_registry"]).read_text())
            secondary, _setup, receipt = construct_secondary(
                secondary_path, "simplicial", weights=weights, assets=assets, registry=registry
            )
            atomic_write_json(window / f"simplicial-rank-{rank:02d}-restore.json", receipt)
            models = {"normal": model, "simplicial": secondary}
            metadata = {
                "format": "archlab-full-eval-window-v1",
                "checkpoint_paths": {"normal": str(checkpoint), "simplicial": str(secondary_path)},
                "cursors": {v: m["cursor"] for v, m in markers.items()},
                "matched_steps": markers["normal"]["cursor"]["step"]
                == markers["simplicial"]["cursor"]["step"],
                "gpu_count": 16,
                "baseline_weights_resident": True,
                "cpu_weight_offload": False,
            }
            if rank == 0:
                atomic_write_json(window / "RUN.json", metadata)
            tokenizer = PreTrainedTokenizerFast.from_pretrained(assets, local_files_only=True)
            data_root = Path(policy["benchmark_data"])
            manifest = json.loads((data_root / "MANIFEST.json").read_text())
            prompts = Path(__file__).parents[1] / "prompts/capability_regression.yaml"
            if (
                sha(data_root / "cases.jsonl") != manifest["cases_sha256"]
                or sha(prompts) != manifest["prompts_sha256"]
            ):
                raise ValueError("sealed benchmark cases or prompts changed")
            cases = read_jsonl(data_root / "cases.jsonl")
            jobs = build_jobs(tokenizer, cases, yaml.safe_load(prompts.read_text()))
            if jobs_digest(jobs) != manifest["jobs_digest"]:
                raise ValueError("benchmark tokenization changed")
            validation = MathPilot(
                Path(policy["validation_pilot"]),
                expected_split="validation",
                expected_budget=1000000,
            )
            if activity:
                activity.stage = "heldout-math"
            math_evaluate(models, validation, window, head_context=inference_head)
            if activity:
                activity.stage = "multiple-choice"
            mc_evaluate(models, cases, jobs, window, head_context=inference_head)
            if rank == 0:
                atomic_write_json(window / "COMPLETE.json", {"passed": True, **metadata})
                for v in models:
                    reference = (
                        Path(policy["benchmark_results"])
                        / f"{v}-step-{markers[v]['cursor']['step']:06d}.json"
                    )
                    if not reference.exists():
                        atomic_write_json(
                            reference,
                            {
                                "variant": v,
                                "cursor": markers[v]["cursor"],
                                "result_directory": str(window),
                                "checkpoint": metadata["checkpoint_paths"][v],
                            },
                        )
                    (Path(metadata["checkpoint_paths"][v]) / "EVAL_PENDING.json").unlink(
                        missing_ok=True
                    )
            dist.barrier()
            renderer = DeepSeekV41Renderer(assets)
            end = None if final else time.time() + policy["chat_window_seconds"]
            if rank == 0:
                activity.phase = "chat"
                activity.stage = "playground"
                front.set_state("chat", checkpoints=metadata["cursors"], available_until_unix=end)
                atomic_write_json(
                    window / "CHAT_READY.json", {"until_unix": end, "cursors": metadata["cursors"]}
                )
            while True:
                command = [None]
                ticket = None
                if rank == 0:
                    next_request = pending_evaluation(policy, "simplicial") if final else None
                    if (end is not None and time.time() >= end) or (
                        final and (next_request or (Path(output) / "STOP_SERVING").exists())
                    ):
                        command[0] = {"action": "close"}
                    else:
                        try:
                            ticket = front.pending.get(
                                timeout=1.0
                                if end is None
                                else min(1.0, max(0.01, end - time.time()))
                            )
                        except Empty:
                            pass
                        command[0] = (
                            {"action": "chat", "payload": ticket.payload}
                            if ticket
                            else {"action": "idle"}
                        )
                dist.broadcast_object_list(command, src=0)
                if command[0]["action"] == "close":
                    break
                if command[0]["action"] == "idle":
                    continue
                payload = command[0]["payload"]
                variant = payload["variant"]
                generate_chat(
                    models[variant],
                    tokenizer,
                    renderer,
                    payload,
                    max_context=policy["chat_context"],
                    ticket=ticket,
                    fingerprint=f"{variant}-step-{markers[variant]['cursor']['step']:06d}",
                )
            if rank == 0:
                front.close_window()
                atomic_write_json(
                    window / "WINDOW_CLOSED.json",
                    {"resumed_after_step": cursor["step"], "closed_unix": time.time()},
                )
    finally:
        if rank == 0 and front:
            front.close_window()
        models = None
        secondary = None
        _setup = None
        gc.collect()
        torch.cuda.empty_cache()
        if activity:
            activity.__exit__(None, None, None)
        dist.barrier()
        if rank == 0 and lease:
            lease.unlink(missing_ok=True)
    after_reference = evaluate_batches(model, verification_batches)
    from archlab.automodel.deepseek_v41_full_training import all_errors

    exact_rng = torch.equal(before_cpu, torch.get_rng_state()) and torch.equal(
        before_cuda, torch.cuda.get_rng_state()
    )
    difference = abs(before_reference["loss"] - after_reference["loss"])
    all_errors(
        []
        if exact_rng and difference < 1e-4
        else ["evaluation/chat changed baseline RNG or next-batch forward"]
    )
    if rank == 0:
        atomic_write_json(
            window / "TRAINING_STATE_VERIFIED.json",
            {"passed": True, "rng_exact": exact_rng, "next_batch_ce_error": difference},
        )
    return window


def qualify_window(model, optimizer, checkpoint, batches, *, variant, assets, weights, source):
    import torch
    import torch.distributed as dist

    from archlab.automodel.deepseek_v41_full_training import _fingerprint, all_errors
    from archlab.automodel.deepseek_v41_full_validation import evaluate_batches, evaluation_state

    marker = json.loads((Path(checkpoint) / "COMPLETE.json").read_text())
    optimizer.zero_grad(set_to_none=True)
    before = _fingerprint(model, optimizer)
    secondary = _setup = None
    with evaluation_state(model):
        secondary, _setup, _ = construct_secondary(
            checkpoint,
            variant,
            weights=weights,
            assets=assets,
            registry={marker["contract"]["project_commit"]: str(source)},
            tiny=True,
        )
        primary = evaluate_batches(model, batches)
        other = evaluate_batches(secondary, batches)
        token_ids = [int(x) for x in batches[0][0][0, :32].tolist()]
        a = next_token_logits(model, token_ids, 128)
        b = next_token_logits(secondary, token_ids, 128)
        error = float((a - b).abs().max())
        all_errors(
            []
            if error < 1e-4 and abs(primary["loss"] - other["loss"]) < 1e-4
            else ["resident training and restored chat model differ"]
        )
        del a, b
        secondary = None
        _setup = None
        gc.collect()
        torch.cuda.empty_cache()
    after = _fingerprint(model, optimizer)
    all_errors(
        [] if before == after else ["evaluation/chat window changed model, optimizer, or RNG state"]
    )
    return {
        "passed": True,
        "state_rng_exact": True,
        "max_logit_error": error,
        "ce_error": primary["loss"] - other["loss"],
        "world_size": dist.get_world_size(),
    }

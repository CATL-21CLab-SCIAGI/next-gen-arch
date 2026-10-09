"""Shared GRPO for native publisher actors and verified Limite adapter warmups."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path

import torch

from archlab.automodel.checkpoint_oracle import assert_state_equal
from archlab.automodel.limite_adapter_common import (
    attention_kernel_name,
    build_model,
    file_hash,
    frozen_fingerprint,
    runtime_contract,
    save_adapter,
)
from archlab.automodel.limite_adapter_communication import (
    communication_contract,
    synchronize_gradients,
)
from archlab.rl.limite_checkpoint import (
    checkpoint_payloads,
    legacy_resume_state,
    resize_rank_state,
    restore_rng,
)
from archlab.rl.limite_data import math_reward
from archlab.rl.limite_rollout import native_rollout


def identity_contract(model, receipt):
    """Separate the immutable publisher identity from trainable checkpoint values."""
    if getattr(model, "archlab_native_checkpoint", False):
        from archlab.automodel.limite_native_checkpoint import native_identity_contract

        return native_identity_contract(model, receipt)
    mode = getattr(model.model, "trainable_mode", "adapter")
    expected = receipt.get("base_snapshot_sha256", receipt.get("frozen_sha256"))
    if mode == "adapter":
        actual = frozen_fingerprint(model)
        if actual != expected:
            raise ValueError("frozen base identity changed")
        return dict(trainable_mode=mode, frozen_sha256=actual)
    actual = model.archlab_base_snapshot_sha256
    if actual != expected:
        raise ValueError("publisher snapshot identity changed")
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("full-weight RL requires every model parameter to be trainable")
    return dict(trainable_mode=mode, base_snapshot_sha256=actual)


def check_warmup_parent(marker, receipt, trainable_mode, *, correctness_fixture=False):
    if marker.get("tokens") != receipt.get("tokens"):
        raise ValueError("warmup marker and completed checkpoint token counts differ")
    if not correctness_fixture:
        if marker.get("tokens") != 10_000_000_000:
            raise ValueError("RL requires exactly 10B warmup tokens")
        if trainable_mode == "full" and receipt.get("trainable_mode") != "full":
            raise ValueError("full-weight RL requires a completed full-weight warmup checkpoint")


def optimizer_for_model(model):
    # Transformer Engine detects the optional FA4 distribution before importing
    # its package. With our official overlay the container FA2 package remains
    # authoritative, so select the pinned CuTe namespace explicitly rather than
    # relying on an earlier FA4 attention invocation to initialize it.
    if getattr(model, "archlab_native_replay", False):
        try:
            metadata.distribution("flash-attn-4")
        except metadata.PackageNotFoundError:
            pass
        else:
            from archlab.architectures.fa4_attention import fa4_runtime_contract

            fa4_runtime_contract(validate=True)
    from archlab.optimizers.rl_adam import SignalFusedAdam

    return SignalFusedAdam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        master_weights=getattr(model.model, "trainable_mode", "adapter") == "full",
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )


def backward_context(model, schedule):
    # The public DDP control covers forward as well as backward. Reduction after
    # the last accumulated backward therefore precedes Trainer gradient clipping.
    if schedule == "deferred" and isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.no_sync()
    return nullcontext()


def finish_backward(parameters, schedule, sync_gradients):
    if schedule == "deferred" and sync_gradients and torch.distributed.is_initialized():
        synchronize_gradients(parameters)


def gradient_evidence(model):
    groups = dict(adapter=[], backbone=[])
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            groups["adapter" if name.startswith("model.adapters.") else "backbone"].append(
                parameter.grad
            )
    result = {}
    for name, gradients in groups.items():
        result[name + "_gradient_tensors"] = len(gradients)
        result[name + "_gradient_norm"] = (
            float(torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients))))
            if gradients else 0.0
        )
    if not all(math.isfinite(value) for key, value in result.items() if key.endswith("norm")):
        raise FloatingPointError("nonfinite gradient")
    return result


def parameter_probe(model, *, backbone_only=False, matrices_only=False):
    parameters = [
        (name, parameter) for name, parameter in model.named_parameters()
        if (not backbone_only or not name.startswith("model.adapters."))
        and (not matrices_only or parameter.ndim >= 2)
    ]
    stride = max(1, len(parameters) // 24)
    return {
        name: parameter.detach().reshape(-1)[:256].clone()
        for name, parameter in parameters[::stride]
    }


def optimizer_probe(value):
    """Copy bounded values and every state counter, without aliasing live state."""
    if isinstance(value, torch.Tensor):
        return dict(shape=tuple(value.shape), dtype=str(value.dtype), values=value.detach().reshape(-1)[:256].clone())
    if isinstance(value, dict):
        return {key: optimizer_probe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(optimizer_probe(item) for item in value)
    return value


def main():
    from datasets import Dataset
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TrainerCallback

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--variant", choices=("normal", "simplicial", "native"), required=True)
    p.add_argument("--attention-backend", choices=("native", "tilelang"), default=None)
    p.add_argument("--normal-kernel", choices=("shared", "gqa"), default=None)
    p.add_argument("--trainable-mode", choices=("adapter", "full"), default=None)
    p.add_argument("--communication-schedule", choices=("bucketed", "deferred"), default=None)
    p.add_argument("--allow-backend-migration", action="store_true")
    p.add_argument("--warmup", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--credentials", type=Path)
    p.add_argument("--oss")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--trl-runtime", type=Path)
    p.add_argument("--stop-file", type=Path, action="append", default=[])
    p.add_argument("--test-rollouts", action="store_true")
    p.add_argument("--test-flat-batch", action="store_true")
    p.add_argument("--test-eval-transition", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--allow-legacy-resume", action="store_true")
    p.add_argument("--resize-resume-world", action="store_true",
                   help="explicitly permit a smaller rank partition; discard old prefetched batches")
    p.add_argument("--runtime-sequence-attention", action="store_true")
    p.add_argument("--checkpoint-steps", type=int, default=10)
    p.add_argument("--math-protocol", type=Path)
    p.add_argument("--curriculum", type=Path)
    p.add_argument("--phase-start", type=int, default=0)
    p.add_argument("--new-tracking-phase", action="store_true")
    p.add_argument("--checkpoint-cache", type=Path)
    a = p.parse_args()
    from archlab.automodel.limite_native_checkpoint import check_native_options

    check_native_options(
        a.variant, a.warmup, a.attention_backend, a.trainable_mode, a.normal_kernel,
        a.runtime_sequence_attention, a.allow_backend_migration,
    )
    native_checkpoint = a.variant == "native"
    if not native_checkpoint and a.warmup is None:
        p.error("adapter RL requires --warmup")
    if a.checkpoint_steps < 1:
        p.error("checkpoint steps must be positive")
    if a.test_flat_batch and not a.test_rollouts:
        p.error("flat-batch injection is permitted only in correctness fixtures")
    if a.test_eval_transition and not a.test_rollouts:
        p.error("eval-transition qualification is permitted only in correctness fixtures")
    if a.trl_runtime:
        sys.path.insert(0, str(a.trl_runtime.resolve()))
    import trl
    from trl import GRPOConfig

    from archlab.rl.limite_trainer import BehaviorGRPO

    if trl.__version__ != "1.4.0":
        raise RuntimeError("Limite RL requires the qualified TRL 1.4.0 overlay")
    protocol = protocol_spec = curriculum = None
    if a.math_protocol:
        import yaml

        from archlab.rl.limite_protocol import MathRolloutProtocol

        protocol_spec = yaml.safe_load(a.math_protocol.read_text())
        protocol = MathRolloutProtocol(**protocol_spec["rollout"])
    if a.curriculum:
        curriculum = json.loads(a.curriculum.read_text())
        if protocol is None:
            raise ValueError("a changed math curriculum requires an explicit RL protocol")
    if protocol_spec and protocol_spec.get("protocol_transition") and not a.resume and not a.test_rollouts:
        raise ValueError("protocol migration requires its declared source RL checkpoint")
    execution = protocol_spec["execution"] if protocol_spec else {}
    if execution.get("async_rollouts") and not execution.get("graph_decode"):
        raise ValueError("asynchronous native sampling requires graph decoding")
    if type(execution.get("overlap_actor_learner", True)) is not bool:
        raise ValueError("actor/learner overlap must be an explicit boolean")
    if execution.get("rollout_rendezvous", "gloo_after_drain") != "gloo_after_drain":
        raise ValueError("unsupported rollout rendezvous")
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if native_checkpoint:
        from archlab.automodel.limite_native_checkpoint import check_native_recipe

        check_native_recipe(protocol_spec, world, a.steps, correctness_fixture=a.test_rollouts)
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.manual_seed(42 + rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=True)
    if native_checkpoint:
        from archlab.automodel.limite_native_checkpoint import (
            build_native_model,
            check_native_publisher,
            publisher_identity,
            save_native_checkpoint,
        )

        parent = Path(a.model)
        parent_receipt = dict(publisher_identity=publisher_identity(parent), trainable_mode="full")
        check_native_publisher(a.model, a.tokenizer, protocol_spec, parent_receipt["publisher_identity"])
        if a.phase_start != protocol_spec["training"]["phase_start"] or (
            not a.test_rollouts and a.checkpoint_steps != protocol_spec["training"]["checkpoint_steps"]
        ):
            raise ValueError("native RL phase clock or checkpoint interval differs from the recipe")
        parent_contract = dict(publisher_snapshot=str(parent))
        save_checkpoint = save_native_checkpoint
    else:
        marker = json.loads((a.warmup / "WARMUP_COMPLETE.json").read_text())
        parent = Path(marker["checkpoint"])
        parent_receipt = json.loads((parent / "COMPLETE.json").read_text())
        parent_contract = dict(warmup_checkpoint=str(parent))
        save_checkpoint = save_adapter
    resume_receipt = None
    protocol_migration = None
    if a.resume:
        resume_receipt = json.loads((a.resume / "COMPLETE.json").read_text())
        parent_matches = (
            resume_receipt.get("model_kind") == "native"
            and resume_receipt.get("publisher_snapshot") == str(parent)
            and resume_receipt.get("publisher_identity") == parent_receipt["publisher_identity"]
        ) if native_checkpoint else (
            resume_receipt.get("warmup_checkpoint") == str(parent)
            and resume_receipt["adapter"]["variant"] == a.variant
        )
        if (not parent_matches or
                resume_receipt.get("trainable_mode", "adapter") != parent_receipt.get("trainable_mode", "adapter")):
            raise ValueError("RL resume checkpoint differs from its initialization parent")
        if not (a.resume / "rl_state.pt").exists() and not a.allow_legacy_resume:
            raise ValueError("legacy RL state lacks rank RNG; explicit migration is required")
        prior_protocol = resume_receipt.get("math_protocol")
        if prior_protocol:
            from archlab.rl.limite_phase import protocol_transition

            protocol_migration = protocol_transition(
                resume_receipt, protocol, protocol_spec, phase_start=a.phase_start,
                curriculum_sha256=file_hash(a.curriculum) if a.curriculum else None,
                new_tracking_phase=a.new_tracking_phase,
            )
    load_checkpoint = a.resume or (None if native_checkpoint else parent)
    if a.checkpoint_cache and load_checkpoint is not None:
        from archlab.automodel.checkpoint_cache import stage_checkpoint

        load_checkpoint = stage_checkpoint(load_checkpoint, a.checkpoint_cache)
    def build_policy(checkpoint):
        if native_checkpoint:
            return build_native_model(
                a.model, f"cuda:{local}", checkpoint, checkpoint_cache=a.checkpoint_cache,
            )
        return build_model(
            a.model, a.variant, f"cuda:{local}", checkpoint,
            attention_backend=a.attention_backend, trainable_mode=a.trainable_mode,
            allow_backend_migration=a.allow_backend_migration, normal_kernel=a.normal_kernel,
            checkpoint_cache=a.checkpoint_cache,
        )

    model = build_policy(load_checkpoint)
    if execution.get("replay_head_chunk_size"):
        from archlab.rl.limite_scoring import enable_chunked_policy_scores

        enable_chunked_policy_scores(model, chunk_size=execution["replay_head_chunk_size"])
    if execution.get("native_decode_gqa"):
        from archlab.architectures.limite_gqa import set_native_decode_gqa

        set_native_decode_gqa(model, backend=execution.get("native_decode_backend", "sdpa"))
    if execution.get("native_replay"):
        from archlab.architectures.limite_replay import enable_native_replay

        enable_native_replay(model, attention_backend=execution.get("native_replay_backend", "sdpa_native"))
    attention_backend = "native" if native_checkpoint else model.model.adapter_config["attention_backend"]
    if a.runtime_sequence_attention:
        from archlab.architectures.limite_adapter import enable_runtime_sequence_attention

        enable_runtime_sequence_attention(model)
    normal_kernel = getattr(model.model, "normal_kernel", "shared")
    native_replay_enabled = native_checkpoint and bool(execution.get("native_replay"))
    native_replay_backend = model.archlab_native_replay["attention_backend"] if native_replay_enabled else None
    normal_backward = ((native_replay_backend if native_replay_enabled else "publisher") if native_checkpoint
                       else getattr(model.model, "normal_backward", "tilelang"))
    attention_kernel = ((f"publisher-{native_replay_backend}-replay" if native_replay_enabled else "publisher-sdpa")
                        if native_checkpoint else attention_kernel_name(
        a.variant, attention_backend, normal_kernel=normal_kernel,
        normal_backward="tilelang" if a.runtime_sequence_attention else normal_backward,
    ))
    identity = identity_contract(model, parent_receipt)
    trainable_mode = identity["trainable_mode"]
    schedule = a.communication_schedule or ("deferred" if trainable_mode == "full" else "bucketed")
    if trainable_mode == "adapter" and schedule != "bucketed":
        raise ValueError("adapter RL retains its qualified bucketed communication")
    communication = communication_contract(model.parameters(), schedule=schedule)
    if not native_checkpoint:
        check_warmup_parent(marker, parent_receipt, trainable_mode, correctness_fixture=a.test_rollouts)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    tok.eos_token_id = 151645
    tok.pad_token_id = 151643
    tok.bos_token_id = 151643
    split = json.loads((a.data / "SPLIT.json").read_text())
    if native_checkpoint and file_hash(a.data / "SPLIT.json") != protocol_spec["data"]["split_sha256"]:
        raise ValueError("native RL split differs from the declared experiment")
    if resume_receipt and resume_receipt.get("rl_split_sha256", file_hash(a.data / "SPLIT.json")) != file_hash(a.data / "SPLIT.json"):
        raise ValueError("RL resume checkpoint belongs to a different dataset split")
    for name, checksum in split["files"].items():
        if file_hash(a.data / name) != checksum:
            raise ValueError("RL split payload changed: " + name)
    datasets = {}
    excluded = []
    for name in ["train", "heldout"]:
        rows = [json.loads(line) for line in (a.data / (name + ".jsonl")).read_text().splitlines()]
        for row in rows:
            # Existing split uses the native base prompt. Preserve the question verbatim,
            # but use the exact math/reasoning template used by warmup.
            prefix = "<|im_start|>user\n"
            end = "<|im_end|>\n<|im_start|>assistant\n"
            question = row["prompt"].split(prefix, 1)[1].rsplit(end, 1)[0]
            row["prompt"] = tok.apply_chat_template(
                [dict(role="user", content=question)], tokenize=False, add_generation_prompt=True
            )
            row["input_ids"] = tok.encode(row["prompt"], add_special_tokens=False)
        excluded.extend(
            dict(
                split=name,
                uuid=r["uuid"],
                reason="serialized_prompt_over_2048",
                tokens=len(r["input_ids"]),
            )
            for r in rows
            if len(r["input_ids"]) > 2048
        )
        datasets[name] = Dataset.from_list([r for r in rows if len(r["input_ids"]) <= 2048])
    if native_checkpoint and len(datasets["heldout"]) != protocol_spec["data"]["heldout"]:
        raise ValueError("native chat serialization changed the declared heldout coverage")
    if set(datasets["train"]["problem_sha256"]) & set(datasets["heldout"]["problem_sha256"]):
        raise ValueError("RL train and heldout overlap")
    if rank == 0:
        (a.output / "exclusions.json").write_text(json.dumps(excluded, indent=2))
    generate = model.generate

    def stop_requested():
        return any(path.exists() for path in [a.output / "STOP_REQUEST", *a.stop_file])

    class Stop(StoppingCriteria):
        deadline = 0.0
        requested = False

        def __call__(self, input_ids, scores, **kwargs):
            # Avoid a network-filesystem stat for every generated token.
            now = time.monotonic()
            if now >= self.deadline:
                self.requested = stop_requested()
                self.deadline = now + 0.5
            return self.requested

    def generate_stoppable(*args, **kwargs):
        kwargs["disable_compile"] = True
        kwargs["stopping_criteria"] = StoppingCriteriaList([Stop()])
        return generate(*args, **kwargs)

    model.generate = generate_stoppable
    evidence = dict(applied_updates=0, flat_batches=0, pending=False)
    from archlab.automodel.checkpoint_publication import CheckpointPublisher

    publisher = CheckpointPublisher() if rank == 0 and protocol is not None else None
    resume_path = a.resume
    resume_migration = None

    def write(name, row):
        with (a.output / name).open("a") as f:
            f.write(json.dumps(dict(time=time.time(), **row)) + "\n")

    def reward(completions, expected_answer, completion_ids, trainer_state, **kwargs):
        scores = None
        if protocol is not None:
            from archlab.rl.limite_protocol import score_math_rollout

            reasons = kwargs.get("finish_reason") or [
                "eos" if ids and ids[-1] in (151643, 151645) else "length"
                for ids in completion_ids
            ]
            budgets = kwargs.get("completion_budget")
            if budgets is None and a.test_rollouts:
                budgets = ([len(ids) for ids in completion_ids] if protocol.budget_mode == "native_context"
                           else [protocol.max_tokens] * len(completions))
            if budgets is None:
                budgets = [None] * len(completions)
            scores = [score_math_rollout(c, y, ids, reason, protocol, training=model.training,
                                        completion_budget=budget)
                      for c, y, ids, reason, budget in zip(completions, expected_answer, completion_ids,
                                                          reasons, budgets, strict=True)]
            vals = [score["reward"] for score in scores]
            measurements = torch.tensor(
                [[score["accuracy"], score["finish_reason"] == "eos", score["finish_reason"] == "length",
                  score["repeated"], score["reasoning_closed"], score["overlong_penalty"], score["unfinished_penalty"]]
                 for score in scores], device=local,
            )
            means = trainer.accelerator.gather(measurements).mean(0).tolist()
            mode = "train" if model.training else "eval"
            for key, value in zip(("accuracy", "natural_eos", "length_cap", "repetition", "reasoning_closed", "overlong_penalty", "unfinished_penalty"), means, strict=True):
                trainer._metrics[mode]["math/" + key].append(value)
        else:
            vals = [
                math_reward(c, y) if ids and ids[-1] in (151643, 151645) else 0.0
                for c, y, ids in zip(completions, expected_answer, completion_ids, strict=True)
            ]
        if model.training:
            contrast = torch.tensor(
                int(
                    any(min(vals[i : i + 4]) < max(vals[i : i + 4]) for i in range(0, len(vals), 4))
                ),
                device=local,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(contrast)
            evidence["flat_batches"] = 0 if bool(contrast) else evidence["flat_batches"] + 1
        write(
            f"rollouts-rank{rank}.jsonl",
            dict(
                step=trainer_state.global_step,
                phase="train" if model.training else "eval",
                rewards=vals,
                uuid=kwargs.get("uuid"),
                completions=completions,
                response_tokens=[len(x) for x in completion_ids],
                scores=scores,
                phase_step=trainer_state.global_step - a.phase_start if protocol else None,
            ),
        )
        return vals

    class Evidence(TrainerCallback):
        tracking_run_id = None

        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            gradients = gradient_evidence(model)
            evidence.update(gradients)
            norm = math.hypot(gradients["adapter_gradient_norm"], gradients["backbone_gradient_norm"])
            signal = norm > 0 and not stop_requested()
            if not signal:
                optimizer.zero_grad(set_to_none=True)
            evidence["pending"] = signal

        def on_optimizer_step(self, args, state, control, **kwargs):
            evidence["applied_updates"] += int(evidence["pending"])
            if evidence["pending"] and getattr(trainer, "archlab_async_rollout", None) is not None:
                from archlab.rl.limite_actor import policy_snapshot

                trainer.archlab_async_rollout.publish(policy_snapshot(model, evidence["applied_updates"]))
            if rank == 0:
                write("updates.jsonl", dict(step=state.global_step + 1, **evidence))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if rank == 0:
                write(
                    "metrics.jsonl",
                    dict(
                        step=state.global_step,
                        **(logs or {}),
                        applied_updates=evidence["applied_updates"],
                    ),
                )
                if a.credentials:
                    import mlflow

                    active = mlflow.active_run()
                    if active:
                        if self.tracking_run_id != active.info.run_id:
                            mlflow.set_tags(
                                {
                                    "active_attention_backend": attention_backend,
                                    "active_attention_kernel": attention_kernel,
                                    "active_trainable_mode": trainable_mode,
                                    "active_communication_schedule": schedule,
                                    "active_source_revision": runtime_contract()["source_revision"],
                                    "active_async_rollouts": str(bool(execution.get("async_rollouts"))),
                                    "active_overlap_actor_learner": str(execution.get("overlap_actor_learner", True)),
                                    "active_max_policy_lag": str(execution.get("max_policy_lag", 0)),
                                    "math_protocol": protocol.contract()["version"] if protocol else "native-v1",
                                    "math_phase_start": str(a.phase_start),
                                    "curriculum_sha256": file_hash(a.curriculum) if a.curriculum else "none",
                                    "model_kind": "native" if native_checkpoint else "adapter",
                                    **({
                                        "publisher_repo": identity["publisher_identity"]["repo"],
                                        "publisher_revision": identity["publisher_identity"]["revision"],
                                        "inserted_layers": "0",
                                    } if native_checkpoint else {}),
                                }
                            )
                            self.tracking_run_id = active.info.run_id
                        (a.output / "MLFLOW.json").write_text(
                            json.dumps(
                                dict(
                                    run_id=active.info.run_id,
                                    experiment_id=active.info.experiment_id,
                                    experiment="Limite — RL",
                                    math_protocol=protocol.contract() if protocol else None,
                                    phase_start=a.phase_start if protocol else None,
                                    curriculum_sha256=file_hash(a.curriculum) if a.curriculum else None,
                                )
                            )
                        )

        def on_step_end(self, args, state, control, **kwargs):
            if publisher is not None:
                publisher.check()
            if evidence["flat_batches"] >= 8 or stop_requested():
                control.should_training_stop = True
            if (
                state.global_step == 1
                or (resume_receipt and state.global_step == resume_receipt["step"] + 1)
                or state.global_step % a.checkpoint_steps == 0
                or state.global_step == a.steps
                or control.should_training_stop
            ):
                if trainable_mode == "adapter" and frozen_fingerprint(model) != identity["frozen_sha256"]:
                    raise RuntimeError("frozen base changed during RL")
                payloads = checkpoint_payloads(trainer, evidence)
                if rank == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        a.output,
                        state.global_step,
                        0,
                        a.oss,
                        extra=dict(
                            **{key: value for key, value in identity.items() if key != "trainable_mode"},
                            **parent_contract, communication=communication,
                            rl_split_sha256=file_hash(a.data / "SPLIT.json"),
                            runtime_sequence_attention=a.runtime_sequence_attention,
                            source_revision=runtime_contract()["source_revision"],
                            math_protocol=protocol.contract() if protocol else None,
                            curriculum_sha256=file_hash(a.curriculum) if a.curriculum else None,
                            phase_start=a.phase_start if protocol else None,
                            protocol_migration=protocol_migration,
                            rollout_execution=execution,
                        ),
                        extra_payloads=payloads,
                        publisher=publisher,
                    )
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            return control

    if a.credentials:
        from archlab.tracking.mlflow_sync import configure_client

        configure_client(a.credentials)
        os.environ["MLFLOW_TRACKING_URI"] = json.loads(a.credentials.read_text())["tracking_uri"]
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "Limite — RL"
        tracking = a.output / "MLFLOW.json"
        previous_tracking = json.loads(tracking.read_text()) if tracking.exists() else {}
        same_tracking_phase = protocol is not None and (
            previous_tracking.get("math_protocol") == protocol.contract()
            and previous_tracking.get("phase_start") == a.phase_start
            and previous_tracking.get("curriculum_sha256") == (file_hash(a.curriculum) if a.curriculum else None)
        )
        new_tracking_phase = a.new_tracking_phase and not same_tracking_phase
        if new_tracking_phase and tracking.exists() and rank == 0:
            phases = a.output / "tracking-history"
            phases.mkdir(exist_ok=True)
            (phases / (json.loads(tracking.read_text())["run_id"] + ".json")).write_bytes(tracking.read_bytes())
        if a.resume and tracking.exists() and not new_tracking_phase:
            os.environ["MLFLOW_RUN_ID"] = json.loads(tracking.read_text())["run_id"]
        else:
            os.environ.pop("MLFLOW_RUN_ID", None)
    responses = execution.get("responses_per_update", world * 4)
    if responses <= 0 or responses % world or responses % 4:
        raise ValueError("response batch must divide evenly across ranks and four-sample groups")
    native = dict(
        output_dir=str(a.output / "trainer"),
        max_steps=a.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=responses // world,
        generation_batch_size=responses,
        num_generations=4,
        per_device_eval_batch_size=4,
        num_generations_eval=4,
        max_completion_length=protocol.max_tokens if protocol else 8192,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=False,
        beta=0.0,
        loss_type="dapo",
        scale_rewards="group",
        num_iterations=1,
        mask_truncated_completions=protocol is None,
        gradient_checkpointing=False,
        logging_steps=1,
        logging_nan_inf_filter=False,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=not bool(a.resume),
        report_to=["mlflow"] if a.credentials else [],
        run_name=("limite-violetto-native-full-rl-grpo" if native_checkpoint
                  else f"limite-base-{a.variant}-{trainable_mode}-grpo")
        + ("-tilelang" if attention_backend == "tilelang" else "")
        + ("-" + protocol.contract()["version"] if protocol else ""),
        seed=42,
        data_seed=42,
        disable_tqdm=True,
        ddp_find_unused_parameters=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        cache_implementation="dynamic",
        generation_kwargs=dict(use_cache=True, eos_token_id=[151645, 151643]),
    )
    if trainable_mode == "full":
        native["ddp_timeout"] = execution.get("rollout_timeout_seconds", 1800)
    optimizer = optimizer_for_model(model)

    class TrainingGRPO(BehaviorGRPO):
        def get_batch_samples(self, epoch_iterator, num_batches, device):
            if not execution.get("async_rollouts"):
                return super().get_batch_samples(epoch_iterator, num_batches, device)
            from archlab.rl.async_rollout import BatchLookahead

            lookahead = getattr(self, "archlab_batch_lookahead", None)
            if lookahead is None:
                lookahead = self.archlab_batch_lookahead = BatchLookahead()
            result = lookahead.collect(super().get_batch_samples, epoch_iterator, num_batches, device)
            self.archlab_next_rollout_prompts = (
                [row["prompt"] for row in lookahead.buffer] if lookahead.buffer is not None else None
            )
            return result

        def _get_train_sampler(self, dataset=None):
            if curriculum is None or a.test_rollouts:
                return super()._get_train_sampler(dataset)
            from archlab.rl.limite_curriculum import MathCurriculumSampler

            return MathCurriculumSampler(
                dataset if dataset is not None else self.train_dataset, curriculum,
                max_steps=a.steps, phase_start=protocol_spec["curriculum"].get("phase_start", a.phase_start),
                prompts_per_batch=self.args.generation_batch_size // self.num_generations,
                num_generations=self.num_generations,
                repeat_count=self.num_iterations * self.args.steps_per_generation,
                anneal_steps=protocol_spec["curriculum"]["anneal_steps"],
                initial_fraction=protocol_spec["curriculum"]["initial_fraction"],
                seed=self.args.seed,
            )

        def _load_from_checkpoint(self, checkpoint, model=None):
            # build_policy already verified and restored the model payloads.
            if Path(checkpoint) != resume_path:
                raise ValueError("trainer resume differs from the verified RL checkpoint")

        def _load_optimizer_and_scheduler(self, checkpoint):
            if checkpoint is None:
                return
            loaded_optimizer = torch.load(load_checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(loaded_optimizer)
            if a.test_rollouts:
                assert_state_equal(self.optimizer.state_dict(), loaded_optimizer)
            del loaded_optimizer
            state_path = load_checkpoint / "rl_state.pt"
            if state_path.exists():
                saved = torch.load(state_path, map_location="cpu", weights_only=True)
                self.lr_scheduler.load_state_dict(saved["scheduler"])
                if a.test_rollouts:
                    assert_state_equal(self.lr_scheduler.state_dict(), saved["scheduler"])
                    write(f"RESUME-rank{rank}.jsonl", dict(
                        kind="optimizer_scheduler", passed=True, checkpoint_step=resume_receipt["step"],
                        optimizer_values_exact=True, scheduler_values_exact=True,
                    ))
            else:
                scheduler = self.lr_scheduler.state_dict()
                scheduler.update(last_epoch=resume_receipt["step"], _step_count=resume_receipt["step"] + 1)
                self.lr_scheduler.load_state_dict(scheduler)

        def _load_rng_state(self, checkpoint):
            state_path = load_checkpoint / "rl_state.pt"
            if state_path.exists():
                saved = torch.load(state_path, map_location="cpu", weights_only=True)
                if a.resize_resume_world:
                    saved = resize_rank_state(saved, world)
                if protocol_migration:
                    from archlab.rl.limite_phase import discard_prefetched_rollouts

                    saved = discard_prefetched_rollouts(saved)
                restore_rng(saved["rank_rng"][rank])
                actor = getattr(self, "archlab_async_rollout", None)
                if actor is not None:
                    actor.restore(saved["rank_rng"][rank].get("rollout"),
                                  initial_rng=saved["rank_rng"][rank]["cuda"])
                if a.test_rollouts:
                    from archlab.rl.limite_checkpoint import capture_rng

                    expected_rng = {key: value for key, value in saved["rank_rng"][rank].items()
                                    if key != "rollout"}
                    assert_state_equal(capture_rng(), expected_rng)
                    if actor is not None:
                        rollout_rng = saved["rank_rng"][rank].get("rollout")
                        expected_actor = rollout_rng["generator"] if rollout_rng else expected_rng["cuda"]
                        assert_state_equal(actor.generator.get_state(), expected_actor)
                    write(f"RESUME-rank{rank}.jsonl", dict(
                        kind="rng", passed=True, checkpoint_step=resume_receipt["step"],
                        learner_rng_exact=True, actor_rng_exact=actor is not None,
                        discarded_old_protocol_prefetched_batches=saved["evidence"].get(
                            "discarded_old_protocol_prefetched_batches", 0),
                    ))
            else:
                import random

                import numpy as np

                saved = torch.load(Path(checkpoint) / "rng.pt", map_location="cpu", weights_only=True)
                torch.set_rng_state(saved["cpu"])
                random.seed(42 + rank)
                np.random.seed(42 + rank)
                if rank == 0:
                    torch.cuda.set_rng_state(saved["cuda"], device=local)
                else:
                    torch.cuda.manual_seed(42 + rank)

        def training_step(self, model, inputs, *args, **kwargs):
            with backward_context(model, schedule):
                loss = super().training_step(model, inputs, *args, **kwargs)
            finish_backward(self.model.parameters(), schedule, self.accelerator.sync_gradients)
            return loss

    rollout = native_rollout
    trainer_class = TrainingGRPO
    if a.test_rollouts:
        # Correctness fixture only: use real model likelihoods for a correct and an
        # incorrect arithmetic answer. Never included in production training.
        native.update(max_completion_length=128, eval_strategy="no", eval_on_start=False)
        import yaml

        fixture = yaml.safe_load(
            (Path(__file__).parents[1] / "prompts/limite_rl_canary_v1.yaml").read_text()
        )
        fixture_prompt = tok.apply_chat_template(
            fixture["messages"], tokenize=False, add_generation_prompt=True
        )
        datasets["train"] = Dataset.from_list(
            [
                dict(
                    prompt=fixture_prompt,
                    expected_answer=fixture["expected_answer"],
                    uuid="test-only",
                )
            ]
            * 32
        )
        if a.test_eval_transition:
            native.update(eval_strategy="steps", eval_steps=1)
            datasets["heldout"] = datasets["train"].select(range(world))

        def fixture_generate(prompts, policy):
            results = dict(prompt_ids=[], completion_ids=[], logprobs=[])
            for i, prompt in enumerate(prompts):
                x = tok.encode(prompt, add_special_tokens=False)
                answer, ending = [
                    ("42", "<|im_end|>"),
                    ("43", "<|im_end|>"),
                    ("42", "<|endoftext|>"),
                    ("42", ""),
                ][i % 4]
                y = tok.encode(
                    "\\boxed{" + answer + "}" + ending,
                    add_special_tokens=False,
                )
                ids = torch.tensor([x + y], device=local)
                with torch.no_grad():
                    logits = (
                        policy(ids, use_cache=False)
                        .logits[:, len(x) - 1 : -1]
                        .float()
                        .log_softmax(-1)
                    )
                    lp = (
                        logits.gather(-1, torch.tensor(y, device=local)[None, :, None])
                        .squeeze(-1)[0]
                        .tolist()
                    )
                results["prompt_ids"].append(x)
                results["completion_ids"].append(y)
                results["logprobs"].append(lp)
            return results

        def rollout(prompts, trainer):
            if getattr(trainer, "archlab_async_rollout", None) is not None and trainer.model.training:
                return native_rollout(prompts, trainer)
            return fixture_generate(prompts, model)

        class FixtureGRPO(TrainingGRPO):
            def _prepare_inputs(self, inputs):
                batch = super()._prepare_inputs(inputs)
                phase = "train" if self.model.training else "eval"
                expected_batch = self.args.per_device_train_batch_size if phase == "train" else self.args.per_device_eval_batch_size
                if a.test_eval_transition:
                    assert batch["completion_ids"].shape[0] == expected_batch
                    write(
                        f"REPLAY-rank{rank}.jsonl",
                        dict(step=self.state.global_step, phase=phase, batch_size=expected_batch),
                    )
                return batch

            def _generate_and_score_completions(self, inputs):
                batch = super()._generate_and_score_completions(inputs)
                mask = batch["completion_mask"]
                assert bool(mask[:3].any(-1).all())
                assert bool(mask[3].any()) if protocol else not bool(mask[3].any())
                zero_active_rank = (trainable_mode == "full" and world > 1 and rank == world - 1
                                    and self.state.global_step == initial_step)
                if zero_active_rank:
                    # Qualification only: one rank contributes no active loss
                    # tokens while its peer retains the real reward contrast.
                    batch["completion_mask"] = torch.zeros_like(mask)
                if a.test_flat_batch and self.state.global_step == 1:
                    # All ranks skip the middle update, then resume a real
                    # update. This proves public DDP scheduling stays live.
                    batch["advantages"] = torch.zeros_like(batch["advantages"])
                write(
                    f"MASKS-rank{rank}.jsonl",
                    dict(
                        passed=True,
                        im_end_retained=True,
                        eod_retained=True,
                        truncated_excluded=protocol is None,
                        truncated_trainable_failure=protocol is not None,
                        zero_active_rank=zero_active_rank,
                    ),
                )
                return batch

        trainer_class = FixtureGRPO

        def fixture_gap():
            prompt_ids = tok.encode(fixture_prompt, add_special_tokens=False)
            scores = []
            with torch.no_grad():
                for answer in ("42", "43"):
                    tail = tok.encode("\\boxed{" + answer + "}<|im_end|>", add_special_tokens=False)
                    ids = torch.tensor([prompt_ids + tail], device=local)
                    lp = (
                        model(ids, use_cache=False)
                        .logits[:, len(prompt_ids) - 1 : -1]
                        .float()
                        .log_softmax(-1)
                    )
                    scores.append(
                        float(lp.gather(-1, torch.tensor(tail, device=local)[None, :, None]).mean())
                    )
            return scores[0] - scores[1]

        gap_before = fixture_gap()
        backbone_before = parameter_probe(model, backbone_only=True, matrices_only=True)

    trainer = trainer_class(
        model=model,
        args=GRPOConfig(**native),
        reward_funcs=reward,
        train_dataset=datasets["train"],
        eval_dataset=datasets["heldout"],
        processing_class=tok,
        rollout_func=rollout,
        optimizers=(optimizer, None),
        callbacks=[Evidence()],
    )
    if protocol is not None:
        trainer.archlab_graph_decode = protocol_spec["execution"]["graph_decode"] and not a.test_rollouts
        trainer.archlab_skip_flat_backward = protocol_spec["execution"]["skip_flat_backward"]
        trainer.archlab_trim_replay = protocol_spec["execution"]["trim_replay"]
        trainer.archlab_stop_requested = stop_requested
        trainer.archlab_compact_decode = execution.get("compact_decode", False)
        trainer.archlab_reuse_decode = execution.get("reuse_decode", False)
        trainer.archlab_max_policy_lag = execution.get("max_policy_lag", 1)
        trainer.archlab_overlap_actor_learner = execution.get("overlap_actor_learner", True)
        if not a.test_rollouts:
            trainer.generation_config.archlab_budget_mode = protocol.budget_mode
        trainer.archlab_chunked_policy_scores = bool(execution.get("replay_head_chunk_size"))
    # Trainer/Accelerate creates the distributed group. Recover a legacy view
    # only after that boundary so every rank observes rank zero's completed file.
    if a.resume:
        if (a.resume / "rl_state.pt").exists():
            saved_rl = torch.load(load_checkpoint / "rl_state.pt", map_location="cpu", weights_only=True)
            if saved_rl["world_size"] != world:
                if not a.resize_resume_world:
                    raise ValueError("RL resume requires the saved distributed world size")
                saved_rl = resize_rank_state(saved_rl, world)
            if protocol_migration:
                from archlab.rl.limite_phase import discard_prefetched_rollouts

                saved_rl = discard_prefetched_rollouts(saved_rl)
            evidence.update(saved_rl["evidence"])
        else:
            if rank == 0:
                resume_path, resume_migration = legacy_resume_state(
                    a.resume, a.output, max_steps=a.steps, train_batch_size=trainer.args.train_batch_size,
                )
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            resume_path, resume_migration = legacy_resume_state(
                a.resume, a.output, max_steps=a.steps, train_batch_size=trainer.args.train_batch_size,
            )
            evidence.update(resume_migration["evidence"])
    if execution.get("async_rollouts"):
        from archlab.architectures.limite_adapter import enable_runtime_sequence_attention
        from archlab.architectures.limite_gqa import set_native_decode_gqa
        from archlab.rl.limite_actor import native_actor_queue

        if not trainer.archlab_overlap_actor_learner:
            from archlab.rl.rollout_rendezvous import RolloutRendezvous

            trainer.archlab_rollout_rendezvous = RolloutRendezvous(
                world, timeout_seconds=execution.get("rollout_timeout_seconds", 1800),
            )

        actor = build_policy(load_checkpoint)
        actor.requires_grad_(False)
        if not native_checkpoint:
            enable_runtime_sequence_attention(actor)
        if execution.get("native_decode_gqa"):
            set_native_decode_gqa(actor, backend=execution.get("native_decode_backend", "sdpa"))
        trainer.archlab_policy_version = lambda: evidence["applied_updates"]
        trainer.archlab_async_rollout = native_actor_queue(
            actor, model, trainer, evidence["applied_updates"],
            fixture_generate if a.test_rollouts else None,
        )
    if rank == 0:
        contract_path = a.output / "CONTRACT.json"
        if a.resume and contract_path.exists():
            history = a.output / "contract-history"
            history.mkdir(exist_ok=True)
            prior = contract_path.read_text()
            (history / (file_hash(contract_path) + ".json")).write_text(prior)
        (a.output / "CONTRACT.json").write_text(
            json.dumps(
                dict(
                    **({} if native_checkpoint else dict(adapter=model.model.adapter_config)),
                    attention_backend=attention_backend,
                    normal_kernel=normal_kernel,
                    normal_backward=normal_backward,
                    normal_backward_runtime=(dict(backend=native_replay_backend, replay=model.archlab_native_replay)
                                             if native_replay_enabled else dict(backend="publisher") if native_checkpoint
                                             else model.archlab_normal_attention_backward_contract),
                    attention_kernel=attention_kernel,
                    **parent_contract,
                    **identity,
                    communication=communication,
                    optimizer=dict(name="SignalFusedAdam", master_weights=trainable_mode == "full", initial_state="resumed_rl" if a.resume else "fresh_rl"),
                    phase_initialization=(
                        dict(publisher_checkpoint=True, fresh_rl_optimizer=not bool(a.resume), rl_seed=42)
                        if native_checkpoint else dict(
                            warmup_optimizer_and_rng_retained=True,
                            continue_warmup_optimizer=False, continue_warmup_rng=False, rl_seed=42,
                        )
                    ),
                    execution=dict(
                        runtime_sequence_attention=a.runtime_sequence_attention,
                        sampling_probabilities="streamed_selected_tokens_native_unwarped",
                        checkpoint_steps=a.checkpoint_steps,
                        rollout_execution=execution,
                        checkpoint_cache=str(a.checkpoint_cache) if a.checkpoint_cache else None,
                        checkpoint_resume="trainer_clock_optimizer_scheduler_all_rank_rng",
                        async_oss_publication=publisher is not None,
                        math_protocol=protocol.contract() if protocol else None,
                        protocol_spec=protocol_spec,
                        curriculum_sha256=file_hash(a.curriculum) if a.curriculum else None,
                        phase_start=a.phase_start if protocol else None,
                    ),
                    resume=dict(checkpoint=str(a.resume), migration=resume_migration,
                                protocol_migration=protocol_migration) if a.resume else None,
                    config=native,
                    correctness_fixture=a.test_rollouts,
                    runtime=runtime_contract(),
                    split_sha256=file_hash(a.data / "SPLIT.json"),
                    dataset_counts={name: len(data) for name, data in datasets.items()},
                ),
                indent=2,
            )
        )
    initial_applied_updates = evidence["applied_updates"]
    initial_step = resume_receipt["step"] if resume_receipt else 0
    try:
        trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
    except BaseException:
        if publisher is not None:
            publisher.close()
        raise
    finally:
        if getattr(trainer, "archlab_async_rollout", None) is not None:
            trainer.archlab_async_rollout.close()
    publication_error = None
    if publisher is not None:
        try:
            publisher.close()
        except BaseException as exc:
            publication_error = exc
    publication_failed = torch.tensor(int(publication_error is not None), device=local)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(publication_failed)
    if bool(publication_failed):
        raise RuntimeError("RL checkpoint publication failed; local snapshot retained") from publication_error
    if trainable_mode == "adapter" and frozen_fingerprint(model) != identity["frozen_sha256"]:
        raise RuntimeError("base changed")
    if a.test_rollouts:
        gap_after = fixture_gap()
        expected_updates = (initial_applied_updates + a.steps - initial_step
                            - int(a.test_flat_batch and initial_step <= 1 < a.steps))
        if gap_after <= gap_before or evidence["applied_updates"] != expected_updates:
            raise AssertionError(
                "RL fixture failed to apply every update and improve correct-answer log odds"
            )
        backbone_after = parameter_probe(model, backbone_only=True, matrices_only=True)
        moved = any(not torch.equal(backbone_before[name], value) for name, value in backbone_after.items())
        if trainable_mode == "full" and (not moved or evidence["backbone_gradient_norm"] <= 0):
            raise AssertionError("full-weight RL fixture did not update the backbone")
        # A flat advantage group has a zero loss derivative. The callback clears
        # zero gradients before this public optimizer call, so Adam state cannot
        # advance. Exercise the exact optimizer used by the training run.
        optimizer.zero_grad(set_to_none=True)
        flat_before = parameter_probe(model)
        state_before = optimizer_probe(optimizer.state_dict())
        optimizer.step()
        assert_state_equal(optimizer_probe(optimizer.state_dict()), state_before)
        assert_state_equal(parameter_probe(model), flat_before)
        if rank == 0:
            checkpoint = a.output / "checkpoints" / f"step-{trainer.state.global_step:07d}"
            restored = build_policy(checkpoint)
            assert_state_equal(restored.state_dict(), model.state_dict())
            restored_optimizer = optimizer_for_model(restored)
            restored_optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
            assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
            # Supply identical native-dtype gradients to both optimizers. This
            # tests the exact next update without conflating atomic kernel
            # ordering with checkpoint/master-state fidelity.
            for parameter, loaded_parameter in zip(model.parameters(), restored.parameters(), strict=True):
                if parameter.requires_grad:
                    parameter.grad = torch.full_like(parameter, 0.125)
                    loaded_parameter.grad = parameter.grad.clone()
            optimizer.step()
            restored_optimizer.step()
            assert_state_equal(restored.state_dict(), model.state_dict())
            assert_state_equal(restored_optimizer.state_dict(), optimizer.state_dict())
            optimizer.zero_grad(set_to_none=True)
            del restored_optimizer, restored
            write("CHECKPOINT_RELOAD.jsonl", dict(passed=True, trainable_mode=trainable_mode, model_values_exact=True, optimizer_values_exact=True, next_update_exact=True, next_update_oracle="identical_native_dtype_gradients"))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        write(
            f"CORRECTNESS-rank{rank}.jsonl",
            dict(
                passed=True,
                correct_answer_gap_before=gap_before,
                correct_answer_gap_after=gap_after,
                **identity,
                backbone_moved=moved,
                flat_optimizer_no_update=True,
                **{key: value for key, value in evidence.items() if "gradient" in key},
                applied_updates=evidence["applied_updates"],
            ),
        )
    status = (
        "no_signal"
        if evidence["flat_batches"] >= 8
        else "stopped"
        if stop_requested()
        else "complete"
    )
    if rank == 0:
        (a.output / "FINISHED.json").write_text(
            json.dumps(
                dict(
                    step=trainer.state.global_step,
                    status=status,
                    **identity,
                    **evidence,
                )
            )
        )
    if status == "no_signal":
        raise RuntimeError("RL paused after eight batches without reward contrast")


if __name__ == "__main__":
    try:
        main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

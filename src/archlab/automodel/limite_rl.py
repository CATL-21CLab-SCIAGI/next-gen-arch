"""Native upstream TRL GRPO launch for the verified Limite base checkpoint.

TRL owns generation, optimization, checkpointing and RNG state. Local callbacks
record reward evidence, skip flat-gradient Adam steps, and honor graceful stops.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path


def append(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(dict(time=time.time(), **row), allow_nan=False) + "\n")


def main():
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TrainerCallback
    from trl import GRPOConfig

    from archlab.architectures.limite_loader import load_model
    from archlab.optimizers.rl_adam import SignalFusedAdam
    from archlab.rl.limite_data import math_reward
    from archlab.rl.limite_rollout import native_rollout
    from archlab.rl.nemotron_data import sha256_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--oss-checkpoints", type=Path, required=True)
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text())
    root = args.run_root
    oss_checkpoints = args.oss_checkpoints
    if (root / "checkpoints").is_symlink():
        raise ValueError("native safetensors saves require NAS staging, not an OSS symlink")
    admission = json.loads(args.admission.read_text())
    if not admission["passed"] or admission["backward_length"] < config["context_length"]:
        raise ValueError("missing successful full-context admission")
    optimizer_admission = json.loads(
        (args.admission.parent / "OPTIMIZER_ADMISSION.json").read_text()
    )
    if not optimizer_admission["passed"]:
        raise ValueError("missing master-weight and optimizer restoration admission")
    if (root / "LAUNCH.json").exists():
        raise ValueError("fresh run required; resume needs an explicit checkpoint")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(config["allocator_fraction"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    generation = json.loads((args.model / "generation_config.json").read_text())
    tokenizer.eos_token_id = generation["eos_token_id"]
    tokenizer.bos_token_id = generation["bos_token_id"]
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = generation["pad_token_id"]
    if tokenizer.eos_token != "<|im_end|>" or tokenizer.pad_token != "<|endoftext|>":
        raise ValueError("unexpected native Limite generation token identities")
    split = json.loads((args.data / "SPLIT.json").read_text())
    for filename, expected in split["files"].items():
        if sha256_file(args.data / filename) != expected:
            raise ValueError("prepared split changed")
    datasets = {}
    context_exclusions = []
    for name in ("train", "heldout"):
        rows = [
            json.loads(line) for line in (args.data / (name + ".jsonl")).read_text().splitlines()
        ]
        keep = []
        for row in rows:
            if len(row["input_ids"]) > config["max_prompt_length"]:
                context_exclusions.append(
                    dict(uuid=row["uuid"], split=name, reason="prompt_exceeds_declared_context")
                )
                continue
            if tokenizer.encode(row["prompt"], add_special_tokens=False) != row["input_ids"]:
                raise ValueError("prepared native prompt tokens changed")
            if tokenizer(row["prompt"])["input_ids"] != row["input_ids"]:
                raise ValueError("default trainer tokenization adds unexpected special tokens")
            keep.append({k: v for k, v in row.items() if k != "input_ids"})
        datasets[name] = Dataset.from_list(keep)
    (root / "context-exclusions.json").write_text(json.dumps(context_exclusions, indent=2) + "\n")
    model = load_model(args.model, attn_implementation="sdpa").to("cuda")

    class GracefulStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return (root / "STOP_REQUEST").exists()

    from functools import wraps

    native_generate = model.generate

    @wraps(native_generate)
    def bounded_generate(*positional, **kwargs):
        criteria = list(kwargs.pop("stopping_criteria", []))
        kwargs["stopping_criteria"] = StoppingCriteriaList(criteria + [GracefulStop()])
        kwargs["disable_compile"] = True
        return native_generate(*positional, **kwargs)

    model.generate = bounded_generate
    evidence = dict(applied_updates=0, flat_batches=0, nonflat_groups=0)
    ending_ids = set(config["trainer"]["generation_kwargs"]["eos_token_id"])

    def reward(completions, expected_answer, completion_ids, trainer_state, **kwargs):
        values = [
            math_reward(c, a) if ids and ids[-1] in ending_ids else 0.0
            for c, a, ids in zip(completions, expected_answer, completion_ids, strict=True)
        ]
        groups = [
            values[i : i + config["num_generations"]]
            for i in range(0, len(values), config["num_generations"])
        ]
        nonflat = sum(min(g) < max(g) for g in groups)
        if model.training:
            evidence["nonflat_groups"] = nonflat
            evidence["flat_batches"] = 0 if nonflat else evidence["flat_batches"] + 1
        append(
            root / "rollouts.jsonl",
            dict(
                step=trainer_state.global_step,
                phase="train" if model.training else "eval",
                rewards=values,
                uuid=kwargs.get("uuid"),
                completions=completions,
                response_tokens=[len(x) for x in completion_ids],
                nonflat_groups=nonflat,
                truncation_rate=sum(not x or x[-1] not in ending_ids for x in completion_ids)
                / len(values),
            ),
        )
        return values

    class Evidence(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            import mlflow

            run = mlflow.active_run()
            if run is not None:
                (root / "MLFLOW.json").write_text(
                    json.dumps(
                        dict(
                            run_id=run.info.run_id,
                            experiment_id=run.info.experiment_id,
                            name=config["name"],
                            experiment="Limite — RL",
                        )
                    )
                    + "\n"
                )

        def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            if not all(bool(torch.isfinite(g).all()) for g in grads):
                raise FloatingPointError("nonfinite gradient; refusing optimizer update")
            has_signal = any(bool(g.count_nonzero()) for g in grads)
            if (root / "STOP_REQUEST").exists():
                has_signal = False
            if not has_signal:
                # Native Adam skips parameters with grad=None, including momentum
                # and weight-decay updates. A zero tensor would not be sufficient.
                optimizer.zero_grad(set_to_none=True)
            evidence["pending_update"] = has_signal

        def on_optimizer_step(self, args, state, control, **kwargs):
            has_signal = evidence.pop("pending_update", False)
            evidence["applied_updates"] += int(has_signal)
            append(
                root / "updates.jsonl",
                dict(
                    step=state.global_step + 1,
                    applied=has_signal,
                    applied_updates=evidence["applied_updates"],
                ),
            )

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1 or (root / "STOP_REQUEST").exists():
                control.should_save = True
            if (root / "STOP_REQUEST").exists() or evidence["flat_batches"] >= config[
                "max_flat_batches"
            ]:
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            values = {k: float(v) for k, v in (logs or {}).items() if isinstance(v, (int, float))}
            if not all(math.isfinite(v) for v in values.values()):
                raise FloatingPointError("nonfinite trainer metric")
            append(
                root / "metrics.jsonl",
                dict(step=state.global_step, applied_updates=evidence["applied_updates"], **values),
            )

        def on_save(self, args, state, control, **kwargs):
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            for source in Path(launch["model"]).glob("*.py"):
                shutil.copyfile(source, checkpoint / source.name)
            files = {
                str(p.relative_to(checkpoint)): sha256_file(p)
                for p in checkpoint.rglob("*")
                if p.is_file()
            }
            if not any(k.endswith(".safetensors") for k in files) or "optimizer.pt" not in files:
                raise ValueError("incomplete native RL checkpoint")
            (checkpoint / "VERIFIED.json").write_text(
                json.dumps(dict(step=state.global_step, files=files), indent=2) + "\n"
            )
            destination = oss_checkpoints / checkpoint.name
            destination.mkdir(parents=True, exist_ok=False)
            for relative, expected in files.items():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(checkpoint / relative, target)
                if sha256_file(target) != expected:
                    raise ValueError("OSS checkpoint checksum mismatch")
            shutil.copyfile(checkpoint / "VERIFIED.json", destination / "VERIFIED.json")
            published = root / "published"
            published.mkdir(exist_ok=True)
            (published / checkpoint.name).symlink_to(destination, target_is_directory=True)

    native = dict(config["trainer"])
    native.update(
        output_dir=str(root / "checkpoints"),
        num_generations=config["num_generations"],
        max_completion_length=config["max_completion_length"],
        report_to=["mlflow"],
        run_name=config["name"],
    )
    from archlab.tracking.mlflow_sync import configure_client

    configure_client(args.credentials)
    credentials = json.loads(args.credentials.read_text())
    os.environ["MLFLOW_TRACKING_URI"] = credentials["tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Limite — RL"

    from archlab.rl.limite_trainer import BehaviorGRPO

    optimizer = SignalFusedAdam(
        model.parameters(),
        lr=native["learning_rate"],
        betas=(native["adam_beta1"], native["adam_beta2"]),
        weight_decay=0.0,
        master_weights=True,
        master_weight_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
    )
    trainer = BehaviorGRPO(
        model=model,
        args=GRPOConfig(**native),
        reward_funcs=reward,
        train_dataset=datasets["train"],
        eval_dataset=datasets["heldout"].select(range(4)),
        processing_class=tokenizer,
        callbacks=[Evidence()],
        rollout_func=native_rollout,
        optimizers=(optimizer, None),
    )
    launch = dict(
        config=config,
        model=str(args.model),
        data=str(args.data),
        admission=admission,
        train_rows=len(datasets["train"]),
        heldout_rows=len(datasets["heldout"]),
        heldout_evaluation="four fixed pilot problems before training and every 32 steps; full 128 reserved",
        versions=admission["versions"],
        optimizer=dict(
            implementation="transformer_engine.pytorch.optimizers.FusedAdam",
            master_weights=True,
            master_dtype="float32",
            moment_dtype="float32",
            no_signal_step_guard=True,
            admission=optimizer_admission,
        ),
        container_image="dev/nemo:26.06",
        image_digest_refresh="unavailable: no configured cloud credential provider",
        upstream_code="unchanged upstream formulas; original BF16/F32 parameter dtypes and oracle_exact head; no AMP; TE Adam FP32 masters",
        verifier="math-verify 0.8.0 symbolic; no judge fallback",
    )
    (root / "LAUNCH.json").write_text(json.dumps(launch, indent=2) + "\n")
    try:
        trainer.train()
        (root / "FINISHED.json").write_text(
            json.dumps(dict(step=trainer.state.global_step, **evidence)) + "\n"
        )
    except BaseException as error:
        (root / "FAILED.json").write_text(
            json.dumps(dict(error_type=type(error).__name__, message=str(error))) + "\n"
        )
        raise


if __name__ == "__main__":
    main()

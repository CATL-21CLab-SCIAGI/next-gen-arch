"""Matched, single-GPU A/B/C pilot using native Megatron training APIs.

Unlike the frozen production launcher, this adapter owns a bounded pilot loop
and evaluates a fixed held-out window independently of the training batch size.
It never changes installed runtime code or launches/stops DLC allocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import sys
import time
from pathlib import Path

import torch

from archlab.architectures.qwen38_flash_next_full import (
    SOURCE_CONFIG_SHA256,
    TOKENIZER_SHA256,
    Qwen38FlashNextFullConfig,
)
from archlab.megatron.backend import validate_runtime
from archlab.megatron.qwen38_flash_next_full_train import (
    TRAIN_STEPS,
    DPRankTokenBatches,
    _assert_dp_only_groups,
    _atomic_json,
    _forward_step,
    _megatron_argv,
    _sha256,
    _tag_native_optimizer_fallbacks,
    _validated_data_prefixes,
    build_model,
)
from archlab.megatron.qwen38_flash_next_full_train import _parser as baseline_parser


def pilot_argv(options):
    baseline = baseline_parser().parse_args([
        "--data-root", str(options.data_root), "--tokenizer", str(options.tokenizer),
        "--run-dir", str(options.run_dir), "--model-variant", "w320-e32-depth48-no-mtp",
        "--parallelism", "dp-only", "--micro-batch-size", str(options.micro_batch),
        "--global-batch-size", str(options.global_batch), "--seed", str(options.seed),
        "--probe-steps", str(options.steps), "--probe-save-interval", str(options.save_interval),
        "--fused-moe", "--fused-cross-entropy", "--no-resume",
    ])
    argv = _megatron_argv(baseline, Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp())
    # Preserve the full baseline's LR horizon even though this run stops early.
    argv.extend(["--lr-decay-iters", str(TRAIN_STEPS),
                 "--no-gradient-accumulation-fusion",
                 "--attention-backend", "unfused",
                 "--eval-global-batch-size", str(options.eval_sequences),
                 "--eval-micro-batch-size", str(options.micro_batch)])
    if options.resume:
        argv.extend(["--load", str(options.run_dir / "checkpoints")])
    return argv


def _write_event(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    print(json.dumps(record, sort_keys=True), flush=True)


def _optimizer_tree(optimizer):
    result = []
    for item in getattr(optimizer, "chained_optimizers", [optimizer]):
        inner = getattr(item, "optimizer", None)
        result.append({"wrapper": type(item).__name__, "inner": type(inner).__name__,
                       "groups": [{key: value for key, value in group.items()
                                   if key != "params" and isinstance(value, (str, int, float, bool, type(None)))}
                                  | {"parameter_tensors": len(group["params"])}
                                  for group in item.param_groups]})
    return result


def optimizer_tensor_hashes(optimizer, *, perturb=False):
    """Audit native FP32 master parameters and momentum/Adam buffers on DP1."""
    hashes = {}
    for oi, wrapper in enumerate(getattr(optimizer, "chained_optimizers", [optimizer])):
        inner = wrapper.optimizer
        for gi, group in enumerate(inner.param_groups):
            for pi, parameter in enumerate(group["params"]):
                tensors = {"master_parameter": parameter, **inner.state.get(parameter, {})}
                for key, value in tensors.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                    hashes[f"{oi}/{gi}/{pi}/{key}"] = hashlib.sha256(raw.tobytes()).hexdigest()
                    if perturb and pi == 0 and value.is_floating_point():
                        with torch.no_grad():
                            value.add_(0.125)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument("--mode", choices=("probe", "train"), default="train")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initialization-reference", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=358)
    parser.add_argument("--global-batch", type=int, default=4096)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--eval-interval", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=64)
    parser.add_argument("--short-window", type=int, default=16)
    parser.add_argument("--long-window", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    options = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("these DSW pilots are DP1 only; no other parallelism is supported")
    if min(options.steps, options.micro_batch, options.global_batch, options.eval_sequences,
           options.eval_interval, options.save_interval) < 1:
        raise ValueError("pilot integer controls must be positive")
    if options.global_batch % options.micro_batch or options.eval_sequences % options.micro_batch:
        raise ValueError("training/evaluation batches must divide by microbatch")
    if options.mode == "train" and (
        options.global_batch != 4096 or options.steps != 358 or options.micro_batch != 4
        or options.eval_sequences != 64 or options.eval_interval != 32 or options.save_interval != 64
    ):
        raise ValueError("training pilot must retain the approved 3B-token/effective-batch contract")
    if (options.short_window, options.long_window) != (16, 128):
        raise ValueError("this named pilot fixes the windows at 16 x 128")
    if options.run_dir.exists() and any(options.run_dir.iterdir()) and not options.resume:
        raise ValueError("fresh pilot refuses to overwrite an existing run directory")
    options.run_dir.mkdir(parents=True, exist_ok=True)
    if options.resume and not (options.run_dir / "checkpoints/latest_checkpointed_iteration.txt").is_file():
        raise ValueError("resume requires a completed native checkpoint")
    ready = json.loads((options.data_root / "DATA_READY.json").read_text())
    if ready.get("tokenizer_sha256") != TOKENIZER_SHA256:
        raise ValueError("dataset tokenizer contract differs from baseline")
    if _sha256(options.tokenizer / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("tokenizer hash drift")
    if _sha256(options.tokenizer / "config.json") != SOURCE_CONFIG_SHA256:
        raise ValueError("source config hash drift")
    train_prefixes, val_prefixes = _validated_data_prefixes(options.data_root)
    if set(train_prefixes) & set(val_prefixes):
        raise ValueError("training and held-out prefixes overlap")
    sys.argv = pilot_argv(options)

    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.enums import ModelType
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.rerun_state_machine import RerunDataIterator
    from megatron.core.utils import get_model_config, unwrap_model
    from megatron.training import get_timers
    from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
    from megatron.training.checkpointing import load_checkpoint, save_checkpoint
    from megatron.training.global_vars import set_global_variables
    from megatron.training.initialize import initialize_megatron
    from megatron.training.training import (
        get_optimizer_param_scheduler, setup_model_and_optimizer, train_step,
    )

    from archlab.megatron.simplicial_attention import (
        EXTRA_MARKERS, install_pilot_attention, parameter_hashes,
    )

    native = validate_args(parse_args())
    native.tensorboard_dir = None
    set_global_variables(native)
    initialize_megatron()
    torch.cuda.set_per_process_memory_fraction(0.70)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    topology = _assert_dp_only_groups(groups)
    architecture = Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()
    runtime = validate_runtime(require_pretrain=False)
    runtime.update(torch_resolved=torch.__version__, cuda=torch.version.cuda)
    import fla
    import emerging_optimizers
    import transformer_engine
    import triton

    runtime.update(fla=fla.__version__, transformer_engine_resolved=transformer_engine.__version__,
                   triton_resolved=triton.__version__, python_executable=sys.executable)
    optimizer_root = Path(emerging_optimizers.__file__).parent
    runtime.update(emerging_optimizers=emerging_optimizers.__version__,
                   emerging_optimizers_source=str(optimizer_root),
                   emerging_optimizers_source_sha256={str(p.relative_to(optimizer_root)): _sha256(p)
                                                     for p in sorted(optimizer_root.rglob("*.py"))},
                   native_attention_backend="unfused", gradient_accumulation_fusion=False)
    source_root = Path(__file__).parents[1]
    hashes = {str(path.relative_to(source_root)): _sha256(path) for path in
              (Path(__file__), Path(__file__).with_name("simplicial_attention.py"),
               source_root / "architectures/simplicial_attention.py",
               source_root / "architectures/simplicial_kernels.py",
               Path(__file__).with_name("qwen38_flash_next_full_train.py"),
               source_root / "architectures/qwen38_flash_next_full.py")}
    contract = {"arm": options.arm, "mode": options.mode, "host": socket.gethostname(),
                "runtime": runtime, "topology": topology, "steps": options.steps,
                "global_batch": options.global_batch, "micro_batch": options.micro_batch,
                "tokens_per_step": options.global_batch * 2048, "seed": options.seed,
                "eval_sequences": options.eval_sequences, "lr_horizon_steps": TRAIN_STEPS,
                "rotary_fraction": 0.25, "rope_theta": 10000000,
                "positional_encoding": "ordinary-RoPE-on-Q-K1-K2-not-relative-invariant-trilinear",
                "windows": [options.short_window, options.long_window],
                "train_prefixes": list(map(str, train_prefixes)),
                "heldout_prefixes": list(map(str, val_prefixes)),
                "manifest_sha256": _sha256(options.data_root / "DATA_READY.json"),
                "source_sha256": hashes, "runtime_installations": False,
                "container": {key: os.environ.get(key) for key in
                              ("NVIDIA_PRODUCT_NAME", "NVIDIA_BUILD_ID", "NVIDIA_PYTORCH_VERSION", "CUDA_VERSION")}}
    if options.resume:
        previous = json.loads((options.run_dir / "RUN_CONTRACT.json").read_text())
        if any(previous.get(k) != contract[k] for k in
               ("arm", "mode", "global_batch", "micro_batch", "seed", "windows", "manifest_sha256", "source_sha256")):
            raise ValueError("resume contract mismatch")
    else:
        _atomic_json(options.run_dir / "RUN_CONTRACT.json", contract)

    def provider(pre_process=True, post_process=True, vp_stage=None, config=None, pg_collection=None):
        cfg = config or core_transformer_config_from_args(native)
        pg = pg_collection or groups
        model = build_model(architecture, cfg, pg, pre_process=pre_process,
                            post_process=post_process, vp_stage=vp_stage)
        baseline_hash = parameter_hashes(model)
        install_pilot_attention(model, options.arm, seed=options.seed,
                                short_window=options.short_window, long_window=options.long_window)
        common_hash = parameter_hashes(model, common_only=True)
        if baseline_hash != common_hash:
            raise RuntimeError("attention replacement changed a common parameter")
        if options.initialization_reference:
            expected = json.loads(options.initialization_reference.read_text())["common_parameter_sha256"]
            if common_hash != expected:
                raise RuntimeError("paired common-weight initialization differs from reference")
        count = sum(p.numel() for p in model.parameters())
        expected_count = 387680960 + (245952 if options.arm == "C" else 0)
        if count != expected_count:
            raise RuntimeError(f"parameter-count contract drift: {count} != {expected_count}")
        partition = _tag_native_optimizer_fallbacks(model)
        shape_map = {name: {"shape": list(p.shape), "count": p.numel(),
                            "optimizer": getattr(p, "archlab_optimizer", "adamw")}
                     for name, p in model.named_parameters()}
        if options.arm == "C":
            for name, entry in shape_map.items():
                if any(marker in name for marker in (".attention.k2.", ".attention.v2.")):
                    if entry["optimizer"] != "muon" or entry["shape"] != [64, 320]:
                        raise RuntimeError("extra K/V optimizer grouping changed")
        _atomic_json(options.run_dir / "INITIALIZATION.json", {"common_parameter_sha256": common_hash,
                     "all_parameter_sha256": parameter_hashes(model), "total_parameters": count,
                     "matches_reference": bool(options.initialization_reference)})
        _atomic_json(options.run_dir / "MODEL_SHAPES.json", shape_map)
        _atomic_json(options.run_dir / "OPTIMIZER_PARTITION.json", partition)
        return model

    context = {}
    model, optimizer, scheduler = setup_model_and_optimizer(provider, ModelType.encoder_or_decoder,
                                                            checkpointing_context=context)
    _atomic_json(options.run_dir / "NATIVE_OPTIMIZER.json", {"optimizers": _optimizer_tree(optimizer)})
    _atomic_json(options.run_dir / "INITIAL_SCHEDULER.json", scheduler.state_dict())
    if scheduler.state_dict()["lr_decay_steps"] != TRAIN_STEPS * options.global_batch:
        raise RuntimeError("native LR schedule no longer uses the full baseline horizon")
    cfg = get_model_config(model[0])
    cfg.grad_scale_func = optimizer.scale_loss
    cfg.timers = get_timers()
    if native.overlap_grad_reduce:
        cfg.no_sync_func = model[0].no_sync
        if native.align_grad_reduce:
            cfg.grad_sync_func = model[0].start_grad_sync
    cfg.finalize_model_grads_func = finalize_model_grads
    schedule = get_forward_backward_func()
    raw_model = unwrap_model(model)[0]
    accumulation = options.global_batch // options.micro_batch
    train_source = DPRankTokenBatches(train_prefixes, batch_size=options.micro_batch,
                     sequence_len=2048, start_batch=native.iteration * accumulation,
                     device=torch.device("cuda", 0))
    train_iterator = RerunDataIterator(train_source)
    event_path = options.run_dir / "metrics.jsonl"
    stop = {"requested": False}

    def request_stop(*_):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def evaluate(iteration):
        # Recreate exactly the same held-out token window at every evaluation.
        validation = DPRankTokenBatches(val_prefixes, batch_size=options.micro_batch,
                         sequence_len=2048, start_batch=0, device=torch.device("cuda", 0))
        for chunk in model:
            chunk.eval()
        digest = hashlib.sha256()

        def validation_forward(iterator, wrapped_model, return_schedule_plan=False):
            batch = next(iterator)
            digest.update(batch["tokens"].detach().cpu().numpy().tobytes())
            return _forward_step(iter([batch]), wrapped_model, return_schedule_plan)

        with torch.no_grad():
            reduced = schedule(forward_step_func=validation_forward, data_iterator=validation,
                         model=model, num_microbatches=options.eval_sequences // options.micro_batch,
                         seq_length=2048, micro_batch_size=options.micro_batch, forward_only=True)
        totals = torch.stack([item["lm loss"] for item in reduced]).sum(dim=0)
        ce = (totals[0] / totals[1]).item()
        if not math.isfinite(ce):
            raise RuntimeError("nonfinite held-out cross entropy")
        _write_event(event_path, {"event": "eval", "step": iteration, "heldout_ce": ce,
                                  "heldout_tokens": int(totals[1]), "time_unix": time.time(),
                                  "heldout_tokens_sha256": digest.hexdigest()})
        for chunk in model:
            chunk.train()

    def checkpoint(iteration):
        native.iteration = iteration
        native.consumed_train_samples = iteration * options.global_batch
        save_checkpoint(iteration, model, optimizer, scheduler, 0,
                        checkpointing_context=context)
        _write_event(event_path, {"event": "checkpoint", "step": iteration, "time_unix": time.time()})

    # Microbatch heartbeat lets a 1024-microbatch DP1 step remain observable.
    progress = {"microbatch": 0, "step": native.iteration + 1, "last": time.monotonic()}
    train_digest = hashlib.sha256()

    def forward_step(iterator, wrapped_model, return_schedule_plan=False):
        batch = next(iterator)
        if progress["microbatch"] < 4:
            train_digest.update(batch["tokens"].detach().cpu().numpy().tobytes())
        progress["microbatch"] += 1
        if time.monotonic() - progress["last"] > 45:
            _atomic_json(options.run_dir / "HEARTBEAT.json", {
                "step": progress["step"], "completed_microbatches": progress["microbatch"] - 1,
                "microbatches_per_step": accumulation, "time_unix": time.time(),
            })
            progress["last"] = time.monotonic()
        return _forward_step(iter([batch]), wrapped_model, return_schedule_plan)

    probe_gradients = {}
    if options.mode == "probe":
        for name, p in raw_model.named_parameters():
            if any(marker in name for marker in EXTRA_MARKERS) or any(
                    suffix in name for suffix in ("layers.7.attention.linear_qkv.gate.weight",
                    "layers.7.attention.q_layernorm.weight", "layers.7.attention.k_layernorm.weight")):
                probe_gradients[name] = False

                def hook(gradient, key=name):
                    if not torch.isfinite(gradient).all():
                        raise RuntimeError(f"nonfinite gradient: {key}")
                    probe_gradients[key] |= bool(torch.count_nonzero(gradient))
                    return gradient

                p.register_hook(hook)
    evaluate(native.iteration)
    initial_hashes = parameter_hashes(raw_model) if options.mode == "probe" else None
    start = native.iteration
    for iteration in range(start, options.steps):
        native.curr_iteration = iteration
        progress.update(step=iteration + 1, microbatch=0)
        train_digest = hashlib.sha256()
        learning_rate = max(group["lr"] for group in optimizer.param_groups)
        torch.cuda.synchronize()
        begin = time.monotonic()
        result = train_step(forward_step, train_iterator, model, optimizer, scheduler, cfg,
                            schedule, iteration=iteration)
        torch.cuda.synchronize()
        seconds = time.monotonic() - begin
        losses, skipped, should_save, should_exit, exit_code, grad_norm, _, _ = result
        ce = float(losses["lm loss"])
        if skipped or should_exit or not math.isfinite(ce) or not math.isfinite(float(grad_norm)):
            raise RuntimeError(f"pilot training rejected: skipped={skipped}, exit={exit_code}, ce={ce}, grad={grad_norm}")
        done = iteration + 1
        stop["requested"] |= (options.run_dir / "STOP_REQUESTED.json").is_file()
        native.iteration = done
        native.consumed_train_samples = done * options.global_batch
        if train_source.batch_index != done * accumulation:
            raise RuntimeError("training data cursor drift")
        _write_event(event_path, {"event": "train", "step": done,
            "consumed_tokens": done * options.global_batch * 2048, "train_ce": ce,
            "gradient_norm": float(grad_norm), "learning_rate_used": learning_rate,
            "step_seconds": seconds, "tokens_per_second": options.global_batch * 2048 / seconds,
            "first_four_microbatches_sha256": train_digest.hexdigest(), "time_unix": time.time()})
        if done % options.eval_interval == 0 or done == options.steps:
            evaluate(done)
        if done % options.save_interval == 0 or done == options.steps or should_save or stop["requested"]:
            checkpoint(done)
        if stop["requested"]:
            _atomic_json(options.run_dir / "STOPPED.json", {"iteration": done})
            return

    if options.mode == "probe":
        before = parameter_hashes(raw_model)
        if not any(value != initial_hashes[key] for key, value in before.items()):
            raise RuntimeError("native optimizer did not change any parameter")
        if not probe_gradients or not all(probe_gradients.values()):
            raise RuntimeError(f"missing restored/extra gradients: {probe_gradients}")
        # Perturb model weights to prove this is a load, not merely a save.
        optimizer_before = optimizer_tensor_hashes(optimizer, perturb=True)
        with torch.no_grad():
            for name, p in raw_model.named_parameters():
                if ".attention." in name and p.ndim <= 2:
                    p.add_(0.125)
        native.load = str(options.run_dir / "checkpoints")
        native.consumed_train_samples = 0
        native.consumed_valid_samples = 0
        saved_scheduler = scheduler.state_dict()
        # Native load_state_dict advances a freshly constructed scheduler by
        # the saved sample count, exactly as setup_model_and_optimizer does.
        scheduler = get_optimizer_param_scheduler(optimizer)
        loaded, _ = load_checkpoint(model, optimizer, scheduler, checkpointing_context=context)
        if loaded != options.steps or parameter_hashes(raw_model) != before:
            raise RuntimeError("native checkpoint reload differs from saved model")
        if scheduler.state_dict() != saved_scheduler:
            raise RuntimeError(f"native scheduler failed checkpoint round-trip: "
                               f"{saved_scheduler} != {scheduler.state_dict()}")
        if optimizer_tensor_hashes(optimizer) != optimizer_before:
            raise RuntimeError("native optimizer master weights or momentum failed checkpoint round-trip")
        _atomic_json(options.run_dir / "PROBE_COMPLETE.json", {
            "iteration": loaded, "all_model_weights_bitwise_restored": True,
            "scheduler_restored": True, "optimizer_tensors_bitwise_restored": True,
            "nonzero_finite_gradients": probe_gradients})
    else:
        _atomic_json(options.run_dir / "TRAINING_COMPLETE.json", {
            "iteration": options.steps, "consumed_tokens": options.steps * options.global_batch * 2048,
            "completed_unix": time.time()})
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

"""Four-node resident RL qualification and bounded execution entrypoint."""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from archlab.megatron.miles_v41_launch import environment, training_argv


def argv_for(args):
    original = training_argv(args)
    remove = {"--offload-train-target": 1, "--offload-train-disk-dir": 1,
              "--offload-train-disk-chunk-mb": 1, "--train-env-vars": 1,
              "--offload-train": 0, "--offload-rollout": 0, "--eval-interval": 1,
              "--eval-prompt-data": 2}
    replace = {"--actor-num-nodes": "4", "--pipeline-model-parallel-size": "4",
               "--rollout-num-gpus-per-engine": "32", "--sglang-tp-size": "32",
               "--sglang-ep-size": "32", "--sglang-dp-size": "4",
               "--sglang-mem-fraction-static": "0.90", "--num-rollout": "1000000",
               "--sglang-chunked-prefill-size": "1024"}
    result, i = [], 0
    while i < len(original):
        key = original[i]
        if key in remove:
            i += 1 + remove[key]
        elif key in replace:
            result.extend((key, replace[key]))
            i += 2
        else:
            result.append(key)
            i += 1
    from torch_memory_saver.utils import get_binary_path_from_package
    env = {"LD_PRELOAD": str(get_binary_path_from_package("torch_memory_saver_hook_mode_preload")),
           "TMS_INIT_ENABLE": "0", "TMS_INIT_ENABLE_CPU_BACKUP": "0",
           "TMS_INIT_ENABLE_DISK_BACKUP": "0",
           "ARCHLAB_RL_OFFLOAD_POLICY": "forbidden", "ARCHLAB_RL_FREEZE_ENGRAM": "1",
           "ARCHLAB_RL_MOMENTUM_DTYPE": args.momentum_dtype}
    # Miles calls discard/reallocation of SGLang's no-backup buffers "offload".
    # This flag enables that lifecycle, not CPU/disk transfer (guarded in model).
    return result + ["--no-offload-train", "--offload-rollout", "--skip-eval-before-train",
                     "--sglang-enable-dp-attention",
                     "--disable-grad-buffers-cpu-backup", "--train-env-vars", json.dumps(env)]


async def train(args, *, pilot, seconds, output, continue_after_pilot=False, variant=None):
    from miles.ray.placement_group import (
        create_rollout_components,
        create_training_models,
        update_weights,
    )
    from miles.ray.wiring import launch_worker_manager
    from miles.utils import object_store
    from miles.utils.audit_utils.process_identity import MainProcessIdentity
    from miles.utils.data import remove_rollout_data_refs
    from miles.utils.logging_utils import configure_logger
    from miles.utils.tracking_utils.tracking import init_tracking
    from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

    configure_logger(args, source=MainProcessIdentity())
    _manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)
    init_tracking(args)
    if args.offload_train or not args.offload_rollout or args.use_critic or args.keep_old_actor:
        raise ValueError("resident driver forbids offloading and extra policies")
    inference = executor = actor = None
    started = None
    completed = 0
    try:
        inference, executor, _ = await create_rollout_components(args)
        # Spawn/import serving workers before 32 checkpoint readers saturate
        # shared storage. Empty completion does not mark dummy weights ready.
        await inference.start_update_weights()
        await inference.end_update_weights(snapshot_cell_id_to_hashes={})
        # ServerCell initialization already discards KV and resumes only weights.
        await inference.offload(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        actor, critic = await create_training_models(args, inference, executor)
        if critic is not None:
            raise ValueError("unexpected critic")
        await inference.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])
        await update_weights(actor, executor)
        await inference.onload_kv()
        started = time.monotonic()
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            await inference.prepare_rollout(rollout_id)
            data = await executor.get.remote(rollout_id)
            # Runtime memory-saver buffers have CPU backup disabled. These calls
            # discard regenerable weights/KV, never offload their contents.
            await inference.offload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS])
            await actor.train(rollout_id, data)
            remove_rollout_data_refs(args, data)
            await actor.clear_memory()
            await inference.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            await update_weights(actor, executor, rollout_id=rollout_id)
            await inference.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            completed += 1
            elapsed = time.monotonic() - started
            (output / "PROGRESS.json").write_text(json.dumps(dict(
                phase="pilot" if pilot else "rl", completed_rollouts=completed,
                rl_seconds=elapsed, momentum_dtype=os.environ["ARCHLAB_RL_MOMENTUM_DTYPE"],
                engram_frozen=True, offload_policy="forbidden")))
            if (pilot and completed >= 2) or (not pilot and elapsed >= seconds):
                await actor.save_model(rollout_id, force_sync=True)
                await executor.save.remote(rollout_id)
                if pilot:
                    from archlab.megatron.miles_v41_resident_admission import record_full_model
                    receipt = record_full_model(output.parent, output, variant,
                                                os.environ["ARCHLAB_RL_MOMENTUM_DTYPE"])
                    if continue_after_pilot:
                        pilot = False
                        started = time.monotonic()
                        (output / "RL_ADMITTED.json").write_text(json.dumps(receipt, indent=2))
                        print("BASELINE_RL_ADMITTED: starting 7200-second productive budget", flush=True)
                        continue
                break
    except BaseException as error:
        (output / "FAILURE.json").write_text(json.dumps(dict(
            error_type=type(error).__name__, error=str(error),
            completed_rollouts=completed, production_admitted=False)))
        raise
    finally:
        # A failed transfer can leave the inference controller's lock held.
        # Bound controller disposal so main's worker-process cleanup still runs.
        async def dispose_components():
            if executor is not None:
                await executor.dispose.remote()
            if inference is not None:
                await inference.dispose()
            if actor is not None:
                await actor.dispose()
        try:
            await asyncio.wait_for(dispose_components(), timeout=45)
        except Exception as error:
            print(f"Controller cleanup deferred to worker manager: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "pilot", "train"])
    parser.add_argument("--variant", choices=["normal", "simplicial"], required=True)
    for name in ("node", "address", "node-ip"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("run-root", "runtime", "miles"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--momentum-dtype", choices=["float16", "float32", "bfloat16"], default="float16")
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--continue-after-pilot", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0 or args.attempt <= 0:
        raise ValueError("RL duration must be positive")
    environment(args)
    os.environ.update(ARCHLAB_RL_OFFLOAD_POLICY="forbidden", ARCHLAB_RL_FREEZE_ENGRAM="1",
                      ARCHLAB_RL_MOMENTUM_DTYPE=args.momentum_dtype,
                      SGLANG_DP_USE_GATHERV="0", SGLANG_DP_USE_REDUCE_SCATTER="0",
                      SGLANG_DP_SHARED_EXPERT_LOCAL="0")
    output = args.run_root / f"{'pilot' if args.action == 'pilot' else 'production'}-{args.variant}-v{args.attempt}"
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "train":
        from archlab.megatron.miles_v41_resident_admission import verify
        verify(args.run_root, args.momentum_dtype, variants=(args.variant,))
    elif args.action == "pilot":
        from archlab.megatron.miles_v41_resident_admission import verify_numerical
        verify_numerical(args.run_root, args.momentum_dtype)
    argv = argv_for(args)
    for flag, directory in (("--save", "checkpoints"), ("--tensorboard-dir", "tensorboard")):
        argv[argv.index(flag) + 1] = str(output / directory)
    (output / "launch-argv.json").write_text(json.dumps(argv, indent=2))
    sys.argv = [str(args.miles / "train.py"), *argv]
    from miles.utils.arguments import parse_args
    parsed = parse_args()
    if parsed.offload_train or not parsed.offload_rollout or parsed.sglang_enable_weights_cpu_backup:
        raise ValueError("runtime enabled forbidden offloading")
    if parsed.load != parsed.hf_checkpoint or parsed.use_kl_loss or parsed.kl_coef != 0:
        raise ValueError("fresh single-policy load contract violated")
    if args.action == "check":
        from miles.backends.sglang_utils.sglang_engine import _compute_server_args
        from sglang.srt.server_args import ServerArgs
        server = ServerArgs(**_compute_server_args(parsed, node_rank=0,
            dist_init_addr="22.0.243.22:19000", nccl_port=19001, host="22.0.243.22",
            port=19002, base_gpu_id=0, disaggregation_bootstrap_port=None,
            engine_info_bootstrap_port=19003, sglang_overrides={}, num_gpus_per_engine=32,
            gated_launch_port=19004, random_seed=20260922))
        server.resolve_once()
        server.check_server_args()
        (output / "resolved-serving.json").write_text(json.dumps(server.resolved_dict(), default=str, indent=2))
        print("RESIDENT_ARGUMENTS_ACCEPTED", flush=True)
        return
    (output / "LAUNCH_PROCESS.json").write_text(json.dumps(dict(
        started_at_epoch=time.time(), pid=os.getpid(), variant=args.variant,
        action=args.action, momentum_dtype=args.momentum_dtype)))
    import ray
    from miles.ray.specs import inference
    from miles.utils.tracking_utils.tracking import finish_tracking
    original_spec = inference._compute_spec_inference_engine

    def resident_spec(*positional, **keywords):
        spec = original_spec(*positional, **keywords)
        ports = [port.model_copy(update={"static_port": {"dist_init": 27300, "nccl": 27400}[port.name],
                                        "allow_dynamic": False})
                 if port.name in ("dist_init", "nccl") else port for port in spec.port_infos]
        return spec.model_copy(update={"port_infos": ports})

    inference._compute_spec_inference_engine = resident_spec
    original = inference.compute_inference_engine_env_vars
    engine_env = {k: v for k, v in os.environ.items() if k.startswith(("ARCHLAB_", "SGLANG_"))}
    inference.compute_inference_engine_env_vars = lambda train_args: original(train_args) | engine_env
    def terminate(signum, frame):
        raise KeyboardInterrupt("resident run terminated")

    signal.signal(signal.SIGTERM, terminate)
    ray.init(address=args.address)
    try:
        asyncio.run(train(parsed, pilot=args.action == "pilot", seconds=args.seconds, output=output,
                          continue_after_pilot=args.continue_after_pilot, variant=args.variant))
    finally:
        # Controller disposal stops watchers; explicitly stop worker processes as
        # well so an exited driver cannot leave an orphaned serving fleet.
        from miles.ray.specs.entrypoint import compute_specs
        from miles.utils.workers.ray_worker_manager import RayWorkerManager
        try:
            manager = RayWorkerManager.get_handle()
            cells = ray.get(manager.get_cell_infos.remote(
                pool_ids=[spec.name for spec in compute_specs(parsed)]), timeout=30)
            ray.get(manager.stop_cells.remote(list(cells)), timeout=60)
            ray.kill(manager)
        finally:
            finish_tracking()
            ray.shutdown()


if __name__ == "__main__":
    main()

"""Inspect an explicitly selected live Miles run before checkpointed retirement.

This operator uses the running Ray actors' own checkpoint and worker lifecycle
APIs. It never changes their training code or resumes an optimizer update.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

from archlab.artifacts import atomic_write_json


def _inspect_actor(actor):
    scheduler = actor.opt_param_scheduler
    return {
        "rank": actor._rank,
        "world_size": actor._world_size,
        "save": actor.args.save,
        "offload_train": actor.args.offload_train,
        "async_save": actor.args.async_save,
        "active_model_tag": actor._active_model_tag,
        "scheduler_num_steps": scheduler.num_steps,
        "global_batch_size": actor.args.global_batch_size,
    }


def selected_actors(ray, run_root):
    managers = [
        ray.get_actor(item["name"], namespace=item["namespace"])
        for item in ray.util.list_named_actors(all_namespaces=True)
        if item["name"] == "ray_worker_manager"
    ]
    if len(managers) != 1:
        raise ValueError("expected one native Miles worker manager")
    manager = managers[0]
    cells = ray.get(manager.get_cell_infos.remote(pool_ids=["trainer-actor"]), timeout=30)
    workers = ray.get([manager.get_worker_infos.remote(cell) for cell in cells], timeout=30)
    train = [worker.handle._actor_handle for group in workers for worker in group]
    states = ray.get([handle.__ray_call__.remote(_inspect_actor) for handle in train], timeout=120)
    expected = str(run_root / "checkpoints")
    if len(states) != 32 or {row["rank"] for row in states} != set(range(32)):
        raise ValueError(
            f"expected 32 training ranks, observed {len(states)}: {[row['rank'] for row in states]}"
        )
    if any(row["save"] != expected or row["world_size"] != 32 for row in states):
        raise ValueError("Ray actors do not all belong to the selected run")
    return train, manager, sorted(states, key=lambda row: row["rank"])


def verify_driver(pid, run_root):
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    values = [value.decode() for value in argv if value]
    if "archlab.megatron.miles_v41_stock_launch" not in values or str(run_root) not in values:
        raise ValueError("driver identity does not match the requested run")


def retire(ray, manager, states, pools, root, output, driver_pid):
    saved = json.loads((output / "RL_CHECKPOINT_SAVED.json").read_text())
    verified = json.loads((output / "RL_CHECKPOINT_READBACK.json").read_text())
    quiesced = json.loads((output / "RL_DRIVER_QUIESCED.json").read_text())
    if (
        not saved["all_native_save_calls_returned"]
        or verified["status"] != "passed"
        or saved["checkpoint"] != verified["checkpoint"]
        or quiesced["driver_pid"] != driver_pid
    ):
        raise ValueError("retirement requires this driver's completed, verified checkpoint")
    expected = {(verified["iteration"] + 1) * row["global_batch_size"] for row in states}
    if {row["scheduler_num_steps"] for row in states} != expected:
        raise ValueError("live optimizer progress differs from saved progress")
    status = Path(f"/proc/{driver_pid}/status").read_text().splitlines()
    if not any(line.startswith("State:") and "T (stopped)" in line for line in status):
        raise ValueError("driver must remain quiesced until retirement")
    cells = ray.get(manager.get_cell_infos.remote(pool_ids=pools), timeout=30)
    ray.get(manager.stop_cells.remote(list(cells)), timeout=180)
    verify_driver(driver_pid, root)
    # Deliver the pending termination before resuming the frozen driver. It must
    # not dispatch another update after the actor group has been retired.
    os.kill(driver_pid, signal.SIGTERM)
    os.kill(driver_pid, signal.SIGCONT)
    record = {
        "observed_at_epoch": time.time(),
        "driver_pid": driver_pid,
        "checkpoint": saved["checkpoint"],
        "completed_optimizer_updates": verified["iteration"] + 1,
        "native_cells_retired": list(cells),
        "checkpoint_readback": str(output / "RL_CHECKPOINT_READBACK.json"),
        "reason": "user requested scratch scaling study on the existing allocation",
        "allocation_stopped": False,
    }
    atomic_write_json(output / "RL_NATIVE_RETIRED.json", record)
    atomic_write_json(root / "STOPPED_FOR_SCALING.json", record)
    (root / "SAVE_REQUEST").unlink(missing_ok=True)
    print("RL_NATIVE_WORKERS_RETIRED", flush=True)


def main():
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "checkpoint", "retire"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument("--driver-pid", type=int, required=True)
    parser.add_argument("--completed-rollout", type=int)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    verify_driver(args.driver_pid, root)
    ray.init(address=args.address)
    handles, manager, states = selected_actors(ray, root)
    pools = list(ray.get(manager.get_addrs.remote(), timeout=30))
    atomic_write_json(
        output / "RL_ACTORS.json",
        {
            "observed_at_epoch": time.time(),
            "ranks": states,
            "pools": pools,
        },
    )
    print(
        json.dumps({"ranks": len(states), "first": states[0], "last": states[-1], "pools": pools}),
        flush=True,
    )
    if args.action == "inspect":
        ray.shutdown()
        return
    if args.action == "retire":
        retire(ray, manager, states, pools, root, output, args.driver_pid)
        ray.shutdown()
        return
    if args.completed_rollout is None:
        parser.error("checkpoint requires --completed-rollout")
    target = root / "checkpoints" / f"iter_{args.completed_rollout:07d}"
    if target.exists():
        raise ValueError("refusing to overwrite an existing checkpoint")
    # The running policy stays GPU-resident during rollout. Freeze only its
    # driver, preventing a new update while the native actors save collectively.
    # The sidecar and Ray workers continue to run. Unfinished generation is not
    # optimizer progress, and is explicitly retired after the save.
    if any(
        row["offload_train"] or row["async_save"] or row["active_model_tag"] != "actor"
        for row in states
    ):
        raise ValueError("operator only supports the resident, synchronous-save policy")
    expected_steps = {(args.completed_rollout + 1) * row["global_batch_size"] for row in states}
    if {row["scheduler_num_steps"] for row in states} != expected_steps:
        raise ValueError("actor scheduler counters differ from the completed rollout")
    verify_driver(args.driver_pid, root)
    os.kill(args.driver_pid, signal.SIGSTOP)
    atomic_write_json(
        output / "RL_DRIVER_QUIESCED.json",
        {
            "driver_pid": args.driver_pid,
            "observed_at_epoch": time.time(),
            "completed_rollout": args.completed_rollout,
            "checkpoint": str(target),
        },
    )
    try:
        # Recheck after dispatch is frozen; this call also waits for any queued
        # native actor operation to finish before taking a consistent snapshot.
        handles, manager, states = selected_actors(ray, root)
        if {row["scheduler_num_steps"] for row in states} != expected_steps:
            raise ValueError("an update crossed the quiescence boundary; re-inspect before saving")
        inference_pools = [pool for pool in pools if pool.startswith("inference-")]
        if not inference_pools:
            raise ValueError("native inference pools are missing")
        cells = ray.get(manager.get_cell_infos.remote(pool_ids=inference_pools), timeout=30)
        ray.get(manager.stop_cells.remote(list(cells)), timeout=120)
        atomic_write_json(
            output / "RL_GENERATION_RETIRED.json",
            {
                "observed_at_epoch": time.time(),
                "cells": list(cells),
                "unfinished_rollout_discarded": args.completed_rollout + 1,
                "completed_optimizer_updates_preserved": args.completed_rollout + 1,
            },
        )
        refs = [
            handle.save_model.remote(args.completed_rollout, force_sync=True) for handle in handles
        ]
        atomic_write_json(
            output / "RL_CHECKPOINT_SAVING.json",
            {
                "observed_at_epoch": time.time(),
                "checkpoint": str(target),
                "ranks": len(refs),
            },
        )
        ray.get(refs, timeout=14400)
        iteration = (root / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip()
        if iteration != str(args.completed_rollout) or not (target / "metadata.json").is_file():
            raise ValueError("native checkpoint completion pointer differs")
        atomic_write_json(
            output / "RL_CHECKPOINT_SAVED.json",
            {
                "observed_at_epoch": time.time(),
                "checkpoint": str(target),
                "completed_rollout": args.completed_rollout,
                "all_native_save_calls_returned": True,
                "readback_pending": True,
            },
        )
        print("RL_NATIVE_CHECKPOINT_SAVED", flush=True)
    except BaseException as error:
        atomic_write_json(
            output / "RL_CHECKPOINT_STOP_FAILURE.json",
            {
                "observed_at_epoch": time.time(),
                "error": repr(error),
                "driver_remains_quiesced": True,
            },
        )
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

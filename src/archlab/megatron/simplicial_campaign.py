"""Fail-closed, sequential A/B/C DSW campaign; no service or allocation control."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--probe-a", type=Path, required=True)
    parser.add_argument("--probe-b", type=Path, required=True)
    parser.add_argument("--probe-c", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--detach", action="store_true")
    options = parser.parse_args()
    root = options.campaign_dir.resolve()
    if options.detach:
        if root.exists() and any(root.iterdir()):
            raise ValueError("refusing to overwrite an existing campaign")
        root.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "archlab.megatron.simplicial_campaign",
                   *(arg for arg in sys.argv[1:] if arg != "--detach")]
        with (root / "supervisor.log").open("x") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / "SUPERVISOR.json", {"pid": process.pid, "command": command})
        print(f"Campaign supervisor PID {process.pid}; evidence: {root}", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    if (root / "CAMPAIGN.json").exists():
        raise ValueError("campaign is fresh-only; resumption needs explicit review")
    # A host-local advisory lock prevents duplicate project campaigns on GPU0.
    with Path("/tmp/archlab-simplicial-dsw-gpu0.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run_campaign(options, root)
        except Exception as error:
            write_json(root / "FAILURE.json", {"error": repr(error), "time_unix": time.time()})
            if (root / "CAMPAIGN.json").exists():
                state = json.loads((root / "CAMPAIGN.json").read_text())
                state.update(status="failed", error=repr(error))
                write_json(root / "CAMPAIGN.json", state)
            raise


def run_campaign(options, root):
    src = Path(__file__).parents[1]
    expected_sources = source_hashes(src)
    reference = options.probe_a / "INITIALIZATION.json"
    expected_init = json.loads(reference.read_text())["common_parameter_sha256"]
    expected_data = None
    expected_contract = None
    expected_heldout = None
    for arm, probe in zip("ABC", (options.probe_a, options.probe_b, options.probe_c), strict=True):
        complete = json.loads((probe / "PROBE_COMPLETE.json").read_text())
        required = ("all_model_weights_bitwise_restored", "scheduler_restored",
                    "optimizer_tensors_bitwise_restored")
        if not all(complete.get(key) is True for key in required):
            raise ValueError(f"{arm} checkpoint/optimizer gate incomplete")
        contract = json.loads((probe / "RUN_CONTRACT.json").read_text())
        if contract["arm"] != arm or contract["mode"] != "probe":
            raise ValueError("wrong probe evidence")
        matched = {key: contract[key] for key in
                   ("runtime", "topology", "seed", "global_batch", "micro_batch",
                    "manifest_sha256", "train_prefixes", "heldout_prefixes", "eval_sequences")}
        if expected_contract is None:
            expected_contract = matched
        if matched != expected_contract:
            raise ValueError("probe runtime/data/optimizer-batch contracts differ")
        if any(expected_sources.get(key) != value for key, value in contract["source_sha256"].items()):
            raise ValueError(f"{arm} probe did not validate these exact training/model sources")
        initialization = json.loads((probe / "INITIALIZATION.json").read_text())
        if initialization["common_parameter_sha256"] != expected_init:
            raise ValueError("probe initialization mismatch")
        records = [json.loads(line) for line in (probe / "metrics.jsonl").read_text().splitlines()]
        digests = [r["first_four_microbatches_sha256"] for r in records if r["event"] == "train"]
        if expected_data is None:
            expected_data = digests
        if len(digests) != 3 or digests != expected_data:
            raise ValueError("probe data-order mismatch")
        heldout = {r["heldout_tokens_sha256"] for r in records if r["event"] == "eval"}
        if expected_heldout is None:
            expected_heldout = heldout
        if len(heldout) != 1 or heldout != expected_heldout:
            raise ValueError("probe held-out window mismatch")
    state = {"source_commit": options.source_commit, "source_sha256": expected_sources,
             "started_unix": time.time(), "supervisor_pid": os.getpid(),
             "arms": {arm: "queued" for arm in "ABC"}, "status": "running"}
    write_json(root / "CAMPAIGN.json", state)
    child = {"process": None, "stop": False}

    def stop(*_):
        child["stop"] = True
        if child["process"] is not None and (root / state["active_arm"]).is_dir():
            # A marker avoids torchrun's shorter worker-kill timeout. The pilot
            # checkpoints at the next complete optimizer step.
            write_json(root / state["active_arm"] / "STOP_REQUESTED.json",
                       {"requested_unix": time.time()})

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for arm in "ABC":
        if child["stop"]:
            break
        if source_hashes(src) != expected_sources:
            raise RuntimeError("source snapshot changed; refusing to launch the next arm")
        run_dir = root / arm
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc-per-node=1", "--module", "archlab.megatron.simplicial_pilot",
                   "--arm", arm, "--mode", "train", "--run-dir", str(run_dir),
                   "--data-root", str(options.data_root.resolve()),
                   "--tokenizer", str(options.tokenizer.resolve()),
                   "--initialization-reference", str(reference.resolve())]
        with (root / f"{arm}.log").open("x") as log:
            state.update(active_arm=arm)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            child["process"] = process
            state.update(active_arm=arm, child_pid=process.pid)
            state["arms"][arm] = "running"
            write_json(root / "CAMPAIGN.json", state)
            while process.poll() is None:
                if child["stop"] and run_dir.is_dir() and not (run_dir / "STOP_REQUESTED.json").exists():
                    write_json(run_dir / "STOP_REQUESTED.json", {"requested_unix": time.time()})
                time.sleep(1)
            result = process.returncode
            child["process"] = None
        completed = run_dir / "TRAINING_COMPLETE.json"
        if result != 0 or not completed.is_file():
            state["arms"][arm] = "stopped" if child["stop"] else "failed"
            state.update(status=state["arms"][arm], exit_code=result)
            write_json(root / "CAMPAIGN.json", state)
            return
        evidence = json.loads(completed.read_text())
        if evidence.get("consumed_tokens") != 3003121664:
            raise RuntimeError("incorrect completion token count")
        state["arms"][arm] = "complete"
        write_json(root / "CAMPAIGN.json", state)
    state.update(status="stopped" if child["stop"] else "complete", finished_unix=time.time())
    write_json(root / "CAMPAIGN.json", state)


if __name__ == "__main__":
    main()

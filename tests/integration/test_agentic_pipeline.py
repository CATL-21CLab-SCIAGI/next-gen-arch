"""Stock verl startup and one-update agentic probes, with stock artifact checks.

Opt in with ARCHLAB_RUN_AGENTIC_STARTUP=1 or ARCHLAB_RUN_AGENTIC_PIPELINE=1.
The latter executes real policy actions in the configured isolated sandboxes.
Neither probe changes the production recipe or claims reward contrast exists.
"""

import json
import math
import os
import signal
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest


def input_lineage(source, experiment_path=None):
    """Hash actual immutable inputs, including model bytes, before and after use."""
    from omegaconf import OmegaConf

    from archlab.artifacts import sha256_file

    source = Path(source).resolve()
    config = OmegaConf.load(source)
    export = Path(config.actor_rollout_ref.model.path).resolve()
    receipt = json.loads((export / "EXPORT.json").read_text())
    files = {"resolved_config": source, "export_receipt": export / "EXPORT.json"}
    variables = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env.env_vars", default={})
    if variables.get("MIXED_HARNESS_SPEC"):
        specification = Path(variables["MIXED_HARNESS_SPEC"]).resolve()
        files["harness_specification"] = specification
        for harness in OmegaConf.load(specification).harnesses:
            path = Path(harness.config)
            files[f"harness/{harness.label}"] = (path if path.is_absolute() else specification.parent / path).resolve()
    for name in receipt["files"]:
        assert Path(name).name == name, "export receipt contains an external path"
        files[f"export/{name}"] = export / name
    parent = receipt["source_checkpoint"]
    parent_receipt = ("ckpt_contents.json"
                      if receipt.get("training", {}).get("format") == "archlab-limite-agentic-sft-v1"
                      else "COMPLETE.json")
    files["parent_receipt"] = Path(parent["path"]) / parent_receipt
    if experiment_path:
        files["experiment"] = Path(experiment_path).resolve()
        experiment = json.loads(files["experiment"].read_text())
        assert Path(experiment["parent_checkpoint"]).resolve() == Path(parent["path"]).resolve()
        files["data_manifest"] = Path(experiment["data_manifest"]).resolve()
    result = {name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
              for name, path in files.items()}
    for name, expected in receipt["files"].items():
        assert result[f"export/{name}"]["sha256"] == expected, f"export bytes changed: {name}"
    assert result["parent_receipt"]["sha256"] == parent["receipt_sha256"]
    return result


def smoke_config(source, output):
    from omegaconf import OmegaConf

    config = OmegaConf.create(OmegaConf.to_container(source, resolve=True))
    settings = {
        # The probe owns its local Ray lifecycle, even if the caller previously
        # selected a shared cluster through RAY_ADDRESS or launch settings.
        "ray_kwargs.ray_init.address": "local",
        "data.train_batch_size": 2,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 2,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.agent.num_workers": 1,
        "actor_rollout_ref.rollout.custom.agent_framework.log_dir": str(output / "sessions"),
        "trainer.total_training_steps": 1,
        "trainer.total_epochs": 1,
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.save_freq": 1,
        "trainer.resume_mode": "disable",
        "trainer.logger": ["console", "file"],
        "trainer.experiment_name": config.trainer.experiment_name + "-pipeline-smoke",
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.default_hdfs_dir": None,
        "trainer.rollout_data_dir": str(output / "rollouts"),
    }
    for key, value in settings.items():
        OmegaConf.update(config, key, value, merge=False)
    config.ray_kwargs.ray_init.runtime_env.env_vars["VERL_FILE_LOGGER_PATH"] = str(output / "metrics.jsonl")
    return config


def inspect_artifacts(output, *, native_context=131072):
    """Check package-written tokens, masks, verdicts, metrics and saved update."""
    from recipes.code.mimoagent_runner import _require_code_verdict

    from archlab.artifacts import sha256_file

    output = Path(output)
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in metrics if row["step"] == 1]
    assert len(updates) == 1, "missing or repeated optimizer-update metrics"
    stats = updates[0]["data"]
    required = ("actor/grad_norm", "actor/loss", "rollout_corr/rollout_is_mean",
                "rollout_corr/rollout_is_max", "rollout_corr/kl",
                "rollout_corr/training_log_ppl", "rollout_corr/rollout_log_ppl")
    for key in required:
        assert isinstance(stats.get(key), (int, float)) and math.isfinite(stats[key]), key
    for key, value in stats.items():
        # Required learner statistics cannot hide NaN as JSON null; optional
        # telemetry can legitimately be unavailable.
        if isinstance(value, (int, float)):
            assert math.isfinite(value), key
    consumed = [json.loads(line) for line in (output / "rollouts/1.jsonl").read_text().splitlines()]
    assert len(consumed) == 8, "the smoke must train on two groups of four responses"
    assert len({row["uid"] for row in consumed}) == 8, "duplicated consumed rollout keys"
    groups = defaultdict(list)
    for row in consumed:
        assert row["score"] in (0, 1)
        groups[row["uid"].rsplit("_", 2)[0]].append(row["score"])
    assert len(groups) == 2 and all(len(values) == 4 for values in groups.values())
    sessions = defaultdict(list)
    for path in sorted((output / "sessions").rglob("session.json")):
        session = json.loads(path.read_text())
        sessions[(session.get("prompt_uid"), session["session_index"])].append((path.parent, session))
    completed, observations, tools, truncated, format_error_steps = 0, 0, 0, 0, 0
    used = set()
    for row in consumed:
        uid, session_index, index = row["uid"].rsplit("_", 2)
        index = int(index)
        matches = sessions[(uid, int(session_index))]
        assert len(matches) == 1, f"missing or ambiguous consumed rollout: {row['uid']}"
        path, session = matches[0]
        used.add(path)
        assert session["gradable"] is True and session["error_type"] is None
        semantic = json.loads((path / "traj.json").read_text())["trajs"]["main" if index == 0 else f"chain_{index + 1}"]
        tools += sum(len(message.get("tool_calls", [])) for message in semantic["messages"])
        trajectory = json.loads((path / "trajectory.json").read_text())["trajectories"][index]
        with np.load(path / "trajectory.npz", allow_pickle=False) as arrays:
            prefix = f"traj{index}_"
            mask = arrays[prefix + "response_mask"]
            response = arrays[prefix + "response_ids"]
            prompt = arrays[prefix + "prompt_ids"]
            logprobs = arrays[prefix + "response_logprobs"]
            assert len(prompt) + len(response) <= native_context
            assert mask.shape == response.shape == logprobs.shape and len(mask) > 0
            expected = np.zeros_like(mask)
            for start, end in arrays[prefix + "generation_spans"]:
                assert 0 <= start < end <= len(mask)
                assert not expected[start:end].any(), "overlapping generated spans"
                expected[start:end] = 1
            np.testing.assert_array_equal(mask, expected, err_msg="tool observations entered policy loss")
            assert mask.any() and np.isfinite(logprobs[mask == 1]).all()
            observations += int(np.count_nonzero(mask == 0))
            reward = trajectory["reward_info"]
            _require_code_verdict({"dataset_type": "opensource-code"}, reward["reward"], reward)
            assert reward["finished"] is True and reward["reward"] == row["score"]
            assert reward.get("agent_status") != "InfraError", "agent infrastructure failure"
            assert not reward.get("codex_transport_error_count", 0), "agent transport failure"
            errors = reward.get("tool_call_error_count", 0)
            assert type(errors) is int and errors >= 0
            if errors:
                # BashOnlyAgent records sampled syntax mistakes as FormatError
                # flags. These are learnable policy outcomes, not failed transport.
                flags = reward.get("tool_call_error_flags", [])
                assert reward.get("agent_type") == "bashonly-agent"
                assert reward.get("tool_call_error_flag_source") == "agent_step"
                assert not reward.get("tool_call_error_types"), "unclassified tool failure"
                assert flags and all(type(flag) is bool for flag in flags)
                assert sum(flags) == errors, "inconsistent sampled format-error accounting"
                format_error_steps += errors
            completed += 1
            truncated += int(reward.get("termination_kind") == "truncated")
    assert completed == 8, "missing completed executable verifiers"
    assert tools > 0 and observations > 0, "no real policy/tool/observation cycle was exercised"
    checkpoint = output / "checkpoints"
    assert (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip() == "1"
    assert (checkpoint / "global_step_1/data.pt").is_file()
    actor = checkpoint / "global_step_1/actor"
    saved = json.loads((actor / "ckpt_contents.json").read_text())
    assert saved["global_step"] == 1 and saved["world_size"] == 8
    for name in ("model", "optimizer", "lr_scheduler", "rng_state"):
        assert name in saved["contents"], f"incomplete resumable checkpoint: {name}"
        location = actor / saved["contents"][name]["path"]
        assert location.is_dir() and any(location.iterdir()), f"missing saved {name}"
    model = actor / saved["contents"]["model"]["path"]
    model_files = sorted(model.glob("*.safetensors"))
    assert model_files and (model / "config.json").is_file(), "missing saved HF model/config"
    model_files.extend(sorted(model.glob("*.json")))
    exported = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in model_files}
    contrast = sum(len(set(values)) > 1 for values in groups.values())
    learning_ready = contrast > 0 and stats["actor/grad_norm"] > 0
    return {"status": "execution_passed", "completed_verifiers": completed,
            "consumed_rollout_uids": [row["uid"] for row in consumed],
            "unconsumed_session_artifacts": sum(map(len, sessions.values())) - len(used),
            "tool_calls": tools, "masked_observation_tokens": observations,
            "sampled_policy_format_error_steps": format_error_steps,
            "truncated_trajectories": truncated, "reward_contrast_groups": contrast,
            "grad_norm": stats["actor/grad_norm"],
            "learnability": {"status": "observed" if learning_ready else "pending",
                             "requires": "mixed rewards within a prompt group and positive finite gradient",
                             "claim": "one update is not evidence of sustained improvement"},
            "checkpoint": {"manifest_sha256": sha256_file(actor / "ckpt_contents.json"),
                           "global_step": saved["global_step"], "world_size": saved["world_size"],
                           "contents": saved["contents"], "export_files": exported},
            "learning_signal": "present" if contrast else "absent_equal_rewards",
            "claim": "one real update; equal rewards establish execution only"}


def _probe(tmp_path, *, startup):
    switch = "ARCHLAB_RUN_AGENTIC_STARTUP" if startup else "ARCHLAB_RUN_AGENTIC_PIPELINE"
    if os.getenv(switch) != "1":
        pytest.skip(f"requires explicit {switch}=1 on an idle deployment node")
    from omegaconf import OmegaConf

    source = Path(os.environ["ARCHLAB_QUALIFICATION_CONFIG"])
    experiment_path = os.getenv("ARCHLAB_QUALIFICATION_EXPERIMENT")
    lineage = input_lineage(source, experiment_path)
    config = smoke_config(OmegaConf.load(source), tmp_path)
    assert config.trainer.n_gpus_per_node == 8 and config.trainer.nnodes == 1
    assert config.actor_rollout_ref.rollout.max_model_len == 131072
    if not startup:
        from types import SimpleNamespace

        from archlab.training.sandbox import check_sandbox

        experiment = json.loads(Path(os.environ["ARCHLAB_QUALIFICATION_EXPERIMENT"]).read_text())
        check_sandbox(SimpleNamespace(**experiment), config)
    config_path = tmp_path / "smoke.yaml"
    OmegaConf.save(config, config_path, resolve=True)
    command = [sys.executable, str(Path(__file__).resolve()),
               "startup" if startup else "train", str(config_path), str(tmp_path)]
    # Stock workers write their artifacts in the same node-local directory.
    with (tmp_path / "package.log").open("w") as log:
        timeout = int(os.getenv("ARCHLAB_STARTUP_TIMEOUT", "1500") if startup
                      else os.getenv("ARCHLAB_PIPELINE_TIMEOUT", "14400"))
        returncode = _run_owned_process(command, log, timeout)
    assert returncode == 0, f"stock verl failed; see {tmp_path / 'package.log'}"
    assert input_lineage(source, experiment_path) == lineage, "qualification changed immutable source inputs"
    lifecycle = json.loads((tmp_path / "ray-lifecycle.json").read_text())
    assert lifecycle["address"] == "local" and lifecycle["shutdown_complete"] is True
    assert lifecycle["remaining_owned_children"] == 0
    if startup:
        evidence = json.loads((tmp_path / "startup.json").read_text())
        assert evidence["gpus"] >= 8 and evidence["agent_requests"] == 0
    else:
        evidence = inspect_artifacts(tmp_path)
    from archlab.artifacts import sha256_file

    evidence.update(source_config_sha256=sha256_file(source), smoke_config_sha256=sha256_file(config_path),
                    input_lineage=lineage, source_inputs_unchanged=True, ray_lifecycle=lifecycle)
    (tmp_path / "pipeline-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


def _run_owned_process(command, log, timeout):
    """Bound this probe's private process group without stopping shared jobs."""
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def test_stock_ray_package_startup(tmp_path):
    _probe(tmp_path, startup=True)


def test_real_policy_grpo_update(tmp_path):
    _probe(tmp_path, startup=False)


def _shutdown_ray(ray):
    """Use official shutdown, then reap only this probe's captured descendants."""
    import inspect

    import psutil

    children = psutil.Process().children(recursive=True)
    supports_wait = "wait_for_processes" in inspect.signature(ray.shutdown).parameters
    ray.shutdown(**({"wait_for_processes": True} if supports_wait else {}))
    _, alive = psutil.wait_procs(children, timeout=30)
    # psutil Process instances retain creation identities: never target a PID
    # reused by unrelated work. Zombies consume no resources and need reaping.
    remaining = []
    for process in alive:
        try:
            if process.status() != psutil.STATUS_ZOMBIE:
                remaining.append(process)
        except psutil.NoSuchProcess:
            pass
    alive = remaining
    forced = len(alive)
    for process in alive:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    assert not alive and not ray.is_initialized(), "owned Ray processes did not exit"
    return {"address": "local", "shutdown_complete": True, "wait_for_processes": supports_wait,
            "remaining_owned_children": 0, "forced_terminations": forced}


def _run(mode, config_path, output):
    import ray
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import main, run_ppo
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    config = OmegaConf.load(config_path)
    assert config.ray_kwargs.ray_init.address == "local", "probe must own its Ray cluster"
    try:
        if mode == "train":
            main(config)
            return

        @ray.remote
        class StartupOnly:
            def run(self, config):
                import transfer_queue as tq
                from verl.trainer.ppo.v1 import get_trainer_cls

                config.transfer_queue.enable = True
                tq.init(config.transfer_queue)
                trainer = None
                try:
                    trainer = get_trainer_cls(config.trainer.v1.trainer_mode)(config=config)
                    trainer.init()  # Stock learner, servers and initial weight sync; no fit/agent loop.
                    Path(output, "startup.json").write_text(json.dumps({
                        "status": "package_initialized", "agent_requests": 0,
                        "gpus": ray.cluster_resources().get("GPU", 0)}))
                finally:
                    if trainer is not None and hasattr(trainer, "_dump_executor"):
                        trainer._shutdown_dump_executor()
                    tq.close()

        validate_config(config, use_reference_policy=need_reference_policy(config), use_critic=need_critic(config))
        run_ppo(config, StartupOnly)
    finally:
        Path(output, "ray-lifecycle.json").write_text(json.dumps(_shutdown_ray(ray)))


@pytest.mark.parametrize("defect", [None, "mask", "verifier", "nonfinite", "checkpoint", "wrong_uid",
                                  "policy_format", "transport", "infra", "unknown_tool_error",
                                  "inconsistent_format_flags"])
def test_artifact_admission_rejects_false_success(tmp_path, defect, monkeypatch):
    """A successful process alone cannot qualify incorrect RL artifacts."""
    upstream = Path(__file__).resolve().parents[2] / "third_party/verl"
    if not (upstream / "recipes/code/mimoagent_runner.py").is_file():
        pytest.skip("requires the initialized pinned MiMo submodule")
    monkeypatch.syspath_prepend(str(upstream))
    stats = {"actor/loss": 0.0, "actor/grad_norm": 0.0, "optional/telemetry": None,
             "rollout_corr/rollout_is_mean": 1.0, "rollout_corr/rollout_is_max": 1.0,
             "rollout_corr/kl": 0.0, "rollout_corr/training_log_ppl": 1.0,
             "rollout_corr/rollout_log_ppl": 1.0}
    if defect == "nonfinite":
        stats["actor/loss"] = None  # orjson's representation of NaN/Inf.
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"step": 1, "data": stats}) + "\n")
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    (rollouts / "1.jsonl").write_text("".join(json.dumps({"uid": f"group{group}_{i}_0", "score": 0}) + "\n"
                                             for group in range(2) for i in range(4)))
    for i in range(8):
        session = tmp_path / "sessions" / str(i)
        session.mkdir(parents=True)
        reward = {"reward": 0, "resolved": False, "verifier_returncode": 1, "finished": True}
        if defect in {"policy_format", "unknown_tool_error", "inconsistent_format_flags"}:
            reward.update(agent_type="bashonly-agent", tool_call_error_count=1,
                          tool_call_error_flag_source="agent_step", tool_call_error_flags=[True, False],
                          tool_call_error_types={})
            if defect == "unknown_tool_error":
                reward["tool_call_error_types"] = {"unclassified": 1}
            if defect == "inconsistent_format_flags":
                reward["tool_call_error_flags"] = [False]
        if defect == "transport":
            reward["codex_transport_error_count"] = 1
        if defect == "infra":
            reward["agent_status"] = "InfraError"
        if defect == "verifier":
            reward.pop("verifier_returncode")
        (session / "session.json").write_text(json.dumps({"gradable": True, "error_type": None,
            "prompt_uid": "unrelated" if defect == "wrong_uid" else f"group{i // 4}", "session_index": i % 4}))
        (session / "traj.json").write_text(json.dumps({"trajs": {"main": {"messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "bash"}}]}]}}}))
        (session / "trajectory.json").write_text(json.dumps({"trajectories": [{"reward_info": reward}]}))
        np.savez(session / "trajectory.npz", traj0_prompt_ids=[1, 2], traj0_response_ids=[3, 4, 5],
                 traj0_response_mask=[1, 1 if defect == "mask" else 0, 1],
                 traj0_generation_spans=[[0, 1], [2, 3]], traj0_response_logprobs=[-1, 0, -1])
    prefetched = tmp_path / "sessions" / "unused-prefetch"
    prefetched.mkdir()
    (prefetched / "session.json").write_text(json.dumps({"prompt_uid": "future-group", "session_index": 0,
                                                       "gradable": False, "error_type": "cancelled"}))
    actor = tmp_path / "checkpoints/global_step_1/actor"
    actor.mkdir(parents=True)
    (actor.parent / "data.pt").touch()
    (actor.parent.parent / "latest_checkpointed_iteration.txt").write_text("1")
    contents = {}
    for name in ("model", "optimizer", "lr_scheduler", "rng_state"):
        (actor / name).mkdir()
        (actor / name / "fixture.distcp").touch()
        contents[name] = {"path": name}
    (actor / "model/model.safetensors").write_bytes(b"updated-model-fixture")
    (actor / "model/config.json").write_text("{}")
    if defect == "checkpoint":
        contents.pop("optimizer")
    (actor / "ckpt_contents.json").write_text(json.dumps({"global_step": 1, "world_size": 8, "contents": contents}))
    if defect and defect != "policy_format":
        from recipes.code.mimoagent_runner import RewardInfrastructureError

        with pytest.raises((AssertionError, RewardInfrastructureError)):
            inspect_artifacts(tmp_path)
    else:
        result = inspect_artifacts(tmp_path)
        assert result["status"] == "execution_passed"
        assert result["learning_signal"] == "absent_equal_rewards"
        assert result["grad_norm"] == 0 and result["learnability"]["status"] == "pending"
        assert result["unconsumed_session_artifacts"] == 1
        assert result["sampled_policy_format_error_steps"] == (8 if defect == "policy_format" else 0)
        # Completed execution and learnability are separate: a mixed group
        # needs a real nonzero gradient before it can admit production RL.
        rows = [json.loads(line) for line in (rollouts / "1.jsonl").read_text().splitlines()]
        rows[0]["score"] = 1
        (rollouts / "1.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        reward = {"reward": 1, "resolved": True, "verifier_returncode": 0, "finished": True}
        (tmp_path / "sessions/0/trajectory.json").write_text(json.dumps({
            "trajectories": [{"reward_info": reward}]}))
        assert inspect_artifacts(tmp_path)["learnability"]["status"] == "pending"
        stats["actor/grad_norm"] = 0.25
        (tmp_path / "metrics.jsonl").write_text(json.dumps({"step": 1, "data": stats}) + "\n")
        result = inspect_artifacts(tmp_path)
        assert result["reward_contrast_groups"] == 1 and result["grad_norm"] == 0.25
        assert result["learnability"]["status"] == "observed"


@pytest.mark.parametrize("package_sft", [False, True])
def test_input_lineage_verifies_actual_export_bytes(tmp_path, package_sft):
    from archlab.artifacts import sha256_file

    export, parent = tmp_path / "export", tmp_path / "parent"
    export.mkdir()
    parent.mkdir()
    parent_receipt = parent / ("ckpt_contents.json" if package_sft else "COMPLETE.json")
    parent_receipt.write_text("{}")
    for name in ("model.safetensors", "config.json"):
        (export / name).write_bytes(b"original")
    (export / "EXPORT.json").write_text(json.dumps({
        "files": {name: sha256_file(export / name) for name in ("model.safetensors", "config.json")},
        "training": {"format": "archlab-limite-agentic-sft-v1"} if package_sft else {},
        "source_checkpoint": {"path": str(parent), "receipt_sha256": sha256_file(parent_receipt)}}))
    config = tmp_path / "resolved.yaml"
    config.write_text(f"actor_rollout_ref:\n  model:\n    path: {export}\n")
    lineage = input_lineage(config)
    assert lineage["parent_receipt"]["path"] == str(parent_receipt)
    assert lineage["export/model.safetensors"]["bytes"] == 8
    parent_receipt.write_text('{"tampered": true}')
    with pytest.raises(AssertionError):
        input_lineage(config)
    parent_receipt.write_text("{}")
    (export / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(AssertionError, match="export bytes changed"):
        input_lineage(config)


def test_smoke_owns_ray_cluster_without_changing_source(tmp_path):
    from omegaconf import OmegaConf

    source = OmegaConf.create({"trainer": {"experiment_name": "production"},
                              "ray_kwargs": {"ray_init": {"address": "ray://shared.invalid:10001",
                                                           "runtime_env": {"env_vars": {}}}}})
    capacity = "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.mimoagent.max_concurrent_sessions"
    OmegaConf.update(source, capacity, 8)
    smoke = smoke_config(source, tmp_path)
    assert smoke.ray_kwargs.ray_init.address == "local"
    assert source.ray_kwargs.ray_init.address == "ray://shared.invalid:10001"
    assert source.trainer.experiment_name == "production"
    assert OmegaConf.select(smoke, capacity) == OmegaConf.select(source, capacity) == 8


def test_probe_timeout_terminates_only_its_own_child(tmp_path):
    with (tmp_path / "child.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_owned_process([sys.executable, "-c", "import time; time.sleep(30)"], log, 0.1)


@pytest.mark.parametrize("supports_wait", [False, True])
def test_shutdown_supports_both_ray_apis(monkeypatch, supports_wait):
    from types import SimpleNamespace

    import psutil

    calls = []

    def old_shutdown():
        calls.append(False)

    def new_shutdown(*, wait_for_processes=False):
        calls.append(wait_for_processes)

    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(children=lambda recursive: []))
    report = _shutdown_ray(SimpleNamespace(shutdown=new_shutdown if supports_wait else old_shutdown,
                                          is_initialized=lambda: False))
    assert calls == [supports_wait]
    assert report["remaining_owned_children"] == 0 and report["forced_terminations"] == 0


if __name__ == "__main__":
    _run(*sys.argv[1:])

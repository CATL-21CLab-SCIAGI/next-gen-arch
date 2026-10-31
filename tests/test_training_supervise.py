import subprocess
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from archlab.training import supervise

_BASE = Path(tempfile.mkdtemp()) / "base.yaml"
_BASE.write_text("trainer: {}\n")
SPEC = {"arm": "a", "data": "/d/new.parquet", "source": "/s/v2", "save_freq": 5, "base": str(_BASE)}


def test_new_data_resumes_actor_only_with_a_fresh_seed_and_repeats_stay_fresh():
    last = {"step": 60, "data": "/d/old.parquet", "source": "/s/v1", "mode": "full", "seed": 1060}
    first = supervise.plan_launch(SPEC, 110, last)
    assert (first["mode"], first["seed"], first["source"]) == ("fresh", 1110, "/s/v2")
    retry = supervise.plan_launch(SPEC, 110, first)  # crashed before a new checkpoint: the loader state is still old
    assert (retry["mode"], retry["seed"]) == ("fresh", 1110)
    later = supervise.plan_launch(SPEC, 115, first)  # checkpoint 115 carries the new dataloader state
    assert (later["mode"], later["seed"]) == ("full", 1110)


def test_switch_is_due_only_when_launch_inputs_change():
    assert not supervise.switch_due(SPEC, {})
    assert not supervise.switch_due(SPEC, {"data": "/d/new.parquet", "source": "/s/v2"})
    assert supervise.switch_due(SPEC, {"data": "/d/old.parquet", "source": "/s/v2"})
    assert supervise.switch_due(SPEC, {"data": "/d/new.parquet", "source": "/s/v1"})
    assert supervise.switch_due(SPEC, {"data": "/d/new.parquet"})  # launched before sources were recorded


def test_resolved_config_overrides_only_resume_inputs(tmp_path):
    base = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create({"data": {"train_files": ["/old"], "seed": 42, "max_prompt_length": 8192},
                                     "trainer": {"save_freq": 10, "resume_mode": "disable"}}), base)
    spec = {**SPEC, "base": str(base), "dir": str(tmp_path), "total_steps": 200}
    launch = supervise.plan_launch(spec, 110, {})
    config = OmegaConf.load(supervise.resolved_config(spec, launch, "/ck/init-step110/global_step_110"))
    assert config.data.train_files == ["/d/new.parquet"] and config.data.seed == 1110
    assert config.data.max_prompt_length == 8192 and config.trainer.save_freq == 5
    assert (config.trainer.resume_mode, config.trainer.resume_from_path) == ("resume_path", "/ck/init-step110/global_step_110")


def test_agent_pods_are_resubmitted_only_when_known_dead(tmp_path, monkeypatch):
    import json

    created = []
    monkeypatch.setattr(supervise.dlc_jobs, "site", lambda path: {"site": path})
    monkeypatch.setattr(supervise.dlc_jobs, "job_body", lambda config, **kwargs: kwargs)
    monkeypatch.setattr(supervise.dlc_jobs, "create", lambda body, config: created.append(body) or "new-job")
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: None)
    spec = {**SPEC, "dir": str(tmp_path), "checkpoints": str(tmp_path), "head_dir": str(tmp_path), "node": "n",
            "dlc_site": "/site.json", "agent_pods": {"name": "pods", "pods": 6, "cpu": 8, "memory": 64, "ray_cpus": 44}}
    arm = supervise.Arm(spec)
    monkeypatch.setattr(arm, "ssh", lambda command, check=False: "5")
    record = tmp_path / "agent-pods-job.json"
    record.write_text(json.dumps({"JobId": "old"}))

    stopped = []
    monkeypatch.setattr(supervise.dlc_jobs, "stop", lambda job, config: stopped.append(job))

    def describe(state, pods=6):
        def answer(job, config):
            if isinstance(state, Exception):
                raise state
            return {"status": state, "pods": pods}
        return answer

    monkeypatch.setattr(supervise.dlc_jobs, "describe", describe("Running"))
    arm.ensure_agent_pods()
    monkeypatch.setattr(supervise.dlc_jobs, "describe", describe(RuntimeError("API timeout")))
    arm.ensure_agent_pods()
    assert created == [] and stopped == []
    monkeypatch.setattr(supervise.dlc_jobs, "describe", describe("Running", pods=4))  # live but undersized
    arm.ensure_agent_pods()
    assert stopped == ["old"] and [body["pods"] for body in created] == [6]
    created.clear()
    record.write_text(json.dumps({"JobId": "old"}))
    monkeypatch.setattr(supervise.dlc_jobs, "describe", describe("Failed"))
    arm.ensure_agent_pods()
    assert [body["pods"] for body in created] == [6] and json.loads(record.read_text()) == {"JobId": "new-job"}
    record.unlink()
    arm.ensure_agent_pods()  # no record at all: start pods
    assert len(created) == 2


def test_a_new_arm_starts_from_its_base_model_without_resume(tmp_path):
    base = tmp_path / "base.yaml"
    OmegaConf.save(OmegaConf.create({"data": {"train_files": ["/old"], "seed": 42},
                                     "trainer": {"save_freq": 10, "resume_mode": "auto", "total_training_steps": 200}}), base)
    spec = {**SPEC, "base": str(base), "dir": str(tmp_path), "checkpoints": str(tmp_path / "checkpoints"),
            "total_steps": 40, "head_dir": str(tmp_path), "node": "n"}
    assert supervise.Arm(spec).latest() == 0
    launch = supervise.plan_launch(spec, 0, {})
    config = OmegaConf.load(supervise.resolved_config(spec, launch, None))
    assert config.trainer.resume_mode == "disable" and "resume_from_path" not in config.trainer
    assert config.trainer.total_training_steps == 40 and config.data.seed == 1000


def test_a_changed_base_config_is_a_switch_but_unrecorded_digests_are_not(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("rollout: {max_concurrent_sessions: 256}\n")
    spec = {**SPEC, "base": str(base)}
    last = supervise.plan_launch(spec, 20, {"data": spec["data"], "seed": 1000, "mode": "full", "step": 10})
    assert not supervise.switch_due(spec, last)
    base.write_text("rollout: {max_concurrent_sessions: 384}\n")
    assert supervise.switch_due(spec, last)
    assert not supervise.switch_due(spec, {k: v for k, v in last.items() if k != "base"})


def test_log_rotation_never_overwrites_an_earlier_rotation(tmp_path):
    log = tmp_path / "train.log"
    assert supervise.rotate_log(log) is None  # nothing to keep
    kept = []
    for text in ("first launch", "second launch"):  # two relaunches within the same second
        log.write_text(text)
        kept.append(supervise.rotate_log(log))
    assert not log.exists() and len(set(kept)) == 2
    assert sorted(p.read_text() for p in kept) == ["first launch", "second launch"]



def test_trainer_pids_matches_only_canonical_arm_directory(tmp_path, monkeypatch):
    wave7 = tmp_path / "wave7"
    wave8 = tmp_path / "wave8"
    wave7.mkdir()
    wave8.mkdir()
    arm = supervise.Arm({**SPEC, "arm": "9b", "dir": str(wave8), "checkpoints": str(wave8), "node": "n"})
    ps_output = "\n".join([
        "  PID COMMAND",
        f"101 /opt/venv/bin/python -u -m archlab.training.runtime {wave7}/resolved-9b-s20.yaml",
        f"102 /opt/venv/bin/python -u -m archlab.training.runtime {wave8}/resolved-9b-extra-s20.yaml",
        f"103 /opt/venv/bin/python -u -m archlab.training.runtime {wave8.resolve()}/resolved-9b-s20.yaml",
    ])
    monkeypatch.setattr(arm, "ssh", lambda command, check=False: ps_output)
    assert arm.trainer_pids() == ["103"]


def test_ssh_failure_and_busy_gpus_never_double_launch(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text("trainer: {}\n")
    spec = {
        **SPEC,
        "base": str(base),
        "dir": str(tmp_path),
        "checkpoints": str(tmp_path / "ckpts"),
        "head_dir": str(tmp_path),
        "node": "n",
        "total_steps": 40,
        "packages": "/pkg",
        "mlflow_config": "/mlflow.json",
    }
    arm = supervise.Arm(spec)
    launched = []
    monkeypatch.setattr(supervise.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(arm, "ensure_agent_pods", lambda: None)
    monkeypatch.setattr(arm, "launch", lambda last: launched.append(last))

    # Case 1: transient SSH exit 255 on ps probe must abort recovery without launching.
    def failing_ssh(command, check=False):
        if check:
            raise subprocess.CalledProcessError(255, ["ssh", "n", command])
        return ""

    monkeypatch.setattr(arm, "ssh", failing_ssh)
    assert arm.run(once=True) == 1
    assert launched == []

    # Case 2: GPUs stay occupied (>= 8000 MiB) across all wait polls -> refuse launch.
    def busy_gpu_ssh(command, check=False):
        if "ps -eo" in command:
            return "  PID COMMAND\n"
        return "220000\n"

    monkeypatch.setattr(arm, "ssh", busy_gpu_ssh)
    assert not arm.wait_for_free_gpus()
    assert arm.run(once=True) == 1
    assert launched == []

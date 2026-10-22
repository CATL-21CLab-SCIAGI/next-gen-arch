from omegaconf import OmegaConf

from archlab.training import supervise

SPEC = {"arm": "a", "data": "/d/new.parquet", "source": "/s/v2", "save_freq": 5}


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
    spec = {**SPEC, "base": str(base), "dir": str(tmp_path)}
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

    def status(state):
        def answer(job, config):
            if isinstance(state, Exception):
                raise state
            return state
        return answer

    monkeypatch.setattr(supervise.dlc_jobs, "status", status("Running"))
    arm.ensure_agent_pods()
    monkeypatch.setattr(supervise.dlc_jobs, "status", status(RuntimeError("API timeout")))
    arm.ensure_agent_pods()
    assert created == []
    monkeypatch.setattr(supervise.dlc_jobs, "status", status("Failed"))
    arm.ensure_agent_pods()
    assert [body["pods"] for body in created] == [6] and json.loads(record.read_text()) == {"JobId": "new-job"}
    record.unlink()
    arm.ensure_agent_pods()  # no record at all: start pods
    assert len(created) == 2

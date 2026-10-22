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

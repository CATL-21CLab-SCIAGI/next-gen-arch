import json

import pytest

from archlab.evaluation import tb21


def export(root, label):
    (root / label).mkdir(parents=True)
    (root / label / "EXPORTED").write_text("{}")


@pytest.fixture
def plan(tmp_path):
    exports = tmp_path / "exports"
    for step in (10, 20, 40, 50, 60, 100):
        export(exports, f"arm-9b-step{step}")
    export(exports, "arm-step20")
    (tmp_path / "native").mkdir()
    spec = {"exports": str(exports), "curve": "9b"}
    return {"points": [
        {"label": "native", "model": str(tmp_path / "native"), "curve": "9b", "step": 0},
        {"label": "missing", "model": str(tmp_path / "absent"), "curve": "9b", "step": 0},
        {"label": "arm-9b-step50", "model": str(exports / "arm-9b-step50"), "curve": "9b", "step": 50},
        {"newest": {**spec, "prefix": "arm-9b", "every": 20}},
        {"every": {**spec, "prefix": "arm-9b", "every": 20}},
        {"every": {"exports": str(exports), "curve": "4b", "prefix": "arm", "every": 20}},
    ]}


def test_priority_order_newest_then_fill_ins(plan):
    labels = [point["label"] for point in tb21.candidates(plan, set())]
    assert labels == ["native", "arm-9b-step50", "arm-9b-step100", "arm-9b-step20", "arm-9b-step40",
                      "arm-9b-step60", "arm-step20"]


def test_newest_waits_for_a_full_interval_after_the_last_evaluated_step(plan, tmp_path):
    done = {"native", "arm-9b-step50", "arm-9b-step100"}
    assert "arm-9b-step110" not in [p["label"] for p in tb21.candidates(plan, done)]
    export(tmp_path / "exports", "arm-9b-step120")
    assert tb21.candidates(plan, done)[0]["label"] == "arm-9b-step120"
    assert tb21.candidates(plan, done)[0]["step"] == 120


def test_harbor_command_keeps_official_budgets_and_mounts_the_toolchain():
    plan = {"harbor": "harbor", "tasks": "/tasks", "jobs": "/jobs", "concurrency": 12,
            "mirror": {"path": "/mirror", "uv_version": "0.9.5", "index": "https://pypi.example/simple/"}}
    command = tb21.harbor_command(plan, "model-a", "point")
    joined = " ".join(command)
    assert "--agent-timeout-multiplier" not in joined and "--verifier-timeout-multiplier" not in joined
    assert "--override-cpus" not in joined and "--override-memory-mb" not in joined
    assert json.loads(command[command.index("--mounts") + 1]) == [
        {"type": "bind", "source": "/mirror", "target": "/opt/tb-mirror", "read_only": True}]
    assert "UV_DOWNLOAD_URL=file:///opt/tb-mirror/astral-sh/uv/releases/download/0.9.5" in command
    assert "record_terminal_session=false" in command and "openai/model-a" in command
    oracle = tb21.harbor_command(plan, "", "oracle", agent="oracle")
    assert "-m" not in oracle and "--ak" not in oracle


def test_errored_trials_score_zero(tmp_path):
    for index, (reward, error) in enumerate([(1.0, None), (None, "AgentTimeoutError"), (0.0, None), (1.0, None)]):
        trial = tmp_path / f"t{index}"
        trial.mkdir()
        trial.joinpath("result.json").write_text(json.dumps({
            "task_name": f"task-{index}", "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
            "exception_info": {"exception_type": error} if error else None}))
    result = tb21.score(tmp_path, bootstrap=200)
    assert result["mean_reward"] == 0.5 and result["trials"] == 4 and result["solved"] == 2
    assert result["errors"] == {"AgentTimeoutError": 1} and result["solved_tasks"] == ["task-0", "task-3"]
    assert 0.0 <= result["ci95"][0] <= 0.5 <= result["ci95"][1] <= 1.0

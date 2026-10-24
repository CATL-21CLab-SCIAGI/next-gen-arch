import json
import sys
from types import SimpleNamespace

from archlab.evaluation import ladder


def fake_mlflow(monkeypatch):
    logged = []

    class Client:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1")

        def search_runs(self, experiment_ids, filter_string, order_by, max_results):
            name = filter_string.split("'")[1]
            return [SimpleNamespace(info=SimpleNamespace(run_id=f"found-{name}"))] if name == "shared" else []

        def create_run(self, experiment_id, run_name, tags):
            return SimpleNamespace(info=SimpleNamespace(run_id=f"new-{run_name}"))

        def log_metric(self, run_id, metric, value, step):
            logged.append((run_id, metric, value, step))

        def set_tag(self, run_id, tag, value):
            logged.append((run_id, tag, value))

    module = SimpleNamespace(set_tracking_uri=lambda uri: None, tracking=SimpleNamespace(MlflowClient=Client))
    monkeypatch.setitem(sys.modules, "mlflow", module)
    return logged


def test_each_point_is_logged_once_and_runs_are_shared_read_only(tmp_path, monkeypatch):
    logged = fake_mlflow(monkeypatch)
    secrets = tmp_path / "mlflow.json"
    secrets.write_text(json.dumps({"tracking_uri": "uri", "token": "t"}))
    heldout = tmp_path / "heldout.json"
    heldout.write_text(json.dumps({"runs": {"9b": "run-9b"}, "logged": ["9b|native-9b"]}))
    config = {"secrets": str(secrets), "ledger": str(tmp_path / "tb.json"), "runs_from": str(heldout),
              "experiment": "e"}
    curves = ladder.Curves(config)
    assert curves.log("9b", "tb21:native-9b", 0, {"tb21/mean_reward": 0.1})
    assert not curves.log("9b", "tb21:native-9b", 0, {"tb21/mean_reward": 0.1})
    assert curves.log("4b", "tb21:native-4b", 0, {"tb21/mean_reward": 0.05})
    assert logged == [("run-9b", "tb21/mean_reward", 0.1, 0), ("new-4b", "tb21/mean_reward", 0.05, 0)]
    assert json.loads(heldout.read_text())["logged"] == ["9b|native-9b"]  # the other ledger is never written
    assert json.loads((tmp_path / "tb.json").read_text())["runs"] == {"9b": "run-9b", "4b": "new-4b"}


def test_pooled_difference_pairs_tasks_across_points_and_seeds():
    base = [{"per_task": {"a": {"successes": 0, "scored": 2}, "b": {"successes": 1, "scored": 2}}},
            {"per_task": {"a": {"successes": 0, "scored": 4}, "b": {"successes": 1, "scored": 4}}}]
    arm = [{"per_task": {"a": {"successes": 1, "scored": 2}, "b": {"successes": 2, "scored": 2}, "c": {"successes": 2, "scored": 2}}}]
    result = ladder.pooled_difference(base, arm, bootstrap=200)
    assert result["tasks"] == 2  # only tasks measured on both sides
    assert abs(result["baseline"] - (0 + 2 / 6) / 2) < 1e-9 and result["trained"] == 0.75
    assert abs(result["difference"] - (0.5 + (1 - 2 / 6)) / 2) < 1e-9
    assert result["ci95"][0] <= result["difference"] <= result["ci95"][1]


def test_a_curve_opened_by_another_ladder_is_reused_by_name(tmp_path, monkeypatch):
    logged = fake_mlflow(monkeypatch)
    secrets = tmp_path / "mlflow.json"
    secrets.write_text(json.dumps({"tracking_uri": "uri", "token": "t"}))
    curves = ladder.Curves({"secrets": str(secrets), "ledger": str(tmp_path / "l.json"), "experiment": "e"})
    assert curves.log("shared", "p", 0, {"m": 1.0}) and curves.log("fresh", "p", 0, {"m": 2.0})
    assert [entry[0] for entry in logged] == ["found-shared", "new-fresh"]

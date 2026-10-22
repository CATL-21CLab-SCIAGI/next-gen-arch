import json
import sys
from types import SimpleNamespace

from archlab.evaluation import ladder


def fake_mlflow(monkeypatch):
    logged = []

    class Client:
        def get_experiment_by_name(self, name):
            return SimpleNamespace(experiment_id="1")

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

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archlab.tracking.mlflow_sync import Sync, point_values, read_rows


def row(step):
    return {
        "step": step,
        "consumed_supervised_tokens": step * 100,
        "supervised_tokens": 100,
        "loss": 1.0 / step,
        "seconds": 2.0,
        "learning_rate": 0.001,
        "gradient_norm_before_clip": 0.5,
        "max_memory_allocated_gib": 10,
    }


class Client:
    def __init__(self):
        self.metrics = []
        self.created = 0
        self.fail_once = False
        self.status = None
        self.tags = {}

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, *args, **kwargs):
        return []

    def create_run(self, *args, **kwargs):
        self.created += 1
        return SimpleNamespace(info=SimpleNamespace(run_id="test"))

    def log_batch(self, run_id, metrics=(), **kwargs):
        self.metrics.extend(metrics)
        self.tags.update({tag.key: tag.value for tag in kwargs.get("tags", [])})
        if metrics and self.fail_once:
            self.fail_once = False
            raise ConnectionError("simulated lost acknowledgement")

    def update_run(self, *args, **kwargs):
        self.status = kwargs["status"]

    def set_terminated(self, *args, **kwargs):
        self.status = kwargs["status"]


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.history = self.root / "train-metrics.jsonl"
        self.history.write_text(json.dumps(row(1)) + "\n" + json.dumps(row(2)) + "\n")
        contract = {
            "runtime": {"parameters": 123},
            "world_size": 8,
            "project_commit": "abc",
            "optimizer": "test",
            "cpu_offload": False,
            "accumulation": 1,
        }
        (self.root / "RUN_CONTRACT.json").write_text(json.dumps(contract))
        self.spec = {
            "id": "test",
            "experiment": "test",
            "name": "normal",
            "phase": "scratch",
            "variant": "normal",
            "path": str(self.root),
            "history": [str(self.history)],
            "budget": 1000,
            "dataset": "fixture",
            "state": "paused",
        }

    def test_incremental_sync_does_not_reimport_history(self):
        client = Client()
        sync = Sync(client, self.root / "state.json")
        sync.sync_run(self.spec)
        count = len(client.metrics)
        sync.sync_run(self.spec)
        self.assertEqual(len(client.metrics), count)
        self.assertEqual(client.created, 1)
        with self.history.open("a") as f:
            f.write(json.dumps(row(3)) + "\n")
        sync.sync_run(self.spec)
        points = [m for m in client.metrics if m.key == "train/cross_entropy"]
        self.assertEqual([m.step for m in points], [1, 2, 3])
        token_points = [m for m in client.metrics if m.key == "train/cross_entropy_by_tokens"]
        self.assertEqual([m.step for m in token_points], [100, 200, 300])

    def test_partial_final_line_is_ignored(self):
        with self.history.open("a") as f:
            f.write('{"step":3')
        self.assertEqual(len(read_rows([self.history])), 2)

    def test_discontinuous_source_rejected(self):
        self.history.write_text(json.dumps(row(1)) + "\n" + json.dumps(row(3)) + "\n")
        with self.assertRaises(ValueError):
            read_rows([self.history])

    def test_failed_batch_retries_identical_points(self):
        client = Client()
        client.fail_once = True
        sync = Sync(client, self.root / "state.json")
        with self.assertRaises(ConnectionError):
            sync.sync_run(self.spec)
        first = [(m.key, m.step, m.timestamp, m.value) for m in client.metrics]
        self.assertIsNone(sync.state["runs"]["test"]["last_step"])
        sync.sync_run(self.spec)
        self.assertEqual(
            first, [(m.key, m.step, m.timestamp, m.value) for m in client.metrics[len(first) :]]
        )

    def test_nonfinite_or_zero_duration_rejected(self):
        r = row(1)
        r["seconds"] = 0
        with self.assertRaises(ValueError):
            point_values(r)

    def test_explicit_native_failure_terminates_mlflow_run(self):
        self.spec["state"] = "failed"
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "failed")
        self.assertEqual(client.status, "FAILED")

    def test_native_launcher_receipt_overrides_running_config(self):
        self.spec["state"] = "running"
        (self.root / "launcher-node0-failure.json").write_text('{"launcher_exit_code":1}')
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "failed")
        self.assertEqual(client.status, "FAILED")

    def test_serving_failure_overrides_fresh_chat_heartbeat(self):
        self.spec["state"] = "running"
        service = self.root / "service"
        service.mkdir()
        self.spec["serving_path"] = str(service)
        (service / "rank-00-failure.json").write_text('{"reason":"serving failure"}')
        (self.root / "ACTIVITY.json").write_text(
            json.dumps({"phase": "chat", "heartbeat_unix": time.time()})
        )
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "failed")
        self.assertEqual(client.status, "FAILED")

    def test_frozen_training_stays_paused_while_chat_is_active(self):
        self.spec["serving_path"] = str(self.root / "service")
        (self.root / "ACTIVITY.json").write_text(
            json.dumps({"phase": "chat", "heartbeat_unix": time.time(), "stage": "playground"})
        )
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "paused")
        self.assertEqual(client.status, "FINISHED")
        self.assertEqual(client.tags["serving.state"], "chat")

    def test_stale_stream_is_distinct_from_checkpointing(self):
        self.spec["state"] = "running"
        os.utime(self.history, (time.time() - 3600, time.time() - 3600))
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "stalled")
        checkpoint = self.root / "checkpoints/step-000002"
        checkpoint.mkdir(parents=True)
        self.assertEqual(sync.sync_run(self.spec)["state"], "checkpointing")
        (checkpoint / "COMPLETE.json").write_text(
            json.dumps({"cursor": {"step": 2, "supervised_tokens": 200}})
        )
        (self.root / "STOP_REQUEST").touch()
        self.assertEqual(sync.sync_run(self.spec)["state"], "paused")
        self.assertEqual(client.status, "FINISHED")

    def test_budget_completion_waits_for_final_checkpoint(self):
        self.spec.update(state="running", budget=200)
        checkpoint = self.root / "checkpoints/step-000002"
        checkpoint.mkdir(parents=True)
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "checkpointing")
        (checkpoint / "COMPLETE.json").write_text(
            json.dumps({"cursor": {"step": 2, "supervised_tokens": 200}})
        )
        self.assertEqual(sync.sync_run(self.spec)["state"], "complete")
        self.assertEqual(client.status, "FINISHED")

    def test_qualified_source_transition_preserves_run_identity(self):
        client = Client()
        sync = Sync(client, self.root / "state.json")
        sync.sync_run(self.spec)
        contract = json.loads((self.root / "RUN_CONTRACT.json").read_text())
        contract["project_commit"] = "def"
        (self.root / "RUN_CONTRACT.json").write_text(json.dumps(contract))
        with self.assertRaises(FileNotFoundError):
            sync.sync_run(self.spec)
        (self.root / "SOURCE_TRANSITION.json").write_text(
            json.dumps(
                {
                    "from_commit": "abc",
                    "to_commit": "def",
                    "training_math_unchanged": True,
                    "cursor": {"step": 2},
                }
            )
        )
        sync.sync_run(self.spec)
        self.assertEqual(client.created, 1)
        self.assertEqual(sync.state["runs"]["test"]["source_commit"], "def")

    def test_fresh_evaluation_activity_is_not_a_stalled_trainer(self):
        self.spec["state"] = "running"
        os.utime(self.history, (time.time() - 3600, time.time() - 3600))
        (self.root / "ACTIVITY.json").write_text(
            json.dumps({"phase": "chat", "heartbeat_unix": time.time(), "stage": "playground"})
        )
        client = Client()
        sync = Sync(client, self.root / "state.json")
        self.assertEqual(sync.sync_run(self.spec)["state"], "chat")
        self.assertEqual(client.status, "RUNNING")

    def test_full_eval_resume_point_need_not_be_in_training_ledger(self):
        self.spec["phase"] = "full-finetuning"
        evaluation = {
            "step": 0,
            "consumed_supervised_tokens": 0,
            "loss": 1.0,
            "cross_entropy": 1.0,
            "perplexity": 2.718,
            "top1_token_accuracy": 0.5,
            "top5_token_accuracy": 0.9,
            "mean_predictive_entropy": 1.0,
            "targets": 64000,
            "seconds": 20.0,
        }
        (self.root / "validation.jsonl").write_text(json.dumps(evaluation) + "\n")
        client = Client()
        sync = Sync(client, self.root / "state.json")
        sync.sync_run(self.spec)
        points = [m for m in client.metrics if m.key == "eval/math64k/cross_entropy"]
        self.assertEqual([(m.step, m.value) for m in points], [(0, 1.0)])
        first_time = points[0].timestamp
        evaluation.update(step=2, consumed_supervised_tokens=200)
        with (self.root / "validation.jsonl").open("a") as stream:
            stream.write(json.dumps(evaluation) + "\n")
        with patch(
            "archlab.tracking.mlflow_sync.time.time", return_value=(first_time + 60000) / 1000
        ):
            sync.sync_run(self.spec)
        points = [m for m in client.metrics if m.key == "eval/math64k/cross_entropy"]
        self.assertEqual(points[-2].timestamp, first_time)
        self.assertEqual(points[-1].timestamp, first_time + 60000)


if __name__ == "__main__":
    unittest.main()

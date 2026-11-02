import io
import json
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from archlab.tracking.rl_mlflow import (
    RLMetricSync,
    _metadata,
    bootstrap_dns,
    evaluation_values,
    main,
    profile_values,
    train_values,
)


@dataclass
class Metric:
    key: str
    value: float
    timestamp: int
    step: int


@dataclass
class Pair:
    key: str
    value: str


class Client:
    def __init__(self):
        self.runs = []
        self.metrics = []
        self.params = {}
        self.fail_once = False
        self.terminated = []

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id="1")

    def search_runs(self, *args, **kwargs):
        return self.runs

    def create_run(self, experiment_id, *, run_name, tags):
        run = SimpleNamespace(info=SimpleNamespace(run_id="new-rl-run"), data=SimpleNamespace(tags=dict(tags)))
        self.runs.append(run)
        return run

    def log_batch(self, run_id, *, metrics=(), tags=(), params=(), **kwargs):
        if run_id != "new-rl-run":
            raise AssertionError("attempted to modify a parent run")
        self.metrics.extend(metrics)
        self.runs[0].data.tags.update({item.key: item.value for item in tags})
        self.params.update({item.key: item.value for item in params})
        if metrics and self.fail_once:
            self.fail_once = False
            raise ConnectionError("lost acknowledgement with a credential-bearing exception")

    def get_metric_history(self, run_id, key):
        return [metric for metric in self.metrics if metric.key == key]

    def set_terminated(self, run_id, *, status):
        if run_id != "new-rl-run":
            raise AssertionError("attempted to terminate a parent run")
        self.terminated.append(status)


def train_row(step, **extra):
    return {
        "update_step": step, "policy_version": f"checkpoint-policy-{step}",
        "reward_mean": .5, "reward_std": .5, "policy_loss": -.1,
        "valid_answer_rate": .75, "update_skipped": False,
        "rollout_tokens": 200, "rollout_seconds": 10., "update_seconds": 3.,
        **extra,
    }


def profile_row():
    return {
        "format": "archlab-rl-rollout-throughput-v1", "world_size": 16,
        "prompt_tokens": 93, "max_new_tokens": 4, "retained_weights": True,
        "batch4_over_batch1_throughput_ratio": 3.5,
        "input_token_count_includes_padding": True,
        "private_prompt": "SHOULD-NOT-BE-UPLOADED", "measured_mfu": .99,
        "batches": [
            {"batch_size_per_rank": 1, "generated_tokens_per_second": 10.,
             "max_rank_seconds_sum": 6.4, "actual_generated_tokens_global": 64,
             "input_tokens_processed_global": 8192, "global_batch_size": 16},
            {"batch_size_per_rank": 4, "generated_tokens_per_second": 35.,
             "max_rank_seconds_sum": 256 / 35., "actual_generated_tokens_global": 256,
             "input_tokens_processed_global": 32768, "global_batch_size": 64},
        ],
    }


class RLMLflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = {
            "project_commit": "abc", "architecture": "normal-additive",
            "algorithm": "rloo", "parent_checkpoint": {"path": "/checkpoint/step4537", "sha256": "parent-sha"},
            "data_manifest": {"path": "/data/manifest.json", "sha256": "data-sha"},
            "group_size": 4, "reward_backend": "exact-rational-scalar-v1",
            "token": "SHOULD-NOT-BE-UPLOADED",
        }
        self.contract_path = self.root / "RUN_CONTRACT.json"
        self.contract_path.write_text(json.dumps(self.contract))
        self.spec = {"id": "rl-normal-v1", "name": "RLOO normal", "variant": "normal", "path": str(self.root), "experiment": "online-RL", "parent_run_id": "parent-supervised-run"}
        self.state = self.root / "state.json"
        self.train = self.root / "rl-metrics.jsonl"
        self.evaluation = self.root / "evaluation.jsonl"
        self.entities = patch.dict(sys.modules, {"mlflow.entities": SimpleNamespace(Metric=Metric, Param=Pair, RunTag=Pair)})
        self.entities.start()
        self.addCleanup(self.entities.stop)

    def write_rows(self, path, rows):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_failed_qualification_is_not_left_running(self):
        client = Client()
        sync = RLMetricSync(client, self.state)
        (self.root / "rank-00-failure.json").write_text('{"traceback":"numerical gate failed"}')
        self.assertEqual(sync.sync_run(self.spec)["state"], "failed")
        self.assertEqual(sync.sync_run(self.spec)["state"], "failed")
        self.assertEqual(client.terminated, ["FAILED"])

    def test_incremental_metrics_and_provenance_are_separate_from_sft(self):
        self.write_rows(self.train, [train_row(1), train_row(2)])
        self.write_rows(self.evaluation, [{"update_step": 0, "policy_version": "parent-sha", "pass_at_1": .25, "valid_answer_rate": .8, "examples": 512}])
        client = Client()
        sync = RLMetricSync(client, self.state)
        first = sync.sync_run(self.spec)
        self.assertEqual((first["new_train_records"], first["new_eval_records"]), (2, 1))
        count = len(client.metrics)
        self.assertEqual(sync.sync_run(self.spec)["new_train_records"], 0)
        self.assertEqual(len(client.metrics), count)
        with self.train.open("a") as stream:
            stream.write(json.dumps(train_row(3)) + "\n")
        self.assertEqual(sync.sync_run(self.spec)["new_train_records"], 1)
        self.assertEqual([m.step for m in client.metrics if m.key == "rl/reward_mean"], [1, 2, 3])
        self.assertEqual(len(client.runs), 1)
        self.assertEqual(client.params["parent_checkpoint_sha256"], "parent-sha")
        self.assertEqual(client.params["data_manifest_sha256"], "data-sha")
        self.assertEqual(client.params["source_commit"], "abc")
        self.assertNotIn("token", client.params)
        self.assertNotIn("SHOULD-NOT-BE-UPLOADED", self.state.read_text())
        self.assertEqual(client.runs[0].data.tags["archlab.parent_run_id"], "parent-supervised-run")
        self.assertFalse(any(m.key.startswith("train/cross_entropy") for m in client.metrics))

    def test_new_run_logs_explicit_recipe_hyperparameters_and_precision_only(self):
        self.contract["recipe"] = {
            "max_new_tokens": 512, "context_limit": 2048, "seed": 20260922,
            "learning_rate": 1e-6, "replay_mode": "sampled-prefix", "replay_prefixes": 4,
            "retain_weights": True, "weight_reserve_gib": 16,
            "numerical_precision": {"cuda_matmul_allow_tf32": False,
                "cuda_matmul_allow_bf16_reduced_precision_reduction": False,
                "private_setting": "DO-NOT-UPLOAD"},
            "credentials": "DO-NOT-UPLOAD", "private_prompt": "DO-NOT-UPLOAD",
        }
        self.contract_path.write_text(json.dumps(self.contract))
        client = Client()
        RLMetricSync(client, self.state).sync_run(self.spec)
        expected = {
            "max_new_tokens": "512", "context_limit": "2048", "seed": "20260922",
            "learning_rate": "1e-06", "replay_mode": "sampled-prefix", "replay_prefixes": "4",
            "retain_weights": "true", "weight_reserve_gib": "16",
            "cuda_matmul_allow_tf32": "false",
            "cuda_matmul_allow_bf16_reduced_precision_reduction": "false",
        }
        for key, value in expected.items():
            self.assertEqual(client.params[key], value)
        self.assertNotIn("DO-NOT-UPLOAD", json.dumps(client.params))

    def test_legacy_contract_has_no_invented_recipe_metadata(self):
        metadata = _metadata(self.spec, self.contract)
        self.assertEqual(metadata["group_size"], "4")
        self.assertEqual(metadata["algorithm"], "rloo")
        for key in ("max_new_tokens", "context_limit", "seed", "learning_rate", "replay_mode",
                    "replay_prefixes", "retain_weights", "weight_reserve_gib",
                    "cuda_matmul_allow_tf32", "cuda_matmul_allow_bf16_reduced_precision_reduction"):
            self.assertNotIn(key, metadata)
        self.contract["recipe"] = {"max_new_tokens": None, "numerical_precision": {
            "cuda_matmul_allow_tf32": "false"}}
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            _metadata(self.spec, self.contract)

    def test_pending_missing_outputs_and_partial_lines(self):
        client = Client()
        sync = RLMetricSync(client, self.state)
        self.contract_path.unlink()
        self.assertEqual(sync.sync_run(self.spec)["state"], "pending")
        self.assertFalse(client.runs)
        self.contract_path.write_text(json.dumps(self.contract))
        self.assertEqual(sync.sync_run(self.spec)["state"], "pending")
        self.train.write_text('{"update_step":1')
        self.assertEqual(sync.sync_run(self.spec)["state"], "pending")
        self.write_rows(self.train, [train_row(1)])
        self.assertEqual(sync.sync_run(self.spec)["new_train_records"], 1)

    def test_lost_acknowledgement_does_not_duplicate_metrics(self):
        self.write_rows(self.train, [train_row(1)])
        client = Client()
        client.fail_once = True
        sync = RLMetricSync(client, self.state)
        with self.assertRaises(ConnectionError):
            sync.sync_run(self.spec)
        count = len(client.metrics)
        self.assertEqual(sync.state["runs"][self.spec["id"]]["streams"]["train"]["offset"], 0)
        restarted = RLMetricSync(client, self.state)
        self.assertEqual(restarted.sync_run(self.spec)["new_train_records"], 1)
        self.assertEqual(len(client.metrics), count)
        self.assertNotIn("pending", restarted.state["runs"][self.spec["id"]])

    def test_state_loss_recovers_server_identity_and_cursors(self):
        self.write_rows(self.train, [train_row(1)])
        client = Client()
        RLMetricSync(client, self.state).sync_run(self.spec)
        count = len(client.metrics)
        self.state.unlink()
        report = RLMetricSync(client, self.state).sync_run(self.spec)
        self.assertEqual(report["new_train_records"], 0)
        self.assertEqual(len(client.runs), 1)
        self.assertEqual(len(client.metrics), count)

    def test_pending_eval_is_retried_before_new_training_rows(self):
        self.write_rows(self.train, [train_row(1)])
        client = Client()
        sync = RLMetricSync(client, self.state)
        sync.sync_run(self.spec)
        self.write_rows(self.evaluation, [{"update_step": 1, "pass_at_1": .5}])
        client.fail_once = True
        with self.assertRaises(ConnectionError):
            sync.sync_run(self.spec)
        self.write_rows(self.train, [train_row(1), train_row(2)])
        report = sync.sync_run(self.spec)
        self.assertEqual((report["new_train_records"], report["new_eval_records"]), (1, 1))
        self.assertEqual(sum(m.key == "eval/pass_at_1" for m in client.metrics), 1)

    def test_source_rewrites_contract_changes_and_old_sft_identity_fail_closed(self):
        self.write_rows(self.train, [train_row(1)])
        client = Client()
        sync = RLMetricSync(client, self.state)
        sync.sync_run(self.spec)
        self.write_rows(self.train, [train_row(1, reward_mean=.4)])
        with self.assertRaisesRegex(ValueError, "prefix changed"):
            sync.sync_run(self.spec)
        self.contract["project_commit"] = "changed"
        self.contract_path.write_text(json.dumps(self.contract))
        with self.assertRaisesRegex(ValueError, "immutable"):
            sync.sync_run(self.spec)
        self.state.unlink()
        client.runs[0].data.tags["archlab.phase"] = "full-finetuning"
        with self.assertRaisesRegex(ValueError, "not this immutable RL source"):
            RLMetricSync(client, self.state).sync_run(self.spec)

    def test_metric_contract_skips_absent_kl_and_allows_legitimate_skips(self):
        values = train_values(train_row(3, update_skipped=True, update_seconds=0., reference_kl_available=False, sampled_kl=0.))
        self.assertEqual(values["rl/update_skipped"], 1.)
        self.assertEqual(values["perf/rollout_tokens_per_second"], 20.)
        self.assertNotIn("rl/kl", values)
        self.assertEqual(train_values(train_row(3, sampled_kl=.1))["rl/kl"], .1)
        with self.assertRaises(ValueError):
            train_values(train_row(1, policy_loss=float("nan")))
        with self.assertRaises(ValueError):
            train_values(train_row(1, rollout_seconds=0.))
        with self.assertRaises(ValueError):
            train_values(train_row(1, rollout_seconds=1e-320))
        with self.assertRaises(ValueError):
            evaluation_values({"pass_at_1": 1.1})

    def test_optional_scientific_metrics_and_null_replay_are_not_fabricated(self):
        expected = {
            "rl/truncation_rate": .25, "rl/nonflat_prompt_groups": 3., "rl/optimizer_step": 2.,
            "numerical/replay_max_abs_error": .001, "numerical/replay_mean_abs_error": .0001,
            "perf/generated_tokens_per_second": 12., "perf/iteration_seconds": 20.,
        }
        values = train_values(train_row(3, truncation_rate=.25, nonflat_prompt_groups=3,
            optimizer_step=2, replay_max_abs_error=.001, replay_mean_abs_error=.0001,
            generated_tokens_per_second=12., iteration_seconds=20., measured_mfu=.8))
        for key, value in expected.items():
            self.assertEqual(values[key], value)
        self.assertFalse(set(expected) & train_values(train_row(3)).keys())
        skipped = train_values(train_row(4, update_skipped=True, nonflat_prompt_groups=0,
            optimizer_step=2, replay_max_abs_error=None, replay_mean_abs_error=None))
        self.assertEqual(skipped["rl/nonflat_prompt_groups"], 0.)
        self.assertEqual(skipped["rl/optimizer_step"], 2.)
        self.assertFalse(any(key.startswith("numerical/") for key in skipped))
        self.assertFalse(any("mfu" in key.lower() for key in values))
        for field, invalid in (("truncation_rate", 1.1), ("nonflat_prompt_groups", -1),
                               ("optimizer_step", 1.5), ("replay_mean_abs_error", -.1),
                               ("replay_max_abs_error", float("nan")), ("iteration_seconds", -1)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                train_values(train_row(3, **{field: invalid}))

    def test_profile_import_is_bounded_whitelisted_and_does_not_reopen_failed_run(self):
        profile = self.root / "PROFILE.json"
        profile.write_text(json.dumps(profile_row()))
        (self.root / "rank-00-failure.json").write_text('{"traceback":"qualification failed"}')
        client = Client()
        sync = RLMetricSync(client, self.state)
        report = sync.sync_run(self.spec)
        expected = profile_values(profile_row())
        self.assertEqual(report["state"], "failed")
        self.assertEqual(report["new_profile_metrics"], len(expected))
        self.assertEqual({metric.key: metric.value for metric in client.metrics}, expected)
        self.assertEqual(expected["profile/batch_4/generated_tokens_per_second"], 35.)
        self.assertEqual(expected["profile/batch4_over_batch1_throughput_ratio"], 3.5)
        self.assertEqual(sync.sync_run(self.spec)["new_profile_metrics"], 0)
        self.assertEqual(len(client.metrics), len(expected))
        self.assertEqual(client.terminated, ["FAILED"])
        self.assertFalse(any("mfu" in metric.key.lower() for metric in client.metrics))
        self.assertNotIn("SHOULD-NOT-BE-UPLOADED", json.dumps(client.runs[0].data.tags))
        self.assertNotIn("SHOULD-NOT-BE-UPLOADED", self.state.read_text())
        self.assertEqual(client.runs[0].data.tags["archlab.rl_profile_input_tokens_include_padding"], "true")
        changed = profile_row()
        changed["batches"][0]["generated_tokens_per_second"] = 11.
        profile.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "profile changed"):
            sync.sync_run(self.spec)

    def test_profile_lost_acknowledgement_and_state_loss_are_idempotent(self):
        (self.root / "PROFILE.json").write_text(json.dumps(profile_row()))
        client = Client()
        client.fail_once = True
        sync = RLMetricSync(client, self.state)
        with self.assertRaises(ConnectionError):
            sync.sync_run(self.spec)
        count = len(client.metrics)
        self.assertIn("pending_profile", sync.state["runs"][self.spec["id"]])
        restarted = RLMetricSync(client, self.state)
        restarted.sync_run(self.spec)
        self.assertEqual(len(client.metrics), count)
        self.assertNotIn("pending_profile", restarted.state["runs"][self.spec["id"]])
        self.state.unlink()
        self.assertEqual(RLMetricSync(client, self.state).sync_run(self.spec)["new_profile_metrics"], 0)
        self.assertEqual(len(client.metrics), count)

    def test_profile_pending_ack_recovers_after_state_loss_and_new_train_rows(self):
        (self.root / "PROFILE.json").write_text(json.dumps(profile_row()))
        client = Client()
        client.fail_once = True
        with self.assertRaises(ConnectionError):
            RLMetricSync(client, self.state).sync_run(self.spec)
        profile_count = sum(metric.key.startswith("profile/") for metric in client.metrics)
        self.state.unlink()
        self.write_rows(self.train, [train_row(1)])
        report = RLMetricSync(client, self.state).sync_run(self.spec)
        self.assertEqual(report["new_train_records"], 1)
        self.assertEqual(report["new_profile_metrics"], 0)
        self.assertEqual(sum(metric.key.startswith("profile/") for metric in client.metrics), profile_count)

    def test_profile_payload_budget_and_metric_validation(self):
        (self.root / "PROFILE.json").write_bytes(b" " * (256 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, "bounded observer budget"):
            RLMetricSync(Client(), self.state).sync_run(self.spec)
        for mutate in (lambda p: p.update(format="other-dataset"),
                       lambda p: p["batches"].append(dict(p["batches"][0])),
                       lambda p: p["batches"][0].update(generated_tokens_per_second=-1)):
            value = profile_row()
            mutate(value)
            with self.assertRaises(ValueError):
                profile_values(value)

    def test_cli_never_prints_credential_bearing_exceptions(self):
        config = self.root / "config.json"
        config.write_text(json.dumps({"runs": [self.spec]}))
        credentials = self.root / "credentials.json"
        credentials.write_text(json.dumps({"tracking_uri": "https://mlflow.example", "token": "SENSITIVE-TOKEN"}))
        output = io.StringIO()
        with patch("archlab.tracking.rl_mlflow.configure_client", side_effect=ConnectionError("Bearer SENSITIVE-TOKEN")), redirect_stdout(output):
            status = main(["--config", str(config), "--credentials", str(credentials), "--state", str(self.state)])
        self.assertEqual(status, 1)
        self.assertNotIn("SENSITIVE-TOKEN", output.getvalue())
        self.assertIn("ConnectionError", output.getvalue())

    def test_dns_bootstrap_is_scoped_and_restored(self):
        calls = []

        def resolver(host, *args, **kwargs):
            calls.append(host)
            return []

        with patch("socket.getaddrinfo", resolver):
            with bootstrap_dns({"mlflow.example": "192.0.2.10"}, "https://mlflow.example"):
                socket.getaddrinfo("mlflow.example", 443)
                socket.getaddrinfo("other.example", 443)
            self.assertIs(socket.getaddrinfo, resolver)
        self.assertEqual(calls, ["192.0.2.10", "other.example"])
        with self.assertRaises(ValueError), bootstrap_dns({"other.example": "192.0.2.10"}, "https://mlflow.example"):
            pass


if __name__ == "__main__":
    unittest.main()

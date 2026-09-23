"""Follow online-RL JSONL ledgers in independent MLflow runs, without CE aliases.

Credentials reuse the private ``tracking_uri``/``token`` schema. An optional
``browser_uri`` is selected only with --use-browser-uri (or the config setting).
Config ``bootstrap_dns`` may map the selected URI's hostname to one literal IP;
the process-local resolver retains the original HTTP Host and TLS hostname.
Nothing changes system DNS, trainer state, or a parent supervised-training run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from functools import partial
import hashlib
import ipaddress
import json
import math
import re
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

from archlab.artifacts import atomic_write_json
from archlab.tracking.mlflow_sync import configure_client

atomic_json = partial(atomic_write_json, sort_keys=False, allow_nan=False, create_parents=False)

_IDENTITY = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_STREAMS = {"train": "rl-metrics.jsonl", "eval": "evaluation.jsonl"}
_EMPTY_SHA = hashlib.sha256(b"").hexdigest()
_PROFILE_TAG = "archlab.rl_profile"
_PROFILE_MAX_BYTES = 256 * 1024


def _scalar(value, key):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid scalar for {key}")
    return float(value)


def _step(row):
    value = row.get("update_step", row.get("step", row.get("policy_version")))
    if type(value) is not int or value < 0:
        raise ValueError("metric record needs nonnegative integer update_step")
    return value


def train_values(row):
    """Whitelist RL metrics; absent KL is not fabricated as zero."""
    aliases = {
        "reward_mean": "rl/reward_mean", "reward_std": "rl/reward_std",
        "policy_loss": "rl/policy_loss", "valid_answer_rate": "rl/valid_answer_rate",
        "truncation_rate": "rl/truncation_rate",
        "nonflat_prompt_groups": "rl/nonflat_prompt_groups", "optimizer_step": "rl/optimizer_step",
        "update_skipped": "rl/update_skipped", "skip_update": "rl/update_skipped",
        "replay_max_abs_error": "numerical/replay_max_abs_error",
        "replay_mean_abs_error": "numerical/replay_mean_abs_error",
        "rollout_tokens_per_second": "perf/rollout_tokens_per_second",
        "generated_tokens_per_second": "perf/generated_tokens_per_second",
        "iteration_seconds": "perf/iteration_seconds",
        "update_seconds": "perf/update_seconds", "rollout_seconds": "perf/rollout_seconds",
        "gradient_norm": "optim/gradient_norm", "learning_rate": "optim/learning_rate",
    }
    values = {}
    for source, target in aliases.items():
        value = row.get(source, row.get(target))
        if value is not None:
            number = _scalar(value, target)
            if target in values and values[target] != number:
                raise ValueError("conflicting metric aliases")
            values[target] = number
    if row.get("reference_kl_available", True):
        value = row.get("sampled_kl", row.get("kl", row.get("rl/kl")))
        if value is not None:
            values["rl/kl"] = _scalar(value, "rl/kl")
    if type(row.get("policy_version")) is int:
        values["rl/policy_version"] = _scalar(row["policy_version"], "policy_version")
    if "perf/rollout_tokens_per_second" not in values and "rollout_tokens" in row:
        seconds = values.get("perf/rollout_seconds", 0)
        if seconds <= 0:
            raise ValueError("rollout throughput requires a positive rollout_seconds")
        values["perf/rollout_tokens_per_second"] = _scalar(row["rollout_tokens"], "rollout_tokens") / seconds
    for key in ("rl/valid_answer_rate", "rl/update_skipped", "rl/truncation_rate"):
        if key in values and not 0 <= values[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    for key, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if (key.startswith(("perf/", "numerical/")) or key == "rl/reward_std") and value < 0:
            raise ValueError(f"{key} must be nonnegative")
    for key in ("rl/nonflat_prompt_groups", "rl/optimizer_step"):
        if key in values and (values[key] < 0 or not values[key].is_integer()):
            raise ValueError(f"{key} must be a nonnegative integer")
    if not values:
        raise ValueError("record has no recognized RL metrics")
    return values


def evaluation_values(row):
    values = {}
    for source in ("pass_at_1", "valid_answer_rate", "truncation_rate", "examples", "seconds"):
        target = "eval/" + source
        value = row.get(source, row.get(target))
        if value is not None:
            values[target] = _scalar(value, target)
    for key in ("eval/pass_at_1", "eval/valid_answer_rate", "eval/truncation_rate"):
        if key in values and not 0 <= values[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    for key in ("eval/examples", "eval/seconds"):
        if key in values and values[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if type(row.get("policy_version")) is int:
        values["eval/policy_version"] = _scalar(row["policy_version"], "policy_version")
    if "eval/pass_at_1" not in values:
        raise ValueError("evaluation record needs pass_at_1")
    return values


def profile_values(profile):
    """Upload only measured profile scalars, never raw inputs, paths, or MFU."""
    if profile.get("format") != "archlab-rl-rollout-throughput-v1":
        raise ValueError("unrecognized rollout profile format")
    values = {}
    for name in ("batch4_over_batch1_throughput_ratio", "prompt_tokens", "max_new_tokens",
                 "world_size", "retained_weights"):
        if profile.get(name) is not None:
            values["profile/" + name] = _scalar(profile[name], name)
    seen = set()
    for batch in profile.get("batches", []):
        size = batch.get("batch_size_per_rank")
        if type(size) is not int or not 1 <= size <= 1024 or size in seen:
            raise ValueError("profile batch sizes must be distinct positive integers <=1024")
        seen.add(size)
        for name in ("generated_tokens_per_second", "max_rank_seconds_sum",
                     "actual_generated_tokens_global", "input_tokens_processed_global", "global_batch_size"):
            if batch.get(name) is not None:
                values[f"profile/batch_{size}/{name}"] = _scalar(batch[name], name)
    if not values or any(value < 0 for value in values.values()):
        raise ValueError("profile needs nonnegative measured scalars")
    return values


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _metadata(spec, contract):
    """Only explicitly named provenance is uploaded; never copy arbitrary config."""
    parent = contract.get("parent_checkpoint", {})
    if isinstance(parent, str):
        parent = {"path": parent}
    data = contract.get("data_manifest", {})
    if isinstance(data, str):
        data = {"path": data}
    algorithm = contract.get("rl", {})
    recipe = contract.get("recipe") or {}
    precision = recipe.get("numerical_precision") or {}
    metadata = {
        "variant": spec["variant"], "phase": "online-rl",
        "architecture": contract.get("architecture", spec.get("architecture", spec["variant"])),
        "source_commit": contract.get("project_commit", contract.get("source_commit")),
        "parent_checkpoint_path": parent.get("path", contract.get("parent_checkpoint_path")),
        "parent_checkpoint_sha256": parent.get("sha256", parent.get("digest", contract.get("parent_checkpoint_sha256"))),
        "data_manifest_path": data.get("path", contract.get("data_manifest_path")),
        "data_manifest_sha256": data.get("sha256", contract.get("data_manifest_sha256")),
        "algorithm": contract.get("algorithm", algorithm.get("algorithm")),
        "group_size": contract.get("group_size", algorithm.get("group_size")),
        "reward_backend": contract.get("reward_backend", algorithm.get("reward_backend")),
        "metric_timestamp_semantics": "record timestamp if supplied, otherwise stable ingestion origin plus record index",
    }
    for key in ("max_new_tokens", "context_limit", "seed", "learning_rate", "replay_mode",
                "replay_prefixes", "retain_weights", "weight_reserve_gib"):
        value = recipe.get(key)
        if value is not None:
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(f"recipe metadata must be scalar: {key}")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"recipe metadata must be finite: {key}")
            metadata[key] = value
    for key in ("cuda_matmul_allow_tf32", "cuda_matmul_allow_bf16_reduced_precision_reduction"):
        value = precision.get(key)
        if value is not None:
            if type(value) is not bool:
                raise ValueError(f"numerical precision metadata must be boolean: {key}")
            metadata[key] = value
    return {key: str(value).lower() if type(value) is bool else str(value)
            for key, value in metadata.items() if value is not None}


def _blank_cursor():
    return {"offset": 0, "count": 0, "prefix_sha256": _EMPTY_SHA, "last_step": -1}


class RLMetricSync:
    """Append-only import with per-stream offsets and lost-acknowledgment recovery."""

    def __init__(self, client, state_path):
        self.client = client
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"format": "archlab-rl-mlflow-v1", "runs": {}}
        if self.state.get("format") != "archlab-rl-mlflow-v1":
            raise ValueError("state belongs to a different synchronizer")

    def save(self):
        atomic_json(self.state_path, self.state)

    def ensure_run(self, spec, contract):
        from mlflow.entities import Param, RunTag

        if not isinstance(spec.get("id"), str) or _IDENTITY.fullmatch(spec["id"]) is None:
            raise ValueError("invalid RL source identity")
        root = str(Path(spec["path"]).resolve())
        digest = _digest(contract)
        local = self.state["runs"].get(spec["id"])
        if local:
            expected = (root, digest, spec["experiment"], spec["variant"])
            if tuple(local[k] for k in ("source_path", "contract_sha256", "experiment", "variant")) != expected:
                raise ValueError("immutable RL source contract or identity changed")
            return local
        experiment = self.client.get_experiment_by_name(spec["experiment"])
        experiment_id = experiment.experiment_id if experiment else self.client.create_experiment(spec["experiment"], tags={"archlab.managed": "true"})
        matches = self.client.search_runs([experiment_id], filter_string=f"tags.`archlab.rl_source_id` = '{spec['id']}'", max_results=2)
        if len(matches) > 1:
            raise ValueError("duplicate RL source identities in MLflow")
        origin = int(time.time() * 1000)
        tags = {
            "archlab.rl_source_id": spec["id"], "archlab.phase": "online-rl",
            "archlab.variant": spec["variant"], "archlab.rl_source_path": root,
            "archlab.rl_contract_sha256": digest, "archlab.rl_ingestion_origin_ms": str(origin),
        }
        if spec.get("parent_run_id"):
            tags["archlab.parent_run_id"] = str(spec["parent_run_id"])
        run = matches[0] if matches else self.client.create_run(experiment_id, run_name=spec["name"], tags=tags)
        remote = run.data.tags
        if matches:
            for key in ("archlab.phase", "archlab.variant", "archlab.rl_source_path", "archlab.rl_contract_sha256"):
                if remote.get(key) != tags[key]:
                    raise ValueError("existing run is not this immutable RL source")
            origin = int(remote["archlab.rl_ingestion_origin_ms"])
        streams = {}
        for name in _STREAMS:
            marker = remote.get(f"archlab.rl_cursor.{name}")
            streams[name] = json.loads(marker) if marker else _blank_cursor()
        local = {
            "run_id": run.info.run_id, "experiment_id": experiment_id,
            "experiment": spec["experiment"], "variant": spec["variant"],
            "source_path": root, "contract_sha256": digest,
            "origin_ms": origin, "streams": streams, "recovered": bool(matches),
            "profile": json.loads(remote[_PROFILE_TAG]) if remote.get(_PROFILE_TAG) else None,
        }
        self.client.log_batch(run.info.run_id, params=[Param(k, v) for k, v in _metadata(spec, contract).items()], tags=[RunTag(k, v) for k, v in tags.items() if k != "archlab.rl_ingestion_origin_ms"])
        self.state["runs"][spec["id"]] = local
        self.save()
        return local

    def _send(self, local, name, row, raw, start, end, prefix_digest):
        from mlflow.entities import Metric, RunTag

        cursor = local["streams"][name]
        step = _step(row)
        if step < cursor["last_step"]:
            raise ValueError("metric steps went backwards")
        values = train_values(row) if name == "train" else evaluation_values(row)
        timestamp = row.get("timestamp_ms", local["origin_ms"] + cursor["count"])
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError("timestamp_ms must be a nonnegative integer")
        pending = {"stream": name, "start": start, "end": end, "sha256": hashlib.sha256(raw).hexdigest(), "timestamp_ms": timestamp}
        recovering = "pending" in local or local.get("recovered", False)
        if "pending" in local and local["pending"] != pending:
            raise ValueError("pending metric record changed before acknowledgement")
        local["pending"] = pending
        self.save()
        metrics = [Metric(key, value, timestamp, step) for key, value in values.items()]
        if recovering:
            missing = []
            for metric in metrics:
                history = self.client.get_metric_history(local["run_id"], metric.key)
                if not any(item.step == metric.step and item.timestamp == metric.timestamp and item.value == metric.value for item in history):
                    missing.append(metric)
            metrics = missing
        if metrics:
            self.client.log_batch(local["run_id"], metrics=metrics, synchronous=True)
        committed = {"offset": end, "count": cursor["count"] + 1, "prefix_sha256": prefix_digest, "last_step": step}
        tags = [RunTag(f"archlab.rl_cursor.{name}", json.dumps(committed, sort_keys=True))]
        if "policy_version" in row:
            tags.append(RunTag(f"archlab.rl_last_{name}_policy_version", str(row["policy_version"])))
        # Publish remote cursor only after acknowledged metric writes.
        self.client.log_batch(local["run_id"], tags=tags, synchronous=True)
        local["streams"][name] = committed
        local.pop("pending", None)
        local["recovered"] = False
        self.save()

    def _sync_stream(self, local, name, path):
        cursor = local["streams"][name]
        if not path.exists():
            if cursor["offset"]:
                raise ValueError("previously imported metric stream disappeared")
            return 0
        if path.stat().st_size < cursor["offset"]:
            raise ValueError("metric stream was truncated")
        imported = 0
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            remaining = cursor["offset"]
            while remaining:
                part = stream.read(min(remaining, 1024 * 1024))
                if not part:
                    raise ValueError("metric stream truncated during read")
                hasher.update(part)
                remaining -= len(part)
            if hasher.hexdigest() != cursor["prefix_sha256"]:
                raise ValueError("previously imported metric prefix changed")
            while True:
                start = stream.tell()
                raw = stream.readline(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError("metric record exceeds one MiB")
                if not raw or not raw.endswith(b"\n"):
                    break  # A live writer has not committed the final line yet.
                if not raw.strip():
                    raise ValueError("blank metric records are not permitted")
                row = json.loads(raw)
                hasher.update(raw)
                self._send(local, name, row, raw, start, stream.tell(), hasher.hexdigest())
                imported += 1
        return imported

    def _sync_profile(self, local, path, *, recovered=False):
        from mlflow.entities import Metric, RunTag

        if not path.exists():
            if local.get("profile") or local.get("pending_profile"):
                raise ValueError("previously observed rollout profile disappeared")
            return 0
        if path.stat().st_size > _PROFILE_MAX_BYTES:
            raise ValueError("rollout profile exceeds the bounded observer budget")
        with path.open("rb") as stream:
            raw = stream.read(_PROFILE_MAX_BYTES + 1)
        if len(raw) > _PROFILE_MAX_BYTES:
            raise ValueError("rollout profile grew beyond the bounded observer budget")
        source_sha = hashlib.sha256(raw).hexdigest()
        if local.get("profile"):
            if local["profile"]["sha256"] != source_sha:
                raise ValueError("previously imported rollout profile changed")
            return 0
        profile = json.loads(raw)
        values = profile_values(profile)
        marker = {"sha256": source_sha, "timestamp_ms": local["origin_ms"], "metric_count": len(values)}
        recovering = bool(local.get("pending_profile") or recovered)
        if local.get("pending_profile") and local["pending_profile"] != marker:
            raise ValueError("pending rollout profile changed before acknowledgement")
        local["pending_profile"] = marker
        self.save()
        metrics = [Metric(key, value, marker["timestamp_ms"], 0) for key, value in values.items()]
        if recovering:
            metrics = [metric for metric in metrics if not any(
                previous.step == metric.step and previous.timestamp == metric.timestamp
                and previous.value == metric.value
                for previous in self.client.get_metric_history(local["run_id"], metric.key))]
        if metrics:
            self.client.log_batch(local["run_id"], metrics=metrics, synchronous=True)
        tags = [RunTag(_PROFILE_TAG, json.dumps(marker, sort_keys=True)),
                RunTag("archlab.rl_profile_scope", "rollout-throughput-profile-not-MFU")]
        if type(profile.get("input_token_count_includes_padding")) is bool:
            tags.append(RunTag("archlab.rl_profile_input_tokens_include_padding",
                               str(profile["input_token_count_includes_padding"]).lower()))
        self.client.log_batch(local["run_id"], tags=tags, synchronous=True)
        local["profile"] = marker
        local.pop("pending_profile", None)
        self.save()
        return len(metrics)

    def sync_run(self, spec):
        root = Path(spec["path"])
        contract_path = root / "RUN_CONTRACT.json"
        if not contract_path.exists():
            return {"id": spec["id"], "state": "pending", "new_train_records": 0, "new_eval_records": 0}
        contract = json.loads(contract_path.read_text())
        local = self.ensure_run(spec, contract)
        profile_recovery = bool(local.get("recovered"))
        counts = {}
        order = list(_STREAMS)
        if local.get("pending"):
            pending_stream = local["pending"]["stream"]
            order.remove(pending_stream)
            order.insert(0, pending_stream)
        for name in order:
            counts[name] = self._sync_stream(local, name, root / _STREAMS[name])
        profile_metrics = self._sync_profile(local, root / "PROFILE.json", recovered=profile_recovery)
        active = sum(cursor["count"] for cursor in local["streams"].values()) > 0
        phase = "tracking" if active else "pending"
        terminal = None
        if any(root.glob("rank-*-failure.json")):
            phase, terminal = "failed", "FAILED"
        elif (root / "COMPLETE.json").exists():
            phase, terminal = "complete", "FINISHED"
        elif (root / "STOPPED.json").exists():
            phase, terminal = "stopped", "FINISHED"
        if terminal and local.get("terminal_status") != terminal:
            from mlflow.entities import RunTag
            self.client.log_batch(local["run_id"], tags=[RunTag("archlab.rl_state", phase)])
            self.client.set_terminated(local["run_id"], status=terminal)
            local["terminal_status"] = terminal
            self.save()
        return {"id": spec["id"], "run_id": local["run_id"], "experiment_id": local["experiment_id"], "state": phase, "new_train_records": counts["train"], "new_eval_records": counts["eval"], "new_profile_metrics": profile_metrics, "last_update_step": local["streams"]["train"]["last_step"]}


@contextmanager
def bootstrap_dns(mapping, tracking_uri):
    """Optional in-process exact-host resolution, never a system hosts-file edit."""
    hostname = urlparse(tracking_uri).hostname
    mapping = mapping or {}
    if not isinstance(mapping, dict) or any(key != hostname for key in mapping):
        raise ValueError("bootstrap_dns may only override the selected tracking hostname")
    addresses = {host: str(ipaddress.ip_address(value)) for host, value in mapping.items()}
    original = socket.getaddrinfo

    def resolve(host, *args, **kwargs):
        return original(addresses.get(host, host), *args, **kwargs)

    if addresses:
        socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--use-browser-uri", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("interval must be positive")
    args.state.parent.mkdir(parents=True, exist_ok=True)
    try:
        config = json.loads(args.config.read_text())
        credentials = json.loads(args.credentials.read_text())
        use_browser = args.use_browser_uri or config.get("use_browser_uri", False)
        uri = credentials["browser_uri" if use_browser else "tracking_uri"]
        with args.state.with_suffix(".lock").open("w") as lock, bootstrap_dns(config.get("bootstrap_dns"), uri):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            client = configure_client(args.credentials)
            if use_browser:
                from mlflow import MlflowClient
                client = MlflowClient(tracking_uri=uri)
            sync = RLMetricSync(client, args.state)
            while True:
                summary = {"utc": datetime.now(timezone.utc).isoformat(), "runs": [], "errors": []}
                for spec in json.loads(args.config.read_text())["runs"]:
                    try:
                        result = sync.sync_run(spec)
                        summary["runs"].append(result)
                        print(json.dumps(result), flush=True)
                    except Exception as error:
                        # HTTP exception messages may contain headers/tokens: log class only.
                        summary["errors"].append({"id": spec.get("id"), "type": type(error).__name__})
                atomic_json(args.state.parent / "RL_SYNC_STATUS.json", summary)
                if not args.watch:
                    return int(bool(summary["errors"]))
                time.sleep(args.interval)
    except Exception as error:
        print(json.dumps({"state": "failed", "type": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

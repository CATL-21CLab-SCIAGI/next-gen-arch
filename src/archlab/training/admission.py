"""Verify persisted GPU admission at every production entrypoint.

Receipts are evidence tied to an immutable checkout, not a transferable license
to run a different configuration. They are not signatures or a security boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from archlab.artifacts import sha256_file
from archlab.training import qualification as checks
from archlab.training.config import fingerprint, read_experiment, validate_science

FORMAT = "archlab-training-qualification-v2"
INPUTS = ("resolved.yaml", "experiment.json", "source.json")


def require_checkout():
    for name in ("src/archlab", "tests/integration", "pyproject.toml", "docs/upstreams.json"):
        if not (checks.ROOT / name).exists():
            raise ValueError("qualification and training require a full immutable source checkout: " + name)


def receipt_identity(output: Path, upstream: Path) -> dict:
    return dict(format=FORMAT, source_root=str(checks.ROOT.resolve()),
                upstream=str(upstream.resolve()),
                inputs_sha256={name: sha256_file(output / name) for name in INPUTS})


def required_checks(experiment) -> set[str]:
    if experiment.family == "qwen3_5":
        raise ValueError("native Qwen admission is not implemented for this source")
    plan = checks.MUSE_INTEGRATION_TESTS if experiment.family == "muse_glimmer" else checks.INTEGRATION_TESTS
    names = {"scientific_config", "upstream_runtime", "upstream_config", "deployment_runtime",
             "agentic_data", "model_export", "execution_bridge", "native_model_fixture",
             "sandbox_coverage", "native_distributed_oracle", "throughput_regression",
             "agentic_learning_signal", "source_unchanged", *plan}
    if experiment.family == "muse_glimmer" or experiment.intervention == "replace_attention":
        names.add("bootstrap_curriculum")
    return names


def verify_admission(config, directory: Path) -> dict:
    """Recheck inputs and environment before importing the upstream trainer."""
    require_checkout()
    directory = Path(directory).resolve()
    report = json.loads((directory / "qualification.json").read_text())
    if report.get("format") != FORMAT or report.get("passed") is not True or report.get("blockers") != []:
        raise ValueError("a complete passing qualification-v2 receipt is required")
    if report.get("source_root") != str(checks.ROOT.resolve()):
        raise ValueError("qualification belongs to another source checkout")
    expected = {name: sha256_file(directory / name) for name in INPUTS}
    if report.get("inputs_sha256") != expected:
        raise ValueError("qualification inputs changed")
    if fingerprint(config) != report.get("config_sha256"):
        raise ValueError("configuration differs from qualification; qualify the changed experiment")
    experiment = read_experiment(directory / "experiment.json")
    rows = report.get("checks", [])
    by_name = {row["name"]: row for row in rows}
    if (len(by_name) != len(rows) or not required_checks(experiment).issubset(by_name)
            or any(row.get("status") != "passed" for row in rows)):
        raise ValueError("qualification evidence is incomplete")
    plan = checks.MUSE_INTEGRATION_TESTS if experiment.family == "muse_glimmer" else checks.INTEGRATION_TESTS
    for name in {*plan, "native_model_fixture"}:
        row = by_name[name]
        if not row.get("reports") or row.get("tests", 0) <= 0:
            raise ValueError("qualification has no executed test evidence: " + name)
        if name in plan and row.get("deployment_sha256") != {
                key: expected[key] for key in ("resolved.yaml", "experiment.json")}:
            raise ValueError("test evidence belongs to a different deployment: " + name)
    for row in rows:
        for path, digest in row.get("reports", {}).items():
            if sha256_file(path) != digest:
                raise ValueError("qualification test report changed: " + path)
    checks.check_source_unchanged(json.loads((directory / "source.json").read_text()))
    validate_science(experiment, config)
    checks.check_package_config(config)
    # These checks verify current data/model bytes as well as package versions.
    # GPU tests are reused only while all of their deployment inputs still match.
    probes = {
        "upstream_runtime": lambda: checks.check_upstream(Path(report["upstream"])),
        "deployment_runtime": lambda: checks.check_runtime(experiment, config),
        "agentic_data": lambda: checks.check_data(experiment, config),
        "model_export": lambda: checks.check_model(experiment, config),
        "execution_bridge": lambda: checks.check_execution(experiment, config),
    }
    if "bootstrap_curriculum" in by_name:
        probes["bootstrap_curriculum"] = lambda: checks.check_curriculum(experiment, config)
    for name, probe in probes.items():
        recorded = {key: value for key, value in by_name[name].items()
                    if key not in {"name", "status", "seconds"}}
        # Normalize tuples returned by runtime probes to their JSON representation.
        if json.loads(json.dumps(probe())) != recorded:
            raise ValueError("deployment changed since qualification: " + name)
    from archlab.training.sandbox import check_sandbox

    sandbox = check_sandbox(experiment, config)
    if sandbox.get("status", "passed") != "passed":
        raise ValueError("sandbox deployment is no longer ready")
    return report

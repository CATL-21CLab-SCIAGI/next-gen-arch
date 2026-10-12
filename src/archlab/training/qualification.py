"""One fail-closed admission report for the actual package-backed deployment.

Metadata, imports and small native fixtures are readiness checks. Production
also requires the concrete GPU integration tests below; skipped tests and old
receipt files never admit a changed deployment.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.training.config import fingerprint, validate_science

ROOT = Path(__file__).resolve().parents[3]
INTEGRATION_TESTS = {
    "native_distributed_oracle": ("test_megatron_agentic_training.py::test_native_forward_backward", True),
    "package_optimizer_restart": ("test_megatron_agentic_training.py::test_package_engine_optimizer_scheduler_checkpoint_restart", True),
    "serving_reload_parity": ("test_agentic_training.py::test_vllm_full_checkpoint_short_context_parity", False),
    "native_context_backward": ("test_agentic_training.py::test_native_context_forward_backward_memory", False),
    "native_context_serving": ("test_agentic_training.py::test_vllm_native_context_capacity", False),
    "stock_ray_startup": ("test_agentic_pipeline.py::test_stock_ray_package_startup", False),
    "learner_throughput": ("test_agentic_throughput.py::test_package_learner_throughput", True),
    "agent_tool_verifier": ("test_agentic_sandbox.py::test_agent_tool_and_successful_verifier", False),
    "real_policy_grpo_update": ("test_agentic_pipeline.py::test_real_policy_grpo_update", False),
}

PREREQUISITES = {
    "serving_reload_parity": ("package_optimizer_restart",),
    "agent_tool_verifier": ("sandbox_coverage",),
    "real_policy_grpo_update": (
        "native_distributed_oracle", "package_optimizer_restart", "serving_reload_parity",
        "native_context_backward", "native_context_serving", "stock_ray_startup",
        "throughput_regression",
        "sandbox_coverage", "agent_tool_verifier",
    ),
}


def source_manifest():
    """Fingerprint actual source bytes, including files not yet committed."""
    files = set()
    for directory in ("src", "tests", "recipes", "scripts/patches"):
        files.update(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".patch"})
    files.update(ROOT / name for name in ("pyproject.toml", "uv.lock", ".gitmodules", "docs/upstreams.json")
                 if (ROOT / name).is_file())
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(files)}


def check_source_unchanged(expected):
    actual = source_manifest()
    changed = sorted(name for name in expected.keys() | actual.keys()
                     if expected.get(name) != actual.get(name))
    if changed:
        raise RuntimeError("source changed during admission: " + ", ".join(changed[:10]))
    return {"files": len(actual), "sha256": hashlib.sha256(
        json.dumps(actual, sort_keys=True).encode()).hexdigest()}


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, timeout=30).stdout


def _import_identity(name, root):
    module = importlib.import_module(name)
    locations = ([module.__file__] if getattr(module, "__file__", None)
                 else list(module.__spec__.submodule_search_locations or []))
    paths = [Path(path).resolve() for path in locations]
    if not paths or any(not path.is_relative_to(Path(root).resolve()) for path in paths):
        raise ValueError(f"{name} imports another checkout: {paths}")
    return [str(path) for path in paths]


def _verify_patch_set(checkout, specification):
    for patch in specification.get("patches", []):
        if sha256_file(ROOT / patch["path"]) != patch["sha256"]:
            raise ValueError("upstream patch differs from provenance: " + patch["path"])
    actual = hashlib.sha256(_git(checkout, "diff", "HEAD", "--")).hexdigest()
    if actual != specification.get("working_diff_sha256"):
        raise ValueError(f"{checkout}: working changes differ from the declared patch set")
    return actual


def check_upstream(upstream):
    upstream = Path(upstream).resolve()
    specification = json.loads((ROOT / "docs/upstreams.json").read_text())["verl"]
    expected = specification["revision"]
    actual = _git(upstream, "rev-parse", "HEAD").decode().strip()
    if actual != expected:
        raise ValueError(f"verl checkout differs from pinned revision: {actual}")
    imports = {"verl": _import_identity("verl", upstream / "verl")}
    pins = {"verl": actual}
    patches = {"verl": _verify_patch_set(upstream, specification)}
    for path, package in (("third_party/uni_agent", "uni_agent"),
                          ("third_party/mimoagent-osr", "mimoagent")):
        record = _git(upstream, "ls-tree", "HEAD", path).decode().split()
        if len(record) != 4 or record[0] != "160000":
            raise ValueError(f"missing pinned agent submodule: {path}")
        current = _git(upstream / path, "rev-parse", "HEAD").decode().strip()
        if current != record[2]:
            raise ValueError(f"agent submodule differs from verl gitlink: {path}")
        pins[package] = current
        imports[package] = _import_identity(package, upstream / path)
        patches[package] = _verify_patch_set(upstream / path, specification["dependencies"][package])
    # Import executable entry points, rather than infer usability from pip metadata.
    for name, root in (("verl.trainer.main_ppo", upstream / "verl"),
                       ("uni_agent.framework.entry", upstream / "third_party/uni_agent"),
                       ("recipes.code.dataset", upstream / "recipes"),
                       ("recipes.code.mimoagent_runner", upstream / "recipes"),
                       ("recipes.code.reward", upstream / "recipes")):
        imports[name] = _import_identity(name, root)
    return dict(revisions=pins, working_patch_sha256=patches, imports=imports)


def check_runtime(experiment, config):
    import torch

    packages = {}
    for distribution in ("torch", "transformers", "ray", "verl", "mimoagent", "uni-agent", "e2b",
                         "megatron-core", "transformer-engine",
                         {"tilelang": "tilelang", "te": "transformer-engine", "native": "flash-attn"}[experiment.attention],
                         config.actor_rollout_ref.rollout.name):
        packages[distribution] = importlib.metadata.version(distribution)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The pinned MiMo Agent requires Python 3.12")
    gpus = torch.cuda.device_count()
    if not torch.cuda.is_available() or gpus < config.trainer.n_gpus_per_node:
        raise ValueError(f"qualification must run on the deployment node; visible GPUs={gpus}")
    image = os.environ.get("ARCHLAB_CONTAINER_IDENTITY")
    if not image:
        raise ValueError("record ARCHLAB_CONTAINER_IDENTITY from the actual allocation before launch")
    return dict(python=sys.version, packages=packages, cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(), container=image,
                gpus=gpus, hardware="NVIDIA B300", reported_names=[
                    torch.cuda.get_device_name(index) for index in range(gpus)])


def check_package_config(config):
    """Run the same upstream validation as the entry point before GPU tests."""
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config

    validate_config(config=config, use_reference_policy=need_reference_policy(config),
                    use_critic=need_critic(config))
    return dict(validator="verl.utils.config.validate_config")


def check_data(experiment, config, output=None):
    from archlab.data.mimo_code import audit_tokenization, verify

    manifest = Path(experiment.data_manifest).resolve()
    result = verify(manifest)
    for key, filename in (("train_files", "train.parquet"), ("val_files", "validation.parquet")):
        actual = [Path(value).resolve() for value in config.data[key]]
        if actual != [manifest.parent / filename]:
            raise ValueError(f"{key} does not use the verified data manifest")
    if (config.data.custom_cls.path != "pkg://recipes.code.dataset"
            or config.data.custom_cls.name != "MimoAgentSWEDataset"
            or config.reward.custom_reward_function.path != "pkg://recipes.code.reward"
            or config.reward.custom_reward_function.name != "compute_score"):
        raise ValueError("agentic pilot must use MiMo's executable task dataset and reward")
    if not 0 < config.data.train_batch_size <= result["train"]:
        raise ValueError("training batch exceeds the verified training split")
    audit = audit_tokenization(manifest, Path(config.actor_rollout_ref.model.path), config.data.max_prompt_length)
    if output is not None:
        atomic_write_json(Path(output) / "data-audit.json", audit)
    return dict(manifest=str(manifest), manifest_sha256=sha256_file(manifest), verification=result,
                tokenization=audit)


def check_sft_parent(experiment):
    """Verify the actual SFT endpoint, rather than trust a 'fresh' run label."""
    parent = Path(experiment.parent_checkpoint).resolve()
    metadata = json.loads((parent / "COMPLETE.json").read_text())
    files = metadata.get("files", {})
    if experiment.parent_stage == "adapter_sft":
        # The original warmup format records the frozen publisher hash rather
        # than trainable_mode; the exporter independently verifies that hash.
        stage_matches = (metadata.get("trainable_mode") in (None, "adapter")
                         and bool(re.fullmatch(r"[0-9a-f]{64}", metadata.get("frozen_sha256", "")))
                         and "adapter.pt" in files and "backbone.pt" not in files
                         and not (parent / "backbone.pt").exists())
    else:
        stage_matches = (experiment.parent_stage == "full_sft"
                         and metadata.get("trainable_mode") == "full"
                         and {"adapter.pt", "backbone.pt"} <= set(files))
    if (not stage_matches or metadata.get("tokens") != experiment.parent_tokens
            or "rl_state.pt" in metadata.get("files", {}) or (parent / "rl_state.pt").exists()
            or any(key in metadata for key in ("math_protocol", "rl_split_sha256"))):
        raise ValueError("fresh RL requires the declared complete SFT endpoint, not post-RL weights")
    return {"path": str(parent), "receipt_sha256": sha256_file(parent / "COMPLETE.json")}


def check_model(experiment, config):
    from archlab.training.limite import verify_export

    if experiment.family != "limite" or experiment.variant not in {"normal", "simplicial"}:
        raise ValueError("this pretrained migration has no qualified model bridge for the selected family/variant")
    lineage = check_sft_parent(experiment)
    root = Path(config.actor_rollout_ref.model.path).resolve()
    receipt = verify_export(root)
    model = json.loads((root / "config.json").read_text())
    if model.get("trainable_mode") != "full":
        raise ValueError("the new RL phase must unfreeze all model weights")
    if receipt.get("source_checkpoint") != lineage:
        raise ValueError("export does not descend from the declared complete parent checkpoint")
    if model.get("model_type") != "archlab_limite" or model["adapter"]["variant"] != experiment.variant:
        raise ValueError("export architecture differs from the experiment")
    if model["adapter"].get("attention_backend") != experiment.attention:
        raise ValueError("export attention backend differs from the experiment")
    if experiment.attention == "native" and model.get("backbone_attention") != "flash_attention_2":
        raise ValueError("native normal RL requires FlashAttention-2 for backbone and preludes")
    if model["publisher_config"]["max_position_embeddings"] != experiment.native_context:
        raise ValueError("exported publisher context differs from the experiment")
    if receipt.get("optimizer_migrated") is not False:
        raise ValueError("this export must explicitly declare a fresh verl optimizer phase")
    return dict(path=str(root), export_sha256=sha256_file(root / "EXPORT.json"), parent=lineage,
                parent_stage=experiment.parent_stage, parent_tokens=experiment.parent_tokens,
                optimizer="fresh RL from SFT; source optimizer remains in the immutable parent")


def check_execution(experiment, config):
    if experiment.engine != "megatron":
        raise ValueError("the native Limite learner requires the Megatron bridge")
    if experiment.attention == "te":
        raise ValueError("the Limite TE attention path has not passed native forward/backward parity")
    if experiment.variant != "normal":
        raise ValueError("simplicial serving needs its short-axis cache and weight-loading integration")
    if config.actor_rollout_ref.model.external_lib != "archlab.training.megatron":
        raise ValueError("the native Megatron registration module is required")
    importlib.import_module("archlab.training.megatron")
    from megatron.bridge import AutoBridge

    from archlab.megatron.limite import NativeLimiteProvider

    provider = AutoBridge.from_hf_pretrained(
        config.actor_rollout_ref.model.path, trust_remote_code=False
    ).to_megatron_provider(load_weights=False)
    if not isinstance(provider, NativeLimiteProvider):
        raise ValueError("Megatron selected a generic model provider")
    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm":
        raise ValueError("the selected serving engine has no qualified Limite implementation")
    if (rollout.tensor_model_parallel_size != 1 or not rollout.enforce_eager
            or rollout.enable_prefix_caching):
        raise ValueError("normal serving currently requires TP=1, eager execution and prefix caching disabled")
    from vllm.config import ModelConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    architecture = json.loads((Path(config.actor_rollout_ref.model.path) / "config.json").read_text())["architectures"]
    serving_config = ModelConfig(model=str(config.actor_rollout_ref.model.path),
                                 dtype="bfloat16", max_model_len=experiment.native_context,
                                 trust_remote_code=False)
    model_class, name = ModelRegistry.resolve_model_cls(architecture, serving_config)
    if model_class.__module__ != "archlab.training.serving":
        raise ValueError("serving selected a generic model implementation")
    return dict(learner=f"{type(provider).__module__}.{type(provider).__qualname__}",
                serving=name, implementation=f"{model_class.__module__}.{model_class.__qualname__}",
                scope="registration only; generation/replay parity is a separate required GPU test")


def run_pytest(nodes, output, *, deployment=None, timeout=1800, world_size=1,
               updated_model=None):
    """Execute this checkout's concrete tests; missing/skipped tests never pass."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for node in nodes:
        if not (ROOT / node.split("::", 1)[0]).is_file():
            return dict(status="pending", reason="concrete integration test not implemented", node=node)
    report = output / "pytest.xml"
    reports = ([output / f"pytest-rank-{rank}.xml" for rank in range(world_size)]
               if world_size > 1 else [report])
    for path in reports:
        path.unlink(missing_ok=True)
    command = [sys.executable]
    if world_size > 1:
        command += ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                    f"--nproc_per_node={world_size}"]
    # Keep numerical receipts and checkpoint fixtures with the admission record;
    # pytest's default /tmp retention can remove them after later test runs.
    command += ["-m", "pytest", "-q", *nodes,
                f"--basetemp={output.resolve() / 'artifacts'}", f"--junitxml={report}"]
    env = os.environ.copy()
    if updated_model is not None:
        updated_model = Path(updated_model).resolve()
        if not (updated_model / "model.safetensors").is_file():
            raise ValueError("serving reload requires the preceding package optimizer export")
        env["ARCHLAB_PARITY_UPDATED_MODEL"] = str(updated_model)
    if deployment is not None:
        # Bind this invocation to a snapshot, even if the next candidate is
        # resolved while independent tests are still running.
        for name in ("resolved.yaml", "experiment.json"):
            (output / name).write_bytes((Path(deployment) / name).read_bytes())
        env["ARCHLAB_QUALIFICATION_CONFIG"] = str((output / "resolved.yaml").resolve())
        env["ARCHLAB_QUALIFICATION_EXPERIMENT"] = str((output / "experiment.json").resolve())
        config = OmegaConf.load(env["ARCHLAB_QUALIFICATION_CONFIG"])
        env.update(ARCHLAB_RUN_DISTRIBUTED="1", ARCHLAB_RUN_ENGINE="1", ARCHLAB_RUN_NATIVE_CONTEXT="1",
                   ARCHLAB_RUN_THROUGHPUT="1",
                   ARCHLAB_RUN_AGENTIC_STARTUP="1", ARCHLAB_RUN_AGENTIC_PIPELINE="1",
                   ARCHLAB_LIMITE_EXPORT=str(config.actor_rollout_ref.model.path),
                   ARCHLAB_MEGATRON_WORK=str(output / "learner"),
                   ARCHLAB_PARITY_GENERATED_TOKENS="128", ARCHLAB_PARITY_TEMPERATURE="1")
    sources = source_manifest()
    atomic_write_json(output / "source.json", sources)
    with (output / "pytest.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout)
    source = check_source_unchanged(sources)
    if not all(path.exists() for path in reports):
        raise RuntimeError(f"pytest did not produce a report (exit {completed.returncode}); see {output / 'pytest.log'}")
    by_rank = {path: list(ET.parse(path).iter("testcase")) for path in reports}
    if any(not cases for cases in by_rank.values()):
        raise RuntimeError(f"qualification tests failed or skipped: empty rank report; see {output / 'pytest.log'}")
    cases = [case for cases in by_rank.values() for case in cases]
    unsuccessful = [case.attrib.get("name") for case in cases
                    if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    if completed.returncode or not cases or unsuccessful:
        raise RuntimeError(f"qualification tests failed or skipped: {unsuccessful}; see {output / 'pytest.log'}")
    return dict(status="passed", tests=len(cases), world_size=world_size,
                source=source,
                deployment_sha256=({name: sha256_file(output / name)
                                    for name in ("resolved.yaml", "experiment.json")}
                                   if deployment is not None else None),
                reports={str(path): sha256_file(path) for path in reports})


def check_learning_signal(output):
    """A completed tool loop alone cannot establish a useful GRPO update."""
    receipts = list((Path(output) / "artifacts").rglob("pipeline-evidence.json"))
    if len(receipts) != 1:
        raise ValueError("real policy update must write exactly one learning-signal receipt")
    evidence = json.loads(receipts[0].read_text())
    contrast = evidence["reward_contrast_groups"]
    grad_norm = evidence["grad_norm"]
    if not isinstance(grad_norm, (int, float)) or not math.isfinite(grad_norm):
        raise ValueError("real policy update did not record a finite gradient norm")
    result = dict(receipt=str(receipts[0]), receipt_sha256=sha256_file(receipts[0]),
                  reward_contrast_groups=contrast, grad_norm=grad_norm)
    if contrast <= 0 or grad_norm <= 0:
        result.update(status="pending", reason="execution passed but this pilot has no nonzero GRPO learning signal")
    return result


def _check(name, function):
    started = time.monotonic()
    try:
        detail = function()
        result = dict(name=name, **{"status": "passed", **detail})
    except Exception as error:
        result = dict(name=name, status="failed", reason=f"{type(error).__name__}: {error}")
    result["seconds"] = time.monotonic() - started
    return result


def qualify(experiment, config, upstream, output):
    """Inspect the deployment and execute tests, without launching a trainer."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "resolved.yaml", resolve=True)
    atomic_write_json(output / "experiment.json", asdict(experiment))
    sources = source_manifest()
    atomic_write_json(output / "source.json", sources)
    checks = [_check("scientific_config", lambda: (validate_science(experiment, config) or {})),
              _check("upstream_runtime", lambda: check_upstream(upstream)),
              _check("upstream_config", lambda: check_package_config(config)),
              _check("deployment_runtime", lambda: check_runtime(experiment, config)),
              _check("agentic_data", lambda: check_data(experiment, config, output)),
              _check("model_export", lambda: check_model(experiment, config)),
              _check("execution_bridge", lambda: check_execution(experiment, config)),
              _check("native_model_fixture", lambda: run_pytest(
                  ["tests/test_training_limite.py"], output / "native_fixture"))]
    # These tests must use this exact resolved config/export and execute on the
    # requested GPU world. Existing tiny fixtures cannot replace production-scale
    # policy parity, distributed restart, or a real completed test verifier.
    ready = all(row["status"] == "passed" for row in checks)
    from archlab.training.sandbox import check_sandbox

    checks.append(_check("sandbox_coverage", lambda: check_sandbox(experiment, config)))
    for test, (node, distributed) in INTEGRATION_TESTS.items():
        if ready:
            passed = {row["name"] for row in checks if row["status"] == "passed"}
            missing = sorted(set(PREREQUISITES.get(test, ())) - passed)
            if missing:
                checks.append(dict(name=test, status="pending", reason="prerequisites not passed: " + ", ".join(missing)))
                continue
            updated_model = None
            if test == "serving_reload_parity":
                updated_model = output / "package_optimizer_restart/learner/engine-checkpoint/model/huggingface"
            checks.append(_check(test, lambda test=test, node=node, distributed=distributed,
                                updated_model=updated_model: run_pytest(
                [f"tests/integration/{node}"], output / test, deployment=output,
                world_size=config.trainer.n_gpus_per_node if distributed else 1,
                timeout=14760 if test == "real_policy_grpo_update" else 1800,
                updated_model=updated_model)))
            if test == "learner_throughput":
                from archlab.training.performance import check_throughput

                if checks[-1]["status"] == "passed":
                    checks.append(_check("throughput_regression", lambda test=test: check_throughput(
                        output / test, experiment.throughput)))
                else:
                    checks.append(dict(name="throughput_regression", status="pending",
                                       reason="learner benchmark has not passed"))
        else:
            checks.append(dict(name=test, status="pending", reason="deployment readiness checks failed"))
    if any(row["name"] == "real_policy_grpo_update" and row["status"] == "passed" for row in checks):
        checks.append(_check("agentic_learning_signal", lambda: check_learning_signal(output / "real_policy_grpo_update")))
    else:
        checks.append(dict(name="agentic_learning_signal", status="pending", reason="real policy update has not passed"))
    checks.append(_check("source_unchanged", lambda: check_source_unchanged(sources)))
    result = dict(format="archlab-training-qualification-v1", passed=all(
        row["status"] == "passed" for row in checks), config_sha256=fingerprint(config),
        checks=checks, blockers=[row["name"] for row in checks if row["status"] != "passed"])
    atomic_write_json(output / "qualification.json", result)
    return result

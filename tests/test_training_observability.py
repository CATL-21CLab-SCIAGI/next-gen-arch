"""Rollout integrity and honest measurements at the package boundaries."""
import ast
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf

from archlab.training.metrics import estimate_limite_flops, step_metrics
from archlab.training.rollout import RolloutContract, validate_trajectory


def trajectory():
    return SimpleNamespace(prompt_ids=[1, 2], response_ids=[10, 9, 50, 11, 9],
                           response_mask=[1, 1, 0, 1, 1], response_logprobs=[-.1, -.2, 0., -.1, -.2],
                           generation_spans=[(0, 2), (3, 5)],
                           extra_fields=dict(min_global_steps=2, max_global_steps=3),
                           reward_info=dict(termination_kind="completed"))


def validate(value):
    return validate_trajectory(value, question_id="task", group_id="group", response_id="group_0_0",
                               session_id=0, context=7, eos_ids={9}, rng_identity="seed42:group_0_0")


def test_two_turn_trajectory_counts_only_generated_eos_and_records_policy_span():
    value = trajectory()
    value.response_ids[2] = 9  # An EOS-like token in a tool observation is not generated.
    result = validate(value)
    assert result.generated_tokens == 4 and result.observation_tokens == 1
    assert result.native_eos_tokens == 2 and result.terminal_native_eos
    assert result.context_limit_reached and result.termination_reason == "completed"
    assert (result.policy_version_first, result.policy_version_last) == (2, 3)
    assert not result.rng_replayable  # Engine seed is not deterministic async replay.


@pytest.mark.parametrize("field,bad", [
    ("prompt_ids", []), ("response_ids", [1] * 6),
    ("response_mask", [1] * 5), ("response_logprobs", None),
    ("response_logprobs", [-.1, -.2, -.3, -.1, -.2]),
    ("response_logprobs", [-.1, float("nan"), 0, -.1, -.2]),
    ("generation_spans", [(0, 3), (2, 5)]), ("generation_spans", []),
    ("extra_fields", {}), ("extra_fields", dict(min_global_steps=3, max_global_steps=2)),
    ("reward_info", dict(termination_kind="framework_timeout")),
])
def test_invalid_trajectory_cannot_reach_learning(field, bad):
    value = trajectory()
    setattr(value, field, bad)
    with pytest.raises(ValueError):
        validate(value)


def test_actual_uni_agent_writer_attaches_validated_metadata_before_transport(tmp_path):
    path = Path("third_party/verl/third_party/uni_agent/uni_agent/framework/framework.py")
    module = ast.parse(path.read_text())
    method = next(n for n in ast.walk(module) if isinstance(n, ast.AsyncFunctionDef)
                  and n.name == "_write_session_trajectories_to_tq")
    # Execute the actual patched method without importing the distributed service.
    namespace = dict(tq=SimpleNamespace(async_kv_batch_put=AsyncMock()),
                     _list_of_tq_fields_to_tensordict=lambda x: x)
    exec(compile("from __future__ import annotations\n" + ast.unparse(method), str(path), "exec"), namespace)
    (tmp_path / "generation_config.json").write_text(json.dumps(dict(eos_token_id=9)))
    config = OmegaConf.create(dict(actor_rollout_ref=dict(model=dict(path=str(tmp_path)),
                                 rollout=dict(name="vllm", seed=42, max_model_len=7))))
    writer = SimpleNamespace(_trajectory_validator=RolloutContract(config),
                             _trajectory_to_tq_field_and_tag=lambda **kw: ({"tokens": kw["trajectory"]}, {"uid": "group"}))
    args = dict(uid="group", session_index=0, trajectories=[trajectory()],
                sample_fields={"extra_info": {"instance_id": "task"}}, global_steps=3, partition_id="train")
    asyncio.run(namespace[method.name](writer, **args))
    call = namespace["tq"].async_kv_batch_put.call_args.kwargs
    assert call["keys"] == ["group_0_0"]
    assert call["tags"][0]["question_id"] == "task"
    assert call["tags"][0]["policy_version_first"] == 2
    assert call["tags"][0]["native_eos_tokens"] == 2
    namespace["tq"].async_kv_batch_put.reset_mock()
    args["trajectories"][0].response_logprobs = None
    with pytest.raises(ValueError):
        asyncio.run(namespace[method.name](writer, **args))
    namespace["tq"].async_kv_batch_put.assert_not_called()


def metric_inputs():
    config = OmegaConf.create(dict(trainer=dict(v1=dict(trainer_mode="colocate_async"), nnodes=1, n_gpus_per_node=8),
                                   actor_rollout_ref=dict(rollout=dict(n=2))))
    tag = asdict(validate(trajectory()))
    tags = [tag, dict(tag, session_id=1, response_id="group_1_0")]
    batch = SimpleNamespace(batch=dict(response_mask=torch.ones(2, 5), rm_scores=torch.tensor([[0., 1.], [0., 0.]])))
    return dict(config=config, batch=batch, tags=tags, timing=dict(step=10., gen=6., update_actor=2.), step=3)


def test_colocated_phase_accounting_does_not_call_generation_idle():
    result = step_metrics(**metric_inputs())
    assert result["perf/allocated_gpu_seconds/step"] == 80
    assert result["perf/allocated_gpu_seconds/gen"] == 48
    assert result["perf/allocated_gpu_seconds/unattributed"] == 16
    assert result["perf/measured_idle_available"] == 0
    assert result["rollout/useful_groups"] == 1
    assert result["rollout/incomplete_groups"] == 0
    assert result["rollout/consumed_generated_tokens"] == 8
    assert "training/optimizer_updates" not in result  # A trainer step can contain several updates.


def test_multi_segment_sessions_are_not_extra_grpo_samples():
    inputs = metric_inputs()
    inputs["tags"][1]["session_id"] = 0
    assert step_metrics(**inputs)["rollout/incomplete_groups"] == 1
    inputs["timing"]["gen"] = 11
    with pytest.raises(ValueError, match="overlap"):
        step_metrics(**inputs)


def test_variable_length_flops_uses_native_geometry():
    from test_limite_performance import native_config

    from archlab.architectures.limite_cost import baseline_flops

    config = native_config()
    config.adapter, config.trainable_mode = {"variant": "normal"}, "full"
    lengths = [17, 40]
    expected = sum(baseline_flops(config, n).per_token * n for n in lengths)
    assert estimate_limite_flops(config, 57, lengths, 2.) == expected / 2e12
    config.adapter["variant"] = "simplicial"
    with pytest.raises(ValueError, match="normal"):
        estimate_limite_flops(config, 57, lengths, 2.)


def test_upstream_mfu_dispatch_uses_explicit_b300_peak(monkeypatch):
    import sys

    from test_limite_performance import native_config

    # Execute the real counter without importing Ray/GPU services on CPU CI.
    path = Path("third_party/verl/verl/utils/flops_counter.py")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module in {"verl.utils.device", "transformers"})]
    namespace = dict(__name__="flops_contract_test", get_torch_device=lambda: torch.cpu,
                     PretrainedConfig=SimpleNamespace)
    exec(compile(tree, str(path), "exec"), namespace)
    monkeypatch.setitem(sys.modules, "verl.utils.flops_counter", SimpleNamespace(**namespace))

    from archlab.training.metrics import register_flops

    register_flops()
    config = native_config()
    config.model_type, config.adapter, config.trainable_mode = "archlab_limite", {"variant": "normal"}, "full"
    monkeypatch.setenv("VERL_MFU_DEVICE_NAME", "B300")
    achieved, peak = namespace["FlopsCounter"](config).estimate_flops([17, 40], 2.)
    assert achieved == estimate_limite_flops(config, 57, [17, 40], 2.)
    assert peak == 2250


def test_upstream_metric_extension_routes_to_existing_logger_and_rejects_overwrite():
    import importlib
    import math

    path = Path("third_party/verl/verl/trainer/ppo/v1/trainer_base.py")
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_metrics")
    offset = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "extension" for t in n.targets))
    nodes = ast.Module(body=method.body[offset:offset+2], type_ignores=[])
    inputs = metric_inputs()
    inputs["config"].trainer.metrics_extension = "archlab.training.metrics.step_metrics"
    namespace = dict(self=SimpleNamespace(config=inputs["config"]), metrics={}, importlib=importlib, math=math,
                     metrics_batch=inputs["batch"], rollout_tags=[*inputs["tags"], {"is_padding": True}],
                     non_padding_mask=[True, True, False], timing_raw=inputs["timing"], global_steps=3)
    code = compile(nodes, str(path), "exec")
    exec(code, namespace)
    assert namespace["metrics"]["rollout/consumed_groups"] == 1
    assert namespace["metrics"]["perf/allocated_gpu_seconds/step"] == 80
    with pytest.raises(ValueError, match="overwrite"):
        exec(code, namespace)

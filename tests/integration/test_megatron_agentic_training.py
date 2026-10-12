"""Real native Limite/Megatron learner checks; run with torchrun on free GPUs."""
from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.slow


def _deployment_config():
    path = os.getenv("ARCHLAB_QUALIFICATION_CONFIG")
    if not path:
        return None
    from omegaconf import OmegaConf

    config = OmegaConf.load(path)
    assert dist.get_world_size() == config.trainer.nnodes * config.trainer.n_gpus_per_node
    return config



def _difference(actual, expected):
    actual, expected = actual.detach().float(), expected.detach().float()
    delta = (actual - expected).abs()
    changed = delta[delta != 0]
    return dict(maximum=float(delta.max()), reference_maximum=float(expected.abs().max()),
                relative_l2=float(delta.norm() / expected.norm().clamp_min(1e-30)),
                p99_changed=float(changed.quantile(.99)) if changed.numel() else 0.,
                mismatched=changed.numel(), count=delta.numel())


def _state_fingerprint(model, optimizer):
    """Hash actual restored master weights and Adam moments, not only metadata."""
    def digest(value):
        if isinstance(value, torch.Tensor):
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            return [str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()]
        if isinstance(value, dict):
            return {str(key): digest(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [digest(item) for item in value]
        return value

    groups = []
    for component in optimizer.chained_optimizers:
        for group in component.optimizer.param_groups:
            groups.append(dict(
                # This grouping label is not an optimizer hyperparameter; MCore
                # reload reconstructs it, while all numerical group fields matter.
                config={key: value for key, value in group.items()
                        if key not in {"params", "_storage_dtype_partition"}
                        # FusedAdam skips empty local groups before reading step.
                        # MCore broadcasts the populated rank's group metadata.
                        and not (key == "step" and not group["params"])},
                parameters=[dict(master=parameter, state=component.optimizer.state[parameter])
                            for parameter in group["params"]],
            ))
    return digest(dict(model=model.state_dict(), optimizer=groups))



def _assert_restored_state(model, optimizer, expected, label):
    actual = _state_fingerprint(model, optimizer)
    differences = []

    def compare(left, right, path):
        if type(left) is not type(right):
            differences.append(dict(path=path, actual=left, expected=right))
        elif isinstance(left, dict):
            for key in left.keys() | right.keys():
                compare(left.get(key), right.get(key), f"{path}/{key}")
        elif isinstance(left, list):
            if len(left) != len(right):
                differences.append(dict(path=path, actual_length=len(left), expected_length=len(right)))
            else:
                for index, (item, target) in enumerate(zip(left, right, strict=True)):
                    compare(item, target, f"{path}/{index}")
        elif left != right:
            differences.append(dict(path=path, actual=left, expected=right))

    compare(actual, expected, "")
    target = Path(os.environ["ARCHLAB_MEGATRON_WORK"]) / f"{label}-state-differences-{dist.get_rank()}.json"
    target.write_text(json.dumps(differences, indent=2))
    assert not differences, differences[:20]


def _optimizer_group_audit(optimizer):
    """Record actual local shard ownership and the installed optimizer backend."""
    from archlab.artifacts import sha256_file

    groups = []
    for component_index, component in enumerate(optimizer.chained_optimizers):
        backend = type(component.optimizer)
        source = inspect.getsourcefile(backend)
        for group_index, group in enumerate(component.optimizer.param_groups):
            groups.append(dict(component=component_index, group=group_index,
                local_parameters=len(group["params"]),
                local_elements=sum(parameter.numel() for parameter in group["params"]),
                step=group.get("step"), betas=list(group["betas"]),
                backend=backend.__module__ + "." + backend.__qualname__,
                backend_source=source,
                backend_source_sha256=sha256_file(source) if source else None))
    return groups


# Fixed before GPU validation: this tests engineering numerical equivalence,
# not bit identity through the native BF16/atomic backward.
GRADIENT_GLOBAL_REL_L2 = .01
GRADIENT_BLOCK_REL_L2 = .02
GRADIENT_COSINE = .999
WEIGHT_GLOBAL_REL_L2 = 1e-5
WEIGHT_BLOCK_REL_L2 = 1e-4
UPDATE_COSINE = .999
UPDATE_NORM_RATIO = (.98, 1.02)


def _block(name):
    import re

    match = re.search(r"(?:base\.layers|layers|adapters)\.\d+", name)
    return match.group(0) if match else "embedding_and_head"


def _backbone_inputs(wrapper, ids):
    if not wrapper.config.conversion:
        return {}
    mask, positions = wrapper.conversion_inputs(ids, None, None)
    return dict(attention_mask=mask, position_ids=positions)


def _pair_metrics(pairs):
    """Global and architectural-block vector metrics, accumulated in FP64."""
    totals = {}
    exact = True
    for name, actual, reference in pairs:
        assert actual.shape == reference.shape, name
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all(), name
        exact = exact and torch.equal(actual, reference)
        a, b = actual.detach().float(), reference.detach().float()
        values = [float(torch.linalg.vector_norm(x, dtype=torch.float64).square())
                  for x in (a, b, a - b)]
        values.append(float((a * b).sum(dtype=torch.float64)))
        for key in ("global", _block(name)):
            row = totals.setdefault(key, [0., 0., 0., 0.])
            for index, value in enumerate(values):
                row[index] += value
    metrics = {}
    for key, (a2, b2, error2, dot) in totals.items():
        metrics[key] = dict(
            actual_l2=a2 ** .5, reference_l2=b2 ** .5, difference_l2=error2 ** .5,
            relative_l2=(error2 / max(b2, 1e-300)) ** .5,
            cosine=max(-1., min(1., dot / (a2 * b2) ** .5)) if a2 and b2 else float(a2 == b2),
            norm_ratio=(a2 / b2) ** .5 if b2 else float(a2 == b2),
        )
    return dict(exact=exact, vectors=metrics)


def _gradient_failures(report):
    errors = []
    for block, row in report["vectors"].items():
        limit = GRADIENT_GLOBAL_REL_L2 if block == "global" else GRADIENT_BLOCK_REL_L2
        if row["relative_l2"] > limit or row["cosine"] < GRADIENT_COSINE:
            errors.append(block)
    return errors


def _update_metrics(actual, expected, baseline):
    assert actual.keys() == expected.keys() == baseline.keys()
    return dict(
        weights=_pair_metrics((name, actual[name], expected[name]) for name in expected),
        step=_pair_metrics((name, actual[name].float() - baseline[name].float(),
                            expected[name].float() - baseline[name].float()) for name in expected),
    )


def _update_failures(report):
    failures = []
    for block, row in report["weights"]["vectors"].items():
        limit = WEIGHT_GLOBAL_REL_L2 if block == "global" else WEIGHT_BLOCK_REL_L2
        if row["relative_l2"] > limit:
            failures.append("weights/" + block)
    step = report["step"]["vectors"]["global"]
    if step["reference_l2"] == 0 or step["cosine"] < UPDATE_COSINE:
        failures.append("step/direction")
    if not UPDATE_NORM_RATIO[0] <= step["norm_ratio"] <= UPDATE_NORM_RATIO[1]:
        failures.append("step/magnitude")
    return failures


def _snapshot(model):
    # Independent on-device copies survive the engine's storage offload. Eight
    # full CPU replicas and comparison temporaries would exhaust host RAM.
    # named_parameters also counts tied embeddings once.
    return {name: value.detach().clone() for name, value in model.named_parameters()}


def _rng_fingerprint(engine):
    import numpy as np

    def normalized(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): normalized(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalized(item) for item in value]
        return value

    state = engine.checkpoint_mananager.get_rng_state(use_dist_ckpt=False)
    return hashlib.sha256(json.dumps(normalized(state), sort_keys=True).encode()).hexdigest()


def _native_gradient_oracle(model, original, backward_native, *, atomic, root, name, native_factory=None):
    references = dict(original.named_parameters())
    assert dict(model.policy.named_parameters()).keys() == references.keys()
    targets = {}
    pairs = []
    for key, parameter in model.policy.named_parameters():
        reference = references[key].grad
        target = torch.zeros_like(parameter.main_grad) if reference is None else reference.detach().clone()
        targets[key] = target
        pairs.append((key, parameter.main_grad, target))
    bridge = _pair_metrics(pairs)
    controls = []
    if atomic:
        # Bounded calibration, evaluated with the SAME predeclared contract.
        # Neither extrema selection nor repeated attempts can change these bounds.
        original.zero_grad(set_to_none=True)
        backward_native(original)
        controls.append(_pair_metrics(
            (key, parameter.grad if parameter.grad is not None else torch.zeros_like(targets[key]), targets[key])
            for key, parameter in original.named_parameters()))
        assert native_factory is not None
        independent = native_factory()
        parameters = dict(independent.named_parameters())
        assert parameters.keys() == references.keys()
        for key, parameter in parameters.items():
            torch.testing.assert_close(parameter, references[key], rtol=0, atol=0)
        backward_native(independent)
        controls.append(_pair_metrics(
            (key, parameter.grad if parameter.grad is not None else torch.zeros_like(targets[key]), targets[key])
            for key, parameter in independent.named_parameters()))
    report = dict(contract="native-equivalence-v2", compared_gradient_tensors=len(targets),
                  bridge=bridge, unchanged_native_controls=controls,
                  exact_forward_and_weights=True,
                  scope="engineering numerical equivalence; native BF16/atomic backward is not bit deterministic")
    (root / f"{name}-gradient-differences-{dist.get_rank()}.json").write_text(json.dumps(report, indent=2))
    if atomic:
        assert all(not _gradient_failures(control) for control in controls), "native control exceeds fixed contract"
        assert not _gradient_failures(bridge), _gradient_failures(bridge)
    else:
        assert bridge["exact"], "deterministic tiny gradient must be bit-exact"
    return report


def test_native_forward_backward():
    """DP-only package learner; deliberately not a complete agentic RL admission."""
    if os.getenv("ARCHLAB_RUN_DISTRIBUTED") != "1":
        pytest.skip("set ARCHLAB_RUN_DISTRIBUTED=1 under torchrun on free GPUs")
    root = Path(os.environ["ARCHLAB_MEGATRON_WORK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))

    import_module("archlab.training.megatron")
    from megatron.bridge import AutoBridge
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallelConfig
    from test_agentic_training import _tiny_policy

    from archlab.training.limite import LimiteForCausalLM, export_model

    snapshot = Path(os.getenv("ARCHLAB_LIMITE_PUBLISHER", "/mnt/oss/models/limite-1b-base-cc612bafcd4a"))
    deployment = _deployment_config()
    source = (deployment.actor_rollout_ref.model.path if deployment is not None
              else os.getenv("ARCHLAB_LIMITE_EXPORT"))
    export = Path(source) if source else root / "tiny-export"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if not source:
            if not export.exists():
                export_model(_tiny_policy(snapshot), snapshot, export)
    dist.barrier()
    bridge = AutoBridge.from_hf_pretrained(export, trust_remote_code=False)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.apply_overrides_and_finalize(dtype=torch.bfloat16)
    sharded_optimizer = os.getenv("ARCHLAB_MEGATRON_SHARDED_OPTIMIZER", "1") == "1"
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True, overlap_grad_reduce=False,
        use_distributed_optimizer=sharded_optimizer,
    )
    models = provider.provide_distributed_model(ddp_config=ddp_config)
    assert len(models) == 1
    model = models[0]
    bridge.load_hf_weights(models)
    native = model.module
    original = LimiteForCausalLM.from_pretrained(export).to(device="cuda").train()
    original.gradient_checkpointing_enable({"use_reentrant": False})
    dtypes = {}
    for name, parameter in native.policy.named_parameters():
        expected = dict(original.named_parameters())[name]
        assert parameter.dtype == expected.dtype
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0, msg=lambda detail, name=name: name + "\n" + detail)
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
    assert "torch.float32" in dtypes and "torch.bfloat16" in dtypes
    model.train()
    ids = torch.tensor([[1, 4 + rank, 8, 5, 6, 10, 13, 3, 9, 7]], device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(model(ids), original(ids, use_cache=False).logits,
                                   rtol=0, atol=0)

    def token_loss(logits, tokens):
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))

    model.zero_grad_buffer()
    token_loss(model(ids), ids).backward()
    token_loss(original(ids, use_cache=False).logits, ids).backward()
    def native_factory(reference=original):
        control = LimiteForCausalLM.from_pretrained(export).to(device="cuda").train()
        control.gradient_checkpointing_enable({"use_reentrant": False})
        with torch.no_grad():
            torch.testing.assert_close(control(ids, use_cache=False).logits,
                                       reference(ids, use_cache=False).logits, rtol=0, atol=0)
        return control

    native_report = _native_gradient_oracle(
        native, original, lambda reference: token_loss(reference(ids, use_cache=False).logits, ids).backward(),
        atomic=bool(source), root=root, name="native", native_factory=native_factory,
    )
    if native.policy.config.conversion:
        assert {f"layers.{index}" for index in range(native.policy.config.num_hidden_layers)} <= set(
            native_report["bridge"]["vectors"]), native_report
    # Exercise the actual verl BSHD dispatch and its unchanged vocabulary-parallel
    # scoring function, including entropy and backward, before optimizer admission.
    from verl.models.mcore import get_mcore_engine_forward_fn
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    processor = partial(
        MegatronEngineWithLMHead._lm_head_logits_processor,
        SimpleNamespace(engine_config=SimpleNamespace(entropy_from_logits_with_chunking=False)),
        calculate_sum_pi_squared=False, calculate_entropy=True,
        distillation_use_topk=False, distillation_only=False,
        logits_processor_func=None, batch=None, data_format="bshd",
    )
    model.zero_grad_buffer()
    original.zero_grad(set_to_none=True)
    nested_ids = torch.nested.nested_tensor([ids[0]], layout=torch.jagged)
    temperatures = torch.full_like(ids, .7, dtype=torch.float32)
    scores = get_mcore_engine_forward_fn(native.policy.config)(
        model, nested_ids, {}, logits_processor=processor,
        logits_processor_args={"label": nested_ids,
                               "temperature": torch.nested.nested_tensor([temperatures[0]], layout=torch.jagged)},
        data_format="bshd",
    )
    from archlab.architectures.limite_head import chunked_logits_processor

    # Use the same bounded-head contract on the standalone native model. A
    # checkpointed tied BF16 head can change addition order; its mathematical
    # gradient oracle is tested separately from the engine boundary.
    reference_hidden = original.policy.model(input_ids=ids, use_cache=False,
                                             **_backbone_inputs(original, ids),
                                             return_dict=True).last_hidden_state
    reference_scores = chunked_logits_processor(
        original.policy._softcapped_logits, reference_hidden, processor,
        {"label": ids.roll(-1, dims=1), "temperature": temperatures.clone()},
    )
    for name in ("log_probs", "entropy"):
        torch.testing.assert_close(scores[name].values(), reference_scores[name].reshape(-1),
                                   rtol=0, atol=0)
    (-scores["log_probs"].values().mean() - .01 * scores["entropy"].values().mean()).backward()
    (-reference_scores["log_probs"].mean() - .01 * reference_scores["entropy"].mean()).backward()
    def native_head_backward(reference):
        hidden = reference.policy.model(input_ids=ids, use_cache=False, return_dict=True,
                                        **_backbone_inputs(reference, ids)).last_hidden_state
        result = chunked_logits_processor(reference.policy._softcapped_logits, hidden, processor,
                                         {"label": ids.roll(-1, dims=1), "temperature": temperatures.clone()})
        (-result["log_probs"].mean() - .01 * result["entropy"].mean()).backward()

    head_report = _native_gradient_oracle(native, original, native_head_backward,
                                          atomic=bool(source), root=root, name="head",
                                          native_factory=native_factory)
    del original, scores, reference_scores, native_head_backward, native_factory
    # A real HF export is also required by the rollout weight refresh.
    exported = dict(bridge.export_hf_weights(models))
    for name, parameter in native.policy.named_parameters():
        exported_name = ("policy.lm_head.weight" if name in {
            "policy.model.base.embed_tokens.weight", "policy.model.embed_tokens.weight"} else name)
        torch.testing.assert_close(exported[exported_name].to(parameter.device), parameter, rtol=0, atol=0)
        replica = parameter.detach().clone()
        dist.broadcast(replica, src=0)
        torch.testing.assert_close(replica, parameter, rtol=0, atol=0)
    report = dict(world_size=dist.get_world_size(), rank=rank, parameter_dtypes=dtypes,
                  native_gradient_comparison=native_report,
                  bounded_head_gradient_comparison=head_report,
                  exact_native_forward=True, exact_hf_export=True,
                  package_logprobs_entropy_backward=True,
                  source=str(export), scope="native model numerical parity; package restart is a separate gate")
    if deployment is not None:
        from archlab.training.config import fingerprint

        report["config_sha256"] = fingerprint(deployment)
    (root / f"rank-{rank}.json").write_text(json.dumps(report, indent=2))
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


def deployment_engine_arguments(deployment):
    """The same resolved engine contract used by ActorRolloutRefWorker."""
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import McoreEngineConfig

    actor = deployment.actor_rollout_ref.actor
    assert actor.ppo_micro_batch_size_per_gpu == 1
    actor_config = omega_conf_to_dataclass(actor)
    model_config = omega_conf_to_dataclass(deployment.actor_rollout_ref.model)
    engine_config = actor_config.engine
    assert isinstance(engine_config, McoreEngineConfig)
    rollout = deployment.actor_rollout_ref.rollout
    engine_config.use_dynamic_bsz = actor.use_dynamic_bsz
    engine_config.max_token_len_per_gpu = actor.ppo_max_token_len_per_gpu
    engine_config.micro_batch_size_per_gpu = actor.ppo_micro_batch_size_per_gpu
    engine_config.infer_max_token_len_per_gpu = rollout.log_prob_max_token_len_per_gpu
    engine_config.infer_micro_batch_size_per_gpu = rollout.log_prob_micro_batch_size_per_gpu
    engine_config.use_remove_padding = model_config.use_remove_padding
    engine_config.use_fused_kernels = model_config.use_fused_kernels
    optimizer_config = actor_config.optim
    optimizer_config.total_training_steps = deployment.trainer.total_training_steps
    return [model_config, engine_config, optimizer_config, actor_config.checkpoint]


def test_package_engine_optimizer_scheduler_checkpoint_restart():
    """Run the upstream engine API, schedule and checkpoint manager together."""
    if os.getenv("ARCHLAB_RUN_ENGINE") != "1":
        pytest.skip("set ARCHLAB_RUN_ENGINE=1 under torchrun on free GPUs")
    from copy import deepcopy

    root = Path(os.environ["ARCHLAB_MEGATRON_WORK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    import_module("archlab.training.megatron")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as mcore_random
    from tensordict import TensorDict
    from test_agentic_training import _tiny_policy
    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from archlab.training.limite import export_model

    snapshot = Path(os.getenv("ARCHLAB_LIMITE_PUBLISHER", "/mnt/oss/models/limite-1b-base-cc612bafcd4a"))
    deployment = _deployment_config()
    source = (deployment.actor_rollout_ref.model.path if deployment is not None
              else os.getenv("ARCHLAB_LIMITE_EXPORT"))
    export = Path(source) if source else root / "tiny-export"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if not source and not export.exists():
            export_model(_tiny_policy(snapshot), snapshot, export)
    dist.barrier()
    original_checkpoint_backward = mcore_random.CheckpointFunction.backward

    def build():
        arguments = [
            HFModelConfig(path=str(export), external_lib="archlab.training.megatron",
                          use_remove_padding=False, override_config={"attn_implementation": "sdpa"}),
            McoreEngineConfig(sequence_parallel=False, use_remove_padding=False,
                              pad_bshd_to_minibatch_max=False, patch_recomputation_backward=False,
                              param_offload=False, grad_offload=False, optimizer_offload=False,
                              override_ddp_config={"grad_reduce_in_fp32": True,
                                                   "overlap_grad_reduce": False}),
            McoreOptimizerConfig(lr=1e-4, min_lr=1e-5, total_training_steps=4,
                                 lr_warmup_steps=0, weight_decay=0., clip_grad=0.,
                                 use_checkpoint_opt_param_scheduler=True),
            CheckpointConfig(save_contents=["model", "optimizer", "extra"],
                             load_contents=["model", "optimizer", "extra"]),
        ]
        if deployment is not None:
            arguments = deployment_engine_arguments(deployment)
        learner = MegatronEngineWithLMHead(*arguments)
        learner.initialize()
        assert mcore_random.CheckpointFunction.backward is original_checkpoint_backward
        if deployment is not None:
            overrides = deployment.actor_rollout_ref.actor.optim.override_optimizer_config
            expected_betas = (overrides.adam_beta1, overrides.adam_beta2)
            for component in learner.optimizer.chained_optimizers:
                assert all(tuple(group["betas"]) == expected_betas
                           for group in component.optimizer.param_groups)
        return learner

    engine = build()
    ids = torch.tensor([1, 4 + rank, 8, 5, 6, 10, 13, 3, 9, 7], device="cuda")

    def loss_function(model_output, data, dp_group):
        mask = data["loss_mask"].values().float()
        loss = -(model_output["log_probs"].values() * mask).sum() / mask.sum()
        return loss, {"token_loss": float(loss.detach())}

    def step(tokens):
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[-1] = False
        batch = TensorDict({
            "input_ids": torch.nested.nested_tensor([tokens], layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor([mask], layout=torch.jagged),
            "temperature": torch.tensor([.7], device="cuda"),
        }, batch_size=[1])
        tu.assign_non_tensor(batch, use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                             calculate_entropy=True)
        with engine.train_mode():
            result = engine.train_batch(batch, loss_function)
            engine.lr_scheduler_step()
        return result

    # Stock offload releases CUDA parameter storage outside train_mode. All
    # numerical observations must materialize the engine through its public API.
    first = step(ids)
    checkpoint = root / "engine-checkpoint"
    saved_rng = _rng_fingerprint(engine)
    engine.save_checkpoint(str(checkpoint), global_step=1)
    assert _rng_fingerprint(engine) == saved_rng
    with engine.train_mode():
        saved_fingerprint = _state_fingerprint(engine.module[0].module, engine.optimizer)
        baseline = _snapshot(engine.module[0].module)
    saved_scheduler = deepcopy(engine.lr_scheduler.state_dict())
    saved_groups = _optimizer_group_audit(engine.optimizer)
    second = step(ids.roll(1))
    with engine.train_mode():
        expected = _snapshot(engine.module[0].module)
    expected_scheduler = deepcopy(engine.lr_scheduler.state_dict())
    # Control: repeat from the same checkpoint in the original engine before
    # reconstructing anything. This measures native arithmetic variation.
    # MCore scheduler loading increments num_steps and assumes a fresh scheduler.
    # Reset only this in-place control to the fresh-engine baseline before load.
    engine.lr_scheduler.num_steps = 0
    engine.load_checkpoint(str(checkpoint), del_local_after_load=False)
    (root / f"control-optimizer-groups-{rank}.json").write_text(json.dumps(dict(
        saved=saved_groups, restored=_optimizer_group_audit(engine.optimizer)), indent=2))
    with engine.train_mode():
        _assert_restored_state(engine.module[0].module, engine.optimizer, saved_fingerprint, "control")
    assert engine.lr_scheduler.state_dict() == saved_scheduler
    assert _rng_fingerprint(engine) == saved_rng
    control_result = step(ids.roll(1))
    assert control_result["loss"] == second["loss"]
    with engine.train_mode():
        control_update = _update_metrics(_snapshot(engine.module[0].module), expected, baseline)
    del engine
    gc.collect()
    engine = build()
    engine.load_checkpoint(str(checkpoint), del_local_after_load=False)
    (root / f"fresh-optimizer-groups-{rank}.json").write_text(json.dumps(dict(
        saved=saved_groups, restored=_optimizer_group_audit(engine.optimizer)), indent=2))
    with engine.train_mode():
        _assert_restored_state(engine.module[0].module, engine.optimizer, saved_fingerprint, "fresh")
    assert engine.lr_scheduler.state_dict() == saved_scheduler
    assert _rng_fingerprint(engine) == saved_rng
    (root / f"restored-state-{rank}.json").write_text(json.dumps(dict(
        exact_model_master_moments_scheduler=True, source=str(export),
        unchanged_engine_repeat=control_update), indent=2))
    restarted = step(ids.roll(1))
    assert restarted["loss"] == second["loss"]
    assert engine.lr_scheduler.state_dict() == expected_scheduler
    with engine.train_mode():
        update = _update_metrics(_snapshot(engine.module[0].module), expected, baseline)
    numerical = dict(contract="native-equivalence-v2", fresh_update=update,
                     unchanged_original_engine_control=control_update,
                     fresh_failures=_update_failures(update),
                     native_control_failures=_update_failures(control_update))
    (root / f"checkpoint-numerical-contract-{rank}.json").write_text(json.dumps(numerical, indent=2))
    if source:
        assert not numerical["native_control_failures"], numerical["native_control_failures"]
        assert not numerical["fresh_failures"], numerical["fresh_failures"]
    else:
        assert update["weights"]["exact"] and control_update["weights"]["exact"]
    report = dict(world_size=dist.get_world_size(), rank=rank, source=str(export),
                  first_loss=first["loss"], second_loss=second["loss"], restart_loss=restarted["loss"],
                  updated_hf_checkpoint=str(checkpoint / "model" / "huggingface"),
                  exact_package_next_update=update["weights"]["exact"], exact_scheduler=True,
                  exact_restored_model_master_moments=True,
                  next_update_comparison=update,
                  exact_rng=True,
                  numerical_contract="native-equivalence-v2",
                  optimizer_betas=sorted({tuple(group["betas"])
                      for component in engine.optimizer.chained_optimizers
                      for group in component.optimizer.param_groups}),
                  unchanged_engine_repeat=control_update,
                  runtime_checkpoint_function_unchanged=True,
                  scope="stock verl learner; Ray rollout/tool/native-context gates separate")
    if deployment is not None:
        from archlab.training.config import fingerprint

        report["config_sha256"] = fingerprint(deployment)
    (root / f"engine-rank-{rank}.json").write_text(json.dumps(report, indent=2))
    dist.barrier()
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

"""Actual native HF/FSDP2 model checks on the declared deployment GPUs."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import statistics
import time
from contextlib import ExitStack
from dataclasses import asdict
from datetime import timedelta
from functools import partial
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.slow


def _local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def _fingerprint(value):
    """Hash every local master/moment and numerical optimizer setting."""
    import numpy as np

    if isinstance(value, torch.Tensor):
        tensor = _local(value).detach().cpu().contiguous()
        digest = hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy()).hexdigest()
        return [str(tensor.dtype), list(tensor.shape), digest]
    if isinstance(value, np.ndarray):
        return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {str(key): _fingerprint(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_fingerprint(item) for item in value]
    return value


def _state(engine):
    return _fingerprint(dict(
        model={name: value for name, value in engine.module.named_parameters()},
        optimizer=engine.optimizer.state_dict(), scheduler=engine.lr_scheduler.state_dict(),
        rng=engine.checkpoint_manager.get_rng_state(),
    ))


def _assert_native_weights(engine, root):
    """Compare every actual FSDP shard with the original safetensors slice."""
    from safetensors import safe_open

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    parameters = dict(engine.module.named_parameters())
    assert parameters.keys() == index.keys(), "missing, unexpected or renamed native weights"
    checked = 0
    with ExitStack() as stack:
        readers = {name: stack.enter_context(safe_open(root / name, framework="pt", device="cpu"))
                   for name in set(index.values())}
        for name, parameter in parameters.items():
            native = readers[index[name]].get_slice(name)
            actual = _local(parameter).detach().cpu()
            if hasattr(parameter, "placements"):
                assert len(parameter.placements) == 1
                placement = parameter.placements[0]
                if placement.is_shard():
                    assert placement.dim == 0, "qualification must account for this shard layout"
                    rows = math.ceil(native.get_shape()[0] / dist.get_world_size())
                    begin = dist.get_rank() * rows
                    expected = native[begin:min(begin + rows, native.get_shape()[0])]
                else:
                    assert placement.is_replicate()
                    expected = native[:]
            else:
                expected = native[:]
            assert actual.dtype == torch.float32, "full training requires FP32 master parameters"
            torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0,
                                       msg=lambda detail, name=name: name + "\n" + detail)
            checked += actual.numel()
    return dict(tensors=len(parameters), local_master_elements=checked,
                every_publisher_weight_verified=True)


def _context_lengths(value, native_context):
    lengths = [int(item) for item in value.split(",")]
    assert lengths == sorted(set(lengths)) and lengths[0] == 64
    assert lengths[-1] <= native_context
    return lengths


def _training_batch(ids, engine_config, pad_token_id, temperature):
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    assert ids.ndim == 2 and ids.shape[0] == 1
    mask = torch.ones_like(ids)
    mask[:, 0] = 0  # No loss on the circular last-token -> BOS label.

    def nested(value):
        return torch.nested.as_nested_tensor(list(value.unbind()), layout=torch.jagged)

    batch = TensorDict({
        "input_ids": nested(ids), "loss_mask": nested(mask),
        "position_ids": nested(torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)),
        "temperature": torch.tensor([temperature], device=ids.device),
    }, batch_size=[1])
    tu.assign_non_tensor(
        batch, pad_mode=DatasetPadMode.NO_PADDING, pad_token_id=pad_token_id,
        use_remove_padding=engine_config.use_remove_padding,
        use_fused_kernels=engine_config.use_fused_kernels,
        use_dynamic_bsz=engine_config.use_dynamic_bsz,
        micro_batch_size_per_gpu=engine_config.micro_batch_size_per_gpu,
        max_token_len_per_gpu=engine_config.max_token_len_per_gpu,
    )
    return batch


def _training_step(engine, actor, ids, temperature):
    """Measure the actual stock logits/log-probability/backward/Adam path."""
    from verl.workers.utils.losses import sft_loss

    batch = _training_batch(ids, actor.engine, engine.model_config.tokenizer.pad_token_id, temperature)
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with engine.train_mode():
        engine.optimizer_zero_grad()
        output = engine.forward_backward_batch(batch, loss_function=partial(sft_loss, config=actor))
        torch.cuda.synchronize()
        forward_backward_seconds = time.perf_counter() - started
        loss = sum(output["loss"])
        assert math.isfinite(loss)
        text_parameters = {name: p for name, p in engine.module.named_parameters()
                           if name.startswith("model.language_model.") or name == "lm_head.weight"}
        text_grads = [name for name, p in text_parameters.items() if p.grad is not None]
        assert set(text_grads) == text_parameters.keys(), "a native text parameter has no gradient"
        assert len({name.split(".layers.")[1].split(".")[0] for name in text_grads
                    if ".layers." in name}) == 52
        gradient = float(engine.optimizer_step())
        assert math.isfinite(gradient) and gradient > 0
        engine.lr_scheduler_step()
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - started
    assert len(engine.optimizer.state) == len(text_parameters), "missing or unexpected Adam parameter state"
    assert all(state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
               for state in engine.optimizer.state.values())
    # Max across ranks is the synchronous batch wall time, never summed GPU time.
    measured = torch.tensor([forward_backward_seconds, step_seconds,
                             torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                            dtype=torch.float64, device="cuda")
    dist.all_reduce(measured, op=dist.ReduceOp.MAX)
    forward_backward_seconds, step_seconds, allocated, reserved = measured.tolist()
    return dict(sequence_length=ids.shape[1], loss=loss, grad_norm=gradient,
                text_gradient_tensors=len(text_grads), forward_backward_seconds=forward_backward_seconds,
                step_seconds=step_seconds, input_tokens_per_second=ids.numel() * dist.get_world_size() / step_seconds,
                peak_allocated_bytes=int(allocated), peak_reserved_bytes=int(reserved))


@pytest.mark.parametrize("value", ["", "32", "64,64", "64,32", "64,131073"])
def test_context_ladder_rejects_invalid_lengths(value):
    with pytest.raises((ValueError, AssertionError)):
        _context_lengths(value, 131072)


def test_context_ladder_keeps_native_bound():
    assert _context_lengths("64", 131072) == [64]
    assert _context_lengths("64,8192,32768,131072", 131072)[-1] == 131072


def test_stock_loss_mask_excludes_circular_bos_target():
    from types import SimpleNamespace

    from verl.utils import tensordict_utils as tu
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from verl.workers.engine.utils import prepare_micro_batches
    from verl.workers.utils.losses import sft_loss

    settings = SimpleNamespace(use_remove_padding=False, use_fused_kernels=False,
                               use_dynamic_bsz=False, micro_batch_size_per_gpu=1, max_token_len_per_gpu=131072)
    ids = torch.tensor([[200000, 100, 101, 200008]])
    batch = _training_batch(ids, settings, 200018, 1.0)
    micro_batches, _ = prepare_micro_batches(batch)
    engine = FSDPEngineWithLMHead.__new__(FSDPEngineWithLMHead)
    model_inputs, _ = engine.prepare_model_inputs(micro_batches[0])
    torch.testing.assert_close(model_inputs["input_ids"], ids)
    assert model_inputs["attention_mask"].all(), "BOS stays visible even though its target is masked"
    tu.assign_non_tensor(batch, dp_size=1, batch_num_tokens=3)
    log_probs = torch.tensor([-1.0, -2.0, -3.0, 10000.0], requires_grad=True)
    nested = torch.nested.nested_tensor_from_jagged(log_probs, batch["input_ids"].offsets())
    loss, _ = sft_loss(None, {"log_probs": nested}, batch)
    torch.testing.assert_close(loss, torch.tensor(2.0))
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.tensor([-1 / 3, -1 / 3, -1 / 3, 0.0]))


def test_package_fsdp_training_restart():
    """Construct the actual model, update it and prove complete stock resume."""
    if os.getenv("ARCHLAB_RUN_ENGINE") != "1":
        pytest.skip("requires the exact deployment under torchrun on free GPUs")
    from omegaconf import OmegaConf
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

    config = OmegaConf.load(os.environ["ARCHLAB_QUALIFICATION_CONFIG"])
    assert config.model_engine == "dp" and config.actor_rollout_ref.actor.strategy == "fsdp2"
    assert config.trainer.nnodes == 1
    assert config.actor_rollout_ref.rollout.max_model_len == 131072, "do not qualify a shortened native context"
    lengths = _context_lengths(os.getenv("ARCHLAB_CONTEXT_LENGTHS", "64"),
                               config.actor_rollout_ref.rollout.max_model_len)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank), timeout=timedelta(minutes=30))
    assert dist.get_world_size() == config.trainer.n_gpus_per_node == 8
    output = Path(os.environ["ARCHLAB_LEARNER_WORK"])
    output.mkdir(parents=True, exist_ok=True)
    # Match the stock trainer's resolved dataclasses, including its real CPU
    # offload transitions and scheduler; do not qualify a cheaper substitute.
    config.actor_rollout_ref.actor.optim.total_training_steps = config.trainer.total_training_steps
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    model = config.actor_rollout_ref.model
    actor.engine.use_dynamic_bsz = actor.use_dynamic_bsz
    actor.engine.max_token_len_per_gpu = actor.ppo_max_token_len_per_gpu
    actor.engine.micro_batch_size_per_gpu = actor.ppo_micro_batch_size_per_gpu
    actor.engine.use_remove_padding = model.use_remove_padding
    assert not actor.engine.use_dynamic_bsz and actor.engine.micro_batch_size_per_gpu == 1
    evidence = dict(status="running", requested_context_lengths=lengths, context_steps=[],
                    full_resume_exact=False, full_context_backward_verified=False,
                    param_offload=actor.engine.param_offload, optimizer_offload=actor.engine.optimizer_offload,
                    scope="synthetic stock SFT loss; actual PPO, agentic reward and serving are separate")
    receipt = output / f"native-fsdp-rank-{rank}.json"

    def record():
        receipt.write_text(json.dumps(evidence, indent=2))

    def token_ids(length):
        tokens = torch.arange(length, device="cuda").remainder(1000).add_(100).unsqueeze(0)
        tokens[0, 0], tokens[0, -1] = 200000, 200008
        return tokens

    try:
        engine = FSDPEngineWithLMHead(
            model_config=omega_conf_to_dataclass(model), engine_config=actor.engine,
            optimizer_config=actor.optim, checkpoint_config=actor.checkpoint,
        )
        engine.initialize()
        weights = _assert_native_weights(engine, Path(model.path))
        evidence["weights"] = weights
        record()
        sentinel = "model.language_model.layers.0.input_layernorm.weight"
        original_sentinel = _fingerprint(dict(engine.module.named_parameters())[sentinel])
        ids = token_ids(64)
        with engine.eval_mode(), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            distributed_logits = engine.module(input_ids=ids, attention_mask=torch.ones_like(ids),
                                               use_cache=False).logits.detach()
        if rank == 0:
            from transformers import AutoModelForImageTextToText

            reference = AutoModelForImageTextToText.from_pretrained(
                model.path, dtype=torch.bfloat16, local_files_only=True,
                attn_implementation="flash_attention_4", trust_remote_code=False).to("cuda").eval()
            with torch.no_grad():
                expected_logits = reference(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            use_cache=False).logits
            torch.testing.assert_close(distributed_logits, expected_logits, rtol=1e-3, atol=5e-3)
            del reference, expected_logits
            gc.collect()
            torch.cuda.empty_cache()
        dist.barrier()
        del distributed_logits
        evidence["context_steps"].append(_training_step(engine, actor, ids, config.actor_rollout_ref.rollout.temperature))
        record()
        engine.optimizer_zero_grad()
        expected = _state(engine)
        assert expected["model"][sentinel] != original_sentinel, "the native optimizer did not change text weights"
        checkpoint = output / "engine-checkpoint"
        engine.save_checkpoint(str(checkpoint), global_step=1)
        # Deliberately damage each component: a successful load must repair it.
        with torch.no_grad():
            _local(next(engine.module.parameters())).add_(.125)
            for state in engine.optimizer.state.values():
                _local(state["exp_avg"]).add_(.125)
        engine.lr_scheduler_step()
        torch.rand(128, device="cuda")
        engine.load_checkpoint(str(checkpoint), del_local_after_load=False)
        assert _state(engine) == expected, "model, Adam, scheduler or RNG resume changed"
        evidence["full_resume_exact"] = True
        record()
        for length in lengths[1:]:
            evidence["active_context_length"] = length
            record()
            evidence["context_steps"].append(_training_step(
                engine, actor, token_ids(length), config.actor_rollout_ref.rollout.temperature))
            record()
        evidence["full_context_backward_verified"] = lengths[-1] == config.actor_rollout_ref.rollout.max_model_len
        if os.getenv("ARCHLAB_RUN_THROUGHPUT") == "1":
            from archlab.artifacts import atomic_write_json, sha256_file
            from archlab.training.performance import ThroughputContract

            experiment = json.loads(Path(os.environ["ARCHLAB_QUALIFICATION_EXPERIMENT"]).read_text())
            specification = ThroughputContract(**experiment["throughput"])
            specification.validate()
            assert specification.sequence_length <= config.actor_rollout_ref.rollout.max_model_len
            evidence["throughput"] = dict(sequence_length=specification.sequence_length, warmups=[], samples=[])
            for index in range(specification.warmup + specification.samples):
                measurement = _training_step(engine, actor, token_ids(specification.sequence_length),
                                             config.actor_rollout_ref.rollout.temperature)
                evidence["throughput"]["warmups" if index < specification.warmup else "samples"].append(measurement)
                record()
            samples = evidence["throughput"]["samples"]
            evidence["throughput"]["median_step_seconds"] = statistics.median(row["step_seconds"] for row in samples)
            evidence["throughput"]["median_input_tokens_per_second"] = statistics.median(
                row["input_tokens_per_second"] for row in samples)
            if rank == 0:
                settings = asdict(specification)
                settings.pop("baseline")
                settings.pop("max_slowdown")
                publisher = json.loads(Path(experiment["publisher_manifest"]).read_text())
                contract = dict(
                    scope="stock FSDP learner forward/backward and Adam with gradient checks; excludes generation/tools",
                    workload="bos-arange-mod1000-plus100-eos-sft-v1", settings=settings,
                    world_size=dist.get_world_size(), publisher_assets=publisher["assets"],
                    actor=OmegaConf.to_container(config.actor_rollout_ref.actor, resolve=True),
                    temperature=config.actor_rollout_ref.rollout.temperature,
                    runtime={name: version(name) for name in ("torch", "transformers", "vllm", "flash-attn-4")},
                    cuda=torch.version.cuda, device_capability=list(torch.cuda.get_device_capability()),
                    effective_hardware="NVIDIA B300", reported_hardware=torch.cuda.get_device_name(),
                    container_identity=os.environ["ARCHLAB_CONTAINER_IDENTITY"],
                    benchmark_sha256=sha256_file(Path(__file__)),
                )
                atomic_write_json(output / "throughput.json", dict(
                    contract=contract, seconds=[row["step_seconds"] for row in samples],
                    model_tokens_per_update=dist.get_world_size() * specification.sequence_length,
                    full_context_backward_verified=evidence["full_context_backward_verified"],
                ))
        evidence.pop("active_context_length", None)
        evidence["status"] = "passed"
        record()
        dist.barrier()
    except Exception as error:
        evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
        record()
        raise
    finally:
        dist.destroy_process_group()

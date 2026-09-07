"""Bounded NeMo AutoModel integration qualification; never launches finetuning."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import socket
import subprocess
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.qwen3_8_flash_next.config import (
    Qwen3_8_FlashNextConfig, Qwen3_8_FlashNextTextConfig,
)
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
from nemo_automodel.components.moe.parallelizer import parallelize_model
from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer

from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig
from archlab.automodel.loading import (
    audit_checkpoint_keys, rebuild_nonpersistent_buffers,
    poison_weights_before_load, assert_loaded_weights_finite,
)
from archlab.automodel.simplicial import (
    ADAPTER_MARKER, UPSTREAM_COMMIT, AdditiveMoERead, install_simplicial_modules,
)
from archlab.automodel.runtime import configure_frozen_gdn_runtime


def emit(event: str, **values):
    print(json.dumps({"event": event, "rank": dist.get_rank(), "time_unix": time.time(), **values},
                     sort_keys=True), flush=True)


def tiny_config(num_experts=8):
    text = Qwen3_8_FlashNextTextConfig(
        vocab_size=1024, hidden_size=256, num_hidden_layers=8,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        moe_intermediate_size=128, shared_expert_intermediate_size=128,
        num_experts=num_experts, num_experts_per_tok=2, hc_count=4, hc_lowrank=32,
        ple_layer_ids=[], indexer_budget=32, indexer_n_heads=2, indexer_head_dim=32,
        indexer_compress_ratio=4, linear_num_key_heads=4, linear_num_value_heads=8,
        linear_key_head_dim=32, linear_value_head_dim=32, max_position_embeddings=32768,
        dtype="bfloat16", rope_parameters={"rope_type": "default", "rope_theta": 10000000.,
                                           "partial_rotary_factor": .25},
    )
    return Qwen3_8_FlashNextConfig(text_config=text, language_model_only=True)


def build_frozen_base(config, *, tiny: bool, checkpoint: Path | None, ep_size: int, activation_checkpointing: bool):
    device = torch.device("cuda", torch.cuda.current_device())
    precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                     output_dtype=torch.bfloat16)
    setup = DistributedSetup.build(strategy=FSDP2Config(mp_policy=precision),
                                   parallelism_sizes=ParallelismSizes(ep_size=ep_size),
                                   world_size=dist.get_world_size())
    mesh = setup.mesh_context
    emit("parallel_mesh", device_mesh_names=list(mesh.device_mesh.mesh_dim_names),
         device_mesh_ranks=mesh.device_mesh.mesh.tolist(),
         moe_mesh_names=list(mesh.moe_mesh.mesh_dim_names), moe_mesh_ranks=mesh.moe_mesh.mesh.tolist())
    backend = BackendConfig(attn="flex", linear="torch", rms_norm="torch_fp32",
                            experts="torch_mm", dispatcher="deepep", rope_fusion=False,
                            gate_precision="float32", enable_hf_state_dict_adapter=not tiny)
    with torch.device("cpu" if tiny else "meta"):
        model = Qwen3_8_FlashNextForConditionalGeneration(
            config, backend=backend, moe_overrides={"aux_loss_coeff": 0.0})
    if tiny:
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
        model.to(device)
    else:
        emit("checkpoint_key_audit", **audit_checkpoint_keys(model, checkpoint))
        # The ordinary initialize_weights() path also does this cast. Skipping
        # random initialization must not leave GroupedExperts' FP32 allocation
        # dtype as an accidental frozen master copy of BF16 checkpoint weights.
        cast_model_to_dtype(model, torch.bfloat16, skip_modules=("_fp32_params",))
        rebuild_nonpersistent_buffers(model, device)
    model.requires_grad_(False)
    parallelize_model(model, mesh.device_mesh, mesh.moe_mesh,
                      **mesh.parallelize_axis_kwargs(), activation_checkpointing=activation_checkpointing,
                      reshard_after_forward=True, mp_policy=precision,
                      reapply_trainability=lambda m: m.requires_grad_(False))
    if not tiny:
        # Use the upstream sharded loader, not a handwritten tensor conversion.
        model._skip_init_weights_on_load = True
        Checkpointer.initialize_model_weights(model, device)
        poison_weights_before_load(model)
        checkpointer = CheckpointingConfig(checkpoint_dir="", model_repo_id=str(checkpoint),
                                          save_consolidated=False, dequantize_base_checkpoint=False).build(
            dp_rank=dist.get_rank(), tp_rank=0, pp_rank=0, moe_mesh=mesh.moe_mesh)
        emit("base_load_begin")
        checkpointer.load_base_model(model, device, None, str(checkpoint))
        emit("base_load_end", local_elements=assert_loaded_weights_finite(model))
    model.requires_grad_(False)
    return model, mesh, precision


def checkpoint_payload(adapters, optimizer):
    # Reuse the upstream PEFT+EP path, including lazy Adam-state materialization.
    # A fresh optimizer otherwise has an empty DCP load skeleton and can silently
    # omit saved moments. These additions are PEFT, although they are not LoRA.
    optimizer_state = OptimizerState(torch.nn.ModuleDict(adapters), optimizer,
                                     is_peft=True, has_expert_parallelism=True).state_dict()["optim"]
    return {"adapters": {name: module.state_dict() for name, module in adapters.items()},
            "optimizer": optimizer_state,
            f"rng_rank_{dist.get_rank()}": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}


def assert_state_equal(actual, expected):
    """Exact optimizer/RNG restoration, independent of kernel nondeterminism."""
    if isinstance(expected, torch.Tensor):
        actual = actual.to_local() if hasattr(actual, "to_local") else actual
        expected = expected.to_local() if hasattr(expected, "to_local") else expected
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_state_equal(a, b)
    else:
        assert actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, help="optional real-corpus qualification, never an epoch launch")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=129)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--check-ep-reference", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--debug-identity", action="store_true")
    parser.add_argument("--capture-gdn-inputs", action="store_true")
    args = parser.parse_args()
    if not args.tiny and args.checkpoint is None:
        parser.error("--checkpoint is required for a pretrained probe")
    if args.ep_size < 2 or args.sequence_length < 1:
        parser.error("use EP >= 2 and a positive sequence length")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.check_ep_reference and not args.tiny:
        parser.error("the combined-batch reference is tiny-model-only")
    if args.data_root is not None and args.tiny:
        parser.error("real-corpus qualification requires the full pretrained tokenizer")
    if args.ep_size > 8 and int(os.environ["LOCAL_WORLD_SIZE"]) != 8:
        parser.error("this installed DeepEP normal-mode backend requires eight GPUs per node for cross-node EP")
    args.run_dir.mkdir(parents=True, exist_ok=False) if int(os.environ["RANK"]) == 0 else None
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        hosts = [None] * dist.get_world_size()
        dist.all_gather_object(hosts, socket.gethostname())
        if args.ep_size > 8 and any(len(set(hosts[i:i+8])) != 1 for i in range(0, len(hosts), 8)):
            raise RuntimeError("DeepEP NVLink peer groups must not cross node boundaries")
        import nemo_automodel
        root = Path(nemo_automodel.__file__).resolve().parent.parent
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if revision != UPSTREAM_COMMIT:
            raise RuntimeError("unexpected AutoModel source revision")
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True):
            raise RuntimeError("the pinned upstream source has tracked modifications")
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        emit("gdn_runtime", **configure_frozen_gdn_runtime())
        config = tiny_config(max(8, args.ep_size * 2)) if args.tiny else Qwen3_8_FlashNextConfig.from_pretrained(
            args.checkpoint, local_files_only=True, language_model_only=True)
        config.language_model_only = True
        emit("runtime", upstream=revision,
             probe_args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
             packages={name: importlib.metadata.version(name) for name in
                       ("torch", "transformers", "transformer-engine", "megatron-core", "fla-core", "triton")},
             ep_size=args.ep_size, world_size=dist.get_world_size(), tp_size=1, pp_size=1, cp_size=1,
             gpu=torch.cuda.get_device_name(), tiny=args.tiny, rank_hosts=hosts,
             cuda=torch.version.cuda, nccl=torch.cuda.nccl.version(),
             container_profile="nemo-26.06", container_image=os.environ.get("NGA_CONTAINER_DIGEST"),
             project_source_sha256={str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted([*Path(__file__).parent.glob("*.py"),
                                                     *Path(__file__).parents[1].joinpath("architectures").glob("*.py")])})
        model, mesh, precision = build_frozen_base(
            config, tiny=args.tiny, checkpoint=args.checkpoint, ep_size=args.ep_size,
            activation_checkpointing=args.activation_checkpointing)
        if args.load_only:
            emit("load_only_pass", production_qualified=False,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
            return
        adapter_config = SimplicialAdapterConfig() if not args.tiny else SimplicialAdapterConfig(
            hidden_size=256, query_heads=4, kv_heads=2, head_dim=64, residual_streams=4,
            residual_low_rank=32, short_window=4, long_window=32)
        adapters = install_simplicial_modules(model, adapter_config, device="cuda", dtype=torch.float32)
        fsdp_mesh = mesh.device_mesh["dp_shard_cp"]
        for adapter in adapters.values():
            fully_shard(adapter, mesh=fsdp_mesh, mp_policy=precision, reshard_after_forward=True)
        emit("constructed", adapters=len(adapters), trainable_global=sum(p.numel() for p in model.parameters()
                                                                          if p.requires_grad))
        generator = torch.Generator(device="cuda").manual_seed(700 + dist.get_rank())
        raw_tokens = torch.randint(2, config.text_config.vocab_size, (1, args.sequence_length + 1),
                                   device="cuda", generator=generator)
        tokens, targets = raw_tokens[:, :-1].contiguous(), raw_tokens[:, 1:].contiguous()
        if args.data_root is not None:
            from archlab.automodel.data import load_fineweb_windows

            train_data, validation_data, provenance = load_fineweb_windows(
                args.data_root, args.checkpoint, args.sequence_length)
            batch = train_data.batch(0, rank=dist.get_rank(), world_size=dist.get_world_size(),
                                     micro_batch=1, device="cuda")
            tokens, targets = batch["input_ids"], batch["labels"]
            emit("data_provenance", **provenance,
                 train_accounting=train_data.accounting(dist.get_world_size(), 1),
                 validation_accounting=validation_data.accounting(dist.get_world_size(), 1))
        reads = [m for m in model.modules() if isinstance(m, AdditiveMoERead)]
        if args.capture_gdn_inputs:
            first_gdn = unwrap_checkpoint_wrapper(model.model.language_model.layers["0"]).linear_attn
            original_gdn = first_gdn.chunk_gated_delta_rule

            def capture_gdn(*inputs, **kwargs):
                if dist.get_rank() == 0:
                    def cpu(value):
                        return value.detach().cpu() if isinstance(value, torch.Tensor) else value

                    torch.save({"args": tuple(cpu(value) for value in inputs),
                                "kwargs": {key: cpu(value) for key, value in kwargs.items()}},
                               args.run_dir / "first-gdn-inputs.pt")
                return original_gdn(*inputs, **kwargs)

            first_gdn.chunk_gated_delta_rule = capture_gdn
        phase = {"name": "baseline"}
        saved_hidden = {}
        handles = []
        original_stage_functions = []
        if args.debug_identity:
            first_layer = unwrap_checkpoint_wrapper(model.model.language_model.layers["0"])

            def observe_stage_value(name, value):
                if not isinstance(value, torch.Tensor):
                    return
                key = f"stage.{name}"
                if phase["name"] == "baseline":
                    saved_hidden[key] = value.detach().clone()
                else:
                    expected = saved_hidden[key]
                    emit("identity_stage", phase=phase["name"], stage=name, dtype=str(value.dtype),
                         equal=torch.equal(value, expected),
                         max_abs=(value.float() - expected.float()).abs().max().item() if value.numel() else 0.)

            def wrap_stage(original, stage_name):
                def observed(*args, **kwargs):
                    for index, value in enumerate(args):
                        observe_stage_value(f"{stage_name}.arg{index}", value)
                    for key, value in kwargs.items():
                        observe_stage_value(f"{stage_name}.{key}", value)
                    output = original(*args, **kwargs)
                    for index, value in enumerate(output if isinstance(output, tuple) else (output,)):
                        observe_stage_value(f"{stage_name}.output{index}", value)
                    return output
                return observed

            for stage_name in ("causal_conv1d_fn", "chunk_gated_delta_rule"):
                original = getattr(first_layer.linear_attn, stage_name)
                original_stage_functions.append((first_layer.linear_attn, stage_name, original))
                setattr(first_layer.linear_attn, stage_name, wrap_stage(original, stage_name))
            submodules = {"embedding": model.model.language_model.embed_tokens}
            submodules.update({f"first.{name}": module for name, module in first_layer.named_modules() if name})
            for name, module in submodules.items():
                def observe_component(module, inputs, output, name=name):
                    values = output if isinstance(output, tuple) else (output,)
                    for index, value in enumerate(values):
                        if not isinstance(value, torch.Tensor):
                            continue
                        key = f"component.{name}.{index}"
                        if phase["name"] == "baseline":
                            saved_hidden[key] = value.detach().clone()
                        else:
                            expected = saved_hidden[key]
                            emit("identity_component", phase=phase["name"], component=key,
                                 dtype=str(value.dtype), equal=torch.equal(value, expected),
                                 max_abs=(value.float() - expected.float()).abs().max().item() if value.numel() else 0.)
                handles.append(module.register_forward_hook(observe_component))
            for name, layer in model.model.language_model.layers.items():
                def observe_layer(module, inputs, output, name=name):
                    if phase["name"] == "baseline":
                        saved_hidden[name] = output.detach().clone()
                    else:
                        expected = saved_hidden[name]
                        emit("identity_layer", phase=phase["name"], layer=name, dtype=str(output.dtype),
                             equal=torch.equal(output, expected), max_abs=(output - expected).abs().max().item())
                handles.append(layer.register_forward_hook(observe_layer))
            for i, read in enumerate(reads):
                def observe_adapter(before, after, i=i):
                    emit("identity_branch", index=i, before_dtype=str(before.dtype), after_dtype=str(after.dtype),
                         equal=torch.equal(before, after), max_abs=(before - after).abs().max().item())
                read.identity_observer = observe_adapter
        for read in reads:
            read.adapter_enabled = False
        with torch.no_grad():
            reference = model(input_ids=tokens).logits.detach().clone()
        if args.capture_gdn_inputs:
            emit("gdn_inputs_captured", path=str(args.run_dir / "first-gdn-inputs.pt"))
            return
        if args.debug_identity:
            phase["name"] = "baseline-repeat"
            with torch.no_grad():
                repeated = model(input_ids=tokens).logits
            emit("baseline_repeat", equal=torch.equal(repeated, reference),
                 max_abs=(repeated - reference).abs().max().item())
            del repeated
        phase["name"] = "adapter"
        for read in reads:
            read.adapter_enabled = True
        with torch.no_grad():
            actual = model(input_ids=tokens).logits
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        if args.data_root is not None:
            with torch.no_grad():
                initial_loss = MaskedCrossEntropy(reduction="mean")(actual, targets)
                if not torch.isfinite(initial_loss):
                    raise RuntimeError("nonfinite initial real-corpus loss")
                dist.all_reduce(initial_loss)
                initial_loss.div_(dist.get_world_size())
            emit("initial_real_corpus_loss", mean_loss=initial_loss.item(),
                 tokens=targets.numel() * dist.get_world_size())
        del actual, reference
        for handle in handles:
            handle.remove()
        for module, name, original in original_stage_functions:
            setattr(module, name, original)
        for read in reads:
            read.identity_observer = None
        saved_hidden.clear()
        emit("identity_pass", sequence_length=args.sequence_length)
        if args.identity_only:
            return
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
                                      betas=(.9, .95), weight_decay=.1, foreach=False)
        loss_fn = MaskedCrossEntropy(reduction="mean")

        captured_gradients = {}

        def update(step, capture=False):
            start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=tokens)
            loss = loss_fn(output.logits, targets)  # all positions; bounded disposable diagnostic updates
            loss.backward()
            torch.cuda.synchronize()
            for name, parameter in model.named_parameters():
                if ADAPTER_MARKER not in name:
                    if parameter.grad is not None:
                        raise RuntimeError(f"frozen gradient: {name}")
                else:
                    if parameter.grad is None:
                        raise RuntimeError(f"missing adapter gradient: {name}")
                    grad = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
                    if not torch.isfinite(grad).all():
                        raise RuntimeError(f"nonfinite gradient: {name}")
                    if step and grad.numel() and grad.abs().max() == 0:
                        raise RuntimeError(f"zero warmed-up adapter gradient: {name}")
                    if capture:
                        captured_gradients[name] = grad.detach().clone()
            optimizer.step()
            emit("diagnostic_update", step=step, loss=loss.item(), seconds=time.monotonic()-start,
                 learning_rate=optimizer.param_groups[0]["lr"],
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
            return loss.detach().clone()

        losses = [update(step).item() for step in range(2)]
        payload = checkpoint_payload(adapters, optimizer)
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_rng = copy.deepcopy(payload[f"rng_rank_{dist.get_rank()}"])
        path = args.run_dir / "adapter-checkpoint"
        dcp.save(payload, checkpoint_id=path)
        saved = {name: tensor.detach().clone() for name, tensor in model.named_parameters() if tensor.requires_grad}
        next_loss = update(2, capture=True)
        next_weights = {name: tensor.detach().clone() for name, tensor in model.named_parameters()
                        if tensor.requires_grad}
        with torch.no_grad():
            for parameter in saved:
                dict(model.named_parameters())[parameter].add_(.125)
        # Deliberately discard the live optimizer: restoration must also work
        # without its already-allocated moment tensors or step counters.
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
                                      betas=(.9, .95), weight_decay=.1, foreach=False)
        restored = checkpoint_payload(adapters, optimizer)
        dcp.load(restored, checkpoint_id=path)
        for name, adapter in adapters.items():
            adapter.load_state_dict(restored["adapters"][name], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        rng = restored[f"rng_rank_{dist.get_rank()}"]
        assert_state_equal(optimizer.state_dict(), saved_optimizer)
        assert_state_equal(rng, saved_rng)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state(rng["cuda"])
        for name, expected in saved.items():
            torch.testing.assert_close(dict(model.named_parameters())[name].to_local(), expected.to_local(), rtol=0, atol=0)
        replay_loss = update(3)
        torch.testing.assert_close(replay_loss, next_loss, rtol=1e-5, atol=1e-6)
        max_resume_difference = 0.0
        gradient_error, gradient_norm, update_error, update_norm = 0., 0., 0., 0.
        largest = []
        for name, expected in next_weights.items():
            parameter = dict(model.named_parameters())[name]
            actual = parameter.to_local()
            expected = expected.to_local()
            difference = (actual - expected).float()
            maximum = difference.abs().max().item() if difference.numel() else 0.0
            largest.append((maximum, name))
            max_resume_difference = max(max_resume_difference, maximum)
            update_error += difference.square().sum().item()
            update_norm += (expected - saved[name].to_local()).float().square().sum().item()
            reference_grad = captured_gradients[name]
            gradient_error += (parameter.grad.to_local() - reference_grad).float().square().sum().item()
            gradient_norm += reference_grad.float().square().sum().item()
        replay_metrics = {"gradient_relative_l2": (gradient_error / max(gradient_norm, 1e-30)) ** .5,
                          "update_relative_l2": (update_error / max(update_norm, 1e-30)) ** .5,
                          "update_max_abs": max_resume_difference,
                          "largest_parameter_differences": sorted(largest, reverse=True)[:5]}
        emit("resume_numerics", **replay_metrics)
        # Atomic shared-K/V sums and BF16 backward are not bitwise deterministic.
        # Test the gradient and UPDATE error, not relative error in near-zero
        # weights. Exact checkpoint parameter/optimizer/RNG tests above remain.
        if replay_metrics["gradient_relative_l2"] > .03 or replay_metrics["update_relative_l2"] > .01:
            raise AssertionError("resume trajectory exceeds BF16 gradient/update numerical bounds")
        emit("probe_pass", losses=losses, checkpoint_parameter_restore=True,
             checkpoint_optimizer_restore=True, checkpoint_rng_restore=True,
             checkpoint_fresh_optimizer_restore=True,
             resume_next_update_max_abs=max_resume_difference, objective="main-next-token-CE-all-positions",
             production_qualified=False, sequence_length=args.sequence_length, tiny=args.tiny,
             pending="all recipe gates plus real-data launch-entry qualification")
        if args.check_ep_reference:
            from archlab.automodel.numerics import compare_combined_batch_reference

            emit("ep_reference_pass", **compare_combined_batch_reference(
                model, config, adapter_config, tokens, targets))
    finally:
        free_buffer()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

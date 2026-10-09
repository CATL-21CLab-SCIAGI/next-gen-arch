"""Project-owned V4.1 adapter recipe using NeMo Automodel components.

Upstream Automodel has no V4.1 model class. Its V4 class must not be used for
this checkpoint. This extension combines Automodel's mesh and indexed-data
components with the separately qualified native V4.1/PyTorch model bridge.
It is not an upstream-supported NeMoAutoModelForCausalLM recipe.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import importlib.metadata
import json
import math
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.automodel.deepseek_v41_execution import (
    adapter_optimizers,
    build_replica,
    enable_activation_checkpointing,
    install_training_branches,
)
from archlab.automodel.deepseek_v41_pytorch import dequantize_base_once, install_pytorch_leaves
from archlab.automodel.deepseek_v41_runtime import (
    implementation_hashes,
    select_container_kernel_packages,
)
from archlab.automodel.deepseek_v41_training import (
    append_metric,
    emit,
    evaluate,
    learning_rate,
    optimizer_step,
    restore_adapter_checkpoint,
    save_adapter_checkpoint,
)


def _path(value: str) -> Path:
    if not value.startswith("env:") or not os.environ.get(value[4:]):
        raise ValueError(f"recipe paths must name populated env: variables: {value}")
    return Path(os.environ[value[4:]]).resolve()


@dataclass(frozen=True)
class V41RecipeConfig:
    assets: str
    weights: str
    train_data: str
    validation_data: str
    qualification: str
    leaf_qualification: str
    output: str
    container_kernel_packages: str
    world_size: int = 32
    ep_size: int = 8
    context: int = 16384
    supervised_tokens: int = 1_000_000_000
    validation_tokens: int = 1_000_000
    validation_interval_tokens: int = 10_000_000
    checkpoint_interval_tokens: int = 50_000_000
    warmup_steps: int = 100
    query_chunk: int = 32

    def __post_init__(self) -> None:
        if (self.world_size, self.ep_size, self.context, self.supervised_tokens) != (32, 8, 16384, 1_000_000_000):
            raise ValueError("this recipe only supports the reviewed DP32/EP8, 16K, 1B-target run")
        if any(type(v) is not int or v < 1 for v in (
            self.world_size, self.ep_size, self.context, self.supervised_tokens,
            self.validation_tokens, self.validation_interval_tokens, self.checkpoint_interval_tokens,
            self.warmup_steps, self.query_chunk,
        )):
            raise ValueError("recipe intervals and chunk dimensions must be positive integers")

    def build(self) -> V41TrainingSession:
        """Build after distributed initialization, admitting only qualified inputs."""
        if dist.get_world_size() != self.world_size:
            raise ValueError("the production recipe requires all 32 ranks")
        leaf_path, qualification = _path(self.leaf_qualification), _path(self.qualification)
        leaf = json.loads(leaf_path.read_text())
        source_hashes = implementation_hashes()
        if leaf.get("implementation_sha256") != source_hashes:
            raise ValueError("leaf qualification was not run against this implementation")
        required_leaf = {"native_fp4_fp8_shared_kv_two_step_backbone",
                         "production_simplicial_geometry_window_edges_and_five_gradients"}
        passed_leaf = {t["name"] for t in leaf.get("tests", []) if t.get("passed")}
        if not leaf.get("passed") or not required_leaf <= passed_leaf:
            raise ValueError("native/PyTorch and production-geometry leaf qualification is incomplete")
        weights = _path(self.weights)
        index_sha = sha256_file(weights / "model.safetensors.index.json")
        receipts = []
        for rank in range(self.ep_size):
            path = qualification / f"rank{rank}.json"
            receipt = json.loads(path.read_text())
            contexts = {test.get("context") for test in receipt.get("tests", [])}
            if (receipt.get("implementation_sha256") != source_hashes
                    or not receipt.get("passed") or receipt["rank"] != rank
                    or receipt["load"]["index_sha256"] != index_sha or self.context not in contexts):
                raise ValueError(f"full-checkpoint/memory qualification is incomplete: {path}")
            receipts.append(sha256_file(path))
        output = _path(self.output)
        if dist.get_rank() == 0:
            output.mkdir(parents=True, exist_ok=False)
        dist.barrier()
        from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
        from nemo_automodel.components.distributed.mesh import ParallelismSizes

        # Reuse Automodel's mesh definition. The frozen native bridge owns EP
        # expert/table placement; no FSDP wrapper is applied to the base here.
        setup = DistributedSetup.build(strategy=FSDP2Config(),
                                       parallelism_sizes=ParallelismSizes(ep_size=self.ep_size),
                                       world_size=self.world_size)
        group = setup.mesh_context.moe_mesh["ep"].get_group()
        hosts = [None] * self.ep_size
        import socket

        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        if len(set(hosts)) != 1:
            raise ValueError("this recipe requires node-local EP8 groups")
        reference, model, loading = build_replica(assets=_path(self.assets), weights=weights,
                                                 group=group, context=self.context)
        conversion = dequantize_base_once(model)
        install_pytorch_leaves(reference, query_chunk=self.query_chunk)
        adapters = install_training_branches(reference, model)
        enable_activation_checkpointing(model)
        from archlab.automodel.deepseek_v41_data import MathPilot

        train = MathPilot(_path(self.train_data), expected_split="train", expected_budget=self.supervised_tokens)
        validation = MathPilot(_path(self.validation_data), expected_split="validation", expected_budget=self.validation_tokens)
        import nemo_automodel

        automodel_root = Path(nemo_automodel.__file__).resolve().parent.parent
        automodel_commit = subprocess.check_output(["git", "-C", str(automodel_root), "rev-parse", "HEAD"], text=True).strip()
        if automodel_commit != "a4ce87c003f08b74d68684d3627f6e6048bc0140":
            raise ValueError("use the existing audited Automodel checkout")
        atomic_write_json(output / f"rank-{dist.get_rank():02d}-loading.json",
                          {"loader": loading, "bf16_memory": conversion}, allow_nan=False)
        contract = {
            "backend": "project-owned-native-v41-bridge-with-automodel-components",
            "stock_automodel_v41_support": False,
            "model_revision": "df42c109f1defefcbfcedbe7d905718a12266e40",
            "checkpoint_index_sha256": index_sha, "qualification_sha256": receipts,
            "implementation_sha256": source_hashes,
            "leaf_qualification_sha256": sha256_file(leaf_path), "world_size": self.world_size,
            "ep_size": self.ep_size, "tp_size": 1, "pp_size": 1, "cp_size": 1,
            "fsdp_base_wrapping": False, "supervised_token_budget": self.supervised_tokens,
            "train_manifest_sha256": sha256_file(_path(self.train_data) / "PILOT_READY.json"),
            "validation_manifest_sha256": sha256_file(_path(self.validation_data) / "PILOT_READY.json"),
            "container_image": os.environ["NGA_CONTAINER_DIGEST"],
            "project_commit": os.environ["NGA_EXPECTED_COMMIT"],
            "runtime": {name: importlib.metadata.version(name) for name in
                        ("torch", "transformers", "transformer-engine", "megatron-core", "triton")},
            "cuda": torch.version.cuda, "nccl": list(torch.cuda.nccl.version()),
            "automodel_commit": automodel_commit,
            "native_activation_rounding": True, "attention_exp_rounding": "BF16-block64",
            "adapter_core_precision": "float32", "adapter_projection_precision": "bfloat16",
        }
        contracts = [None] * self.world_size
        dist.all_gather_object(contracts, contract)
        if any(value != contract for value in contracts):
            raise ValueError("rank-dependent production contract")
        if dist.get_rank() == 0:
            atomic_write_json(output / "RUN_CONTRACT.json", contract, allow_nan=False)
        return V41TrainingSession(self, reference, model, adapters, train, validation, output, contract)


class V41TrainingSession:
    def __init__(self, config, reference, model, adapters, train, validation, output, contract):
        self.config, self.reference, self.model, self.adapters = config, reference, model, adapters
        self.train, self.validation, self.output, self.contract = train, validation, output, contract
        self.optimizers = adapter_optimizers(model, adapters)
        self.stop_requested = False
        signal.signal(signal.SIGUSR1, self._request_stop)

    def _request_stop(self, signum, frame):
        self.stop_requested = True

    def qualify_mesh(self) -> None:
        """Check 32-rank gradients/reload, then restore fresh adapter/RNG state."""
        initial = {i: {name: p.detach().cpu().clone() for name, p in adapter.state_dict().items()}
                   for i, adapter in self.adapters.items()}
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        inputs, labels, _ = self.train.batch(dist.get_rank(), device="cuda", smoke_context=128)
        frozen = [(p, p._version) for p in self.model.parameters() if not p.requires_grad]
        for step in range(2):
            metric = optimizer_step(self.reference, self.model, self.optimizers, inputs, labels, learning_rate=1e-5)
            emit("production_mesh_qualification", step=step, **metric)
            if step == 1 and any(p.grad is None or not bool(p.grad.count_nonzero())
                                 for adapter in self.adapters.values() for p in adapter.parameters()):
                raise ValueError("missing second-step adapter gradients on the production mesh")
        if any(p._version != version or p.grad is not None for p, version in frozen):
            raise ValueError("the frozen base changed during production-mesh qualification")
        path = self.output / "qualification-checkpoint"
        cursor = {"step": 2, "supervised_tokens": 0, "qualification_only": True}
        save_adapter_checkpoint(path, self.adapters, self.optimizers, cursor, self.contract)
        optimizer_step(self.reference, self.model, self.optimizers, inputs, labels, learning_rate=1e-5)
        expected = {i: {name: p.detach().cpu().clone() for name, p in adapter.state_dict().items()}
                    for i, adapter in self.adapters.items()}
        expected_optimizers = copy.deepcopy([optimizer.state_dict() for optimizer in self.optimizers])
        restore_adapter_checkpoint(path, self.adapters, self.optimizers, contract=self.contract)
        optimizer_step(self.reference, self.model, self.optimizers, inputs, labels, learning_rate=1e-5)
        for i, adapter in self.adapters.items():
            for name, parameter in adapter.state_dict().items():
                torch.testing.assert_close(parameter.cpu(), expected[i][name], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close([optimizer.state_dict() for optimizer in self.optimizers],
                                   expected_optimizers, rtol=1e-5, atol=1e-7)
        del expected, expected_optimizers
        long_inputs, long_labels, _ = self.train.batch(dist.get_rank(), device="cuda",
                                                      smoke_context=self.config.context, pad_to_full=True)
        torch.cuda.reset_peak_memory_stats()
        metric = optimizer_step(self.reference, self.model, self.optimizers, long_inputs, long_labels, learning_rate=1e-7)
        emit("production_mesh_context_qualification", context=self.config.context, **metric)
        del long_inputs, long_labels
        for i, adapter in self.adapters.items():
            adapter.load_state_dict(initial[i], strict=True)
        self.optimizers = adapter_optimizers(self.model, self.adapters)
        self.model.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        del initial
        torch.cuda.empty_cache()
        emit("production_mesh_qualified_fresh_state_restored")

    def run(self, *, resume_from: Path | None = None) -> None:
        self.qualify_mesh()
        start_step, consumed, warmup_tokens, last_saved = 0, 0, 0, -1
        total_steps = math.ceil(len(self.train) / self.config.world_size)
        if resume_from is not None:
            cursor = restore_adapter_checkpoint(resume_from, self.adapters, self.optimizers, contract=self.contract)
            start_step, consumed, warmup_tokens = cursor["step"], cursor["supervised_tokens"], cursor["warmup_tokens"]
            expected = sum(window["targets"] for window in self.train.windows[:start_step * self.config.world_size])
            if not 0 <= start_step <= total_steps or consumed != expected:
                raise ValueError("checkpoint cursor disagrees with the sealed data order")
        append_metric(self.output / "validation.jsonl", evaluate(self.reference, self.model, self.validation, step=start_step))
        next_eval = (consumed // self.config.validation_interval_tokens + 1) * self.config.validation_interval_tokens
        next_save = (consumed // self.config.checkpoint_interval_tokens + 1) * self.config.checkpoint_interval_tokens
        for step in range(start_step, total_steps):
            inputs, labels, _ = self.train.batch(step * self.config.world_size + dist.get_rank(), device="cuda")
            rate = learning_rate(step, consumed, warmup_tokens, budget=self.config.supervised_tokens,
                                 warmup_steps=self.config.warmup_steps)
            metric = optimizer_step(self.reference, self.model, self.optimizers, inputs, labels, learning_rate=rate)
            consumed += metric["supervised_tokens"]
            if step < self.config.warmup_steps:
                warmup_tokens = consumed
            if consumed > self.config.supervised_tokens:
                raise RuntimeError("supervised token budget exceeded")
            record = {"step": step + 1, "consumed_supervised_tokens": consumed, **metric}
            append_metric(self.output / "train.jsonl", record)
            emit("train_step", **record)
            requested = self.stop_requested or (self.output / "STOP_REQUEST").exists()
            stop = torch.tensor(int(requested), device="cuda", dtype=torch.int32)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if consumed >= next_eval:
                append_metric(self.output / "validation.jsonl", evaluate(self.reference, self.model, self.validation, step=step + 1))
                next_eval = (consumed // self.config.validation_interval_tokens + 1) * self.config.validation_interval_tokens
            if consumed >= next_save or stop.item() or step + 1 == total_steps:
                cursor = {"step": step + 1, "supervised_tokens": consumed, "warmup_tokens": warmup_tokens}
                save_adapter_checkpoint(self.output / "checkpoints" / f"step-{step + 1:06d}",
                                        self.adapters, self.optimizers, cursor, self.contract)
                last_saved = step + 1
                next_save = (consumed // self.config.checkpoint_interval_tokens + 1) * self.config.checkpoint_interval_tokens
            if stop.item():
                emit("training_paused", step=step + 1, supervised_tokens=consumed, checkpoint_step=last_saved)
                return
        if consumed != self.config.supervised_tokens:
            raise RuntimeError("the sealed data did not reach exactly the requested budget")
        if dist.get_rank() == 0:
            atomic_write_json(self.output / "TRAINING_COMPLETE.json", {"steps": total_steps, "supervised_tokens": consumed})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    config = V41RecipeConfig(**yaml.safe_load(args.recipe.read_text()))
    if not os.environ.get("NGA_CONTAINER_DIGEST") or not os.environ.get("NGA_EXPECTED_COMMIT"):
        raise ValueError("record the existing container image and committed source before launch")
    select_container_kernel_packages(_path(config.container_kernel_packages))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15), device_id=torch.device("cuda", local_rank))
    try:
        config.build().run(resume_from=args.resume_from)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

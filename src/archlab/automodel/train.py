"""One-pass frozen-pretrained finetuning; reuses the qualified AutoModel boundary.

Only the new simplicial modules are optimized. No baseline job, MTP, unfreezing,
data wrapping, runtime installation, or node lifecycle operation is performed.
The tiny mode is a bounded entry/recovery fixture, not a pretrained experiment.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import math
import os
import signal
import time
import uuid
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextConfig
from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from torch.distributed.fsdp import fully_shard

from archlab.architectures.simplicial_adapter import SimplicialAdapterConfig
from archlab.automodel.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
    write_json,
)
from archlab.automodel.data import load_fineweb_windows
from archlab.automodel.execution import build_frozen_base, emit, tiny_config
from archlab.automodel.runtime import configure_frozen_gdn_runtime, runtime_provenance
from archlab.automodel.simplicial import ADAPTER_MARKER, AdditiveMoERead, install_simplicial_modules
from archlab.automodel.training_config import TrainingConfig


class _SmokeWindows:
    """Deterministic, disjoint-target windows for the tiny production-entry fixture."""

    def __init__(self, config: TrainingConfig, seed: int):
        self.config = config
        self.total_steps = 8
        self.total_tokens = self.total_steps * config.world_size * config.sequence_length + 1
        self.tokens = torch.randint(2, 1024, (self.total_tokens,), generator=torch.Generator().manual_seed(seed))

    def full_microbatches(self, world_size: int, micro_batch: int) -> int:
        if (world_size, micro_batch) != (self.config.world_size, 1):
            raise ValueError("smoke fixture topology mismatch")
        return self.total_steps

    def batch(self, cursor: int, *, rank: int, world_size: int, micro_batch: int, device: str) -> dict:
        """Return contiguous input_ids/labels, each [1, sequence], with no wrapping."""
        if not 0 <= cursor < self.full_microbatches(world_size, micro_batch) or not 0 <= rank < world_size:
            raise IndexError("smoke corpus exhausted or invalid rank")
        start = (cursor * world_size + rank) * self.config.sequence_length
        raw = self.tokens[start:start + self.config.sequence_length + 1].to(device).unsqueeze(0)
        return {"input_ids": raw[:, :-1].contiguous(), "labels": raw[:, 1:].contiguous()}

    def accounting(self, world_size: int, micro_batch: int) -> dict:
        return {"source_tokens": self.total_tokens, "full_microbatches": self.full_microbatches(world_size, micro_batch),
                "consumed_target_tokens": self.total_tokens - 1, "unused_final_target_tokens": 0,
                "initial_context_only_tokens": 1, "wrapped_tokens": 0}


def _path(value: str) -> Path:
    if value.startswith("env:"):
        name = value[4:]
        if not os.environ.get(name):
            raise ValueError(f"missing path environment variable: {name}")
        value = os.environ[name]
    return Path(value).resolve()


class TrainingSession:
    """Own the optimizer-step, evaluation and durable-cursor lifecycle of one run."""

    def __init__(self, *, config: TrainingConfig, model, adapters: dict, mesh, train_data, validation_data,
                 run_dir: Path, contract: dict, attempt: str):
        self.config, self.model, self.adapters, self.mesh = config, model, adapters, mesh
        self.train_data, self.validation_data = train_data, validation_data
        self.run_dir, self.contract, self.attempt = run_dir, contract, attempt
        self.optimizer = config.build_optimizer(model)
        self.scheduler = config.build_scheduler(self.optimizer, total_steps=contract["total_steps"])
        self.loss_fn = MaskedCrossEntropy(reduction="mean")
        self.stop_requested = False
        self.save_requested = False
        self.last_saved_cursor = -1
        for name, parameter in model.named_parameters():
            if parameter.requires_grad != (ADAPTER_MARKER in name):
                raise RuntimeError(f"unexpected trainability: {name}")
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        actual = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        if actual != expected:
            raise RuntimeError("optimizer does not contain exactly the added parameters")

    def record(self, event: str, **values) -> None:
        if dist.get_rank() == 0:
            record = {"event": event, "time_unix": time.time(), "attempt": self.attempt, **values}
            with (self.run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            emit(event, **values)

    def batch(self, data, cursor: int) -> dict:
        """Obtain input_ids/labels [micro_batch, sequence] for this DP rank."""
        return data.batch(cursor, rank=dist.get_rank(), world_size=self.config.world_size,
                          micro_batch=self.config.micro_batch, device="cuda")

    @torch.no_grad()
    def check_initialization(self) -> None:
        """Check the declared initialization and record its first-batch loss impact.

        Nonzero initialization may worsen loss arbitrarily; only nonfinite
        output or a silently inactive branch is an error in that mode.
        """
        batch = self.batch(self.train_data, 0)
        reads = [m for m in self.model.modules() if isinstance(m, AdditiveMoERead)]
        mode = self.contract["adapter"]["output_initialization"]
        self.model.eval()
        try:
            for read in reads:
                read.adapter_enabled = False
            reference = self.model(input_ids=batch["input_ids"]).logits.detach().clone()
            for read in reads:
                read.adapter_enabled = True
            actual = self.model(input_ids=batch["input_ids"]).logits
            finite = (torch.isfinite(actual).all() & torch.isfinite(reference).all()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError("nonfinite initial logits")
            identical = torch.tensor(int(torch.equal(actual, reference)), device="cuda")
            dist.all_reduce(identical, op=dist.ReduceOp.MIN)
            if mode == "zeros" and not identical.item():
                raise RuntimeError("zero-output initialization did not preserve exact logits")
            changed = torch.tensor(int(not torch.equal(actual, reference)), device="cuda")
            dist.all_reduce(changed, op=dist.ReduceOp.MIN)
            if mode == "normal" and not changed.item():
                raise RuntimeError("nonzero-output initialization did not change logits on every rank")
            losses = torch.stack((self.loss_fn(reference, batch["labels"]),
                                  self.loss_fn(actual, batch["labels"]))).double()
            dist.all_reduce(losses)
            losses /= self.config.world_size
            # Bound temporary FP32 storage rather than converting both entire
            # [batch, sequence, vocabulary] 16K logits tensors at once.
            squared_delta = torch.zeros((), device="cuda", dtype=torch.float64)
            elements = torch.tensor(actual.numel(), device="cuda", dtype=torch.float64)
            for start in range(0, actual.shape[1], 256):
                delta = actual[:, start:start + 256].float() - reference[:, start:start + 256].float()
                squared_delta += delta.square().sum(dtype=torch.float64)
            dist.all_reduce(squared_delta)
            dist.all_reduce(elements)
            logit_delta_rms = (squared_delta / elements).sqrt().item()
        finally:
            for read in reads:
                read.adapter_enabled = True
            self.model.train()
        self.record("initial_identity_pass" if mode == "zeros" else "initial_nonzero_pass",
                    sequence_length=self.config.sequence_length, output_initialization=mode,
                    backbone_loss=losses[0].item(), initialized_loss=losses[1].item(),
                    loss_delta=(losses[1] - losses[0]).item(), logit_delta_rms=logit_delta_rms)

    @torch.no_grad()
    def validate(self, cursor: int) -> None:
        start = time.monotonic()
        count = min(self.config.validation_batches,
                    self.validation_data.full_microbatches(self.config.world_size, self.config.micro_batch))
        loss = torch.zeros((), device="cuda", dtype=torch.float64)
        self.model.eval()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                for index in range(count):
                    batch = self.batch(self.validation_data, index)
                    logits = self.model(input_ids=batch["input_ids"]).logits
                    loss += self.loss_fn(logits, batch["labels"]).double()
                    del logits
        finally:
            self.model.train()
        dist.all_reduce(loss)
        mean = (loss / (count * self.config.world_size)).item()
        if not math.isfinite(mean):
            raise RuntimeError("nonfinite held-out loss")
        self.record("validation", step=cursor, mean_loss=mean, batches=count,
                    target_tokens=count * self.tokens_per_step, fixed_heldout_prefix=True,
                    seconds=time.monotonic() - start)

    @property
    def tokens_per_step(self) -> int:
        return self.config.world_size * self.config.micro_batch * self.config.sequence_length

    def train_step(self, cursor: int) -> None:
        start = time.monotonic()
        batch = self.batch(self.train_data, cursor)
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(input_ids=batch["input_ids"]).logits
        loss = self.loss_fn(logits, batch["labels"])
        loss.backward()
        del logits
        checks, issues, nonzero = [torch.isfinite(loss)], [], []
        check_first_nonzero = cursor == 0 and self.contract["adapter"]["output_initialization"] == "normal"
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    issues.append(name)
            elif parameter.grad is None:
                issues.append(name)
            else:
                local = parameter.grad.to_local()
                checks.append(torch.isfinite(local).all())
                if check_first_nonzero:
                    nonzero.append(torch.count_nonzero(local) > 0)
        valid = (torch.stack(checks).all() & (not issues)).to(torch.int32)
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise RuntimeError(f"nonfinite loss/gradient or trainability violation: {issues}")
        if check_first_nonzero:
            # An individual FSDP shard can be empty; require nonzero gradients
            # for every GLOBAL added parameter, not on every local shard.
            nonzero_parameters = torch.stack(nonzero).to(torch.int32)
            dist.all_reduce(nonzero_parameters, op=dist.ReduceOp.MAX)
            if not nonzero_parameters.all().item():
                names = [n for n, p in self.model.named_parameters() if p.requires_grad]
                missing = [n for n, present in zip(names, nonzero_parameters.tolist(), strict=True) if not present]
                raise RuntimeError(f"zero first-step gradients with nonzero initialization: {missing}")
            self.record("first_step_nonzero_gradients_pass", parameter_tensors=len(nonzero),
                        all_added_parameters=True, frozen_gradients_absent=True)
        norm = scale_grads_and_clip_grad_norm(
            self.config.gradient_clip, [self.model], device_mesh=self.mesh.device_mesh,
            moe_mesh=self.mesh.moe_mesh, ep_axis_name="ep", dp_group_size=self.config.world_size, foreach=False)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite global adapter gradient norm")
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step(1)
        self.optimizer.zero_grad(set_to_none=True)
        reporting_loss = loss.detach().clone()
        dist.all_reduce(reporting_loss)
        mean_loss = reporting_loss.item() / self.config.world_size
        torch.cuda.synchronize()
        timing = torch.tensor([time.monotonic() - start, torch.cuda.max_memory_allocated()],
                              device="cuda", dtype=torch.float64)
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        seconds, peak = timing.tolist()
        self.record("train", step=cursor + 1, cursor=cursor + 1, mean_loss=mean_loss,
                    learning_rate=lr, gradient_norm=norm.item(), gradient_clip=self.config.gradient_clip,
                    consumed_target_tokens=(cursor + 1) * self.tokens_per_step,
                    seconds=seconds, tokens_per_second=self.tokens_per_step / seconds,
                    peak_allocated_bytes=int(peak))

    def save(self, cursor: int) -> None:
        if cursor == self.last_saved_cursor:
            return
        path = self.run_dir / "checkpoints" / f"step-{cursor:08d}-{self.attempt}"
        save_training_checkpoint(path, adapters=self.adapters, optimizer=self.optimizer,
                                 scheduler=self.scheduler, contract=self.contract, cursor=cursor)
        self.last_saved_cursor = cursor
        if dist.get_rank() == 0:
            write_json(self.run_dir / "LATEST.json", {"checkpoint": str(path.resolve()), "cursor": cursor})
        self.record("checkpoint", step=cursor, path=str(path.resolve()), exact_state_digests=True)

    def run(self, *, start_cursor: int, stop_after: int | None) -> None:
        total = self.contract["total_steps"]
        end = min(total, stop_after) if stop_after is not None else total
        if end <= start_cursor:
            raise ValueError("stop boundary must exceed the restored cursor")
        self.model.train()
        if start_cursor == 0:
            self.check_initialization()
            self.validate(0)
        for cursor in range(start_cursor, end):
            self.train_step(cursor)
            completed = cursor + 1
            flags = torch.tensor([int(self.save_requested), int(self.stop_requested)], device="cuda")
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            save_now, stop_now = flags.tolist()
            if completed == self.config.early_validation_step or completed % self.config.validation_interval == 0:
                self.validate(completed)
            if (save_now or stop_now or completed in (1, self.config.warmup_steps, end)
                    or completed % self.config.checkpoint_interval == 0):
                self.save(completed)
                self.save_requested = False
            if stop_now:
                break
        self.record("complete" if completed == total else "stopped", step=completed,
                    consumed_target_tokens=completed * self.tokens_per_step,
                    one_pass_complete=completed == total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--stop-after", type=int, help="bounded qualification/stop boundary; never changes the LR horizon")
    parser.add_argument("--tiny", action="store_true", help="eight-step synthetic entry/recovery fixture")
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text())
    config = TrainingConfig(**recipe["training"])
    adapter_config = SimplicialAdapterConfig(**recipe["model"]["adapter"])
    if not recipe["model"]["freeze_existing_parameters"] or recipe["model"]["added_ffn"]:
        parser.error("this entry requires the agreed frozen backbone and no added FFN")
    if (config.world_size, config.ep_size, config.sequence_length) != (
            recipe["execution"]["data_parallel_size"], recipe["execution"]["expert_parallel_size"],
            recipe["data"]["sequence_length"]):
        parser.error("training settings disagree with the declared experiment topology/context")
    if args.tiny:
        world = int(os.environ["WORLD_SIZE"])
        config = replace(config, world_size=world, ep_size=min(8, world), sequence_length=129,
                         warmup_steps=2, validation_interval=2, early_validation_step=2,
                         validation_batches=1, checkpoint_interval=2)
        adapter_config = replace(adapter_config, hidden_size=256, query_heads=4, kv_heads=2, head_dim=64,
                                 residual_low_rank=32, short_window=4, long_window=32)
    if int(os.environ["WORLD_SIZE"]) != config.world_size or (args.stop_after is not None and args.stop_after < 1):
        parser.error("world size mismatch or invalid stop boundary")
    checkpoint = None if args.tiny else (args.checkpoint or _path(recipe["model"]["checkpoint"])).resolve()
    data_root = None if args.tiny else (args.data_root or _path(recipe["data"]["root"])).resolve()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_lock = None
    try:
        if dist.get_rank() == 0:
            args.run_dir.mkdir(parents=True, exist_ok=args.resume_from is not None)
            run_lock = (args.run_dir / "writer.lock").open("a")
            fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt = [uuid.uuid4().hex[:12] if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(attempt)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        provenance = runtime_provenance(ep_size=config.ep_size)
        gdn_runtime = configure_frozen_gdn_runtime()
        emit("runtime", **provenance, gdn_runtime=gdn_runtime, pid=os.getpid(), tiny=args.tiny)
        base_config = tiny_config(max(8, config.ep_size * 2)) if args.tiny else Qwen3_8_FlashNextConfig.from_pretrained(
            checkpoint, local_files_only=True, language_model_only=True)
        base_config.language_model_only = True
        selected_layers = [i + 1 for i, kind in enumerate(base_config.text_config.layer_types) if kind == "full_attention"]
        if not args.tiny and selected_layers != recipe["model"]["layers_1based"]:
            raise ValueError("recipe insertion layers do not match the pretrained full-attention blocks")
        model, mesh, precision = build_frozen_base(base_config, tiny=args.tiny, checkpoint=checkpoint,
                                                  ep_size=config.ep_size, activation_checkpointing=True)
        adapters = install_simplicial_modules(model, adapter_config, seed=config.adapter_seed,
                                              device="cuda", dtype=torch.float32)
        for adapter in adapters.values():
            fully_shard(adapter, mesh=mesh.device_mesh["dp_shard_cp"], mp_policy=precision, reshard_after_forward=True)
        if args.tiny:
            train_data, validation_data = _SmokeWindows(config, 700), _SmokeWindows(config, 701)
            data_provenance = {"fixture": "deterministic-disjoint-target-windows-v1"}
        else:
            train_data, validation_data, data_provenance = load_fineweb_windows(data_root, checkpoint, config.sequence_length)
        total_steps = train_data.full_microbatches(config.world_size, config.micro_batch)
        contract = {"training": asdict(config), "adapter": asdict(adapter_config), "tiny": args.tiny,
                    "total_steps": total_steps, "data": data_provenance,
                    "pretrained_metadata_sha256": {name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
                        for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "ARCHLAB_VERIFIED_COPY.json")
                        if (checkpoint / name).exists()} if checkpoint is not None else None,
                    "runtime": {k: v for k, v in provenance.items() if k != "rank_hosts"},
                    "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad]}
        # JSON is the boundary for immutable resume metadata (e.g. NCCL version tuples).
        contract = json.loads(json.dumps(contract, sort_keys=True))
        if dist.get_rank() == 0:
            write_json(args.run_dir / f"manifest-{attempt[0]}.json", {"contract": contract,
                       "recipe": recipe, "checkpoint": str(checkpoint), "data_root": str(data_root),
                       "rank_hosts": provenance["rank_hosts"], "gdn_runtime": gdn_runtime,
                       "train_accounting": train_data.accounting(config.world_size, config.micro_batch),
                       "validation_accounting": validation_data.accounting(config.world_size, config.micro_batch),
                       "resume_from": str(args.resume_from) if args.resume_from else None})
        session = TrainingSession(config=config, model=model, adapters=adapters, mesh=mesh,
                                  train_data=train_data, validation_data=validation_data, run_dir=args.run_dir,
                                  contract=contract, attempt=attempt[0])
        cursor = 0
        if args.resume_from is not None:
            cursor = load_training_checkpoint(args.resume_from, adapters=adapters, optimizer=session.optimizer,
                                               scheduler=session.scheduler, contract=contract)
            emit("fresh_process_restore_pass", cursor=cursor, exact_state_digest=True)

        def handle_signal(signum, frame):
            session.save_requested = True
            session.stop_requested = session.stop_requested or signum in (signal.SIGTERM, signal.SIGINT)

        for signum in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, handle_signal)
        session.record("started", resumed_cursor=cursor, total_steps=total_steps,
                       trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                       training_config=asdict(config))
        session.run(start_cursor=cursor, stop_after=args.stop_after)
    finally:
        if run_lock is not None:
            run_lock.close()
        free_buffer()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

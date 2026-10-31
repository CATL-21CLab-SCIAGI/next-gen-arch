"""Stage-2 end-to-end off-policy forward-KL distillation for Qwen3.5 mechanism switches.

Follows OPAL Stage 2: Stage 1 initializes each replacement mixer independently on frozen
teacher hidden states, leaving cross-layer representation drift and unadapted retained layers.
Stage 2 composes the full student model end-to-end and minimizes token-level forward KL
``KL(p_teacher || p_student)`` over packed document windows, using two-timescale AdamW
learning rates (1.0x peak LR on replacement mixers, ``backbone_lr_ratio`` [default 0.1x]
on retained backbone/mixers/embeddings/head) and memory-efficient vocabulary-chunked backward.

Run: torchrun --nproc_per_node N -m archlab.training.stage2_distill run.json
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from datetime import timedelta
from importlib import metadata
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from archlab.artifacts import sha256_file
from archlab.training.latent_match import (
    MixtureWindows,
    SourceConfig,
    contract_hash,
    lr_at,
    rng_state,
    run_directory,
    teacher_identity,
    text_model,
    tokenizer_identity,
)


@dataclass(frozen=True)
class Stage2Config:
    teacher: str
    stage1_checkpoint: str
    output: str
    variant: Literal["simplicial", "triadic"]
    sources: tuple[SourceConfig, ...]
    sequence_length: int
    batch_tokens: int
    tokens: int
    eval_windows: int
    lr: float = 1.5e-4
    backbone_lr_ratio: float = 0.1
    seed: int = 42
    init_std: float = 0.02
    feature_dim: int = 4
    backend: Literal["kernel", "reference"] = "kernel"
    window_1: int | None = None
    window_2: int | None = None
    microbatch_windows: int = 1
    heldout_fraction: float = 0.1
    split_seed: int = 42
    warmup_fraction: float = 0.03
    final_lr_fraction: float = 0.1
    clip_norm: float = 1.0
    eval_every: int = 50
    log_every: int = 1
    kl_chunk_tokens: int = 128
    gradient_checkpointing: bool = False
    expected_layers: int | None = None

    def __post_init__(self):
        if self.variant not in ("simplicial", "triadic") or self.backend not in ("kernel", "reference"):
            raise ValueError("unknown switch variant/backend")
        for name in (
            "sequence_length",
            "batch_tokens",
            "tokens",
            "eval_windows",
            "microbatch_windows",
            "eval_every",
            "log_every",
            "kl_chunk_tokens",
            "feature_dim",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("window_1", "window_2"):
            val = getattr(self, name)
            if val is not None and (type(val) is not int or val < 1):
                raise ValueError(f"{name} must be None or a positive integer")
        if self.sequence_length < 2 or self.batch_tokens % self.sequence_length or self.tokens % self.batch_tokens:
            raise ValueError("budgets must contain whole sequences and whole optimizer updates")
        if not self.sources or not self.teacher or not self.stage1_checkpoint or not self.output:
            raise ValueError("teacher, stage1_checkpoint, output and sources are required")
        if not 0 < self.heldout_fraction < 1 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("invalid heldout/warmup fraction")
        if not 0 <= self.final_lr_fraction <= 1:
            raise ValueError("invalid final LR fraction")
        for name in ("lr", "clip_norm"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.backbone_lr_ratio) or self.backbone_lr_ratio < 0:
            raise ValueError("backbone_lr_ratio must be finite and nonnegative")
        if not math.isfinite(self.init_std) or self.init_std < 0:
            raise ValueError("init_std must be nonnegative and finite")
        if type(self.seed) is not int or self.seed < 0 or type(self.split_seed) is not int:
            raise ValueError("seed must be a nonnegative integer and split_seed an integer")

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        values["sources"] = tuple(SourceConfig(**source) for source in values["sources"])
        return cls(**values)

    def windows_per_rank(self, world):
        if self.batch_tokens % (self.sequence_length * world):
            raise ValueError("batch_tokens must divide exactly over ranks and sequence_length")
        return self.batch_tokens // (self.sequence_length * world)


class InstalledSwitch(nn.Module):
    """Adapter installing a Qwen3.5 switch directly on a decoder layer for end-to-end training."""

    def __init__(self, switch: nn.Module, *, is_attention: bool):
        super().__init__()
        self.switch = switch
        self.is_attention = is_attention

    def forward(self, *args, **kwargs):
        hidden = kwargs.pop("hidden_states", args[0] if args else None)
        kwargs.pop("attention_mask", None)
        with torch.autocast(
            device_type=hidden.device.type,
            dtype=torch.bfloat16,
            enabled=hidden.device.type == "cuda",
        ):
            output = self.switch(hidden, **kwargs)
        return (output, None) if self.is_attention else output


def causal_lm_parameters(model) -> list[nn.Parameter]:
    """Unique parameters on the causal-LM path (text backbone + output LM head) in deterministic order."""
    seen: set[int] = set()
    ordered: list[nn.Parameter] = []
    head = model.get_output_embeddings()
    modules = (text_model(model),) if head is None else (text_model(model), head)
    for mod in modules:
        for p in mod.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                ordered.append(p)
    return ordered


def install_student_switches(student, layer_types, variant: str, switch_cfg, *, seed: int, stage1_ckpt=None):
    """Replace targeted mixers on ``student`` with independent switch modules and optionally load Stage-1/2 weights."""
    from archlab.architectures import qwen35_switch as switch_lib

    layers = text_model(student).layers
    switches = switch_lib.build_switches(layers, layer_types, variant, switch_cfg, seed=seed)
    for idx, module in switches.items():
        attr_name, orig_mixer = switch_lib.mixer(layers[idx])
        is_attn = hasattr(orig_mixer, "q_proj")
        setattr(layers[idx], attr_name, InstalledSwitch(module, is_attention=is_attn))
    def _load_init():
        use_mmap = isinstance(stage1_ckpt, (str, os.PathLike))
        state = torch.load(stage1_ckpt, map_location="cpu", weights_only=True, mmap=use_mmap)
        fmt = state.get("format")
        if fmt not in (
            "archlab-latent-checkpoint-v1",
            "archlab-stage2-checkpoint-v1",
            "archlab-stage3-checkpoint-v1",
        ):
            raise ValueError("unsupported initialization checkpoint format")
        if set(state["switches"]) != {str(i) for i in switches}:
            raise ValueError("initialization checkpoint switched layers do not match student variant")
        if fmt == "archlab-latent-checkpoint-v1":
            for idx, module in switches.items():
                module.load_state_dict(state["switches"][str(idx)], strict=True)
        else:
            student.load_state_dict(state["student"], strict=True)
        del state
        gc.collect()

    if stage1_ckpt is not None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            rank, world = dist.get_rank(), dist.get_world_size()
            for r in range(world):
                if rank == r:
                    _load_init()
                dist.barrier()
        else:
            _load_init()
    return switches


def build_two_timescale_optimizer(student, switches, *, lr: float, backbone_lr_ratio: float):
    """Partition active causal-LM parameters into switches (1.0x LR) and retained backbone (ratio * LR)."""
    for p in student.parameters():
        p.requires_grad_(False)
    switch_params = [p for idx in sorted(switches) for p in switches[idx].parameters()]
    switch_ids = {id(p) for p in switch_params}
    active_lm_params = causal_lm_parameters(student)
    active_ids = {id(p) for p in active_lm_params}
    backbone_params = [p for p in active_lm_params if id(p) not in switch_ids]
    frozen_non_text = [p for p in student.parameters() if id(p) not in active_ids]
    for p in switch_params:
        p.requires_grad_(True)
    train_backbone = backbone_lr_ratio > 0.0
    for p in backbone_params:
        p.requires_grad_(train_backbone)
    groups = [{"name": "switches", "params": switch_params, "lr": lr, "lr_scale": 1.0}]
    if train_backbone and backbone_params:
        groups.append(
            {
                "name": "backbone",
                "params": backbone_params,
                "lr": lr * backbone_lr_ratio,
                "lr_scale": backbone_lr_ratio,
            }
        )
    trainable = switch_params + (backbone_params if train_backbone else [])
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0)
    return optimizer, trainable, {
        "switch_parameters": sum(p.numel() for p in switch_params),
        "backbone_parameters": sum(p.numel() for p in backbone_params) if train_backbone else 0,
        "frozen_non_text_parameters": sum(p.numel() for p in frozen_non_text),
        "total_trainable_parameters": sum(p.numel() for p in trainable),
    }


def chunked_forward_kl_step(
    teacher,
    student,
    optimizer,
    trainable_params,
    ids: torch.Tensor,
    *,
    microbatch_windows: int = 1,
    chunk_tokens: int = 128,
    clip_norm: float = 1.0,
):
    """End-to-end forward-KL update using vocabulary-chunked autograd to avoid [B, L, V] tensor allocation."""
    if not trainable_params or ids.ndim != 2 or ids.shape[1] < 2 or microbatch_windows < 1 or chunk_tokens < 1:
        raise ValueError("invalid inputs for chunked forward-KL training step")
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("teacher parameters must remain frozen")
    if {id(p) for p in trainable_params} & {id(p) for p in teacher.parameters()}:
        raise ValueError("student and teacher must not share parameter tensors")

    device = ids.device
    use_amp = device.type == "cuda"
    total_pred_tokens = ids.shape[0] * (ids.shape[1] - 1)
    optimizer.zero_grad(set_to_none=True)

    kl_total = ids.new_zeros((), dtype=torch.float32)
    ce_student_total = ids.new_zeros((), dtype=torch.float32)
    ce_teacher_total = ids.new_zeros((), dtype=torch.float32)

    teacher_head = teacher.get_output_embeddings()
    student_head = student.get_output_embeddings()

    for start_b in range(0, ids.shape[0], microbatch_windows):
        batch = ids[start_b : start_b + microbatch_windows]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                t_hidden = text_model(teacher)(input_ids=batch, use_cache=False).last_hidden_state

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            s_hidden = text_model(student)(input_ids=batch, use_cache=False).last_hidden_state

        s_pred = s_hidden[:, :-1]
        s_leaf = s_pred.detach().requires_grad_(True)
        pred_len = batch.shape[1] - 1

        for tok_start in range(0, pred_len, chunk_tokens):
            tok_end = min(tok_start + chunk_tokens, pred_len)
            targets = batch[:, tok_start + 1 : tok_end + 1].reshape(-1)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    t_logits = teacher_head(t_hidden[:, tok_start:tok_end])
                t_logp = t_logits.float().log_softmax(dim=-1)
                t_prob = t_logp.exp()
                ce_teacher_total += F.nll_loss(t_logp.flatten(0, 1), targets, reduction="sum")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                s_logits = student_head(s_leaf[:, tok_start:tok_end])
            s_logp = s_logits.float().log_softmax(dim=-1)
            with torch.no_grad():
                ce_student_total += F.nll_loss(s_logp.detach().flatten(0, 1), targets, reduction="sum")

            chunk_kl_sum = (t_prob * (t_logp - s_logp)).sum()
            if not torch.isfinite(chunk_kl_sum):
                raise FloatingPointError("nonfinite chunk forward-KL loss")
            (chunk_kl_sum / total_pred_tokens).backward()
            kl_total += chunk_kl_sum.detach() / total_pred_tokens

        s_pred.backward(s_leaf.grad)

    if any(p.grad is None for p in trainable_params):
        raise RuntimeError("a trainable student parameter received no gradient")

    if dist.is_initialized():
        world = dist.get_world_size()
        for p in trainable_params:
            dist.all_reduce(p.grad)
            p.grad.div_(world)
        for metric in (kl_total, ce_student_total, ce_teacher_total):
            dist.all_reduce(metric)
            metric.div_(world)

    norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable_params):
        raise FloatingPointError("optimizer produced nonfinite student parameters")

    return {
        "train_kl": float(kl_total),
        "train_ce_student": float(ce_student_total / total_pred_tokens),
        "train_ce_teacher": float(ce_teacher_total / total_pred_tokens),
        "grad_norm": float(norm),
    }


@torch.no_grad()
def evaluate_composed(teacher, student, windows: torch.Tensor, device, *, chunk_tokens: int = 128):
    """Evaluate composed end-to-end student forward-KL, reverse-KL, and next-token cross-entropy."""
    if not len(windows) or chunk_tokens < 1:
        raise ValueError("evaluation requires windows and a positive chunk size")
    use_amp = device.type == "cuda"
    f_kl = r_kl = ce_t = ce_s = 0.0
    tokens = 0
    t_head = teacher.get_output_embeddings()
    s_head = student.get_output_embeddings()

    was_training = student.training
    student.eval()
    try:
        for window in windows:
            ids = window[None].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                t_hidden = text_model(teacher)(input_ids=ids, use_cache=False).last_hidden_state
                s_hidden = text_model(student)(input_ids=ids, use_cache=False).last_hidden_state
            for start in range(0, ids.shape[1] - 1, chunk_tokens):
                end = min(start + chunk_tokens, ids.shape[1] - 1)
                targets = ids[:, start + 1 : end + 1].reshape(-1)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    t_logp = t_head(t_hidden[:, start:end]).float().log_softmax(-1)
                    s_logp = s_head(s_hidden[:, start:end]).float().log_softmax(-1)
                f_kl += float((t_logp.exp() * (t_logp - s_logp)).sum())
                r_kl += float((s_logp.exp() * (s_logp - t_logp)).sum())
                ce_t += float(F.nll_loss(t_logp.flatten(0, 1), targets, reduction="sum"))
                ce_s += float(F.nll_loss(s_logp.flatten(0, 1), targets, reduction="sum"))
                tokens += targets.numel()
    finally:
        student.train(was_training)

    res = {
        "kl": f_kl / tokens,
        "reverse_kl": r_kl / tokens,
        "ce_teacher": ce_t / tokens,
        "ce_student": ce_s / tokens,
        "eval_prediction_tokens": tokens,
    }
    if not all(math.isfinite(res[k]) for k in ("kl", "reverse_kl", "ce_teacher", "ce_student")):
        raise FloatingPointError("nonfinite Stage-2 evaluation metrics")
    return res


def runtime_versions() -> dict[str, str | None]:
    return {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "transformers": metadata.version("transformers"),
        "numpy": np.__version__,
        "triton": metadata.version("triton"),
    }


def atomic_torch_save(destination: Path, state: dict) -> Path:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def restore_rank_rng(saved: dict, rng, device: torch.device) -> None:
    rng.bit_generator.state = saved["numpy"]
    torch.set_rng_state(saved["torch"].to(device="cpu", dtype=torch.uint8))
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"].to(device="cpu", dtype=torch.uint8), device)


def save_stage2_checkpoint(output, step, student, switches, optimizer, rngs, fingerprint):
    destination = Path(output) / "checkpoints" / f"step-{step:08d}.pt"
    return atomic_torch_save(
        destination,
        {
            "format": "archlab-stage2-checkpoint-v1",
            "contract_sha256": fingerprint,
            "step": step,
            "student": student.state_dict(),
            "switches": {str(i): m.state_dict() for i, m in switches.items()},
            "optimizer": optimizer.state_dict(),
            "rngs": rngs,
        },
    )


def _load_stage2_single_rank(path, student, optimizer, rng, fingerprint, *, rank=0, world=1, device=None) -> int:
    device = torch.device("cpu") if device is None else device
    use_mmap = isinstance(path, (str, os.PathLike))
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=use_mmap)
    if state.get("format") != "archlab-stage2-checkpoint-v1" or state.get("contract_sha256") != fingerprint:
        raise ValueError("checkpoint belongs to another Stage-2 experiment contract")
    if len(state["rngs"]) != world:
        raise ValueError("checkpoint rank count differs")
    student.load_state_dict(state["student"], strict=True)
    del state["student"]
    optimizer.load_state_dict(state["optimizer"])
    del state["optimizer"]
    restore_rank_rng(state["rngs"][rank], rng, device)
    step = int(state["step"])
    del state
    gc.collect()
    return step


def load_stage2_checkpoint(path, student, optimizer, rng, fingerprint, *, rank=0, world=1, device=None):
    if dist.is_available() and dist.is_initialized() and world > 1:
        step = 0
        for r in range(world):
            if rank == r:
                step = _load_stage2_single_rank(
                    path, student, optimizer, rng, fingerprint, rank=rank, world=world, device=device
                )
            dist.barrier()
        return step
    return _load_stage2_single_rank(path, student, optimizer, rng, fingerprint, rank=rank, world=world, device=device)


def main(argv=None):
    from transformers import AutoConfig, AutoModelForCausalLM

    from archlab.architectures import qwen35_switch as switch_lib
    from archlab.architectures.triadic_attention import official_dependency_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--resume", type=Path, help="explicit latest checkpoint in the same output directory")
    parser.add_argument("--stop-after", type=int, help="stop and checkpoint at this total optimizer-step count")
    args = parser.parse_args(argv)
    run = Stage2Config.read(args.run)

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
        count = run.windows_per_rank(world)
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        torch.cuda.set_device(device)
        torch.manual_seed(run.seed)

        identity = tokenizer_identity(run.teacher)
        pinned = [
            (teacher_identity(run.teacher), sha256_file(run.stage1_checkpoint)) if rank == 0 else None
        ]
        if distributed:
            dist.broadcast_object_list(pinned, src=0)
        teacher_files, stage1_sha = pinned[0]

        hf_config = AutoConfig.from_pretrained(run.teacher, local_files_only=True, trust_remote_code=False)
        config = hf_config.get_text_config()
        data = MixtureWindows(
            run.sources,
            run.sequence_length,
            tokenizer_files=identity,
            vocab_size=config.vocab_size,
            heldout_fraction=run.heldout_fraction,
            split_seed=run.split_seed,
        )
        rng = np.random.default_rng(np.random.SeedSequence([run.seed, rank]))
        heldout = data.sample(np.random.default_rng(run.seed + 7), run.eval_windows, heldout=True)

        teacher = AutoModelForCausalLM.from_pretrained(
            run.teacher,
            dtype=torch.bfloat16,
            device_map={"": device},
            local_files_only=True,
            trust_remote_code=False,
        )
        teacher.eval().requires_grad_(False)
        teacher.config.use_cache = False

        # Keep FP32 master weights on student so small backbone LR updates (1.5e-5) never underflow BF16.
        student = AutoModelForCausalLM.from_pretrained(
            run.teacher,
            dtype=torch.float32,
            device_map={"": "cpu"},
            local_files_only=True,
            trust_remote_code=False,
        )
        student.config.use_cache = False
        switch_cfg = switch_lib.SwitchConfig(
            init_std=run.init_std,
            feature_dim=run.feature_dim,
            backend=run.backend,
            window_1=run.window_1,
            window_2=run.window_2,
        )
        switches = install_student_switches(
            student,
            config.layer_types,
            run.variant,
            switch_cfg,
            seed=run.seed,
            stage1_ckpt=None if args.resume else run.stage1_checkpoint,
        )
        expected = run.expected_layers if run.expected_layers is not None else {"simplicial": 8, "triadic": 24}[run.variant]
        if len(switches) != expected:
            raise ValueError(f"expected {expected} switched layers, got {len(switches)}")
        student.to(device=device, dtype=torch.float32)
        if run.gradient_checkpointing:
            student.gradient_checkpointing_enable()
        student.train()

        optimizer, trainable_params, param_counts = build_two_timescale_optimizer(
            student, switches, lr=run.lr, backbone_lr_ratio=run.backbone_lr_ratio
        )

        architecture_dir = Path(switch_lib.__file__).parent
        source_paths = (
            Path(__file__),
            Path(__file__).parent / "latent_match.py",
            *(
                architecture_dir / name
                for name in (
                    "qwen35_switch.py",
                    "simplicial_attention.py",
                    "simplicial_kernels.py",
                    "triadic_attention.py",
                )
            ),
        )
        versions = runtime_versions()
        contract = {
            "format": "archlab-stage2-run-v1",
            "config": json.loads(json.dumps(asdict(run))),
            "world": world,
            "switched_layers": sorted(switches),
            "parameter_counts": param_counts,
            "teacher_files_sha256": teacher_files,
            "stage1_checkpoint_sha256": stage1_sha,
            "tokenizer_files": identity,
            "versions": versions,
            "data": data.receipts,
            "source_sha256": {p.name: sha256_file(p) for p in source_paths},
            "eval_windows_sha256": hashlib.sha256(heldout.numpy().tobytes()).hexdigest(),
            "objective": "forward_kl_teacher_to_student",
        }
        if run.variant == "triadic" and run.backend == "kernel":
            contract["triadic_dependency"] = official_dependency_contract()
        fingerprint = contract_hash(contract)

        with ExitStack() as stack:
            output = Path(run.output)
            if rank == 0:
                stack.enter_context(run_directory(output, contract, args.resume))
            if distributed:
                dist.barrier()
            start_step = (
                load_stage2_checkpoint(
                    args.resume, student, optimizer, rng, fingerprint, rank=rank, world=world, device=device
                )
                if args.resume
                else 0
            )
            steps = run.tokens // run.batch_tokens
            if not 0 <= start_step <= steps:
                raise ValueError("checkpoint step exceeds this run's budget")
            stop_step = steps if args.stop_after is None else args.stop_after
            if not 0 < stop_step <= steps or stop_step < start_step:
                raise ValueError("stop-after must be within the run budget and not before the resumed step")

            metrics = None
            if rank == 0:
                attempts = output / "attempts"
                attempts.mkdir(exist_ok=True)
                metrics = stack.enter_context((attempts / f"{uuid.uuid4().hex}.jsonl").open("x"))

            def log(record):
                if metrics:
                    metrics.write(json.dumps({"at": time.time(), **record}, allow_nan=False) + "\n")
                    metrics.flush()

            if rank == 0:
                log(
                    {
                        "step": start_step,
                        "tokens": start_step * run.batch_tokens,
                        "resumed": bool(args.resume),
                        **evaluate_composed(teacher, student, heldout, device, chunk_tokens=run.kl_chunk_tokens),
                    }
                )
            if distributed:
                dist.barrier()

            started = time.monotonic()
            for step in range(start_step, stop_step):
                base_lr = lr_at(step, steps, run.lr, run.warmup_fraction, run.final_lr_fraction)
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * group["lr_scale"]
                ids = data.sample(rng, count).to(device)
                result = chunked_forward_kl_step(
                    teacher,
                    student,
                    optimizer,
                    trainable_params,
                    ids,
                    microbatch_windows=run.microbatch_windows,
                    chunk_tokens=run.kl_chunk_tokens,
                    clip_norm=run.clip_norm,
                )
                done = (step + 1) * run.batch_tokens
                if (step + 1) % run.log_every == 0:
                    log(
                        {
                            "step": step + 1,
                            "tokens": done,
                            "lr_switches": base_lr,
                            "lr_backbone": base_lr * run.backbone_lr_ratio,
                            **result,
                            "tokens_per_second": (step + 1 - start_step)
                            * run.batch_tokens
                            / (time.monotonic() - started),
                        }
                    )
                if (step + 1) % run.eval_every == 0 or step + 1 == stop_step:
                    if rank == 0:
                        log(
                            {
                                "step": step + 1,
                                "tokens": done,
                                **evaluate_composed(
                                    teacher, student, heldout, device, chunk_tokens=run.kl_chunk_tokens
                                ),
                            }
                        )
                    local_rng = rng_state(rng, device)
                    states = [None] * world if rank == 0 else None
                    if distributed:
                        dist.gather_object(local_rng, states, dst=0)
                    else:
                        states = [local_rng]
                    if rank == 0:
                        save_stage2_checkpoint(output, step + 1, student, switches, optimizer, states, fingerprint)
                    if distributed:
                        dist.barrier()

            elapsed = time.monotonic() - started
            log(
                {
                    "event": "stopped" if stop_step < steps else "complete",
                    "step": stop_step,
                    "tokens": stop_step * run.batch_tokens,
                    "attempt_seconds": elapsed,
                    "attempt_tokens_per_second": (stop_step - start_step) * run.batch_tokens / max(elapsed, 1e-9),
                }
            )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

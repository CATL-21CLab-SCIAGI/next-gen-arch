"""Stage-3 on-policy reverse-KL distillation for Qwen3.5 mechanism switches.

Follows OPAL Stage 3: teacher-forced off-policy recovery (Stages 1 and 2) aligns
representations and short/medium-context retrieval, but long-form generation and
multi-step reasoning suffer from exposure bias once the morphed student conditions
on its own generated trajectories. Stage 3 chains from a Stage-2 checkpoint, samples
prompts from reasoning/instruction corpora, rolls out student continuations on-policy
at temperature ``T`` (default 1.0), and minimizes token-level reverse KL
``KL(p_student || p_teacher)`` strictly over generated positions (with optional
off-policy forward-KL anchoring ``anchor_kl_weight`` to prevent backbone drift), using:

- Vocabulary-chunked reverse-KL + anchor forward-KL backward (avoiding ``[B, L, V]`` tensors),
- Token-parameterized Warmup-Stable-Decay (WSD) learning-rate schedule,
- Non-divergent OPAL generation-length curriculum (doubling ``gen_tokens`` from
  ``min_gen_tokens`` up to ``max_gen_tokens`` only when smoothed reverse KL plateaus
  without diverging above the phase-entry baseline).

Run: torchrun --nproc_per_node N -m archlab.training.stage3_onpolicy run.json
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
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from archlab.artifacts import sha256_file
from archlab.training.latent_match import (
    MixtureWindows,
    SourceConfig,
    contract_hash,
    rng_state,
    run_directory,
    teacher_identity,
    text_model,
    tokenizer_identity,
)
from archlab.training.stage2_distill import (
    atomic_torch_save,
    build_two_timescale_optimizer,
    evaluate_composed,
    install_student_switches,
    restore_rank_rng,
    runtime_versions,
)


def wsd_lr_at(
    tokens_after_step: int,
    total_tokens: int,
    peak_lr: float,
    warmup_fraction: float = 0.03,
    decay_fraction: float = 0.10,
    final_lr_fraction: float = 0.10,
) -> float:
    """Token-parameterized Warmup-Stable-Decay (WSD) learning-rate schedule."""
    if total_tokens <= 0 or tokens_after_step < 0:
        raise ValueError("token counts must be positive total and nonnegative progress")
    if not (0 <= warmup_fraction < 1 and 0 <= decay_fraction < 1 and warmup_fraction + decay_fraction <= 1):
        raise ValueError("invalid WSD warmup/decay fractions")
    if not 0 <= final_lr_fraction <= 1 or not math.isfinite(peak_lr) or peak_lr <= 0:
        raise ValueError("invalid WSD peak/final learning rate")

    progress = min(float(total_tokens), float(max(1, tokens_after_step)))
    warmup_tokens = total_tokens * warmup_fraction
    decay_start = total_tokens * (1.0 - decay_fraction)

    if warmup_tokens > 0.0 and progress < warmup_tokens:
        return peak_lr * (progress / warmup_tokens)
    if decay_fraction > 0.0 and progress > decay_start:
        decay_span = max(1e-9, total_tokens - decay_start)
        ratio = min(1.0, max(0.0, (progress - decay_start) / decay_span))
        return peak_lr * (1.0 - ratio * (1.0 - final_lr_fraction))
    return peak_lr


@dataclass
class CurriculumState:
    """OPAL generation-length curriculum doubling horizon on non-divergent smoothed reverse-KL plateau."""

    gen_tokens: int
    phase: int = 0
    smoothed_kl: float | None = None
    best_smoothed_kl: float | None = None
    phase_initial_kl: float | None = None
    stale_steps: int = 0

    def update(
        self,
        step_reverse_kl: float,
        *,
        max_gen_tokens: int,
        ema_alpha: float = 0.2,
        improvement_rel: float = 0.02,
        plateau_steps: int = 30,
        max_divergence_rel: float = 0.05,
    ) -> bool:
        if not math.isfinite(step_reverse_kl):
            raise FloatingPointError("nonfinite reverse KL in curriculum update")
        if self.phase_initial_kl is None:
            self.phase_initial_kl = step_reverse_kl if self.smoothed_kl is None else self.smoothed_kl
        if self.smoothed_kl is None:
            self.smoothed_kl = step_reverse_kl
        else:
            self.smoothed_kl = (1.0 - ema_alpha) * self.smoothed_kl + ema_alpha * step_reverse_kl

        if self.best_smoothed_kl is None or self.smoothed_kl <= self.best_smoothed_kl * (1.0 - improvement_rel):
            self.best_smoothed_kl = self.smoothed_kl
            self.stale_steps = 0
        else:
            self.stale_steps += 1

        ceiling = max(self.phase_initial_kl * (1.0 + max_divergence_rel), self.phase_initial_kl + 1e-6)
        if self.stale_steps >= plateau_steps and self.gen_tokens < max_gen_tokens and self.smoothed_kl <= ceiling:
            self.gen_tokens = min(max_gen_tokens, self.gen_tokens * 2)
            self.phase += 1
            self.smoothed_kl = None
            self.best_smoothed_kl = None
            self.phase_initial_kl = None
            self.stale_steps = 0
            return True
        return False


def _curriculum_horizons(min_gen_tokens: int, max_gen_tokens: int) -> tuple[int, ...]:
    if min_gen_tokens < 1 or max_gen_tokens < min_gen_tokens:
        raise ValueError("invalid min_gen_tokens / max_gen_tokens")
    horizons = [min_gen_tokens]
    cur = min_gen_tokens
    while cur < max_gen_tokens:
        cur *= 2
        if cur > max_gen_tokens:
            raise ValueError("max_gen_tokens must be a power-of-two multiple of min_gen_tokens")
        horizons.append(cur)
    return tuple(horizons)


@dataclass(frozen=True)
class Stage3Config:
    teacher: str
    init_checkpoint: str
    output: str
    variant: Literal["simplicial", "triadic"]
    sources: tuple[SourceConfig, ...]
    prompt_length: int
    min_gen_tokens: int
    max_gen_tokens: int
    gen_batch_tokens: int
    tokens: int
    eval_windows: int
    lr: float = 1e-5
    backbone_lr_ratio: float = 0.1
    anchor_kl_weight: float = 0.0
    temperature: float = 1.0
    seed: int = 42
    init_std: float = 0.02
    feature_dim: int = 4
    backend: Literal["kernel", "reference"] = "kernel"
    window_1: int | None = None
    window_2: int | None = None
    microbatch_prompts: int = 2
    heldout_fraction: float = 0.1
    split_seed: int = 42
    warmup_fraction: float = 0.03
    decay_fraction: float = 0.10
    final_lr_fraction: float = 0.10
    ema_alpha: float = 0.2
    improvement_rel: float = 0.02
    max_divergence_rel: float = 0.05
    plateau_steps: int = 30
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
            "prompt_length",
            "min_gen_tokens",
            "max_gen_tokens",
            "gen_batch_tokens",
            "tokens",
            "eval_windows",
            "microbatch_prompts",
            "plateau_steps",
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
        if self.prompt_length < 2:
            raise ValueError("prompt_length must be at least 2")
        horizons = _curriculum_horizons(self.min_gen_tokens, self.max_gen_tokens)
        if any(self.gen_batch_tokens % h for h in horizons) or self.tokens % self.gen_batch_tokens:
            raise ValueError("gen_batch_tokens must divide tokens and be divisible by every curriculum horizon")
        if self.variant == "triadic" and self.backend == "kernel":
            if self.prompt_length % 64 or any((self.prompt_length + h) % 64 for h in horizons):
                raise ValueError("official Triadic kernel requires prompt_length and sequence lengths divisible by 64")
        if not self.sources or not self.teacher or not self.init_checkpoint or not self.output:
            raise ValueError("teacher, init_checkpoint, output and sources are required")
        if not 0 < self.heldout_fraction < 1:
            raise ValueError("invalid heldout_fraction")
        wsd_lr_at(
            self.gen_batch_tokens,
            self.tokens,
            self.lr,
            self.warmup_fraction,
            self.decay_fraction,
            self.final_lr_fraction,
        )
        if not math.isfinite(self.clip_norm) or self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive and finite")
        if not math.isfinite(self.backbone_lr_ratio) or self.backbone_lr_ratio < 0:
            raise ValueError("backbone_lr_ratio must be finite and nonnegative")
        if not math.isfinite(self.anchor_kl_weight) or not 0.0 <= self.anchor_kl_weight < 1.0:
            raise ValueError("anchor_kl_weight must be in [0.0, 1.0)")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not (0 < self.ema_alpha <= 1 and 0 <= self.improvement_rel < 1):
            raise ValueError("invalid curriculum ema_alpha or improvement_rel")
        if not math.isfinite(self.max_divergence_rel) or self.max_divergence_rel < 0:
            raise ValueError("max_divergence_rel must be finite and nonnegative")
        if not math.isfinite(self.init_std) or self.init_std < 0:
            raise ValueError("init_std must be nonnegative and finite")
        if type(self.seed) is not int or self.seed < 0 or type(self.split_seed) is not int:
            raise ValueError("seed must be a nonnegative integer and split_seed an integer")

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        values["sources"] = tuple(SourceConfig(**source) for source in values["sources"])
        return cls(**values)

    def prompts_per_rank(self, world: int, gen_tokens: int) -> int:
        if self.gen_batch_tokens % (gen_tokens * world):
            raise ValueError("gen_batch_tokens must divide evenly over world * gen_tokens")
        return self.gen_batch_tokens // (gen_tokens * world)

    def check_world_curriculum(self, world: int) -> None:
        for h in _curriculum_horizons(self.min_gen_tokens, self.max_gen_tokens):
            self.prompts_per_rank(world, h)


@torch.no_grad()
def rollout_on_policy(
    student,
    prompts: torch.Tensor,
    gen_tokens: int,
    *,
    temperature: float = 1.0,
    microbatch_prompts: int = 4,
    generator: torch.Generator | None = None,
    pad_to: int | None = None,
) -> torch.Tensor:
    """Autoregressively sample ``gen_tokens`` student continuations with chunk-aligned static causal lengths."""
    if prompts.ndim != 2 or prompts.shape[0] < 1 or prompts.shape[1] < 1 or gen_tokens < 1 or microbatch_prompts < 1:
        raise ValueError("invalid inputs for on-policy student rollout")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")

    device = prompts.device
    use_amp = device.type == "cuda"
    pad_multiple = (64 if device.type == "cuda" else 1) if pad_to is None else pad_to
    if type(pad_multiple) is not int or pad_multiple < 1:
        raise ValueError("pad_to must be a positive integer")

    prompt_len = prompts.shape[1]
    total_len = prompt_len + gen_tokens
    max_padded = ((total_len + pad_multiple - 1) // pad_multiple) * pad_multiple

    head = student.get_output_embeddings()
    was_training = student.training
    student.eval()
    outputs: list[torch.Tensor] = []
    try:
        for start in range(0, prompts.shape[0], microbatch_prompts):
            batch_p = prompts[start : start + microbatch_prompts]
            buf = torch.zeros((batch_p.shape[0], max_padded), dtype=prompts.dtype, device=device)
            buf[:, :prompt_len] = batch_p
            for k in range(gen_tokens):
                cur_len = prompt_len + k
                step_len = ((cur_len + pad_multiple - 1) // pad_multiple) * pad_multiple
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    hidden = text_model(student)(input_ids=buf[:, :step_len], use_cache=False).last_hidden_state
                    last_logits = head(hidden[:, cur_len - 1, :])
                logits = last_logits.float()
                if temperature == 0.0:
                    next_tok = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_tok = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
                buf[:, cur_len] = next_tok
            outputs.append(buf[:, :total_len].clone())
    finally:
        student.train(was_training)
    return torch.cat(outputs, dim=0)


def chunked_reverse_kl_step(
    teacher,
    student,
    optimizer,
    trainable_params,
    sequences: torch.Tensor,
    *,
    prompt_length: int,
    microbatch_prompts: int = 2,
    chunk_tokens: int = 128,
    clip_norm: float = 1.0,
    anchor_windows: torch.Tensor | None = None,
    anchor_kl_weight: float = 0.0,
    anchor_microbatch_windows: int = 1,
):
    """Reverse-KL ``KL(p_student || p_teacher)`` update over generated positions, with optional forward-KL anchor."""
    if not trainable_params or sequences.ndim != 2 or microbatch_prompts < 1 or chunk_tokens < 1:
        raise ValueError("invalid inputs for chunked reverse-KL training step")
    if not math.isfinite(anchor_kl_weight) or not 0.0 <= anchor_kl_weight < 1.0:
        raise ValueError("anchor_kl_weight must be in [0.0, 1.0)")
    use_anchor = anchor_kl_weight > 0.0
    if use_anchor and (
        anchor_windows is None
        or anchor_windows.ndim != 2
        or anchor_windows.shape[0] < 1
        or anchor_windows.shape[1] < 2
        or anchor_microbatch_windows < 1
    ):
        raise ValueError("valid 2D anchor_windows required when anchor_kl_weight > 0")
    gen_tokens = sequences.shape[1] - prompt_length
    if prompt_length < 1 or gen_tokens < 1:
        raise ValueError("sequences must contain at least one prompt token and one generated token")
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("teacher parameters must remain frozen")
    if {id(p) for p in trainable_params} & {id(p) for p in teacher.parameters()}:
        raise ValueError("student and teacher must not share parameter tensors")

    device = sequences.device
    use_amp = device.type == "cuda"
    total_gen_tokens = sequences.shape[0] * gen_tokens
    onpolicy_scale = (1.0 - anchor_kl_weight) if use_anchor else 1.0
    optimizer.zero_grad(set_to_none=True)

    r_kl_total = sequences.new_zeros((), dtype=torch.float32)
    f_kl_total = sequences.new_zeros((), dtype=torch.float32)
    ent_total = sequences.new_zeros((), dtype=torch.float32)
    teacher_nll_total = sequences.new_zeros((), dtype=torch.float32)
    anchor_f_kl_total = sequences.new_zeros((), dtype=torch.float32)

    teacher_head = teacher.get_output_embeddings()
    student_head = student.get_output_embeddings()

    pred_start = prompt_length - 1
    pred_end = prompt_length + gen_tokens - 1

    for start_b in range(0, sequences.shape[0], microbatch_prompts):
        batch = sequences[start_b : start_b + microbatch_prompts]
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                t_hidden = text_model(teacher)(input_ids=batch, use_cache=False).last_hidden_state
            t_gen_hidden = t_hidden[:, pred_start:pred_end]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            s_hidden = text_model(student)(input_ids=batch, use_cache=False).last_hidden_state

        s_pred = s_hidden[:, pred_start:pred_end]
        s_leaf = s_pred.detach().requires_grad_(True)

        for tok_start in range(0, gen_tokens, chunk_tokens):
            tok_end = min(tok_start + chunk_tokens, gen_tokens)
            sampled_targets = batch[:, prompt_length + tok_start : prompt_length + tok_end].reshape(-1)
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    t_logits = teacher_head(t_gen_hidden[:, tok_start:tok_end])
                t_logp = t_logits.float().log_softmax(dim=-1)
                teacher_nll_total += F.nll_loss(t_logp.flatten(0, 1), sampled_targets, reduction="sum")

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                s_logits = student_head(s_leaf[:, tok_start:tok_end])
            s_logp = s_logits.float().log_softmax(dim=-1)
            s_prob = s_logp.exp()

            chunk_r_kl_sum = (s_prob * (s_logp - t_logp)).sum()
            if not torch.isfinite(chunk_r_kl_sum):
                raise FloatingPointError("nonfinite chunk reverse-KL loss")
            ((onpolicy_scale * chunk_r_kl_sum) / total_gen_tokens).backward()

            with torch.no_grad():
                r_kl_total += chunk_r_kl_sum.detach() / total_gen_tokens
                t_prob = t_logp.exp()
                f_kl_total += (t_prob * (t_logp - s_logp.detach())).sum() / total_gen_tokens
                ent_total += -(s_prob.detach() * s_logp.detach()).sum() / total_gen_tokens

        s_pred.backward(s_leaf.grad)

    if use_anchor and anchor_windows is not None:
        anchor_ids = anchor_windows.to(device)
        anchor_pred_len = anchor_ids.shape[1] - 1
        total_anchor_tokens = anchor_ids.shape[0] * anchor_pred_len
        for start_b in range(0, anchor_ids.shape[0], anchor_microbatch_windows):
            batch = anchor_ids[start_b : start_b + anchor_microbatch_windows]
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    ta_hidden = text_model(teacher)(input_ids=batch, use_cache=False).last_hidden_state
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                sa_hidden = text_model(student)(input_ids=batch, use_cache=False).last_hidden_state
            a_pred = sa_hidden[:, :-1]
            a_leaf = a_pred.detach().requires_grad_(True)
            for tok_start in range(0, anchor_pred_len, chunk_tokens):
                tok_end = min(tok_start + chunk_tokens, anchor_pred_len)
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                        ta_logits = teacher_head(ta_hidden[:, tok_start:tok_end])
                    ta_logp = ta_logits.float().log_softmax(dim=-1)
                    ta_prob = ta_logp.exp()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    sa_logits = student_head(a_leaf[:, tok_start:tok_end])
                sa_logp = sa_logits.float().log_softmax(dim=-1)
                chunk_f_kl_sum = (ta_prob * (ta_logp - sa_logp)).sum()
                if not torch.isfinite(chunk_f_kl_sum):
                    raise FloatingPointError("nonfinite chunk anchor forward-KL loss")
                ((anchor_kl_weight * chunk_f_kl_sum) / total_anchor_tokens).backward()
                anchor_f_kl_total += chunk_f_kl_sum.detach() / total_anchor_tokens
            a_pred.backward(a_leaf.grad)

    if any(p.grad is None for p in trainable_params):
        raise RuntimeError("a trainable student parameter received no gradient")

    if dist.is_initialized():
        world = dist.get_world_size()
        for p in trainable_params:
            dist.all_reduce(p.grad)
            p.grad.div_(world)
        metrics = (
            (r_kl_total, f_kl_total, ent_total, teacher_nll_total, anchor_f_kl_total)
            if use_anchor
            else (r_kl_total, f_kl_total, ent_total, teacher_nll_total)
        )
        for metric in metrics:
            dist.all_reduce(metric)
            metric.div_(world)

    norm = torch.nn.utils.clip_grad_norm_(trainable_params, clip_norm, error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in trainable_params):
        raise FloatingPointError("optimizer produced nonfinite student parameters")

    out = {
        "train_reverse_kl": float(r_kl_total),
        "train_forward_kl": float(f_kl_total),
        "train_student_entropy": float(ent_total),
        "train_teacher_nll_on_rollout": float(teacher_nll_total / total_gen_tokens),
        "grad_norm": float(norm),
    }
    if use_anchor:
        out["train_anchor_forward_kl"] = float(anchor_f_kl_total)
    return out


@torch.no_grad()
def evaluate_on_policy(
    teacher,
    student,
    heldout_windows: torch.Tensor,
    *,
    prompt_length: int,
    gen_tokens: int,
    device: torch.device,
    temperature: float = 0.0,
    microbatch_prompts: int = 2,
    chunk_tokens: int = 128,
):
    """Evaluate both teacher-forced held-out windows and on-policy student continuations."""
    composed = evaluate_composed(teacher, student, heldout_windows, device, chunk_tokens=chunk_tokens)
    prompts = heldout_windows[:, :prompt_length].to(device)
    gen = torch.Generator(device=device.type)
    gen.manual_seed(12345)
    rollouts = rollout_on_policy(
        student,
        prompts,
        gen_tokens,
        temperature=temperature,
        microbatch_prompts=microbatch_prompts,
        generator=gen if temperature > 0.0 else None,
    )

    use_amp = device.type == "cuda"
    t_head = teacher.get_output_embeddings()
    s_head = student.get_output_embeddings()
    pred_start = prompt_length - 1
    pred_end = prompt_length + gen_tokens - 1

    r_kl = f_kl = ent = t_nll = 0.0
    total = 0
    was_training = student.training
    student.eval()
    try:
        for seq in rollouts:
            ids = seq[None]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                t_h = text_model(teacher)(input_ids=ids, use_cache=False).last_hidden_state[:, pred_start:pred_end]
                s_h = text_model(student)(input_ids=ids, use_cache=False).last_hidden_state[:, pred_start:pred_end]
            for start in range(0, gen_tokens, chunk_tokens):
                end = min(start + chunk_tokens, gen_tokens)
                targets = ids[:, prompt_length + start : prompt_length + end].reshape(-1)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    t_logp = t_head(t_h[:, start:end]).float().log_softmax(-1)
                    s_logp = s_head(s_h[:, start:end]).float().log_softmax(-1)
                s_prob = s_logp.exp()
                r_kl += float((s_prob * (s_logp - t_logp)).sum())
                f_kl += float((t_logp.exp() * (t_logp - s_logp)).sum())
                ent += float(-(s_prob * s_logp).sum())
                t_nll += float(F.nll_loss(t_logp.flatten(0, 1), targets, reduction="sum"))
                total += targets.numel()
    finally:
        student.train(was_training)

    res = {
        **composed,
        "onpolicy_reverse_kl": r_kl / total,
        "onpolicy_forward_kl": f_kl / total,
        "onpolicy_student_entropy": ent / total,
        "onpolicy_teacher_nll": t_nll / total,
        "onpolicy_eval_tokens": total,
    }
    if not all(
        math.isfinite(res[k])
        for k in ("onpolicy_reverse_kl", "onpolicy_forward_kl", "onpolicy_student_entropy", "onpolicy_teacher_nll")
    ):
        raise FloatingPointError("nonfinite Stage-3 on-policy evaluation metrics")
    return res


def save_stage3_checkpoint(
    output, step: int, consumed_tokens: int, curriculum: CurriculumState, student, switches, optimizer, rngs, fingerprint
):
    destination = Path(output) / "checkpoints" / f"step-{step:08d}.pt"
    return atomic_torch_save(
        destination,
        {
            "format": "archlab-stage3-checkpoint-v1",
            "contract_sha256": fingerprint,
            "step": step,
            "consumed_tokens": consumed_tokens,
            "curriculum": asdict(curriculum),
            "student": student.state_dict(),
            "switches": {str(i): m.state_dict() for i, m in switches.items()},
            "optimizer": optimizer.state_dict(),
            "rngs": rngs,
        },
    )


def _load_stage3_single_rank(
    path, student, optimizer, rng, rollout_gen, fingerprint, *, rank=0, world=1, device=None
):
    device = torch.device("cpu") if device is None else device
    use_mmap = isinstance(path, (str, os.PathLike))
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=use_mmap)
    if state.get("format") != "archlab-stage3-checkpoint-v1" or state.get("contract_sha256") != fingerprint:
        raise ValueError("checkpoint belongs to another Stage-3 experiment contract")
    if len(state["rngs"]) != world:
        raise ValueError("checkpoint rank count differs")
    student.load_state_dict(state["student"], strict=True)
    del state["student"]
    optimizer.load_state_dict(state["optimizer"])
    del state["optimizer"]
    saved = state["rngs"][rank]
    restore_rank_rng(saved, rng, device)
    if rollout_gen is not None and "rollout_generator" in saved:
        rollout_gen.set_state(saved["rollout_generator"].to(device="cpu", dtype=torch.uint8))
    curriculum = CurriculumState(**state["curriculum"])
    step, consumed = int(state["step"]), int(state["consumed_tokens"])
    del state
    gc.collect()
    return step, consumed, curriculum


def load_stage3_checkpoint(path, student, optimizer, rng, rollout_gen, fingerprint, *, rank=0, world=1, device=None):
    if dist.is_available() and dist.is_initialized() and world > 1:
        result = None
        for r in range(world):
            if rank == r:
                result = _load_stage3_single_rank(
                    path,
                    student,
                    optimizer,
                    rng,
                    rollout_gen,
                    fingerprint,
                    rank=rank,
                    world=world,
                    device=device,
                )
            dist.barrier()
        assert result is not None
        return result
    return _load_stage3_single_rank(
        path, student, optimizer, rng, rollout_gen, fingerprint, rank=rank, world=world, device=device
    )


def main(argv=None):
    from transformers import AutoConfig, AutoModelForCausalLM

    from archlab.architectures import qwen35_switch as switch_lib
    from archlab.architectures.triadic_attention import official_dependency_contract

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--resume", type=Path, help="explicit latest checkpoint in the same output directory")
    parser.add_argument("--stop-after", type=int, help="stop and checkpoint at this total optimizer-step count")
    args = parser.parse_args(argv)
    run = Stage3Config.read(args.run)

    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
        run.check_world_curriculum(world)
        device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        torch.cuda.set_device(device)
        torch.manual_seed(run.seed)

        rollout_gen = torch.Generator(device=device)
        rollout_gen.manual_seed(run.seed * 1009 + rank)

        identity = tokenizer_identity(run.teacher)
        pinned = [(teacher_identity(run.teacher), sha256_file(run.init_checkpoint)) if rank == 0 else None]
        if distributed:
            dist.broadcast_object_list(pinned, src=0)
        teacher_files, init_sha = pinned[0]

        hf_config = AutoConfig.from_pretrained(run.teacher, local_files_only=True, trust_remote_code=False)
        config = hf_config.get_text_config()
        prompt_data = MixtureWindows(
            run.sources,
            run.prompt_length,
            tokenizer_files=identity,
            vocab_size=config.vocab_size,
            heldout_fraction=run.heldout_fraction,
            split_seed=run.split_seed,
        )
        eval_data = MixtureWindows(
            run.sources,
            run.prompt_length + run.min_gen_tokens,
            tokenizer_files=identity,
            vocab_size=config.vocab_size,
            heldout_fraction=run.heldout_fraction,
            split_seed=run.split_seed,
        )
        anchor_len = run.prompt_length + run.max_gen_tokens
        anchor_data = (
            MixtureWindows(
                run.sources,
                anchor_len,
                tokenizer_files=identity,
                vocab_size=config.vocab_size,
                heldout_fraction=run.heldout_fraction,
                split_seed=run.split_seed,
            )
            if run.anchor_kl_weight > 0.0
            else None
        )
        anchor_per_rank = max(1, run.gen_batch_tokens // (anchor_len * world))
        rng = np.random.default_rng(np.random.SeedSequence([run.seed, rank]))
        heldout = eval_data.sample(np.random.default_rng(run.seed + 7), run.eval_windows, heldout=True)

        teacher = AutoModelForCausalLM.from_pretrained(
            run.teacher,
            dtype=torch.bfloat16,
            device_map={"": device},
            local_files_only=True,
            trust_remote_code=False,
        )
        teacher.eval().requires_grad_(False)
        teacher.config.use_cache = False

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
            stage1_ckpt=None if args.resume else run.init_checkpoint,
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
            Path(__file__).parent / "stage2_distill.py",
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
        contract = {
            "format": "archlab-stage3-run-v1",
            "config": json.loads(json.dumps(asdict(run))),
            "world": world,
            "switched_layers": sorted(switches),
            "parameter_counts": param_counts,
            "teacher_files_sha256": teacher_files,
            "init_checkpoint_sha256": init_sha,
            "tokenizer_files": identity,
            "versions": runtime_versions(),
            "data": prompt_data.receipts,
            "source_sha256": {p.name: sha256_file(p) for p in source_paths},
            "eval_windows_sha256": hashlib.sha256(heldout.numpy().tobytes()).hexdigest(),
            "objective": (
                "anchored_onpolicy_reverse_kl_plus_offpolicy_forward_kl"
                if run.anchor_kl_weight > 0.0
                else "onpolicy_reverse_kl_student_to_teacher"
            ),
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

            if args.resume:
                start_step, consumed_tokens, curriculum = load_stage3_checkpoint(
                    args.resume,
                    student,
                    optimizer,
                    rng,
                    rollout_gen,
                    fingerprint,
                    rank=rank,
                    world=world,
                    device=device,
                )
            else:
                start_step = 0
                consumed_tokens = 0
                curriculum = CurriculumState(gen_tokens=run.min_gen_tokens)

            steps = run.tokens // run.gen_batch_tokens
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
                        "tokens": consumed_tokens,
                        "gen_tokens": curriculum.gen_tokens,
                        "curriculum_phase": curriculum.phase,
                        "resumed": bool(args.resume),
                        **evaluate_on_policy(
                            teacher,
                            student,
                            heldout,
                            prompt_length=run.prompt_length,
                            gen_tokens=run.min_gen_tokens,
                            device=device,
                            temperature=0.0,
                            microbatch_prompts=run.microbatch_prompts,
                            chunk_tokens=run.kl_chunk_tokens,
                        ),
                    }
                )
            if distributed:
                dist.barrier()

            started = time.monotonic()
            for step in range(start_step, stop_step):
                next_tokens = consumed_tokens + run.gen_batch_tokens
                base_lr = wsd_lr_at(
                    next_tokens,
                    run.tokens,
                    run.lr,
                    run.warmup_fraction,
                    run.decay_fraction,
                    run.final_lr_fraction,
                )
                for group in optimizer.param_groups:
                    group["lr"] = base_lr * group["lr_scale"]

                cur_gen = curriculum.gen_tokens
                prompt_count = run.prompts_per_rank(world, cur_gen)
                prompts = prompt_data.sample(rng, prompt_count).to(device)
                anchor_batch = anchor_data.sample(rng, anchor_per_rank).to(device) if anchor_data is not None else None
                t_roll = time.monotonic()
                sequences = rollout_on_policy(
                    student,
                    prompts,
                    cur_gen,
                    temperature=run.temperature,
                    microbatch_prompts=run.microbatch_prompts,
                    generator=rollout_gen,
                )
                rollout_seconds = time.monotonic() - t_roll

                t_train = time.monotonic()
                result = chunked_reverse_kl_step(
                    teacher,
                    student,
                    optimizer,
                    trainable_params,
                    sequences,
                    prompt_length=run.prompt_length,
                    microbatch_prompts=run.microbatch_prompts,
                    chunk_tokens=run.kl_chunk_tokens,
                    clip_norm=run.clip_norm,
                    anchor_windows=anchor_batch,
                    anchor_kl_weight=run.anchor_kl_weight,
                )
                train_seconds = time.monotonic() - t_train
                consumed_tokens = next_tokens
                advanced = curriculum.update(
                    result["train_reverse_kl"],
                    max_gen_tokens=run.max_gen_tokens,
                    ema_alpha=run.ema_alpha,
                    improvement_rel=run.improvement_rel,
                    plateau_steps=run.plateau_steps,
                    max_divergence_rel=run.max_divergence_rel,
                )

                if (step + 1) % run.log_every == 0:
                    log(
                        {
                            "step": step + 1,
                            "tokens": consumed_tokens,
                            "gen_tokens": cur_gen,
                            "next_gen_tokens": curriculum.gen_tokens,
                            "curriculum_phase": curriculum.phase,
                            "curriculum_advanced": advanced,
                            "lr_switches": base_lr,
                            "lr_backbone": base_lr * run.backbone_lr_ratio,
                            "rollout_seconds": rollout_seconds,
                            "train_seconds": train_seconds,
                            **result,
                            "tokens_per_second": (step + 1 - start_step)
                            * run.gen_batch_tokens
                            / (time.monotonic() - started),
                        }
                    )

                if (step + 1) % run.eval_every == 0 or step + 1 == stop_step:
                    if rank == 0:
                        log(
                            {
                                "step": step + 1,
                                "tokens": consumed_tokens,
                                "gen_tokens": curriculum.gen_tokens,
                                "curriculum_phase": curriculum.phase,
                                **evaluate_on_policy(
                                    teacher,
                                    student,
                                    heldout,
                                    prompt_length=run.prompt_length,
                                    gen_tokens=run.min_gen_tokens,
                                    device=device,
                                    temperature=0.0,
                                    microbatch_prompts=run.microbatch_prompts,
                                    chunk_tokens=run.kl_chunk_tokens,
                                ),
                            }
                        )
                    local_rng = rng_state(rng, device)
                    local_rng["rollout_generator"] = rollout_gen.get_state().cpu()
                    states = [None] * world if rank == 0 else None
                    if distributed:
                        dist.gather_object(local_rng, states, dst=0)
                    else:
                        states = [local_rng]
                    if rank == 0:
                        save_stage3_checkpoint(
                            output,
                            step + 1,
                            consumed_tokens,
                            curriculum,
                            student,
                            switches,
                            optimizer,
                            states,
                            fingerprint,
                        )
                    if distributed:
                        dist.barrier()

            elapsed = time.monotonic() - started
            log(
                {
                    "event": "stopped" if stop_step < steps else "complete",
                    "step": stop_step,
                    "tokens": consumed_tokens,
                    "attempt_seconds": elapsed,
                    "attempt_tokens_per_second": (stop_step - start_step) * run.gen_batch_tokens / max(elapsed, 1e-9),
                }
            )
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

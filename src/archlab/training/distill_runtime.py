"""Student construction, evaluation and persistence shared by distillation stages."""

from __future__ import annotations

import gc
import math
import os
import uuid
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from archlab.training.latent_match import text_model


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


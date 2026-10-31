"""Tests for Stage-2 end-to-end off-policy forward-KL distillation."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from archlab.architectures.qwen35_switch import SwitchConfig
from archlab.training import stage2_distill as s2
from archlab.training.latent_match import rng_state


class TinyAttentionMixer(nn.Module):
    def __init__(self, width=16, heads=2, kv=1, dim=8):
        super().__init__()
        self.head_dim = dim
        self.q_proj = nn.Linear(width, heads * dim * 2, bias=False)
        self.k_proj = nn.Linear(width, kv * dim, bias=False)
        self.v_proj = nn.Linear(width, kv * dim, bias=False)
        self.o_proj = nn.Linear(heads * dim, width, bias=False)
        self.q_norm = nn.RMSNorm(dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        b, n, _ = hidden_states.shape
        q, g = self.q_proj(hidden_states).chunk(2, dim=-1)
        out = self.o_proj(q * torch.sigmoid(g))
        return out, None


class TinyDecoderLayer(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.norm1 = nn.RMSNorm(width, eps=1e-6)
        self.self_attn = TinyAttentionMixer(width=width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, hidden_states, position_embeddings):
        attn_out, _ = self.self_attn(hidden_states=self.norm1(hidden_states), position_embeddings=position_embeddings)
        hidden_states = hidden_states + attn_out
        return hidden_states + self.mlp(hidden_states)


class TinyBody(nn.Module):
    def __init__(self, vocab=19, width=16, n_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
        self.layers = nn.ModuleList([TinyDecoderLayer(width=width) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(width, eps=1e-6)

    def forward(self, input_ids, use_cache=False):
        b, n = input_ids.shape
        x = self.embed_tokens(input_ids)
        angles = torch.arange(n, device=input_ids.device, dtype=torch.float32)[None, :] * 0.1
        angles = angles.unsqueeze(-1).expand(b, n, 4)
        rotary = (angles.cos(), angles.sin())
        for layer in self.layers:
            x = layer(x, rotary)
        return SimpleNamespace(last_hidden_state=self.norm(x))


class TinyCausalLM(nn.Module):
    def __init__(self, vocab=19, width=16, tied=False):
        super().__init__()
        self.model = TinyBody(vocab=vocab, width=width, n_layers=2)
        self.lm_head = nn.Linear(width, vocab, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head


@pytest.mark.parametrize("tied", [False, True])
def test_chunked_forward_kl_matches_unchunked_autograd_and_preserves_teacher(tied):
    torch.manual_seed(7)
    teacher = TinyCausalLM(tied=tied).eval().requires_grad_(False)
    teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}

    student_a = copy.deepcopy(teacher).train()
    switches_a = s2.install_student_switches(
        student_a, ["full_attention", "full_attention"], "simplicial",
        SwitchConfig(backend="reference", window_1=None, window_2=3), seed=42
    )
    opt_a, params_a, counts = s2.build_two_timescale_optimizer(student_a, switches_a, lr=1e-3, backbone_lr_ratio=0.1)
    assert counts["switch_parameters"] > 0 and counts["backbone_parameters"] > 0

    student_b = copy.deepcopy(student_a).train()
    switches_b = {0: student_b.model.layers[0].self_attn.switch, 1: student_b.model.layers[1].self_attn.switch}
    opt_b, params_b, _ = s2.build_two_timescale_optimizer(student_b, switches_b, lr=1e-3, backbone_lr_ratio=0.1)

    ids = torch.randint(0, 19, (4, 7))
    res_a = s2.chunked_forward_kl_step(
        teacher, student_a, opt_a, params_a, ids, microbatch_windows=2, chunk_tokens=2, clip_norm=10.0
    )

    # Full unchunked reference update on student_b
    opt_b.zero_grad(set_to_none=True)
    with torch.no_grad():
        t_h = teacher.model(ids).last_hidden_state[:, :-1]
        t_logp = teacher.lm_head(t_h).float().log_softmax(-1)
    s_h = student_b.model(ids).last_hidden_state[:, :-1]
    s_logp = student_b.lm_head(s_h).float().log_softmax(-1)
    loss = (t_logp.exp() * (t_logp - s_logp)).sum() / (ids.shape[0] * (ids.shape[1] - 1))
    loss.backward()
    norm_b = torch.nn.utils.clip_grad_norm_(params_b, 10.0)
    opt_b.step()

    assert abs(res_a["train_kl"] - float(loss.detach())) < 1e-6
    assert abs(res_a["grad_norm"] - float(norm_b)) < 1e-5
    for pa, pb in zip(params_a, params_b, strict=True):
        torch.testing.assert_close(pa, pb, atol=1e-6, rtol=1e-5)
    for k, v in teacher.state_dict().items():
        assert torch.equal(v, teacher_before[k])


def test_stage2_reduces_composed_heldout_kl_and_roundtrips_checkpoint(tmp_path):
    torch.manual_seed(13)
    teacher = TinyCausalLM().eval().requires_grad_(False)
    student = copy.deepcopy(teacher)
    switches = s2.install_student_switches(
        student, ["full_attention", "full_attention"], "simplicial",
        SwitchConfig(backend="reference", init_std=0.2), seed=19
    )
    # Save a mock Stage-1 checkpoint and reload into a fresh student
    s1_ckpt = tmp_path / "stage1.pt"
    torch.save({"format": "archlab-latent-checkpoint-v1", "switches": {str(i): m.state_dict() for i, m in switches.items()}}, s1_ckpt)

    student2 = copy.deepcopy(teacher).train()
    switches2 = s2.install_student_switches(
        student2, ["full_attention", "full_attention"], "simplicial",
        SwitchConfig(backend="reference", init_std=0.01), seed=99, stage1_ckpt=s1_ckpt
    )
    for i in switches:
        for k, v in switches[i].state_dict().items():
            assert torch.equal(v, switches2[i].state_dict()[k])

    opt, params, _ = s2.build_two_timescale_optimizer(student2, switches2, lr=5e-3, backbone_lr_ratio=0.1)
    windows = torch.randint(0, 19, (6, 6))
    before = s2.evaluate_composed(teacher, student2, windows, torch.device("cpu"), chunk_tokens=3)
    for _ in range(12):
        s2.chunked_forward_kl_step(teacher, student2, opt, params, windows, microbatch_windows=3, chunk_tokens=2)
    after = s2.evaluate_composed(teacher, student2, windows, torch.device("cpu"), chunk_tokens=3)
    assert after["kl"] < before["kl"] * 0.5

    rng = np.random.default_rng(5)
    ckpt = s2.save_stage2_checkpoint(tmp_path / "out", 12, student2, switches2, opt, [rng_state(rng, torch.device("cpu"))], "fp123")
    fresh = copy.deepcopy(teacher).train()
    fresh_sw = s2.install_student_switches(
        fresh, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=1
    )
    fresh_opt, _, _ = s2.build_two_timescale_optimizer(fresh, fresh_sw, lr=5e-3, backbone_lr_ratio=0.1)
    step = s2.load_stage2_checkpoint(ckpt, fresh, fresh_opt, np.random.default_rng(0), "fp123")
    assert step == 12
    reloaded = s2.evaluate_composed(teacher, fresh, windows, torch.device("cpu"), chunk_tokens=3)
    assert reloaded == after


def _gloo_worker(rank, world, port, teacher_sd, student_sd, ids, out_path):
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    try:
        teacher = TinyCausalLM().eval().requires_grad_(False)
        teacher.load_state_dict(teacher_sd)
        student = TinyCausalLM().train()
        switches = s2.install_student_switches(
            student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=42
        )
        student.load_state_dict(student_sd)
        opt, params, _ = s2.build_two_timescale_optimizer(student, switches, lr=2e-3, backbone_lr_ratio=0.1)
        shard = ids[rank * 2 : (rank + 1) * 2]
        res = s2.chunked_forward_kl_step(teacher, student, opt, params, shard, microbatch_windows=1, chunk_tokens=2)
        if rank == 0:
            torch.save(
                {"kl": res["train_kl"], "sd": {k: v.detach().cpu() for k, v in student.state_dict().items()}},
                out_path,
            )
    finally:
        dist.destroy_process_group()


def test_stage2_two_rank_gloo_matches_single_rank_global_batch(tmp_path):
    torch.manual_seed(31)
    teacher = TinyCausalLM().eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train()
    switches = s2.install_student_switches(
        student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=42
    )
    t_sd = copy.deepcopy(teacher.state_dict())
    s_sd = copy.deepcopy(student.state_dict())
    ids = torch.randint(0, 19, (4, 6))

    opt, params, _ = s2.build_two_timescale_optimizer(student, switches, lr=2e-3, backbone_lr_ratio=0.1)
    single = s2.chunked_forward_kl_step(teacher, student, opt, params, ids, microbatch_windows=1, chunk_tokens=2)

    out_path = str(tmp_path / "rank0.pt")
    mp.spawn(_gloo_worker, args=(2, 29783, t_sd, s_sd, ids, out_path), nprocs=2)
    payload = torch.load(out_path, map_location="cpu", weights_only=True)
    assert abs(single["train_kl"] - payload["kl"]) < 1e-6
    for k, v in student.state_dict().items():
        torch.testing.assert_close(v, payload["sd"][k], atol=1e-6, rtol=1e-5)


def test_stage2_freezes_non_text_submodules_and_loads_stage2_init_checkpoint(tmp_path):
    torch.manual_seed(41)
    teacher = TinyCausalLM().eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train()
    # Attach an unused vision-like tower outside text_model(student) and get_output_embeddings()
    student.visual = nn.Linear(8, 8)
    visual_before = {k: v.clone() for k, v in student.visual.state_dict().items()}

    switches = s2.install_student_switches(
        student, ["full_attention", "full_attention"], "simplicial", SwitchConfig(backend="reference"), seed=11
    )
    opt, params, counts = s2.build_two_timescale_optimizer(student, switches, lr=3e-3, backbone_lr_ratio=0.1)
    assert counts["frozen_non_text_parameters"] == sum(p.numel() for p in student.visual.parameters())
    assert all(not p.requires_grad for p in student.visual.parameters())

    ids = torch.randint(0, 19, (3, 6))
    s2.chunked_forward_kl_step(teacher, student, opt, params, ids, microbatch_windows=2, chunk_tokens=2)
    for k, v in student.visual.state_dict().items():
        assert torch.equal(v, visual_before[k])

    ckpt = s2.save_stage2_checkpoint(
        tmp_path / "s2out",
        1,
        student,
        switches,
        opt,
        [s2.rng_state(np.random.default_rng(3), torch.device("cpu"))],
        "fp-s2",
    )
    downstream = copy.deepcopy(teacher).train()
    downstream.visual = copy.deepcopy(student.visual)
    s2.install_student_switches(
        downstream,
        ["full_attention", "full_attention"],
        "simplicial",
        SwitchConfig(backend="reference"),
        seed=99,
        stage1_ckpt=ckpt,
    )
    for k, v in student.state_dict().items():
        torch.testing.assert_close(v, downstream.state_dict()[k])

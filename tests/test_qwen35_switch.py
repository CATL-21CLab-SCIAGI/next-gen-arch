"""Rotation, ownership, gradient and GDN reduction contracts for layer replacements."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from archlab.architectures.qwen35_switch import (
    SimplicialSwitch,
    SwitchConfig,
    TriadicSwitch,
    _chunk_rope,
)
from archlab.architectures.triadic_attention import triadic_gdn_reference


class GatedNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(dim) + 0.5)

    def forward(self, x, gate):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight
        return x * F.silu(gate)


def attention_teacher(width=32, heads=4, kv=2, dim=16):
    t = nn.Module()
    t.head_dim = dim
    t.q_proj, t.k_proj = nn.Linear(width, heads * dim * 2, bias=False), nn.Linear(width, kv * dim, bias=False)
    t.v_proj, t.o_proj = nn.Linear(width, kv * dim, bias=False), nn.Linear(heads * dim, width, bias=False)
    t.q_norm, t.k_norm = nn.RMSNorm(dim, eps=1e-6), nn.RMSNorm(dim, eps=1e-6)
    return t


def test_determinant_switch_depends_on_relative_positions():
    torch.manual_seed(0)
    t, x = attention_teacher(), torch.randn(1, 7, 32)
    frequencies = torch.rand(4)

    def rotary(offset):
        angles = (torch.arange(7).float() + offset)[None, :, None] * frequencies
        angles = torch.cat((angles, angles), -1)
        return angles.cos(), angles.sin()

    switch = SimplicialSwitch(t, SwitchConfig(backend="reference"), seed=1)
    torch.testing.assert_close(switch(x, rotary(0)), switch(x, rotary(11)), rtol=1e-4, atol=1e-5)


def test_chunk_rope_preserves_norm_and_unrotated_tail():
    torch.manual_seed(2)
    x = torch.randn(2, 7, 4, 16)
    angle = torch.randn(2, 7, 1, 4)
    angle = torch.cat((angle, angle), -1)
    rotated = _chunk_rope(x, angle.cos(), angle.sin())
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1))
    assert torch.equal(rotated[..., 12:], x[..., 12:])


def gdn_teacher(width=32, k_heads=2, v_heads=4, dim=8):
    t = nn.Module()
    t.hidden_size, t.num_k_heads, t.num_v_heads, t.head_k_dim, t.head_v_dim = width, k_heads, v_heads, dim, dim
    t.key_dim, t.value_dim, t.conv_kernel_size = k_heads * dim, v_heads * dim, 4
    channels = 2 * t.key_dim + t.value_dim
    t.in_proj_qkv = nn.Linear(width, channels, bias=False)
    t.conv1d = nn.Conv1d(channels, channels, 4, groups=channels, bias=False, padding=3)
    t.in_proj_z, t.in_proj_b = nn.Linear(width, t.value_dim, bias=False), nn.Linear(width, v_heads, bias=False)
    t.in_proj_a = nn.Linear(width, v_heads, bias=False)
    t.A_log, t.dt_bias = nn.Parameter(torch.rand(v_heads).log()), nn.Parameter(torch.rand(v_heads))
    t.norm, t.out_proj = GatedNorm(dim), nn.Linear(t.value_dim, width, bias=False)
    return t


def gdn_forward(t, x):
    """One-slice GDN (E=1) with the teacher's projections: the recurrence the switch must reproduce."""
    b, n = x.shape[:2]
    mixed = F.silu(t.conv1d(t.in_proj_qkv(x).transpose(1, 2))[..., :n]).transpose(1, 2)
    q, k, v = torch.split(mixed, [t.key_dim, t.key_dim, t.value_dim], dim=-1)
    rep = t.num_v_heads // t.num_k_heads
    q = q.reshape(b, n, -1, t.head_k_dim).repeat_interleave(rep, 2)
    k = k.reshape(b, n, -1, t.head_k_dim).repeat_interleave(rep, 2)
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    v = v.reshape(b, n, -1, t.head_v_dim)
    beta = t.in_proj_b(x).sigmoid()
    g = (-t.A_log.exp() * F.softplus(t.in_proj_a(x) + t.dt_bias)).unsqueeze(-1)
    ones = torch.ones(b, n, t.num_v_heads, 1)
    core = triadic_gdn_reference(q, k, v, ones, ones, g, beta)
    gate = t.in_proj_z(x).reshape(-1, t.head_v_dim)
    return t.out_proj(t.norm(core.reshape(-1, t.head_v_dim), gate).reshape(b, n, -1))


@pytest.mark.parametrize("qk_scale", [1.0, 1e-3])
def test_triadic_switch_reduces_to_the_teachers_gdn(qk_scale):
    torch.manual_seed(0)
    t, x = gdn_teacher(), torch.randn(2, 10, 32)
    with torch.no_grad():
        t.in_proj_qkv.weight[:2 * t.key_dim].mul_(qk_scale)
    switch = TriadicSwitch(t, SwitchConfig(backend="reference", feature_dim=4), seed=1)
    with torch.no_grad():
        switch.k2.weight.zero_(), switch.q2.weight.zero_()
    torch.testing.assert_close(switch(x), gdn_forward(t, x), rtol=1e-4, atol=1e-10)
    names = {name for name, _ in switch.named_parameters()}
    assert {"k2.weight", "q2.weight", "decay_offset"} <= names
    assert "inherited.in_proj_qkv.weight" in names


def test_new_branches_start_away_from_the_identity():
    torch.manual_seed(0)
    t, x = gdn_teacher(), torch.randn(1, 10, 32)
    switch = TriadicSwitch(t, SwitchConfig(backend="reference", feature_dim=4, init_std=0.5), seed=1)
    assert (switch(x) - gdn_forward(t, x)).abs().max() > 1e-3


@pytest.mark.parametrize("variant", ["simplicial", "triadic"])
def test_all_replacement_parameters_train_without_changing_the_teacher(variant):
    torch.manual_seed(3)
    teacher = attention_teacher() if variant == "simplicial" else gdn_teacher()
    teacher.requires_grad_(False)
    before = {name: p.clone() for name, p in teacher.named_parameters()}
    cls = SimplicialSwitch if variant == "simplicial" else TriadicSwitch
    switch = cls(teacher, SwitchConfig(backend="reference"), seed=4)
    x = torch.randn(1, 5, 32)
    kwargs = {}
    if variant == "simplicial":
        kwargs["position_embeddings"] = (torch.ones(1, 5, 8), torch.zeros(1, 5, 8))
    optimizer = torch.optim.AdamW(switch.parameters(), lr=1e-3)
    switch(x, **kwargs).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in switch.parameters())
    assert any(p.grad.abs().sum() > 0 for p in switch.inherited.parameters())
    optimizer.step()
    for name, parameter in teacher.named_parameters():
        assert parameter.grad is None
        assert torch.equal(parameter, before[name])
        assert parameter.data_ptr() != dict(switch.inherited.named_parameters())[name].data_ptr()


@pytest.mark.parametrize("name", ["attention_mask", "past_key_values", "cache_params"])
def test_switches_refuse_unsupported_padding_and_caches(name):
    switch = TriadicSwitch(gdn_teacher(), SwitchConfig(backend="reference"), seed=0)
    with pytest.raises(ValueError, match="unpadded"):
        switch(torch.randn(1, 5, 32), **{name: object()})


def test_simplicial_switch_sliding_windows_and_validation():
    for bad in (0, -3, True, 2.5):
        with pytest.raises(ValueError, match="window_1"):
            SwitchConfig(window_1=bad)
        with pytest.raises(ValueError, match="window_2"):
            SwitchConfig(window_2=bad)
    torch.manual_seed(11)
    t = attention_teacher()
    x = torch.randn(1, 7, 32)
    rotary = (torch.ones(1, 7, 8), torch.zeros(1, 7, 8))
    full = SimplicialSwitch(t, SwitchConfig(backend="reference"), seed=5)
    capped = SimplicialSwitch(t, SwitchConfig(backend="reference", window_1=7, window_2=7), seed=5)
    torch.testing.assert_close(full(x, rotary), capped(x, rotary), atol=1e-6, rtol=1e-5)

    # Inherited long (None = full), auxiliary short (window_2=2) vs both short/long
    win_switch = SimplicialSwitch(t, SwitchConfig(backend="reference", window_1=None, window_2=2), seed=5)
    out = win_switch(x, rotary)
    assert out.shape == x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in win_switch.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA BF16 autocast")
def test_fp32_master_simplicial_switch_preserves_outer_bf16_autocast():
    torch.manual_seed(21)
    device = torch.device("cuda")
    t = attention_teacher(width=64, heads=4, kv=2, dim=32).to(device=device, dtype=torch.float32)
    sw = SimplicialSwitch(t, SwitchConfig(backend="kernel", window_1=None, window_2=8), seed=7).to(
        device=device, dtype=torch.float32
    )
    x = torch.randn(1, 16, 64, device=device, dtype=torch.float32, requires_grad=True)
    rotary = (torch.ones(1, 16, 16, device=device), torch.zeros(1, 16, 16, device=device))

    seen_dtypes = []
    orig_fn = __import__("archlab.architectures.qwen35_switch", fromlist=["oriented_determinant_simplicial_attention"])
    real_kernel = orig_fn.oriented_determinant_simplicial_attention

    def spy(*args, **kwargs):
        seen_dtypes.append(tuple(a.dtype for a in args[:5]))
        return real_kernel(*args, **kwargs)

    orig_fn.oriented_determinant_simplicial_attention = spy
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = sw(x, rotary)
        out.float().square().mean().backward()
    finally:
        orig_fn.oriented_determinant_simplicial_attention = real_kernel

    assert seen_dtypes == [(torch.bfloat16,) * 5]
    assert all(p.dtype == torch.float32 and p.grad is not None and torch.isfinite(p.grad).all() for p in sw.parameters())

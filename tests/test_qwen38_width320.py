"""Width-320 contract and frozen-container numerical gates (no runtime patches)."""

import os
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from archlab.architectures.qwen38_flash_next_full import (
    DistributedPLE,
    FourStreamGatedResidual,
    GatedDeltaNet,
    GroupRMSNorm,
    Qwen38FlashNextFullConfig,
    parameter_count_contract,
)
from archlab.megatron.qwen38_flash_next_full_train import (
    _effective_probe_gradient,
    _megatron_argv,
    _parser,
)


def test_probe_uses_native_te_main_gradient_not_autograd_placeholder():
    weight = torch.nn.Parameter(torch.ones(2, 2))
    normal = torch.full_like(weight, 3)
    assert _effective_probe_gradient(weight, normal) is normal
    weight.main_grad = torch.full_like(weight, 4)
    weight.grad_added_to_main_grad = True
    placeholder = torch.full_like(weight, float("nan"))
    assert _effective_probe_gradient(weight, placeholder) is weight.main_grad


def test_width320_exact_contract_and_feature_flags(tmp_path, monkeypatch):
    config = Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()
    assert parameter_count_contract(config)["total"] == 387_680_960
    assert config.ngram_padded_rows == 2_622_720
    assert config.ngram_branch_dim == 20
    assert config.residual_streams == 4
    assert (config.num_experts_per_token + 1) * config.moe_intermediate_size == 880
    assert config.attention_heads * config.attention_head_dim == 768
    assert config.linear_v_heads * config.linear_value_dim == 768
    for field, value in (
        ("residual_streams", 1),
        ("attention_output_gate", False),
        ("qk_layernorm", False),
        ("num_experts_per_token", 3),
    ):
        with pytest.raises(ValueError, match="contract drift"):
            replace(config, **{field: value})
    args = _parser().parse_args(
        [
            "--data-root",
            str(tmp_path),
            "--tokenizer",
            str(tmp_path),
            "--run-dir",
            str(tmp_path),
            "--model-variant",
            "w320-e32-depth48-no-mtp",
            "--parallelism",
            "dp-only",
            "--micro-batch-size",
            "4",
        ]
    )
    argv = _megatron_argv(args, config)
    for flag in ("--attention-output-gate", "--qk-layernorm", "--apply-layernorm-1p"):
        assert flag in argv
    for flag in (
        "--tensor-model-parallel-size",
        "--pipeline-model-parallel-size",
        "--expert-model-parallel-size",
        "--expert-tensor-parallel-size",
        "--context-parallel-size",
    ):
        assert argv[argv.index(flag) + 1] == "1"
    assert "--load" not in argv
    native = pytest.importorskip("megatron.training.arguments")
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr("sys.argv", argv)
    parsed = native.validate_args(native.parse_args())
    native_config = native.core_transformer_config_from_args(parsed)
    assert parsed.data_parallel_size == 32
    assert native_config.attention_output_gate and native_config.qk_layernorm
    assert native_config.layernorm_zero_centered_gamma


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_source_zero_centered_group_norm_output_and_gradients(dtype):
    torch.manual_seed(19)
    norm = GroupRMSNorm(320, 4, 1e-6, zero_centered=True).to(dtype)
    assert torch.count_nonzero(norm.weight) == 0
    with torch.no_grad():
        norm.weight.normal_(std=0.1)
    x = torch.randn(3, 2, 1280, dtype=dtype, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_w = norm.weight.detach().clone().requires_grad_()
    grouped = ref_x.float().reshape(3, 2, 4, 320)
    reference = (
        (
            grouped
            * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
            * (1 + ref_w.float().reshape(4, 320))
        )
        .reshape_as(x)
        .to(dtype)
    )
    actual = norm(x)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual.float().square().sum().backward()
    reference.float().square().sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(norm.weight.grad, ref_w.grad, rtol=0, atol=0)


def test_restored_gr_and_ple_use_source_norms():
    config = Qwen38FlashNextFullConfig.tiny(zero_centered_gamma=True)
    gr = FourStreamGatedResidual(config)
    ple = DistributedPLE(config, owner_rank=0, owner_world_size=1)
    for norm in (gr.norm, ple.norm_key, ple.norm_query, ple.norm_conv):
        assert norm.zero_centered
        assert torch.count_nonzero(norm.weight) == 0
    packed = torch.randn(8, 2, 128, requires_grad=True)
    mixed, original, injection = gr(packed)
    output = gr.inject(original, mixed, injection)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    ple.embedding.reset_parameters()
    (output + ple(tokens, output)).square().mean().backward()
    assert torch.isfinite(packed.grad).all() and torch.count_nonzero(packed.grad)
    assert torch.count_nonzero(gr.norm.weight.grad)
    assert torch.count_nonzero(ple.norm_query.weight.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires frozen FLA CUDA kernels")
def test_width320_gdn_kernel_matches_recurrent_cpu_oracle():
    torch.manual_seed(34)
    config = Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()
    # The recurrent CPU oracle deliberately accumulates/returns FP32; compare
    # it with the BF16 FLA path using identical BF16-representable weights.
    reference = GatedDeltaNet(config).to(torch.bfloat16).float()
    actual = deepcopy(reference).cuda().to(torch.bfloat16)
    x = torch.randn(16, 1, 320, dtype=torch.bfloat16, requires_grad=True)
    gpu_x = x.detach().cuda().requires_grad_()
    expected, observed = reference(x.float()), actual(gpu_x)
    torch.testing.assert_close(observed.cpu().float(), expected, atol=0.003, rtol=0.06)
    expected.float().square().mean().backward()
    observed.float().square().mean().backward()
    torch.testing.assert_close(gpu_x.grad.cpu(), x.grad, atol=1e-5, rtol=0.1)
    for (name, p), (_, gpu_p) in zip(
        reference.named_parameters(), actual.named_parameters(), strict=True
    ):
        assert p.grad is not None and torch.isfinite(gpu_p.grad).all(), name
        torch.testing.assert_close(gpu_p.grad.cpu().float(), p.grad, atol=0.0001, rtol=0.1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or "RANK" not in os.environ,
    reason="run with torchrun in frozen NeMo (one or two DP ranks)",
)
def test_native_gated_attention_matches_sdpa_and_dp_gradient_oracle():
    from megatron.core import parallel_state
    from megatron.core.extensions.transformer_engine import TERowParallelLinear
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.enums import AttnMaskType
    from megatron.core.transformer.transformer_config import TransformerConfig

    from archlab.megatron.gated_qkv import SplitGatedQKV

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(22)
    torch.manual_seed(22)
    config = TransformerConfig(
        num_layers=48,
        hidden_size=320,
        num_attention_heads=24,
        num_query_groups=2,
        kv_channels=32,
        normalization="RMSNorm",
        qk_layernorm=True,
        layernorm_zero_centered_gamma=True,
        attention_output_gate=True,
        add_bias_linear=False,
        attention_dropout=0,
        hidden_dropout=0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    attention = (
        SelfAttention(
            config,
            SelfAttentionSubmodules(
                linear_qkv=SplitGatedQKV,
                linear_proj=TERowParallelLinear,
                core_attention=TESpecProvider().core_attention(),
                q_layernorm=None,
                k_layernorm=None,
            ),
            layer_number=4,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=groups,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    try:
        assert attention.linear_qkv.q.weight.shape == (768, 320)
        assert attention.linear_qkv.gate.weight.shape == (768, 320)
        assert attention.linear_qkv.k.weight.shape == (64, 320)
        assert attention.q_layernorm.weight.shape == (32,)
        assert not any("linear_qkv.weight" in name for name, _ in attention.named_parameters())
        assert torch.count_nonzero(attention.q_layernorm.weight) == 0
        # All replicas use identical weights; different samples per DP rank.
        for p in attention.parameters():
            torch.distributed.broadcast(p.data, src=0)
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        all_inputs = torch.randn(16, world, 320, device="cuda", dtype=torch.bfloat16)
        torch.distributed.broadcast(all_inputs, src=0)
        x = all_inputs[:, rank : rank + 1].clone().requires_grad_()
        actual = attention(x, attention_mask=None)[0]
        actual.float().square().mean().backward()
        grads = {}
        for name, p in attention.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
            grads[name] = p.grad.float().clone()
            torch.distributed.all_reduce(grads[name])
            grads[name] /= world
        attention.zero_grad(set_to_none=True)
        ref_x = all_inputs.detach().clone().requires_grad_()
        q = F.linear(ref_x, attention.linear_qkv.q.weight).reshape(16, world, 24, 32)
        k = F.linear(ref_x, attention.linear_qkv.k.weight).reshape(16, world, 2, 32)
        v = F.linear(ref_x, attention.linear_qkv.v.weight).reshape(16, world, 2, 32)

        def norm(t, weight):
            f = t.float()
            return (
                f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-5) * (1 + weight.float())
            ).to(t.dtype)

        q, k = norm(q, attention.q_layernorm.weight), norm(k, attention.k_layernorm.weight)
        attended = (
            F.scaled_dot_product_attention(
                q.permute(1, 2, 0, 3),
                k.permute(1, 2, 0, 3),
                v.permute(1, 2, 0, 3),
                is_causal=True,
                enable_gqa=True,
            )
            .permute(2, 0, 1, 3)
            .reshape(16, world, 768)
        )
        gate = F.linear(ref_x, attention.linear_qkv.gate.weight).sigmoid()
        reference = F.linear(attended * gate, attention.linear_proj.weight)
        torch.testing.assert_close(actual, reference[:, rank : rank + 1], atol=0.002, rtol=0.04)
        reference.float().square().mean().backward()
        for name, p in attention.named_parameters():
            torch.testing.assert_close(grads[name], p.grad.float(), atol=2e-5, rtol=0.06)
        assert torch.count_nonzero(grads["linear_qkv.gate.weight"])
        assert torch.count_nonzero(grads["q_layernorm.weight"])
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

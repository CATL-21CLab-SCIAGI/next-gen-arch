"""DP1/DP2 native adapter checks in the existing container runtime."""

import os
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(torch.cuda.is_available() and "RANK" in os.environ,
                     "requires torchrun and the existing native CUDA runtime")
class NativePilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The production native initializer applies this environment switch
        # from --attention-backend unfused; this standalone test has no initializer.
        os.environ["NVTE_FUSED_ATTN"] = "0"
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        if int(os.environ["WORLD_SIZE"]) not in (1, 2):
            raise ValueError("native numerical tests use DP1 or DP2")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group("nccl")
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)

    @classmethod
    def tearDownClass(cls):
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()

    def attention(self):
        from megatron.core.extensions.transformer_engine import TERowParallelLinear
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.enums import AttnBackend
        from megatron.core.transformer.transformer_config import TransformerConfig

        from archlab.megatron.gated_qkv import SplitGatedQKV

        cfg = TransformerConfig(num_layers=48, hidden_size=320, num_attention_heads=24,
            num_query_groups=2, kv_channels=32, normalization="RMSNorm", qk_layernorm=True,
            layernorm_zero_centered_gamma=True, attention_output_gate=True,
            layernorm_epsilon=1e-6, add_bias_linear=False, attention_dropout=0,
            hidden_dropout=0, bf16=True, params_dtype=torch.bfloat16,
            use_cpu_initialization=True, gradient_accumulation_fusion=False,
            apply_rope_fusion=False, attention_backend=AttnBackend.unfused)
        return SelfAttention(cfg, SelfAttentionSubmodules(linear_qkv=SplitGatedQKV,
            linear_proj=TERowParallelLinear, core_attention=TESpecProvider().core_attention(),
            q_layernorm=None, k_layernorm=None), layer_number=8,
            attn_mask_type=AttnMaskType.causal,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups()).cuda().to(torch.bfloat16)

    def positions(self, length):
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

        return RotaryEmbedding(32, rotary_percent=0.25, rotary_base=10000000)(length)

    def test_dp_mean_gradients_match_combined_batch(self):
        from archlab.megatron.simplicial_attention import PilotAttention

        world = torch.distributed.get_world_size()
        if world != 2:
            self.skipTest("requires two GPUs")
        rank = torch.distributed.get_rank()
        for arm in ("A", "B", "C"):
            native = self.attention()
            module = native if arm == "A" else PilotAttention(native, arm)
            for p in module.parameters():
                torch.distributed.broadcast(p.data, 0)
            torch.manual_seed(900)
            x = torch.randn(20, 4, 320, device="cuda", dtype=torch.bfloat16)
            dy = torch.randn_like(x)
            rotary = self.positions(20)
            output = module(x[:, rank * 2:(rank + 1) * 2].contiguous(),
                            attention_mask=None, rotary_pos_emb=rotary)[0]
            (output.float() * dy[:, rank * 2:(rank + 1) * 2].float()).mean().backward()
            distributed = {}
            for name, p in module.named_parameters():
                value = p.grad.float()
                torch.distributed.all_reduce(value)
                distributed[name] = value / world
            module.zero_grad(set_to_none=True)
            output = module(x, attention_mask=None, rotary_pos_emb=rotary)[0]
            (output.float() * dy.float()).mean().backward()
            for name, p in module.named_parameters():
                actual, expected = distributed[name], p.grad.float()
                self.assertTrue(torch.isfinite(actual).all(), name)
                relative = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                self.assertLess(relative.item(), 0.04, f"{arm}: {name}")

    def test_local_equals_global_inside_window_and_preserves_parameters(self):
        from archlab.megatron.simplicial_attention import PilotAttention, parameter_hashes

        native = self.attention()
        before = parameter_hashes(native)
        local = PilotAttention(native, "B", long_window=128)
        self.assertEqual(before, parameter_hashes(local))
        x = torch.randn(32, 2, 320, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        rotary = self.positions(32)
        out = native(x, attention_mask=None, rotary_pos_emb=rotary)[0]
        dy = torch.randn_like(out)
        out.backward(dy)
        dx = x.grad.clone()
        grads = {name: p.grad.clone() for name, p in native.named_parameters()}
        native.zero_grad(set_to_none=True)
        x.grad = None
        actual = local(x, rotary_pos_emb=rotary)[0]
        actual.backward(dy)
        torch.testing.assert_close(actual, out, atol=0.002, rtol=0.04)
        torch.testing.assert_close(x.grad, dx, atol=0.005, rtol=0.05)
        for name, p in local.named_parameters():
            torch.testing.assert_close(p.grad, grads[name], atol=0.08, rtol=0.08, msg=name)

    def test_simplicial_block_matches_oracle_and_keeps_rng_and_all_branches(self):
        from archlab.architectures.simplicial_attention import reference_simplicial
        from archlab.megatron.simplicial_attention import PilotAttention, parameter_hashes

        native = self.attention()
        hashes = parameter_hashes(native)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        actual = PilotAttention(native, "C")
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        for name, expected in hashes.items():
            self.assertEqual(parameter_hashes(actual)[name], expected)
        self.assertEqual(sum(p.numel() for p in actual.parameters()) -
                         sum(p.numel() for p in native.parameters()), 40992)
        x = torch.randn(20, 2, 320, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        rotary = self.positions(20)
        out = actual(x, rotary_pos_emb=rotary)[0]
        dy = torch.randn_like(out)
        out.backward(dy)
        grads = {name: p.grad.clone() for name, p in actual.named_parameters()}
        dx = x.grad.clone()
        actual.zero_grad(set_to_none=True)
        x.grad = None

        def oracle(q, k1, k2, v1, v2, short, long):
            return reference_simplicial(*(t.double() for t in (q, k1, k2, v1, v2)),
                                       short, long).to(q.dtype)

        with patch("archlab.megatron.simplicial_attention.simplicial_attention", oracle):
            expected = actual(x, rotary_pos_emb=rotary)[0]
            expected.backward(dy)
        torch.testing.assert_close(out, expected, atol=0.0005, rtol=0.05)
        torch.testing.assert_close(dx, x.grad, atol=0.002, rtol=0.08)
        for name, p in actual.named_parameters():
            self.assertTrue(torch.isfinite(grads[name]).all(), name)
            self.assertTrue(torch.count_nonzero(grads[name]), name)
            error = (grads[name].float() - p.grad.float()).norm()
            scale = p.grad.float().norm().clamp_min(1e-8)
            self.assertLess(float(error / scale), 0.025, name)

    def test_local_window_boundary_matches_explicit_mask(self):
        from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb

        from archlab.megatron.simplicial_attention import PilotAttention

        local = PilotAttention(self.attention(), "B", long_window=128)
        x = torch.randn(160, 1, 320, device="cuda", dtype=torch.bfloat16)
        positions = self.positions(160)
        with torch.no_grad():
            actual = local(x, rotary_pos_emb=positions)[0]
            q, k, v, gate = local.qkv_gate(x)
            q = apply_rotary_pos_emb(q, positions, config=local.config, cp_group=local.pg_collection.cp)
            k = apply_rotary_pos_emb(k, positions, config=local.config, cp_group=local.pg_collection.cp)
            i = torch.arange(160, device="cuda")[:, None]
            j = torch.arange(160, device="cuda")[None, :]
            allowed = (j <= i) & (j > i - 128)
            attended = torch.nn.functional.scaled_dot_product_attention(
                q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3),
                attn_mask=allowed, enable_gqa=True).permute(2, 0, 1, 3).flatten(-2)
            gated = (attended * gate.flatten(-2).float().sigmoid()).to(attended.dtype)
            expected = local.linear_proj(gated)[0]
        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.04)


if __name__ == "__main__":
    unittest.main()

"""Leaf tests only: these do not claim pretrained-model or FSDP support."""

import unittest
from dataclasses import replace

import torch
from torch import nn

from archlab.architectures.simplicial_adapter import (
    SimplicialAdapterConfig,
    SimplicialResidualAdapter,
    partial_rope,
)


def small_config():
    return SimplicialAdapterConfig(hidden_size=16, query_heads=4, kv_heads=2,
                                  head_dim=16, residual_low_rank=2, short_window=2, long_window=3)


class SimplicialAdapterTests(unittest.TestCase):
    def test_parameter_count_and_full_shapes_on_meta(self):
        config = SimplicialAdapterConfig()
        with torch.device("meta"):
            module = SimplicialResidualAdapter(config)
        self.assertEqual(config.parameter_count(), 59_034_368)
        self.assertEqual(12 * config.parameter_count(), 708_412_416)
        self.assertEqual(sum(p.numel() for p in module.parameters()), config.parameter_count())
        self.assertEqual(module.q.weight.shape, (6144, 2560))
        self.assertEqual(module.k2.weight.shape, (512, 2560))
        self.assertEqual(module.output.weight.shape, (2560, 6144))
        self.assertEqual(module.residual.input_mix_weight_down.weight.shape, (320, 10240))

    def test_configuration_rejects_invalid_geometry(self):
        for values in ({"head_dim": 192}, {"query_heads": 3}, {"short_window": 129},
                       {"rotary_fraction": 0.001}, {"hidden_size": 0},
                       {"query_heads": 512}, {"rms_norm_eps": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(SimplicialAdapterConfig(), **values)

    def test_initialization_preserves_rng_and_is_exact_identity(self):
        torch.manual_seed(200)
        state = torch.get_rng_state()
        module = SimplicialResidualAdapter(small_config(), backend="reference")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        x = torch.randn(2, 4, 64, requires_grad=True)
        out = module(x)
        self.assertTrue(torch.equal(out, x))
        dy = torch.randn_like(out)
        out.backward(dy)
        self.assertTrue(torch.equal(x.grad, dy))
        self.assertGreater(module.output.weight.grad.abs().sum().item(), 0)
        for name, p in module.named_parameters():
            self.assertIsNotNone(p.grad, name)
            if name != "output.weight":
                self.assertEqual(torch.count_nonzero(p.grad).item(), 0, name)

    def test_two_updates_reach_every_added_parameter_through_frozen_suffix(self):
        module = SimplicialResidualAdapter(small_config(), backend="reference")
        prefix, suffix = nn.Linear(64, 64), nn.Linear(64, 5)
        for layer in (prefix, suffix):
            layer.requires_grad_(False)
        before = [p.detach().clone() for layer in (prefix, suffix) for p in layer.parameters()]
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.001)
        x, target = torch.randn(2, 4, 64), torch.randn(2, 4, 5)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = suffix(module(prefix(x)))
            (output - target).square().mean().backward()
            optimizer.step()
        for name, p in module.named_parameters():
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(torch.count_nonzero(p.grad).item(), 0, name)
        after = [p for layer in (prefix, suffix) for p in layer.parameters()]
        for old, p in zip(before, after):
            self.assertTrue(torch.equal(old, p))
            self.assertIsNone(p.grad)

    def test_adapter_state_roundtrip(self):
        a = SimplicialResidualAdapter(small_config(), backend="reference")
        b = SimplicialResidualAdapter(small_config(), seed=91, backend="reference")
        with torch.no_grad():
            a.output.weight.normal_(std=0.02)
        b.load_state_dict(a.state_dict(), strict=True)
        x = torch.randn(1, 5, 64)
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)

    def test_partial_rope_only_rotates_selected_coordinates(self):
        config = small_config()
        x = torch.randn(1, 4, 4, 16)
        positions = torch.arange(4).view(1, 4)
        y = partial_rope(x, positions, config)
        self.assertTrue(torch.equal(y[:, 0], x[:, 0]))
        self.assertTrue(torch.equal(y[..., 4:], x[..., 4:]))
        torch.testing.assert_close(y[..., :4].square().sum(-1), x[..., :4].square().sum(-1))

    def test_added_branch_is_causal_after_output_unzeroed(self):
        module = SimplicialResidualAdapter(small_config(), backend="reference")
        with torch.no_grad():
            module.output.weight.normal_(std=0.02)
        x = torch.randn(1, 6, 64)
        changed = x.clone()
        changed[:, 3:] += 10
        torch.testing.assert_close(module(x)[:, :3], module(changed)[:, :3], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

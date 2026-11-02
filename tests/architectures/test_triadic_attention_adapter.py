"""Matched branch budgets, initialization and useful compensation gradients."""

import unittest

import torch

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
from archlab.architectures.deepseek_v41_matched_mixer import (
    MatchedMixerAdapter,
    matched_mixer_parameter_contract,
)
from archlab.architectures.deepseek_v41_triadic_adapter import (
    V41TriadicAttentionAdapter,
    triadic_adapter_parameter_count,
)


class TriadicAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = V41AdapterConfig(
            width=640, query_heads=2, kv_heads=2, head_dim=128, short_window=32
        )

    def test_actual_four_arm_budgets_and_common_initialization(self):
        adapters = {
            name: MatchedMixerAdapter(self.config, variant=name, backend="reference")
            for name in ("linear", "linsimp", "gdn", "triadic")
        }
        target = None
        for name, adapter in adapters.items():
            contract = adapter.parameter_contract()
            self.assertEqual(
                sum(p.numel() for p in adapter.parameters()), contract["total_parameters"]
            )
            self.assertEqual(contract["compensation_intermediate"] % 16, 0)
            self.assertLess(contract["relative_parameter_error"], 0.005)
            if target is None:
                target = contract["target_parameters"]
            self.assertEqual(contract["target_parameters"], target)
            self.assertEqual(contract, matched_mixer_parameter_contract(self.config, name))
        baseline = adapters["linear"].core
        for adapter in adapters.values():
            torch.testing.assert_close(adapter.core.q.weight, baseline.q.weight, rtol=0, atol=0)
            torch.testing.assert_close(
                adapter.core.output_gate.weight, baseline.output_gate.weight, rtol=0, atol=0
            )
            self.assertEqual(adapter.config, self.config)
            self.assertEqual(adapter.initialization_contract()["common_core_sha256"],
                             adapters["linear"].initialization_contract()["common_core_sha256"])
        gdn, triadic = adapters["gdn"].core, adapters["triadic"].core
        torch.testing.assert_close(gdn.beta_projection.weight, triadic.beta_projection.weight,
                                   rtol=0, atol=0)
        torch.testing.assert_close(gdn.conv_weight, triadic.conv_weight[:gdn.qkv_channels],
                                   rtol=0, atol=0)
        self.assertEqual(adapters["gdn"].initialization_contract()["delta_shared_sha256"],
                         adapters["triadic"].initialization_contract()["delta_shared_sha256"])

    def test_named_initialization_is_deterministic_and_preserves_ambient_rng(self):
        torch.manual_seed(113)
        ambient = torch.get_rng_state().clone()
        first = MatchedMixerAdapter(self.config, variant="triadic", seed=421, backend="reference")
        self.assertTrue(torch.equal(ambient, torch.get_rng_state()))
        second = MatchedMixerAdapter(self.config, variant="triadic", seed=421, backend="reference")
        for name, parameter in first.named_parameters():
            torch.testing.assert_close(parameter, dict(second.named_parameters())[name], rtol=0, atol=0)
        gdn = MatchedMixerAdapter(self.config, variant="gdn", seed=421, backend="reference")
        self.assertEqual(first.initialization_contract()["delta_shared_sha256"],
                         gdn.initialization_contract()["delta_shared_sha256"])
        different = MatchedMixerAdapter(self.config, variant="gdn", seed=422, backend="reference")
        self.assertNotEqual(gdn.initialization_contract()["delta_shared_sha256"],
                            different.initialization_contract()["delta_shared_sha256"])
        initial = first.initialization_contract()
        with torch.no_grad():
            first.core.beta_projection.weight.add_(1)
        self.assertEqual(first.initialization_contract(), initial)
        initial["common_core_sha256"].clear()
        self.assertTrue(first.initialization_contract()["common_core_sha256"])

    def test_zero_initialized_branches_preserve_residual(self):
        streams = torch.randn(1, 3, 4, 640)
        for name in ("linear", "linsimp", "gdn", "triadic"):
            with self.subTest(variant=name):
                adapter = MatchedMixerAdapter(self.config, variant=name, backend="reference")
                torch.testing.assert_close(adapter(streams), streams, rtol=0, atol=0)

    def test_gdn_has_no_unidentifiable_second_feature_parameters(self):
        adapter = V41TriadicAttentionAdapter(self.config, feature_dim=1, backend="reference")
        self.assertIsNone(adapter.k2)
        self.assertIsNone(adapter.q2)
        self.assertEqual(adapter.conv_weight.shape, (3 * 2 * 128, 4))
        self.assertEqual(
            sum(p.numel() for p in adapter.parameters()),
            triadic_adapter_parameter_count(self.config, feature_dim=1),
        )

    def test_activated_triadic_and_compensation_have_finite_useful_gradients(self):
        torch.manual_seed(921)
        adapter = MatchedMixerAdapter(self.config, variant="triadic", backend="reference")
        with torch.no_grad():
            adapter.core.output.weight.normal_(std=0.02)
            adapter.compensation_down.weight.normal_(std=0.02)
        streams = torch.randn(1, 5, 4, 640, requires_grad=True)
        result = adapter(streams)
        (result * torch.randn_like(result)).sum().backward()
        for name, parameter in adapter.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(parameter.grad.isfinite().all())
                self.assertGreater(parameter.grad.norm().item(), 0)
        self.assertTrue(streams.grad.isfinite().all())

    def test_geometry_and_variant_contracts_reject_mismatches(self):
        with self.assertRaisesRegex(ValueError, "equal Q/KV"):
            V41TriadicAttentionAdapter(V41AdapterConfig(), backend="reference")
        with self.assertRaisesRegex(ValueError, "more than one feature"):
            MatchedMixerAdapter(self.config, variant="triadic", feature_dim=1)


if __name__ == "__main__":
    unittest.main()

"""Geometry contracts of the single-stream GDN/Triadic adapter used by Limite delta preludes."""

import unittest

import torch

from archlab.architectures.deepseek_v41_adapter import V41AdapterConfig
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

    def test_gdn_has_no_unidentifiable_second_feature_parameters(self):
        adapter = V41TriadicAttentionAdapter(self.config, feature_dim=1, backend="reference")
        self.assertIsNone(adapter.k2)
        self.assertIsNone(adapter.q2)
        self.assertEqual(adapter.conv_weight.shape, (3 * 2 * 128, 4))
        self.assertEqual(
            sum(p.numel() for p in adapter.parameters()),
            triadic_adapter_parameter_count(self.config, feature_dim=1),
        )

    def test_geometry_and_variant_contracts_reject_mismatches(self):
        with self.assertRaisesRegex(ValueError, "equal Q/KV"):
            V41TriadicAttentionAdapter(V41AdapterConfig(), backend="reference")


if __name__ == "__main__":
    unittest.main()

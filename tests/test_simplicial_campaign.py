"""CPU-only fail-closed campaign and native optimizer audit helper checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from archlab.megatron.simplicial_campaign import run_campaign, source_hashes


class CampaignTests(unittest.TestCase):
    def test_source_fingerprint_tracks_python_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.py").write_text("width = 320\n")
            before = source_hashes(root)
            (root / "log.json").write_text("{}")
            self.assertEqual(before, source_hashes(root))
            (root / "model.py").write_text("width = 640\n")
            self.assertNotEqual(before, source_hashes(root))

    def test_missing_optimizer_gate_prevents_any_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probe = root / "probe"
            probe.mkdir()
            (probe / "INITIALIZATION.json").write_text(
                json.dumps({"common_parameter_sha256": {"weight": "hash"}}))
            (probe / "PROBE_COMPLETE.json").write_text(json.dumps({
                "all_model_weights_bitwise_restored": True, "scheduler_restored": True}))
            options = SimpleNamespace(probe_a=probe, probe_b=probe, probe_c=probe)
            with self.assertRaisesRegex(ValueError, "checkpoint/optimizer gate incomplete"):
                run_campaign(options, root)
            self.assertFalse((root / "CAMPAIGN.json").exists())

    def test_optimizer_audit_includes_master_weights_and_momentum(self):
        import torch

        from archlab.megatron.simplicial_pilot import optimizer_tensor_hashes

        parameter = torch.nn.Parameter(torch.ones(3, 4))
        inner = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
        parameter.grad = torch.full_like(parameter, 0.25)
        inner.step()
        wrapper = SimpleNamespace(optimizer=inner)
        before = optimizer_tensor_hashes(wrapper)
        self.assertTrue(any("master_parameter" in key for key in before))
        self.assertTrue(any("momentum_buffer" in key for key in before))
        self.assertEqual(before, optimizer_tensor_hashes(wrapper, perturb=True))
        self.assertNotEqual(before, optimizer_tensor_hashes(wrapper))


if __name__ == "__main__":
    unittest.main()

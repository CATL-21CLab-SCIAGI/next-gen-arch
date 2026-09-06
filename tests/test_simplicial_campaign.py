"""CPU-only fail-closed campaign and native optimizer audit helper checks."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from archlab.megatron.simplicial_campaign import run_campaign, source_hashes


class CampaignTests(unittest.TestCase):
    def test_dlc_rejects_dp1_proof(self):
        from archlab.megatron.simplicial_dlc_campaign import validate_probes

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "A").mkdir()
            (root / "A/PROBE_COMPLETE.json").write_text(json.dumps({
                "all_model_weights_bitwise_restored": True, "scheduler_restored": True,
                "optimizer_tensors_bitwise_restored": True, "all_ranks_passed": True,
                "dp_world_size": 1}))
            with self.assertRaisesRegex(ValueError, "DP32 checkpoint/optimizer proof"):
                validate_probes(root, root)

    def test_dp_stream_union_and_resume_equal_dp1(self):
        import numpy as np
        import torch

        from archlab.megatron.simplicial_pilot import StridedTokenBatches

        with tempfile.TemporaryDirectory() as directory:
            prefixes = [Path(directory) / f"part{i}" for i in range(3)]
            for i, prefix in enumerate(prefixes):
                np.arange(i * 1000, i * 1000 + 513, dtype=np.int32).tofile(f"{prefix}.bin")
            kwargs = dict(batch_size=4, sequence_len=8, device=torch.device("cpu"))
            serial = StridedTokenBatches(prefixes, start_batch=0, **kwargs)
            expected = [next(serial) for _ in range(64)]
            for world in (1, 2, 32):
                for rank in range(world):
                    stream = StridedTokenBatches(prefixes, rank=rank, world_size=world,
                                                start_batch=0, **kwargs)
                    for local in range(64 // world):
                        actual = next(stream)
                        for key in actual:
                            self.assertTrue(torch.equal(actual[key], expected[local * world + rank][key]))
                    resumed = StridedTokenBatches(prefixes, rank=rank, world_size=world,
                                                  start_batch=32 // world, **kwargs)
                    self.assertTrue(torch.equal(next(resumed)["tokens"], expected[32 + rank]["tokens"]))

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

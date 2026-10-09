"""Independent encoded-store and analytic metric checks for local evaluation."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from archlab.automodel.deepseek_v41_cpu_store import ResidentExpertStore
from archlab.evaluation.deepseek_v41_local_eval import batch_jobs, math_metrics


class LocalMetricTests(unittest.TestCase):
    def test_resident_store_preserves_encoded_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {}
            data = bytearray()
            expected = {}
            for projection in ("w1", "w2", "w3"):
                for suffix, dtype, shape in (
                    ("weight", "I8", [32, 16]),
                    ("scale", "F8_E8M0", [32, 1]),
                ):
                    name = f"layers.0.ffn.experts.0.{projection}.{suffix}"
                    size = shape[0] * shape[1]
                    values = bytes(i % 256 for i in range(size))
                    header[name] = {
                        "dtype": dtype,
                        "shape": shape,
                        "data_offsets": [len(data), len(data) + size],
                    }
                    expected[name] = values
                    data += values
            raw = json.dumps(header, separators=(",", ":")).encode()
            raw += b" " * ((-len(raw)) % 8)
            (root / "weights.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + data)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {name: "weights.safetensors" for name in header}})
            )
            store = ResidentExpertStore(root, threads=2, pinned=False, reserve_gib=0)
            self.assertEqual(store.report["bytes"], len(data))
            for name, values in expected.items():
                self.assertEqual(store.get(name).view(torch.uint8).numpy().tobytes(), values)

    def test_batching_preserves_jobs_and_token_bound(self):
        jobs = [{"id": i, "input_ids": list(range(size))} for i, size in enumerate((5, 2, 7, 3, 4))]
        batches = list(batch_jobs(jobs, 3, 24))
        self.assertEqual(sorted(j["id"] for batch in batches for j in batch), list(range(5)))
        for batch in batches:
            self.assertLessEqual(2 * len(batch) * max(len(j["input_ids"]) for j in batch), 24)

    @unittest.skipUnless(torch.cuda.is_available(), "requires B300")
    def test_math_mask_shift_and_pairing_against_analytic_predictor(self):
        class Head:
            def __call__(self, x, full_logits=True):
                logits = (
                    -torch.arange(16, device=x.device, dtype=torch.float32).expand(
                        *x.shape[:-1], 16
                    )
                    / 10
                )
                logits = logits.clone()
                preferred = (x[..., 0] + 1 + x[..., 1]).long()
                return logits.scatter(-1, preferred.unsqueeze(-1), 4.0)

        class Engine:
            model = SimpleNamespace(head=Head())

            def forward(self, ids, adapted, return_hidden=True):
                ids = ids.cuda()
                m = torch.tensor(adapted, device="cuda").expand(ids.shape[1], -1).T
                hidden = torch.stack((ids, m), -1).float()
                return None, hidden

        class Pilot:
            def batch(self, index, device):
                return torch.tensor([[2, 3, 4, 5]]), torch.tensor([[-100, 4, 5, -100]]), 2

        selected = [
            {
                "pilot_index": i,
                "mode": mode,
                "targets": 2,
                "length": 4,
                "has_tools": False,
                "problem_sha256": str(i),
            }
            for i, mode in enumerate(("low", "medium", "high"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = math_metrics(
                Engine(),
                Pilot(),
                selected,
                Path(directory),
                {"max_batch_size": 4, "max_padded_tokens": 24},
            )["overall"]
        self.assertEqual(result["supervised_targets"], 6)
        self.assertEqual(result["base"]["top1_token_accuracy"], 1.0)
        self.assertEqual(result["adapted"]["top1_token_accuracy"], 0.0)
        self.assertGreater(result["adapted"]["cross_entropy"], result["base"]["cross_entropy"])
        self.assertGreater(result["base_to_adapted_kl"], 0.0)
        self.assertEqual(result["token_argmax_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

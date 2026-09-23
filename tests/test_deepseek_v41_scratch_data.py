import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from archlab.automodel.deepseek_v41_scratch_data import ScratchData
from archlab.preprocessing.v41_scratch import document_windows


def make_data(tmp_path, budget=6):
    (tmp_path / "CONTRACT.json").write_text(
        json.dumps(
            {"training_targets": budget, "validation_targets": 4, "sequence": 4, "pad_id": 2}
        )
    )
    p = tmp_path / "chunks/train-000000"
    p.mkdir(parents=True)
    np.array([0, 11, 12, 13, 1, 0, 21, 22, 1], dtype="<u4").tofile(p / "tokens.u32")
    np.save(p / "windows.npy", np.array([[0, 4, 0], [5, 3, 4]], dtype="<u8"))
    m = {
        "index": 0,
        "split": "train",
        "sequence": 4,
        "windows": 2,
        "targets": 7,
        "tokens_sha256": hashlib.sha256((p / "tokens.u32").read_bytes()).hexdigest(),
        "windows_sha256": hashlib.sha256((p / "windows.npy").read_bytes()).hexdigest(),
    }
    (p / "COMPLETE.json").write_text(json.dumps(m))
    return ScratchData(tmp_path, wait_seconds=0)


class ScratchDataTests(unittest.TestCase):
    def test_document_targets_covered_once(self):
        tokens = list(range(11))
        covered = []
        for start, count in document_windows(tokens, 4):
            covered.extend(tokens[start + 1 : start + count + 1])
        self.assertEqual(covered, tokens[1:])

    def test_budget_and_document_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = make_data(Path(tmp))
            inputs, labels, count = data.batch([0, 1], device="cpu")
            self.assertEqual(inputs.tolist(), [[0, 11, 12, 13], [0, 21, 22, 2]])
            self.assertEqual(labels.tolist(), [[11, 12, 13, 1], [21, 22, -100, -100]])
            self.assertEqual(
                (count, data.targets_before(0), data.targets_before(1), data.targets_before(2)),
                (6, 0, 4, 6),
            )
            self.assertEqual(data.window(2)[1], 0)

    def test_exact_resume_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_data(root, budget=7)
            first = data.batch([1], device="cpu")
            restored = ScratchData(root, wait_seconds=0)
            second = restored.batch([1], device="cpu")
            self.assertEqual(data.targets_before(1), 4)
            self.assertEqual(restored.targets_before(1), 4)
            self.assertTrue(first[0].equal(second[0]) and first[1].equal(second[1]))
            self.assertEqual((first[2], second[2]), (3, 3))

    def test_payload_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = make_data(root)
            (root / "chunks/train-000000/tokens.u32").write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "checksum"):
                data.window(0)

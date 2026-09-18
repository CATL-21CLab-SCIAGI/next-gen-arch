import json
import tempfile
import unittest
from pathlib import Path

from archlab.reporting.checkpoint_retention import sweep, validate


class RetentionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoints = self.root / "checkpoints"
        self.checkpoints.mkdir()
        self.audit = self.root / "audit"
        self.config = {"groups": [{"name": "test", "roots": [str(self.checkpoints)]}]}

    def checkpoint(self, step, complete=True):
        path = self.checkpoints / f"step-{step:06}"
        rank = path / "rank-00"
        rank.mkdir(parents=True)
        (rank / "tensor.pt").write_bytes(b"1234")
        (rank / "optimizer.pt").write_bytes(b"1234")
        (rank / "rng.pt").write_bytes(b"1234")
        marker = {
            "format": "archlab-v41-full-sharded-v1",
            "world_size": 1,
            "contract": {"variant": "test"},
            "cursor": {"step": step, "supervised_tokens": step * 100},
            "manifests": ["rank-00/MANIFEST.json"],
        }
        manifest = {
            **marker,
            "rank": 0,
            "optimizer_states": ["optimizer.pt"],
            "tensors": [
                {
                    "name": "weight",
                    "shape": [2],
                    "dtype": "torch.bfloat16",
                    "chunks": [{"file": "tensor.pt", "elements": 2}],
                }
            ],
        }
        (rank / "MANIFEST.json").write_text(json.dumps(manifest))
        if complete:
            (path / "COMPLETE.json").write_text(json.dumps(marker))
        return path

    def test_keep_last_good_and_in_progress_then_replace(self):
        old = self.checkpoint(1)
        new = self.checkpoint(2)
        writing = self.checkpoint(3, False)
        result = sweep(self.config, self.audit, True)
        self.assertFalse(result["errors"])
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(writing.exists())
        receipts = list((self.audit / "deleted").glob("*/RECEIPT.json"))
        self.assertEqual(len(receipts), 1)
        self.assertTrue(json.loads(receipts[0].read_text())["complete"])
        self.assertTrue((receipts[0].parent / "rank-00/MANIFEST.json").exists())
        again = sweep(self.config, self.audit, True)
        self.assertFalse(again["groups"][0]["removed"])

    def test_invalid_newest_preserves_older(self):
        old = self.checkpoint(1)
        new = self.checkpoint(2)
        (new / "rank-00/tensor.pt").unlink()
        self.assertTrue(sweep(self.config, self.audit, True)["errors"])
        self.assertTrue(old.exists())

    def test_dry_run_and_protected_checkpoint(self):
        old = self.checkpoint(1)
        self.checkpoint(2)
        self.assertFalse(sweep(self.config, self.audit)["errors"])
        self.assertTrue(old.exists())
        self.config["protected"] = {str(old): "required initializer"}
        self.assertFalse(sweep(self.config, self.audit, True)["errors"])
        self.assertTrue(old.exists())

    def test_symlinks_refused(self):
        old = self.checkpoint(1)
        new = self.checkpoint(2)
        (new / "rank-00/tensor.pt").unlink()
        (new / "rank-00/tensor.pt").symlink_to(old / "rank-00/tensor.pt")
        self.assertTrue(sweep(self.config, self.audit, True)["errors"])
        self.assertTrue(old.exists())

    def test_path_traversal_refused(self):
        new = self.checkpoint(1)
        file = new / "COMPLETE.json"
        marker = json.loads(file.read_text())
        marker["manifests"] = ["../../outside/MANIFEST.json"]
        file.write_text(json.dumps(marker))
        with self.assertRaises(ValueError):
            validate(new)

    def test_resume_interrupted_removal(self):
        old = self.checkpoint(1)
        new = self.checkpoint(2)
        staged = self.checkpoints / ".retention-delete-test"
        old.rename(staged)
        receipt = self.audit / "deleted/test/RECEIPT.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps({"original": str(old), "staged": str(staged), "complete": False})
        )
        self.assertFalse(sweep(self.config, self.audit, True)["errors"])
        self.assertFalse(staged.exists())
        self.assertTrue(new.exists())
        self.assertTrue(json.loads(receipt.read_text())["complete"])


if __name__ == "__main__":
    unittest.main()

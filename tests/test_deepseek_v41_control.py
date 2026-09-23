import json
import tempfile
import unittest
from pathlib import Path

from archlab.automodel.deepseek_v41_control import (
    checkpoint_due,
    evaluation_due,
    pending_evaluation,
    pin_checkpoint,
    publish_evaluation_request,
    read_policy,
)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.policy = {
            "format": "archlab-full-eval-policy-v1",
            "checkpoint_interval_steps": 500,
            "evaluation_interval_steps": 500,
            "chat_window_seconds": 1800,
            "pin_steps": [4537],
            "pinned_tokens": {"4537": 756364650},
        }

    def test_global_optimizer_boundaries(self):
        for step, expected in [
            (0, False),
            (499, False),
            (500, True),
            (4999, False),
            (5000, True),
            (5001, False),
        ]:
            self.assertEqual(checkpoint_due(step, self.policy), expected)
            self.assertEqual(evaluation_due(step, self.policy), expected)

    def test_pin_requires_exact_complete_matched_cursor(self):
        checkpoint = self.root / "step-004537"
        checkpoint.mkdir()
        cursor = {"step": 4537, "phase_step": 4230, "supervised_tokens": 756364650}
        with self.assertRaises(FileNotFoundError):
            pin_checkpoint(checkpoint, cursor, self.policy)
        (checkpoint / "COMPLETE.json").write_text(
            json.dumps({"cursor": cursor, "contract": {"variant": "normal"}})
        )
        self.assertTrue(pin_checkpoint(checkpoint, cursor, self.policy))
        self.assertFalse(
            json.loads((checkpoint / "PINNED.json").read_text())["automatic_retention"]
        )
        with self.assertRaises(ValueError):
            pin_checkpoint(checkpoint, {**cursor, "supervised_tokens": 1}, self.policy)

    def test_matched_checkpoint_is_saved_outside_periodic_grid(self):
        self.assertFalse(checkpoint_due(4536, self.policy))
        self.assertTrue(checkpoint_due(4537, self.policy))
        self.assertFalse(checkpoint_due(4538, self.policy))
        self.assertFalse(evaluation_due(4537, self.policy))

    def test_requests_are_idempotent_and_cursor_checked(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        cursor = {"step": 4500, "supervised_tokens": 750000000}
        marker = {"cursor": cursor, "contract": {"variant": "simplicial"}}
        (checkpoint / "COMPLETE.json").write_text(json.dumps(marker))
        path = publish_evaluation_request(self.root / "queue", checkpoint, cursor, "simplicial")
        self.assertEqual(
            path, publish_evaluation_request(self.root / "queue", checkpoint, cursor, "simplicial")
        )
        with self.assertRaises(ValueError):
            publish_evaluation_request(self.root / "queue", checkpoint, cursor, "normal")

    def test_invalid_policy_rejected(self):
        path = self.root / "policy.json"
        path.write_text(json.dumps(self.policy))
        self.assertEqual(read_policy(path), self.policy)
        path.write_text(json.dumps({**self.policy, "evaluation_interval_steps": 0}))
        with self.assertRaises(ValueError):
            read_policy(path)

    def test_only_unfinished_checkpoint_requests_are_scheduled(self):
        queue = self.root / "queue"
        queue.mkdir()
        results = self.root / "results"
        results.mkdir()
        item = {
            "variant": "simplicial",
            "checkpoint": "/fixture/checkpoint",
            "cursor": {"step": 4000},
        }
        name = "simplicial-step-004000.json"
        (queue / name).write_text(json.dumps(item))
        policy = {
            **self.policy,
            "evaluation_requests": str(queue),
            "benchmark_results": str(results),
        }
        self.assertEqual(pending_evaluation(policy, "simplicial"), item)
        (results / name).write_text("{}")
        self.assertIsNone(pending_evaluation(policy, "simplicial"))


if __name__ == "__main__":
    unittest.main()

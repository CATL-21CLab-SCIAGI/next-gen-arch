"""CPU-only tests; runnable with unittest when pytest is not installed."""

import copy
import tempfile
import unittest
from pathlib import Path

from archlab.preprocessing.nemotron_math import (
    EFFORT,
    choose_split,
    file_sha,
    inventory,
    normalize_row,
    problem_key,
    verify_part,
)


class NemotronMathPreprocessingTests(unittest.TestCase):
    def test_problem_split_groups_variants_and_normalizes_whitespace(self):
        left = problem_key("Find  x\n for é.")
        right = problem_key("Find x for e\u0301.")
        self.assertEqual(left, right)
        self.assertEqual(choose_split(left), choose_split(right))
        self.assertEqual(choose_split(left, 0), "train")
        self.assertEqual(choose_split(left, 10000), "validation")

    def test_missing_problem_is_not_silently_split_by_empty_uuid(self):
        for problem in (None, "", "  "):
            with self.assertRaises(ValueError):
                problem_key(problem)

    def test_tool_argument_decode_preserves_source_and_reasoning(self):
        row = {"tools": [], "messages": [
            {"role": "user", "content": "Compute 2+2"},
            {"role": "assistant", "content": "", "reasoning_content": "Use Python",
             "tool_calls": [{"function": {"name": "python", "arguments": '{"code":"2+2"}'}}]},
            {"role": "tool", "content": "4"},
            {"role": "assistant", "content": "4", "reasoning_content": "Checked"},
        ]}
        before = copy.deepcopy(row)
        messages, has_tools = normalize_row(row)
        self.assertTrue(has_tools)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"], {"code": "2+2"})
        self.assertEqual(messages[1]["reasoning_content"], "Use Python")
        self.assertEqual(messages[2]["content"], "4")
        self.assertEqual(row, before)

    def test_bad_tool_arguments_fail_instead_of_dropping_trajectory(self):
        row = {"messages": [{"role": "assistant", "tool_calls": [
            {"function": {"name": "python", "arguments": "[1, 2]"}},
        ]}]}
        with self.assertRaises(ValueError):
            normalize_row(row)

    def test_native_effort_names(self):
        self.assertEqual(EFFORT, {"high": "xhigh", "medium": "medium", "low": "low"})

    def test_inventory_covers_extra_high_shard_without_jsonl_duplicates(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "data"
            directory.mkdir()
            for name in ("high_part00", "high_part01", "high_part02", "medium", "low"):
                pq.write_table(pa.table({"problem": ["a", "b", "c"]}), directory / (name + ".parquet"), row_group_size=2)
                (directory / (name + ".jsonl")).write_text('{"problem":"duplicate"}\n')
            sources, tasks = inventory(temporary, 1, 0)
            self.assertEqual(len(sources), 5)
            self.assertEqual(sum(task["rows"] for task in tasks), 15)
            self.assertEqual(len(tasks), 10)
            self.assertEqual((sources, tasks), inventory(temporary, 1, 0))
            self.assertEqual([task["row_start"] for task in tasks[:2]], [0, 2])

    def test_checksum_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokens.bin"
            path.write_bytes(b"abcd")
            manifest = {"files": [{"name": path.name, "bytes": 4, "sha256": file_sha(path)}]}
            verify_part(temporary, manifest)
            path.write_bytes(b"abce")
            with self.assertRaises(ValueError):
                verify_part(temporary, manifest)


if __name__ == "__main__":
    unittest.main()

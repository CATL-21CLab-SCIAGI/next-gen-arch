"""Leakage and provenance regressions for unseen-problem RL data."""

from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from archlab.rl.nemotron_data import (
    audit_exposure,
    prepare,
    problem_key,
    select_records,
    sha256_file,
)
from archlab.rl.rewards import canonical_math_answer


class NemotronDataTest(unittest.TestCase):
    def exposure(self):
        return {"problem_hashes": {problem_key("already seen")}, "uuids": {"old-uuid"}}

    def row(self, problem, answer="2", uuid="", index=0):
        return {
            "problem": problem,
            "expected_answer": answer,
            "uuid": uuid,
            "messages": [{"role": "assistant", "content": "SECRET SOLUTION"}],
            "_source": {"parquet": "new.parquet", "row": index, "row_group": 0},
        }

    def select(self, rows, **kwargs):
        return select_records(
            rows,
            self.exposure(),
            train_size=1,
            heldout_size=1,
            seed=123,
            canonicalizer=canonical_math_answer,
            **kwargs,
        )

    def test_normalization_matches_historical_split_hash(self):
        from archlab.preprocessing.nemotron_math import problem_key as old_key

        self.assertEqual(problem_key("  cafe\u0301 \n x"), old_key("café x"))
        with self.assertRaises(ValueError):
            problem_key(" ")

    def test_cross_mode_repeats_and_exposure_do_not_leak(self):
        rows = [
            self.row(" already\nseen "),
            self.row("new wording", uuid="old-uuid"),
            self.row("clean A", "0.5", index=2),
            self.row("clean A", "1/2", index=3),
            self.row("clean B", "3", index=4),
        ]
        train, heldout, stats = self.select(rows)
        selected = train + heldout
        self.assertEqual({x["prompt"][0]["content"] for x in selected}, {"clean A", "clean B"})
        self.assertEqual(len({x["problem_sha256"] for x in selected}), 2)
        self.assertEqual(stats["previously_exposed_problem_or_uuid"], 2)
        self.assertNotIn("SECRET SOLUTION", json.dumps(selected))

    def test_conflicting_gold_and_uuid_aliases_are_discarded(self):
        rows = [
            self.row("conflict", "2"),
            self.row("conflict", "3"),
            self.row("alias a", uuid="same-id"),
            self.row("alias b", uuid="same-id"),
            self.row("valid a"),
            self.row("valid b"),
        ]
        train, heldout, stats = self.select(rows)
        self.assertEqual(
            {x["prompt"][0]["content"] for x in train + heldout}, {"valid a", "valid b"}
        )
        self.assertEqual(stats["conflicting_or_aliased_problems"], 3)

    def test_order_does_not_change_split_or_duplicate_choice(self):
        rows = [self.row("a", index=3), self.row("a", index=1), self.row("b", index=2)]
        first = self.select(rows)
        second = self.select(reversed(rows))
        self.assertEqual(first, second)
        self.assertEqual(
            next(x for x in first[0] + first[1] if x["prompt"][0]["content"] == "a")["source"][
                "row"
            ],
            1,
        )

    def fixture(self, root):
        """One exposed group with BOTH partitions; unexposed candidates follow."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        source = root / "raw"
        (source / "data").mkdir(parents=True)
        data = source / "data" / "low.parquet"
        rows = [
            {
                "problem": f"question {i}",
                "expected_answer": str(i),
                "uuid": f"uuid-{i}",
                "messages": "SECRET SOLUTION",
            }
            for i in range(8)
        ]
        pq.write_table(pa.Table.from_pylist(rows), data, row_group_size=2)
        old = root / "prepared"
        part_id = "low-rg00000-00001"
        directory = old / "parts" / part_id
        directory.mkdir(parents=True)
        files = []
        for split, index in (("train", 0), ("validation", 1)):
            path = directory / f"{split}.metadata.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                stream.write(
                    json.dumps(
                        {
                            "problem_sha256": problem_key(f"question {index}"),
                            "uuid": f"uuid-{index}",
                        }
                    )
                    + "\n"
                )
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        ready = {
            "status": "complete",
            "contract_sha256": "contract",
            "parts": [{"id": part_id, "documents": 2, "files": files}],
        }
        (old / "DATA_READY.json").write_text(json.dumps(ready))
        (old / "manifest.json").write_text(
            json.dumps({"sources": [{"path": str(data), "bytes": data.stat().st_size, "rows": 8}]})
        )
        pilot = root / "pilot"
        pilot.mkdir()
        (pilot / "windows.jsonl").write_text(
            json.dumps(
                {"prefix": f"parts/{part_id}/train", "problem_sha256": problem_key("question 0")}
            )
            + "\n"
        )
        # Only training is explicitly selected. Audit must exclude validation
        # too because it treats the entire referenced part as exposed.
        spec = files[0]
        manifest = {
            "source": str(old),
            "source_ready_sha256": sha256_file(old / "DATA_READY.json"),
            "source_contract_sha256": "contract",
            "split": "train",
            "supervised_tokens": 10,
            "selected_conversations": 1,
            "windows_sha256": sha256_file(pilot / "windows.jsonl"),
            "verified_source_files": {f"parts/{part_id}/{spec['name']}": spec},
        }
        marker = pilot / "PILOT_READY.json"
        marker.write_text(json.dumps(manifest))
        return source, marker, directory / files[1]["name"]

    def test_entire_part_excluded_and_changed_sidecar_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, pilot, validation = self.fixture(Path(temporary))
            audit = audit_exposure([pilot])
            self.assertEqual(
                audit["problem_hashes"], {problem_key("question 0"), problem_key("question 1")}
            )
            self.assertEqual(audit["row_groups"], {"low.parquet": {0}})
            with validation.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Changed historical sidecar"):
                audit_exposure([pilot])

    def test_real_parquet_projection_and_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, pilot, _ = self.fixture(root)
            output = root / "output"
            manifest = prepare(source, output, [pilot], train_size=3, heldout_size=2, max_rows=6)
            records = [
                json.loads(line)
                for name in ("train", "heldout")
                for line in (output / f"{name}.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 5)
            self.assertNotIn("SECRET SOLUTION", json.dumps(records))
            self.assertTrue(all(row["source"]["row_group"] > 0 for row in records))
            self.assertEqual(manifest["statistics"]["scanned_rows"], 6)
            self.assertFalse(manifest["training_authorized"])
            self.assertEqual(manifest["solution_columns_read"], [])
            self.assertEqual(
                manifest["files"]["train"]["sha256"], sha256_file(output / "train.jsonl")
            )
            with self.assertRaises(FileExistsError):
                prepare(source, output, [pilot], train_size=3, heldout_size=2, max_rows=6)


if __name__ == "__main__":
    unittest.main()

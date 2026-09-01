from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from archlab.megatron import data

pyarrow = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq  # noqa: E402


def _write_source(path: Path, texts: list[str], counts: list[int]) -> None:
    table = pyarrow.table({"text": texts, "token_count": counts})
    pq.write_table(table, path)


def test_parquet_inventory_and_assignment_hold_out_one_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _write_source(source / "000.parquet", ["a", "b"], [1, 2])
    _write_source(source / "001.parquet", ["c"], [3])
    _write_source(source / "002.parquet", ["validation"], [4])
    manifest = tmp_path / "SOURCE_MANIFEST.json"

    data._inventory_parquet(
        argparse.Namespace(
            source_root=source,
            pattern="*.parquet",
            expected_shards=3,
            validation_source="002.parquet",
            text_column="text",
            source_token_column="token_count",
            output=manifest,
        )
    )

    payload = json.loads(manifest.read_text())
    assert payload["total_shards"] == 3
    assert payload["total_rows"] == 4
    assert payload["total_source_tokens"] == 10
    assert payload["validation_source"] == "002.parquet"

    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")
    common = {
        "source_root": source,
        "source_manifest": manifest,
        "output_root": tmp_path / "output",
        "tokenizer": tokenizer,
        "nodes": 1,
        "node_rank": 0,
        "workers": 1,
        "qwen_eos_id": 151_643,
        "document_batch_size": 16,
        "text_column": "text",
        "source_token_column": "token_count",
    }
    train = data._assigned_parquet_jobs(
        argparse.Namespace(**common, split="train", expected_shards=2)
    )
    valid = data._assigned_parquet_jobs(
        argparse.Namespace(**common, split="val", expected_shards=1)
    )

    assert [Path(path).name for path in train[0].sources] == ["000.parquet", "001.parquet"]
    assert [Path(path).name for path in valid[0].sources] == ["002.parquet"]


def test_parquet_assignment_rejects_changed_source_size(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "000.parquet"
    _write_source(path, ["a"], [1])
    manifest = tmp_path / "SOURCE_MANIFEST.json"
    data._inventory_parquet(
        argparse.Namespace(
            source_root=source,
            pattern="*.parquet",
            expected_shards=1,
            validation_source="000.parquet",
            text_column="text",
            source_token_column="token_count",
            output=manifest,
        )
    )
    path.write_bytes(path.read_bytes() + b"drift")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}")

    with pytest.raises(RuntimeError, match="source artifact changed"):
        data._assigned_parquet_jobs(
            argparse.Namespace(
                source_root=source,
                source_manifest=manifest,
                output_root=tmp_path / "output",
                tokenizer=tokenizer,
                split="val",
                expected_shards=1,
                nodes=1,
                node_rank=0,
                workers=1,
                qwen_eos_id=151_643,
                document_batch_size=16,
                text_column="text",
                source_token_column="token_count",
            )
        )

import gzip
import json

import numpy as np
import pytest

from archlab.automodel.limite_adapter_data import MathWindows, document_window_offsets, prepare


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prefixes = []
    documents = {}
    for split, lengths in (("train", (8, 9, 10, 16, 17, 18, 12)), ("validation", (6, 11))):
        prefix = f"part/{split}-text_document"
        (source / "part").mkdir(exist_ok=True)
        prefixes.append(prefix)
        offset, tokens, rows = 0, [], []
        for index, length in enumerate(lengths):
            key = f"{split}-{index}"
            values = [index + 1] * (length - 2) + [151645, 198]
            documents[key] = dict(prefix=prefix, offset=offset, tokens=values)
            rows.append(dict(problem_sha256=key, tokens=length))
            tokens.extend(values)
            offset += length
        np.asarray(tokens, dtype="<i4").tofile(source / (prefix + ".bin"))
        with gzip.open(source / (prefix + ".metadata.jsonl.gz"), "wt") as handle:
            handle.write("".join(json.dumps(row) + "\n" for row in rows))
    (source / "DATA_READY.json").write_text(json.dumps({
        "prefixes": {"high": prefixes}, "contract_sha256": "fixture-source-contract",
    }))
    heldout = tmp_path / "heldout.jsonl"
    heldout.write_text(json.dumps({"problem_sha256": "train-6"}) + "\n")
    return source, heldout, documents


def test_legacy_order_and_index_payload_remain_unchanged(tmp_path):
    source, heldout, _ = _source(tmp_path)
    output = tmp_path / "legacy"
    prepare(source, output, heldout, context=8)
    expected = np.asarray([[0, x] for x in (8, 17, 27, 43, 51, 60, 68)], dtype=np.int64)
    np.random.default_rng(42).shuffle(expected)
    np.testing.assert_array_equal(np.load(output / "train.npy"), expected)
    manifest = json.loads((output / "READY.json").read_text())
    assert "tail_policy" not in manifest
    assert manifest["counts"] == {"train": 7, "validation": 1}
    assert manifest["excluded_rl_heldout_documents"] == 1
    assert MathWindows(output).contract == "fixture-source-contract"


def test_corrected_windows_retain_every_long_document_end_without_padding(tmp_path):
    source, heldout, documents = _source(tmp_path)
    output = tmp_path / "corrected"
    prepare(source, output, heldout, context=8, tail_policy="overlap-final-window")
    dataset = MathWindows(output)
    manifest = dataset.spec
    assert manifest["format"] == "archlab-limite-math-windows-v2"
    assert dataset.contract == manifest["window_contract_sha256"]
    assert dataset.contract != manifest["source_contract"]
    assert manifest["counts"] == {"train": 10, "validation": 2}
    assert manifest["coverage"]["train"] == dict(
        documents=6, retained_documents=5, excluded_short_documents=1,
        excluded_short_tokens=8, unique_target_tokens=65, repeated_target_tokens=15,
        additional_tail_windows=3,
    )
    for name, document in documents.items():
        if not name.startswith("train-") or name == "train-6":
            continue
        start, values = document["offset"], document["tokens"]
        indices = [i for i, (_, row_start) in enumerate(dataset.order)
                   if start <= row_start < start + len(values)]
        if len(values) <= 8:
            assert indices == []
            continue
        terminal_targets = []
        for index in indices:
            window = dataset[index]
            local = int(dataset.order[index, 1]) - start
            np.testing.assert_array_equal(window, values[local:local + 9])
            assert len(window) == 9
            terminal_targets.extend(range(local + 1, local + 9))
        assert len(values) - 2 in terminal_targets  # Actual EOS is a target.
        assert len(values) - 1 in terminal_targets  # Trailing newline is preserved.
        assert max(terminal_targets) == len(values) - 1
        assert len({int(dataset.order[i, 1]) for i in indices}) == len(indices)


def test_policy_mismatch_cannot_silently_reuse_a_sealed_legacy_dataset(tmp_path):
    source, heldout, _ = _source(tmp_path)
    output = tmp_path / "sealed"
    prepare(source, output, heldout, context=8)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="different tail policy"):
        prepare(source, output, heldout, context=8, tail_policy="overlap-final-window")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


@pytest.mark.parametrize("length,expected", [(8, []), (9, [0]), (10, [0, 1]),
                                            (17, [0, 8]), (18, [0, 8, 9])])
def test_tail_boundary_and_exact_multiple_deduplication(length, expected):
    assert document_window_offsets(length, 8, tail_policy="overlap-final-window") == expected


def test_empty_split_is_a_valid_two_column_index(tmp_path):
    source, heldout, _ = _source(tmp_path)
    output = tmp_path / "empty"
    prepare(source, output, heldout, context=128, tail_policy="overlap-final-window")
    assert np.load(output / "train.npy").shape == (0, 2)
    assert np.load(output / "validation.npy").shape == (0, 2)


def test_corrected_index_contract_is_deterministic_and_refuses_context_change(tmp_path):
    source, heldout, _ = _source(tmp_path)
    first, second = tmp_path / "first", tmp_path / "second"
    for output in (first, second):
        prepare(source, output, heldout, context=8, tail_policy="overlap-final-window")
    assert MathWindows(first).contract == MathWindows(second).contract
    with pytest.raises(ValueError, match="different data contract"):
        prepare(source, first, heldout, context=7, tail_policy="overlap-final-window")
    broken = MathWindows(first)
    del broken.spec["window_contract_sha256"]
    with pytest.raises(ValueError, match="lacks its data-order contract"):
        _ = broken.contract

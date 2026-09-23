import json

import pytest

from archlab.reporting.runs import (
    decode_jsonl,
    matched_rows,
    read_training_run,
    token_smooth,
    token_window_mean,
)


def history():
    return [dict(step=1, supervised_tokens=2, consumed_supervised_tokens=12, loss=2., learning_rate=.1),
            dict(step=2, supervised_tokens=6, consumed_supervised_tokens=18, loss=4., learning_rate=.1)]


def test_live_reader_preserves_unicode_and_rejects_corrupt_published_records():
    line = json.dumps({'label':'a\u2028b\u2029c'}, ensure_ascii=False).encode() + b'\n'
    assert decode_jsonl(line + b'{"partial":') == [{'label':'a\u2028b\u2029c'}]
    with pytest.raises(json.JSONDecodeError):
        decode_jsonl(line + b'{"corrupt":\n')
    with pytest.raises(ValueError):
        decode_jsonl(b'1\n')


def test_token_weighting_uses_exact_partial_overlap():
    rows = history()
    assert token_window_mean(rows, 11, 15) == 3.5
    x, y = token_smooth(rows, 7)
    assert x.tolist() == [12, 18]
    assert y.tolist() == pytest.approx([2, 26 / 7])


def test_matching_rejects_cursor_and_schedule_mismatches(tmp_path):
    rows = history()
    assert matched_rows(rows, rows[:1]) == (rows[:1], rows[:1])
    for field, value in [('learning_rate', .2), ('supervised_tokens', 3), ('step', 4)]:
        bad = [dict(r) for r in rows]
        bad[0][field] = value
        with pytest.raises(ValueError):
            matched_rows(rows, bad)
    (tmp_path/'prior-phase-metrics.jsonl').write_text(json.dumps(rows[0])+'\n')
    (tmp_path/'train-metrics.jsonl').write_text(json.dumps(rows[1])+'\n')
    actual, sources = read_training_run(tmp_path)
    assert actual == rows and len(sources) == 2


def test_formatted_source_receipts_bind_current_bytes_and_ast():
    import hashlib
    from pathlib import Path

    from archlab.source_compatibility import formatting_predecessor

    root = Path(__file__).resolve().parents[1] / 'src/archlab'
    records = json.loads((root/'data/source-formatting-20260923.json').read_text())['files']
    assert records
    for relative, record in records.items():
        path = root/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        previous, proof = formatting_predecessor(relative, path, actual)
        assert proof == record and previous == record['before_sha256']
        assert formatting_predecessor(relative, path, '0'*64) == ('0'*64, None)

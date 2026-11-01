"""Native-replay admission of served Limite samples and paired held-out pass@1."""

import json
import math
from argparse import Namespace

import pytest
from test_training_limite import SNAPSHOT, native, policy  # noqa: F401  (native is a fixture)

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.evaluation import limite_heldout as heldout
from archlab.evaluation.limite_math import paired_cluster_bootstrap
from archlab.training import limite


def test_logratio_stats_use_the_serving_contract_estimators():
    stats = heldout.logratio_stats([0.1, -0.2, 0.0, 0.3])
    assert stats["tokens"] == 4 and stats["maximum_absolute"] == pytest.approx(0.3)
    assert stats["mean"] == pytest.approx(0.05)
    assert stats["k3_kl_estimate"] == pytest.approx(
        sum(math.exp(v) - v - 1 for v in (0.1, -0.2, 0.0, 0.3)) / 4)
    with pytest.raises(ValueError, match="finite"):
        heldout.logratio_stats([0.0, float("nan")])


def test_paired_cluster_bootstrap_resamples_problems_and_reports_direction():
    tie = paired_cluster_bootstrap([0.25, 0.5, 1.0], [0.25, 0.5, 1.0], draws=200)
    assert tie["difference"] == 0 and tie["cluster_bootstrap_95pct"] == [0, 0] and not tie["excludes_zero"]
    gain = paired_cluster_bootstrap([0.0] * 20, [0.5] * 10 + [1.0] * 10, draws=500)
    assert gain["difference"] == pytest.approx(0.75) and gain["excludes_zero"]
    assert gain == paired_cluster_bootstrap([0.0] * 20, [0.5] * 10 + [1.0] * 10, draws=500)
    with pytest.raises(ValueError, match="aligned"):
        paired_cluster_bootstrap([0.0], [0.0, 1.0])


def test_admission_uses_the_first_draw_of_the_first_split_problems():
    records = [dict(uuid=u, sample=s) for u in ("c", "a", "b") for s in (1, 0)]
    assert [r["uuid"] for r in heldout.admission_records(records, ["a", "b", "c"], 2)] == ["a", "b"]
    assert all(r["sample"] == 0 for r in heldout.admission_records(records, ["a", "b", "c"], 3))
    with pytest.raises(ValueError, match="first draw"):
        heldout.admission_records(records[:1], ["a", "b"], 2)


def _traces(root, export, rows, serving=heldout.SERVING):
    root.mkdir()
    shard = root / "heldout-shard-000-of-001.jsonl"
    shard.write_text("".join(json.dumps(row) + "\n" for row in rows))
    atomic_write_json(root / "heldout-shard-000-of-001.complete.json", dict(
        file=shard.name, file_sha256=sha256_file(shard), response_logprobs="raw", serving=serving,
        export_receipt_sha256=sha256_file(export / "EXPORT.json")))


def test_native_replay_admits_matching_serving_logprobs_and_rejects_a_shifted_token(
        tmp_path, native):  # noqa: F811
    export = tmp_path / "export"
    limite.export_model(policy(native), SNAPSHOT, export)
    model = limite.LimiteForCausalLM.from_pretrained(export)
    split = tmp_path / "heldout.jsonl"
    split.write_text("".join(json.dumps(dict(uuid=u)) + "\n" for u in ("p0", "p1")))
    rows = []
    for index, uuid in enumerate(("p0", "p1")):
        for sample in range(2):
            prompt, response = [1, 4 + index, 8], [5, 9 + sample, 3, 2]
            served = [value + 1e-3 for value in heldout.native_logprobs(model, prompt, response)]
            rows.append(dict(uuid=uuid, sample=sample, prompt_ids=prompt, response_ids=response,
                             response_logprobs=served, correct=index == sample,
                             finish_reason="stop" if sample == 0 else "length"))
    _traces(tmp_path / "good", export, rows)
    shifted = [dict(row, response_logprobs=list(row["response_logprobs"])) for row in rows]
    shifted[0]["response_logprobs"][1] += 0.6
    _traces(tmp_path / "shifted", export, shifted)

    def admit(name):
        return heldout.admit(Namespace(export=export, traces=tmp_path / name, split=split, problems=2,
                                       device="cpu", output=tmp_path / name / "ADMISSION.json"))

    good = admit("good")
    assert good["passed"] and good["aggregate"]["tokens"] == 8
    assert good["aggregate"]["maximum_absolute"] == pytest.approx(1e-3, abs=1e-6)
    bad = admit("shifted")
    assert not bad["passed"] and bad["aggregate"]["maximum_absolute"] > heldout.MAXIMUM_ABSOLUTE_LOGRATIO
    _traces(tmp_path / "prefix-cached", export, rows, serving=dict(heldout.SERVING, enable_prefix_caching=True))
    with pytest.raises(ValueError, match="serving contract"):
        admit("prefix-cached")

    summary = heldout.summarize(Namespace(
        split=split, samples=2, reference=["teacher"], output=tmp_path / "SUMMARY.json",
        model=[("teacher", tmp_path / "good"), ("arm", tmp_path / "good"), ("broken", tmp_path / "shifted")]))
    assert summary["models"]["teacher"]["heldout"]["pass_at_1"] == pytest.approx(0.5)
    assert summary["models"]["teacher"]["heldout"]["truncation_rate"] == pytest.approx(0.5)
    assert "heldout" not in summary["models"]["broken"] and not summary["models"]["broken"]["admission"]["passed"]
    assert set(summary["paired_pass_at_1"]) == {"arm-minus-teacher"}
    assert summary["paired_pass_at_1"]["arm-minus-teacher"]["difference"] == 0

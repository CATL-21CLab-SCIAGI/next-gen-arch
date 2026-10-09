"""Scientific scoring and repeated-sample contracts for the Limite benchmark."""

import pytest

from archlab.evaluation.limite_math import (
    benchmark_seed,
    compare_aime,
    first_eos_prefix,
    pass_at_k,
    score_aime,
    summarize_aime,
)


@pytest.mark.parametrize("text,expected", [
    (r"\boxed{7}", True),
    (r"\boxed{007}", True),
    (r"\boxed{14/2}", True),
    (r"\boxed{8}", False),
    (r"<think>\boxed{7}", False),
    (r"<think>\boxed{7}</think>\boxed{8}", False),
    (r"<think>\boxed{8}</think>\boxed{7}", True),
    (r"<think>7</think><think>\boxed{7}", False),
    (r"The candidates include 7, 8 and 9.", False),
])
def test_symbolic_final_answer(text, expected):
    assert score_aime(text, "7", "eos")["correct"] is expected


def test_benchmark_censoring_is_visible_without_training_reward_rule():
    result = score_aime(r"\boxed{7}", "7", "length")
    assert result["correct"]
    assert result["truncated"]
    assert not result["natural_eos"]
    with pytest.raises(ValueError):
        score_aime(r"\boxed{7}", "7", "repetition")


def test_pass_at_k_has_valid_sample_count():
    assert pass_at_k(2, 4, 1) == .5
    assert pass_at_k(2, 4, 2) == pytest.approx(5 / 6)
    assert pass_at_k(2, 4, 4) == 1
    assert pass_at_k(0, 4, 4) == 0
    with pytest.raises(ValueError):
        pass_at_k(1, 1, 4)


def test_dual_eos_correction_preserves_prefix_without_resampling():
    original = [7, 151643, 151643, 8, 151645]
    assert first_eos_prefix(original, [151643, 151645]) == ([7, 151643], 'eos')
    assert first_eos_prefix(original, [151645]) == (original, 'eos')
    assert first_eos_prefix([7, 8], [151643, 151645]) == ([7, 8], 'length')
    assert original == [7, 151643, 151643, 8, 151645]


def records(scores):
    return [dict(problem_id=f"aime25:{index}", sample_index=sample,
                 correct=bool(correct), generated_tokens=10, finish_reason="eos")
            for index, values in enumerate(scores) for sample, correct in enumerate(values)]


def test_problem_cluster_summary_and_partial_records():
    rows = records([[1, 0, 1, 0], [0, 0, 0, 0]])
    summary = summarize_aime(rows, ["aime25:0", "aime25:1"], 4, bootstrap_draws=100)
    assert summary["complete"]
    assert summary["pass_at_1"] == .25
    assert summary["pass_at_4"] == .5
    assert summary["problems"] == 2
    assert summary["independent_responses"] == 8
    assert summary["first_sample_pass_at_1"] == .5
    assert "problem" in summary["uncertainty_unit"]
    assert summarize_aime(rows[:-1], ["aime25:0", "aime25:1"], 4)["complete"] is False
    with pytest.raises(ValueError, match="duplicate"):
        summarize_aime(rows + rows[:1], ["aime25:0", "aime25:1"], 4)


def test_paired_comparison_and_independent_seed():
    first = records([[0, 1, 0, 0], [1, 0, 0, 0]])
    second = records([[1, 0, 0, 0], [1, 0, 0, 0]])
    result = compare_aime(first, second, ["aime25:0", "aime25:1"])
    assert result["gains"] == 1
    assert result["regressions"] == 0
    assert result["delta_percentage_points"] == 50
    assert result["mcnemar_exact_two_sided_p"] == 1
    seeds = {benchmark_seed(42, problem, sample) for problem in ["aime24:0", "aime25:0"] for sample in range(4)}
    assert len(seeds) == 8
    assert benchmark_seed(42, "aime24:0", 0) == benchmark_seed(42, "aime24:0", 0)


def test_live_report_does_not_score_partial_exam_or_partial_append(tmp_path):
    import json

    from archlab.evaluation.limite_math_report import build_report, read_complete_rows

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'cases.jsonl').write_text(''.join(json.dumps(dict(id=f'aime25:{i}', task='aime25')) + '\n'
                                             for i in range(2)))
    output = tmp_path / 'output'
    shard = output / 'model' / 'shard-0'
    shard.mkdir(parents=True)
    rows = records([[1]])
    rows[0].update(model='model', task='aime25')
    path = shard / 'records.jsonl'
    path.write_text(json.dumps(rows[0]) + '\n' + '{"incomplete":')
    assert read_complete_rows(path) == rows
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(models=[dict(name='model')], output=str(output),
                                   sampling=dict(samples_per_problem=4))))
    registry = dict(bundle=str(bundle), models=[dict(name='model', plan=str(plan))], interpretation=[])
    report = build_report(registry, bootstrap_draws=100)
    assert not report['complete']
    result = report['models']['model']['exams']['aime25']
    assert not result['first_sample']['complete']
    assert 'pass_at_1' not in result['first_sample']
    assert report['paired_first_sample'] == {}


def test_report_waits_for_final_rl_added_by_queue(tmp_path):
    import json

    from archlab.evaluation.limite_math_report import build_report

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'cases.jsonl').write_text(json.dumps(dict(id='aime25:0', task='aime25')) + '\n')
    registry = dict(bundle=str(bundle), models=[], interpretation=[],
                    required_final_rl_variants=['normal', 'simplicial'])

    def register(name, kind, variant):
        shard = tmp_path / name / 'shard-0'
        shard.mkdir(parents=True)
        row = dict(records([[1]])[0], model=name, task='aime25')
        (shard / 'records.jsonl').write_text(json.dumps(row) + '\n')
        plan = tmp_path / (name + '.json')
        plan.write_text(json.dumps(dict(models=[dict(name=name, kind=kind, variant=variant)],
                                       output=str(tmp_path), sampling=dict(samples_per_problem=1))))
        registry['models'].append(dict(name=name, plan=str(plan)))

    register('base', 'base', 'base')
    register('published-violetto', 'published_rl', 'base')
    report = build_report(registry, bootstrap_draws=100)
    assert not report['complete']
    assert report['pending_final_rl_variants'] == ['normal', 'simplicial']
    register('normal-final', 'rl', 'normal')
    assert not build_report(registry, bootstrap_draws=100)['complete']
    register('simplicial-final', 'rl', 'simplicial')
    assert build_report(registry, bootstrap_draws=100)['complete']


def test_overlap_uses_verified_training_metadata_and_actual_cursor(tmp_path):
    import gzip
    import json

    import numpy as np

    from archlab.artifacts import sha256_file
    from archlab.evaluation.limite_math_overlap import audit_sft
    from archlab.preprocessing.nemotron_math import problem_key

    source = tmp_path / 'source'
    part = source / 'parts' / 'one'
    part.mkdir(parents=True)
    metadata = part / 'train-text.metadata.jsonl.gz'
    with gzip.open(metadata, 'wt') as handle:
        for index, question in enumerate(['training-only', 'AIME question']):
            handle.write(json.dumps(dict(problem_sha256=problem_key(question), tokens=8,
                                         source='original.parquet', source_row=index)) + '\n')
    (part / 'READY.json').write_text(json.dumps(dict(files=[dict(
        name=metadata.name, bytes=metadata.stat().st_size, sha256=sha256_file(metadata))])))
    (source / 'manifest.json').write_text(json.dumps(dict(sources=[])))
    data = tmp_path / 'data'
    data.mkdir()
    np.save(data / 'train.npy', np.array([[0, 0], [0, 8]]))
    (data / 'READY.json').write_text(json.dumps(dict(source=str(source), context=2,
        prefixes=['parts/one/train-text'], files={'train.npy': sha256_file(data / 'train.npy')})))
    cases = [dict(id='aime25:0', question='AIME question')]
    assert audit_sft(cases, data, tokens=2, global_windows_per_step=1)['seen_problem_ids'] == []
    assert audit_sft(cases, data, tokens=4, global_windows_per_step=1)['seen_problem_ids'] == ['aime25:0']
    with metadata.open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='metadata changed'):
        audit_sft(cases, data, tokens=4, global_windows_per_step=1)

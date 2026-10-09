"""Full-context budgeting and pinned external evaluation dependency checks."""

import pytest

from archlab.artifacts import sha256_file
from archlab.evaluation.limite_pipeline import (
    SealedDataset,
    completion_budget,
    validate_pipeline_contract,
)


def test_completion_budget_removes_32k_cap_but_reserves_prompt():
    assert completion_budget(317, 131072, 131072) == 130755
    assert completion_budget(317, 32768, 131072) == 32768
    assert completion_budget(131071, 131072, 131072) == 1
    for prompt in (0, -1, 131072, 131073):
        with pytest.raises(ValueError, match="prompt"):
            completion_budget(prompt, 131072, 131072)


def test_no_toy_fallback_and_dependency_identity(tmp_path):
    module = tmp_path / 'custom_eval/eval_aime26.py'
    module.parent.mkdir()
    module.write_text('# pinned dependency')
    cases = [dict(task='aime26') for _ in range(30)]
    plan = dict(sampling=dict(context_limit=131072, max_new_tokens=131072,
                             budget_policy='native_context_minus_prompt', top_k=0, repetition_watchdog=False),
                eval_pipeline=dict(source=str(tmp_path), upstream_revision='fixed', integration_patch_sha256='patch',
                                   files_sha256={'custom_eval/eval_aime26.py': sha256_file(module)}))
    validate_pipeline_contract(plan, cases)
    with pytest.raises(ValueError, match='complete sealed'):
        validate_pipeline_contract(plan, cases[:20])
    module.write_text('# changed dependency')
    with pytest.raises(ValueError, match='source changed'):
        validate_pipeline_contract(plan, cases)


def test_sealed_dataset_selection_preserves_exact_questions_and_answers():
    rows = SealedDataset([dict(question='real question', answer='0'), dict(question='other', answer='123')])
    selected = rows.select(range(1, 2))
    assert isinstance(selected, SealedDataset) and selected == [rows[1]]


def test_native_runner_uses_remaining_context_and_resumes_without_resampling(tmp_path, monkeypatch):
    pytest.importorskip('transformers')
    from types import SimpleNamespace

    import archlab.automodel.limite_pipeline_benchmark as worker

    plan = dict(sampling=dict(temperature=.6, top_p=.95, max_new_tokens=131072, context_limit=131072,
                              samples_per_problem=4, seed=42, eos_token_ids=[151643, 151645],
                              budget_policy='native_context_minus_prompt'),
                implementation_sha256='frozen', eval_pipeline=dict(upstream_revision='upstream'))
    phase = dict(name='sft-normal-10B')
    cases = [dict(id='aime26:0', task='aime26', question='Q', answer='4')]
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1] * 317)
    pipeline = SimpleNamespace(extract_final_answer_robust=lambda text: '4', is_math_correct=lambda a, b: a == b)
    monkeypatch.setattr(worker.torch, 'tensor', lambda value, **kwargs: value)
    monkeypatch.setattr(worker.torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(worker, 'score_aime', lambda *a: dict(correct=True, truncated=False))
    calls = []

    def sample(model, ids, **kwargs):
        calls.append(kwargs)
        return dict(completion='</think>\\boxed{4}', generated_ids=[4, 151643], generated_tokens=2,
                    finish_reason='eos', seconds=1., seed=kwargs['seed'])

    monkeypatch.setattr(worker, 'sample_math', sample)
    request = dict(messages_list=[[dict(role='user', content='Q')]], max_new_tokens=131072,
                   temperature=.6, top_p=.95, enable_thinking=True)
    for _ in range(2):
        runner = worker.NativeRunner(object(), tokenizer, None, pipeline, plan, phase, cases, tmp_path)
        assert runner.chat_generate(**request) == ['</think>\\boxed{4}']
    assert len(calls) == 1 and calls[0]['max_new_tokens'] == 130755
    assert calls[0]['eos_token_ids'] == (151643, 151645)
    with pytest.raises(ValueError, match='contract'):
        runner.chat_generate(**dict(request, top_p=.9))


def test_report_keeps_pipeline_credit_separate_from_final_answer_accuracy(tmp_path):
    import json

    from archlab.evaluation.limite_math_report import build_report

    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'cases.jsonl').write_text(json.dumps(dict(id='aime26:0', task='aime26')) + '\n')
    output = tmp_path / 'responses' / 'sft' / 'shard-0'
    output.mkdir(parents=True)
    row = dict(model='sft', problem_id='aime26:0', sample_index=0, correct=False, pipeline_correct=True,
               finish_reason='length', generated_tokens=130755)
    (output / 'records.jsonl').write_text(json.dumps(row) + '\n')
    plan = tmp_path / 'PLAN.json'
    plan.write_text(json.dumps(dict(models=[dict(name='sft', kind='sft')], sampling=dict(samples_per_problem=1),
                                    output=str(tmp_path / 'responses'), evaluation_backend='eval_pipeline')))
    report = build_report(dict(bundle=str(bundle), models=[dict(name='sft', plan=str(plan))], interpretation=[]),
                          bootstrap_draws=100)
    exam = report['models']['sft']['exams']['aime26']
    assert exam['repeated']['pass_at_1'] == 0 and exam['pipeline_repeated']['pass_at_1'] == 1
    assert exam['repeated']['truncation_rate'] == 1

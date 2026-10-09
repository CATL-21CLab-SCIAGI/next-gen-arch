"""Complete-exam reporting for independently queued Limite benchmark models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.evaluation.limite_math import compare_aime, summarize_aime


def read_complete_rows(path):
    # A live writer can have an incomplete last append. Never score that row.
    return [json.loads(line) for line in path.read_text().split('\n')[:-1] if line.strip()]


def build_report(registry, *, bootstrap_draws=10000):
    cases = read_complete_rows(Path(registry['bundle']) / 'cases.jsonl')
    tasks = {task: [row['id'] for row in cases if row['task'] == task]
             for task in sorted({row['task'] for row in cases})}
    tasks['all_aime'] = [row['id'] for row in cases]
    report = dict(format='archlab-limite-math-report-v1', updated_unix=time.time(),
                  interpretation=registry['interpretation'], models={}, paired_first_sample={})
    records = {}
    all_complete = True
    for entry in registry['models']:
        plan_path = Path(entry['plan'])
        plan = json.loads(plan_path.read_text())
        phase = next(row for row in plan['models'] if row['name'] == entry['name'])
        name = phase['name']
        paths = sorted((Path(plan['output']) / name).glob('shard-*/records.jsonl'))
        rows = [row for path in paths for row in read_complete_rows(path)]
        if any(row['model'] != name for row in rows):
            raise ValueError('benchmark model identity differs from registered output')
        records[name] = rows
        exams = {}
        for task, ids in tasks.items():
            subset = [row for row in rows if row['problem_id'] in ids]
            exams[task] = dict(
                first_sample=summarize_aime([row for row in subset if row['sample_index'] == 0], ids, 1,
                                            bootstrap_draws=bootstrap_draws),
                repeated=summarize_aime(subset, ids, plan['sampling']['samples_per_problem'],
                                       bootstrap_draws=bootstrap_draws),
            )
            all_complete &= exams[task]['repeated']['complete']
        if plan.get('evaluation_backend') == 'eval_pipeline':
            for task, ids in tasks.items():
                selected = [dict(row, correct=row['pipeline_correct']) for row in rows if row['problem_id'] in ids]
                exams[task]['pipeline_first_sample'] = summarize_aime(
                    [row for row in selected if row['sample_index'] == 0], ids, 1,
                    bootstrap_draws=bootstrap_draws)
                exams[task]['pipeline_repeated'] = summarize_aime(
                    selected, ids, plan['sampling']['samples_per_problem'], bootstrap_draws=bootstrap_draws)
        report['models'][name] = dict(phase=phase, sampling=plan['sampling'], exams=exams,
                                     plan_sha256=sha256_file(plan_path), recorded_samples=len(rows),
                                     expected_samples=len(cases) * plan['sampling']['samples_per_problem'])
    for left, right in combinations(records, 2):
        pair = {}
        for task, ids in tasks.items():
            if all(report['models'][name]['exams'][task]['first_sample']['complete'] for name in (left, right)):
                pair[task] = compare_aime(
                    [row for row in records[left] if row['problem_id'] in ids],
                    [row for row in records[right] if row['problem_id'] in ids], ids,
                )
        if pair:
            report['paired_first_sample'][left + ' vs ' + right] = pair
    registered_rl = {model['phase'].get('variant') for model in report['models'].values()
                     if model['phase'].get('kind') == 'rl'}
    report['pending_final_rl_variants'] = sorted(set(registry.get('required_final_rl_variants', []))
                                                 - registered_rl)
    report['complete'] = bool(all_complete and report['models'] and not report['pending_final_rl_variants'])
    queue_outputs = registry.get('queue_outputs', [registry['queue_output']] if registry.get('queue_output') else [])
    for queue_output in queue_outputs:
        state_path = Path(queue_output) / 'STATE.json'
        if state_path.exists():
            state = json.loads(state_path.read_text())
            report.setdefault('queues', {})[str(queue_output)] = dict(status=state['status'], stage=state['stage'])
            if len(queue_outputs) == 1:
                report['queue'] = dict(status=state['status'], stage=state['stage'])
        oom_path = Path(queue_output) / 'OOM_EVENT.json'
        if oom_path.exists():
            report['memory_overflow'] = json.loads(oom_path.read_text())
    return report


def markdown_report(report):
    has_pipeline = any('pipeline_repeated' in exam for model in report['models'].values()
                       for exam in model['exams'].values())
    extra_heading = ' Pipeline pass@1 | Truncated |' if has_pipeline else ''
    extra_separator = ' --- | --- |' if has_pipeline else ''
    lines = ['# Limite public math evaluation', '',
             'Completed exams only are scored. Partial response counts are progress, not accuracy.', '',
             '| Model | Exam | Completed draws / expected | First-draw accuracy | pass@1 (4 draws) | pass@4 |' + extra_heading,
             '| --- | --- | --- | --- | --- | --- |' + extra_separator]
    if report.get('memory_overflow'):
        lines[2:2] = ['**GPU memory overflow detected; queue held. See JSON event details.**', '']
    for name, model in report['models'].items():
        for task, results in model['exams'].items():
            first, repeat = results['first_sample'], results['repeated']
            count = repeat.get('independent_responses', repeat.get('recorded_samples', 0))
            expected = repeat.get('expected_samples', repeat.get('independent_responses'))
            first_text = f"{first['pass_at_1']:.1%}" if first['complete'] else 'pending'
            one = f"{repeat['pass_at_1']:.1%}" if repeat['complete'] else 'pending'
            four = f"{repeat['pass_at_4']:.1%}" if repeat['complete'] and 'pass_at_4' in repeat else 'pending'
            extra = ''
            if has_pipeline:
                pipeline = results.get('pipeline_repeated', {})
                pipeline_text = f"{pipeline['pass_at_1']:.1%}" if pipeline.get('complete') else 'pending'
                truncation = f"{repeat['truncation_rate']:.1%}" if repeat['complete'] else 'pending'
                extra = f' {pipeline_text} | {truncation} |'
            lines.append(f'| {name} | {task} | {count} / {expected} | {first_text} | {one} | {four} |' + extra)
    lines += ['', 'JSON includes problem-cluster bootstrap intervals, Wilson intervals, paired tests, '
              'truncation rates, token lengths, and checkpoint/training exposure.', '']
    lines.extend('- ' + item for item in report['interpretation'])
    return '\n'.join(lines) + '\n'


def publish_report(client, registry, report, output):
    """Attach evidence to existing runs; never create an empty experiment/run."""
    contract = registry.get('contract_name', 'limite-math-v1')
    for entry in registry['models']:
        run_id = entry.get('mlflow_run_id')
        if run_id is None:
            continue
        client.get_run(run_id)
        model = report['models'][entry['name']]
        for task, exams in model['exams'].items():
            for stage, values in exams.items():
                if not values['complete']:
                    continue
                for key in ('pass_at_1', 'pass_at_2', 'pass_at_4', 'truncation_rate', 'natural_eos_rate',
                            'mean_completion_tokens'):
                    if key in values:
                        prefix = registry.get('metric_prefix', 'benchmark')
                        client.log_metric(run_id, f'{prefix}/{task}/{stage}/{key}', float(values[key]),
                                          step=model['phase']['step'])
        client.log_artifact(run_id, str(output), artifact_path='benchmarks/' + contract)
        client.log_artifact(run_id, str(output.with_suffix('.md')), artifact_path='benchmarks/' + contract)
        client.set_tag(run_id, 'benchmark.' + contract, 'complete' if report['complete'] else 'running')
        if 'queue' in report:
            client.set_tag(run_id, 'benchmark.queue_status', report['queue']['status'])
        if 'memory_overflow' in report:
            client.set_tag(run_id, 'benchmark.memory_overflow', json.dumps(report['memory_overflow'],
                           sort_keys=True)[:4000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credentials', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--max-hours', type=float, default=168)
    args = parser.parse_args()
    client = None
    if args.credentials:
        from archlab.tracking.mlflow_sync import configure_client

        client = configure_client(args.credentials)
    published = None
    deadline = time.monotonic() + args.max_hours * 3600
    while True:
        # Each queue adds a final RL checkpoint after its configured training dependencies finish.
        registry = json.loads(args.registry.read_text())
        report = build_report(registry)
        atomic_write_json(args.output, report)
        args.output.with_suffix('.md').write_text(markdown_report(report))
        # Publish when a complete-exam metric changes, not on every response.
        metric_state = {name: {task: {stage: value for stage, value in exams.items()
                  if value['complete']} for task, exams in model['exams'].items()}
                  for name, model in report['models'].items()}
        signature = hashlib.sha256(json.dumps(dict(metrics=metric_state, queue=report.get('queue'), queues=report.get('queues'),
                          memory_overflow=report.get('memory_overflow')), sort_keys=True).encode()).hexdigest()
        if client is not None and signature != published:
            publish_report(client, registry, report, args.output)
            published = signature
        if report['complete'] or not args.watch or time.monotonic() >= deadline:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()

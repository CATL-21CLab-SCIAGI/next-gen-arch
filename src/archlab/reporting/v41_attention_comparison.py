"""Compare identical update/data prefixes of the two V4.1 attention variants.

Read-only with respect to training runs. Outputs JSON and CSV evidence; refuses
missing, duplicate, nonfinite, or data/schedule-mismatched measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def ledger(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if [row['step'] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"expected an uninterrupted fresh-run ledger: {path}")
    for row in rows:
        if any(not math.isfinite(row[key]) for key in ('loss', 'seconds', 'gradient_norm_before_clip')):
            raise ValueError(f"nonfinite training measurement: {path}")
        if row['seconds'] <= 0 or row['supervised_tokens'] <= 0:
            raise ValueError(f"invalid throughput denominator or token count: {path}")
    return rows


def percentile(values, fraction):
    values = sorted(values)
    position = fraction * (len(values) - 1)
    first = math.floor(position)
    last = math.ceil(position)
    return values[first] + (values[last] - values[first]) * (position - first)


def statistics_for(rows):
    seconds = [row['seconds'] for row in rows]
    targets = sum(row['supervised_tokens'] for row in rows)
    padded = sum(row['input_tokens'] for row in rows)
    return {'steps': len(rows), 'supervised_targets': targets, 'padded_input_tokens': padded,
            'total_update_seconds': sum(seconds), 'mean_update_seconds': statistics.mean(seconds),
            'median_update_seconds': statistics.median(seconds), 'p10_update_seconds': percentile(seconds, .1),
            'p90_update_seconds': percentile(seconds, .9), 'supervised_targets_per_second': targets / sum(seconds),
            'padded_input_tokens_per_second': padded / sum(seconds),
            'peak_allocated_gib_including_qualification': max(row['max_memory_allocated_gib'] for row in rows),
            'gradient_norm_range': [min(row['gradient_norm_before_clip'] for row in rows),
                                    max(row['gradient_norm_before_clip'] for row in rows)]}


def events(path):
    text, start, decoded = path.read_text(errors='replace'), 0, []
    decoder = json.JSONDecoder()
    while True:
        start = text.find('{"event"', start)
        if start < 0:
            return decoded
        try:
            value, size = decoder.raw_decode(text[start:])
            start += size
            if value.get('event') == 'train_step' and value.get('rank') == 0:
                decoded.append(value)
        except json.JSONDecodeError:
            start += 1


def compare(baseline, normal, *, first=21, last=50):
    if not 2 <= first <= last:
        raise ValueError('comparison must follow a logged prior update')
    all_rows = [ledger(path / 'train.jsonl') for path in (baseline, normal)]
    if min(map(len, all_rows)) < last:
        raise ValueError(f'wait for the predefined step {last}; latest are {[len(rows) for rows in all_rows]}')
    for left, right in zip(*all_rows, strict=False):
        for key in ('step', 'supervised_tokens', 'input_tokens', 'consumed_supervised_tokens', 'learning_rate'):
            if left[key] != right[key]:
                raise ValueError(f'paired update {left["step"]} differs in {key}')
    paired = [rows[first - 1:last] for rows in all_rows]
    measured = [statistics_for(rows) for rows in paired]
    ratio = measured[0]['total_update_seconds'] / measured[1]['total_update_seconds']
    generator = random.Random(2234)
    resampled = []
    for _ in range(10000):
        indices = generator.choices(range(last - first + 1), k=last - first + 1)
        resampled.append(sum(paired[0][i]['seconds'] for i in indices) / sum(paired[1][i]['seconds'] for i in indices))
    for path, rows, stats in zip((baseline, normal), paired, measured, strict=True):
        timed = {row['step']: row['unix_time'] for row in events(path.parent / f'{path.name}-node0.log')}
        elapsed = timed[last] - timed[first - 1]
        stats['wall_seconds_between_matched_step_boundaries'] = elapsed
        stats['supervised_targets_per_wall_second'] = stats['supervised_targets'] / elapsed
    validations = [[json.loads(line) for line in (path / 'validation.jsonl').read_text().splitlines()]
                   for path in (baseline, normal)]
    if validations[0][0] != validations[1][0]:
        raise ValueError('fresh initial held-out validation differs')
    return {'matched_steps': [first, last], 'paired_update_count': last - first + 1,
            'exact_data_and_schedule_match_through_step': min(map(len, all_rows)),
            'baseline': measured[0], 'normal': measured[1], 'throughput_ratio': ratio,
            'throughput_increase_percent': (ratio - 1) * 100,
            'update_time_reduction_percent': (1 - 1 / ratio) * 100,
            'paired_bootstrap_ratio_interval_95': [percentile(resampled, .025), percentile(resampled, .975)],
            'bootstrap_scope': 'variation among these paired steps; not independent repeated-run uncertainty',
            'initial_validation_exact_match': True, 'initial_validation': validations[1][0],
            'latest_normal_validation': validations[1][-1], 'latest_normal_update': all_rows[1][-1],
            'all_normal_updates_finite': True, 'normal_ledger_steps': len(all_rows[1]),
            'qualification_and_initial_compilation_excluded': True,
            'parameter_matched': False, 'core_precision_matched': False}, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--normal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, default=21)
    parser.add_argument('--last', type=int, default=50)
    args = parser.parse_args()
    report, paired = compare(args.baseline, args.normal, first=args.first, last=args.last)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'THROUGHPUT_COMPARISON.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    with (args.output / 'matched-updates.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['step', 'supervised_targets', 'padded_input_tokens', 'learning_rate',
                         'simplicial_seconds', 'normal_seconds', 'simplicial_loss', 'normal_loss'])
        for baseline, normal in zip(*paired, strict=True):
            writer.writerow([normal['step'], normal['supervised_tokens'], normal['input_tokens'], normal['learning_rate'],
                             baseline['seconds'], normal['seconds'], baseline['loss'], normal['loss']])
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

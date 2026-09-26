"""Mirror native Miles scalar logs and selected evidence without controlling training."""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from archlab.artifacts import atomic_write_json
from archlab.tracking.mlflow_sync import configure_client
from archlab.tracking.rl_mlflow import bootstrap_dns

ANSI = re.compile(r'\x1b\[[0-9;]*m')
EVENT = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) [^]]+\].*? - (step|perf|rollout|eval) (\d+): (\{.*\})')


def parse_event(line):
    match = EVENT.search(ANSI.sub('', line))
    if not match:
        return None
    timestamp, family, step, payload = match.groups()
    values = ast.literal_eval(payload)
    scalars = {k: float(v) for k, v in values.items()
               if isinstance(k, str) and type(v) in (int, float) and math.isfinite(v)}
    return {'timestamp': int(datetime.fromisoformat(timestamp).replace(tzinfo=timezone.utc).timestamp() * 1000),
            'family': family, 'step': int(step), 'metrics': scalars}


def parameters(argv):
    result = {}
    for token in argv:
        if token.startswith('--'):
            key = token[2:]
            result[key] = []
        else:
            result[key].append(token)
    return {key: 'true' if not values else ' '.join(values) for key, values in result.items()}


class MilesSync:
    def __init__(self, client, root, log, state_path, experiment):
        self.client, self.root, self.log, self.state_path = client, root, log, state_path
        self.state = json.loads(state_path.read_text()) if state_path.exists() else {}
        source = str(log.resolve())
        if self.state and self.state['source'] != source:
            raise ValueError('state belongs to a different log')
        if not self.state:
            source_id = hashlib.sha256(source.encode()).hexdigest()
            exp = client.get_experiment_by_name(experiment)
            exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
            matches = client.search_runs([exp_id], filter_string=f"tags.`archlab.miles_source_id` = '{source_id}'", max_results=2)
            if len(matches) > 1:
                raise ValueError('duplicate Miles source identities')
            with log.open('rb') as stream:
                header = stream.read(16384).decode(errors='replace')
            started = re.search(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[,\.]\d+', header)
            start_ms = int(datetime.fromisoformat(started[0].replace(',', '.')).replace(
                tzinfo=timezone.utc).timestamp() * 1000) if started else None
            run = matches[0] if matches else client.create_run(
                exp_id, run_name=f'{root.name}/{log.stem}',
                tags={'archlab.miles_source_id': source_id, 'archlab.source_log': source,
                      'archlab.phase': 'online-rl', 'archlab.variant': json.loads(
                          (root / 'model/config.json').read_text())['archlab']['variant'],
                      'archlab.tracking': 'external-log-sidecar',
                      'archlab.metric_timestamps': 'source log UTC; step is native zero-based Miles coordinate',
                      'archlab.full_training_resume': 'not_demonstrated'},
                start_time=start_ms)
            self.state = {'source': source, 'run_id': run.info.run_id, 'experiment_id': exp_id,
                          'offset': 0, 'events': 0, 'artifacts': {}}
            self.save()

    def save(self):
        atomic_write_json(self.state_path, self.state)

    def sync(self):
        from mlflow.entities import Metric

        if self.log.stat().st_size < self.state['offset']:
            raise ValueError('source log truncated; use a distinct attempt')
        count = 0
        with self.log.open('rb') as stream:
            stream.seek(self.state['offset'])
            while True:
                line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    break
                event = parse_event(line.decode(errors='replace'))
                if event and event['metrics']:
                    metrics = [Metric(key, value, event['timestamp'], event['step'])
                               for key, value in event['metrics'].items()]
                    self.client.log_batch(self.state['run_id'], metrics=metrics)
                    self.state['events'] += 1
                    self.state['latest_source_timestamp_ms'] = event['timestamp']
                    self.state[f"last_{event['family']}_step"] = event['step']
                    count += 1
                self.state['offset'] = stream.tell()
                if event:
                    self.save()
        self.save()
        self.client.set_tag(self.state['run_id'], 'archlab.last_sync_utc', datetime.now(timezone.utc).isoformat())
        return count

    def evidence(self):
        from mlflow.entities import Param

        argv = json.loads((self.root / 'launch-argv.json').read_text())
        manifest = json.loads((self.root / 'MANIFEST.json').read_text())
        params = parameters(argv) | {
            'actual_project_commit': manifest['project_commit'],
            'actual_miles_commit': manifest['miles_commit'],
            'hardware': '32 NVIDIA B300 (NVML mislabeled L20D)',
            'optimizer_storage': 'node_local_disk', 'engram_tables': 'frozen BF16',
            'runtime_image': json.dumps(manifest['image']),
        }
        params.update({f'runtime/{k}': v for k, v in manifest['runtime'].items()})
        items = [Param(k, str(v)) for k, v in params.items()]
        for first in range(0, len(items), 100):
            self.client.log_batch(self.state['run_id'], params=items[first:first + 100])
        files = [self.root / name for name in (
            'MANIFEST.json', 'QUALIFICATION.json', 'launch-argv.json', 'resolved-launch.json',
            'DATA_PROVENANCE.json', 'DRIVER_STATUS.json', 'ROLLOUT_BOOTSTRAP.json',
            'full-checkpoint-verification.json', 'saved-optimizer-verification.json',
            'attempt6-checkpoint-hostmem-summary.json', 'attempt6-post-checkpoint-gpu-memory.json',
            'attempt6-qualified-gpu-memory.json', 'backup-io-isolation.json',
            'model/config.json', 'model/DIRECT_CHECKPOINT.json',
            'checkpoints/latest_checkpointed_iteration.txt',
        )]
        files += list(self.root.glob('runtime-worker-*.json'))
        files += [self.root / 'runtime-master-0.json']
        files += list(self.root.glob('parent-load-rank-*.json'))
        files += list(self.root.glob('attempt6-rollout-*-replay-check.json'))
        references = []
        for checkpoint in sorted((self.root / 'checkpoints').glob('iter_*')):
            references.append({'path': str(checkpoint), 'native_completion_metadata':
                               (checkpoint / 'metadata.json').is_file(),
                               'distributed_metadata': (checkpoint / '.metadata').is_file(),
                               'payload_uploaded': False})
        evidence_dir = self.root / 'mlflow-evidence'
        evidence_dir.mkdir(exist_ok=True)
        atomic_write_json(evidence_dir / 'CHECKPOINT_REFERENCES.json', references)
        files += list(evidence_dir.glob('*'))
        for path in files:
            if not path.is_file() or path.stat().st_size > 20 * 2**20:
                continue
            relative = str(path.relative_to(self.root))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.state['artifacts'].get(relative) == digest:
                continue
            parent = path.relative_to(self.root).parent
            artifact_dir = 'evidence' if str(parent) == '.' else 'evidence/' + str(parent)
            self.client.log_artifact(self.state['run_id'], str(path), artifact_path=artifact_dir)
            if path.name == 'README.md':
                self.client.set_tag(self.state['run_id'], 'mlflow.note.content', path.read_text())
            self.state['artifacts'][relative] = digest
            self.save()

    def driver_status(self, now=None):
        """Tracking activity must never stand in for trainer liveness."""
        path = self.root / 'DRIVER_STATUS.json'
        if not path.exists():
            self.client.set_tag(self.state['run_id'], 'archlab.driver_state', 'unobserved')
            return False
        status = json.loads(path.read_text())
        state = status['state']
        age = (time.time() if now is None else now) - status['observed_at_epoch']
        if state == 'running' and age > 180:
            state = 'unobserved_stale_heartbeat'
        self.client.set_tag(self.state['run_id'], 'archlab.driver_state', state)
        self.client.set_tag(self.state['run_id'], 'archlab.driver_heartbeat_age_seconds', str(round(age)))
        if state not in ('failed', 'finished'):
            return False
        terminal = 'FINISHED' if state == 'finished' else 'FAILED'
        if self.state.get('terminal_status') != terminal:
            self.client.set_terminated(self.state['run_id'], status=terminal)
            self.state['terminal_status'] = terminal
            self.save()
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--log', default='train-attempt6.log')
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--dns-cache', type=Path)
    parser.add_argument('--experiment', default='DeepSeek-V4.1 Miles GRPO')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('--interval must be positive')
    root = args.root.resolve()
    credentials = json.loads(args.credentials.read_text())
    dns = json.loads(args.dns_cache.read_text()) if args.dns_cache else None
    mapping = {dns['host']: dns['address']} if dns else {}
    with (root / 'mlflow-sync.lock').open('w') as lock, bootstrap_dns(mapping, credentials['tracking_uri']):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        client = configure_client(args.credentials)
        sync = MilesSync(client, root, root / args.log, root / 'mlflow-state.json', args.experiment)
        while True:
            try:
                count = sync.sync()
                sync.evidence()
                terminal = sync.driver_status()
                atomic_write_json(root / 'MLFLOW_STATUS.json', {
                    **sync.state, 'last_sync_utc': datetime.now(timezone.utc).isoformat(), 'error': None})
                if count or not args.watch:
                    print(json.dumps({'run_id': sync.state['run_id'], 'new_events': count}), flush=True)
                if terminal:
                    break
            except Exception as error:
                atomic_write_json(root / 'MLFLOW_STATUS.json', {'run_id': sync.state['run_id'], 'error': type(error).__name__})
                if not args.watch:
                    raise SystemExit(type(error).__name__) from None
            if not args.watch:
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

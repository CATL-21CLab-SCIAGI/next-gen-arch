import json
from types import SimpleNamespace

from archlab.tracking.miles_mlflow import MilesSync, parameters, parse_event


def test_native_metrics_keep_names_steps_and_source_utc():
    row = parse_event("\x1b[36m(actor)\x1b[0m [2026-09-25 02:45:16.640 actor_cell0_rank24] model.py:855 - step 3: {'train/loss': 0.0, 'train/grad_norm': 0.159, 'skip': 1e999}")
    assert row['step'] == 3
    assert row['metrics'] == {'train/loss': 0.0, 'train/grad_norm': 0.159}
    assert row['timestamp'] == 1790304316640
    assert parse_event('other logging') is None


def test_effective_params_preserve_last_override():
    assert parameters(['--lr', '1e-6', '--bf16', '--lr', '2e-6']) == {'lr': '2e-6', 'bf16': 'true'}


class Client:
    def __init__(self):
        self.metrics = []

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id='7')

    def search_runs(self, *a, **k):
        return []

    def create_run(self, *a, **k):
        return SimpleNamespace(info=SimpleNamespace(run_id='test'))

    def log_batch(self, run, *, metrics):
        self.metrics.extend(metrics)

    def set_tag(self, *a):
        pass


def test_cursor_keeps_partial_line_and_resume_does_not_repeat(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, 'mlflow.entities', SimpleNamespace(Metric=lambda *a: a))
    (tmp_path / 'model').mkdir()
    (tmp_path / 'model/config.json').write_text('{"archlab":{"variant":"simplicial"}}')
    log = tmp_path / 'train.log'
    line = "[2026-09-25 02:45:16.640 actor] model.py:855 - step 3: {'train/grad_norm': 0.159}"
    log.write_text(line)
    client = Client()
    state = tmp_path / 'state.json'
    sync = MilesSync(client, tmp_path, log, state, 'test')
    assert sync.sync() == 0
    assert json.loads(state.read_text())['offset'] == 0
    with log.open('a') as stream:
        stream.write('\n')
    assert sync.sync() == 1
    restored = MilesSync(client, tmp_path, log, state, 'test')
    assert restored.sync() == 0
    assert len(client.metrics) == 1


def test_native_fixed_evaluation_is_not_dropped():
    event = parse_event("[2026-09-25 02:45:16.640 eval] metrics.py:53 - eval 0: {'eval/heldout_4096': 0.25}")
    assert event['family'] == 'eval'
    assert event['metrics'] == {'eval/heldout_4096': 0.25}

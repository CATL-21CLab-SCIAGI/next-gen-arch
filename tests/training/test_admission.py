import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json, sha256_file
from archlab.training import admission, qualification, resume, runtime, sandbox
from archlab.training.config import Experiment, fingerprint


@pytest.fixture
def admitted(tmp_path, monkeypatch):
    config = OmegaConf.create({'trainer': {'default_local_dir': str(tmp_path / 'run')}})
    experiment = Experiment(name='pilot', parent_checkpoint='parent', data_manifest='data')
    OmegaConf.save(config, tmp_path / 'resolved.yaml')
    atomic_write_json(tmp_path / 'experiment.json', asdict(experiment))
    atomic_write_json(tmp_path / 'source.json', {'src/model.py': 'source-hash'})
    monkeypatch.setattr(qualification, 'check_source_unchanged', lambda expected: None)
    monkeypatch.setattr(admission, 'validate_science', lambda *args: None)
    monkeypatch.setattr(qualification, 'check_package_config', lambda *args: None)
    monkeypatch.setattr(sandbox, 'check_sandbox', lambda *args: {'status': 'passed'})
    probes = ('check_upstream', 'check_runtime', 'check_data', 'check_model', 'check_execution')
    for name in probes:
        monkeypatch.setattr(qualification, name, lambda *args: {'identity': 'same'})
    xml = tmp_path / 'pytest.xml'
    xml.write_text('<testsuite><testcase name="passed"/></testsuite>')
    rows = [dict(name=name, status='passed', identity='same') for name in admission.required_checks(experiment)]
    for row in rows:
        if row['name'] in {*qualification.INTEGRATION_TESTS, 'native_model_fixture'}:
            row.update(tests=1, reports={str(xml): sha256_file(xml)}, deployment_sha256={
                name: sha256_file(tmp_path / name) for name in ('resolved.yaml', 'experiment.json')})
    report = dict(**admission.receipt_identity(tmp_path, Path('third_party/verl')), passed=True,
                  blockers=[], config_sha256=fingerprint(config), checks=rows)
    atomic_write_json(tmp_path / 'qualification.json', report)
    return config, tmp_path, report


def test_exact_admission_passes_and_drift_does_not(admitted, monkeypatch):
    config, directory, report = admitted
    assert admission.verify_admission(config, directory) == report
    monkeypatch.setattr(qualification, 'check_runtime', lambda *args: {'identity': 'changed'})
    with pytest.raises(ValueError, match='deployment_runtime'):
        admission.verify_admission(config, directory)


@pytest.mark.parametrize('defect', ['old', 'failed', 'missing', 'empty', 'config', 'input', 'xml'])
def test_incomplete_or_changed_admission_never_launches(admitted, defect):
    config, directory, report = admitted
    if defect == 'old':
        report['format'] = 'archlab-training-qualification-v1'
    elif defect == 'failed':
        report['checks'][0]['status'] = 'failed'
    elif defect == 'missing':
        report['checks'].pop()
    elif defect == 'empty':
        next(row for row in report['checks'] if row['name'] == 'native_model_fixture')['reports'] = {}
    elif defect == 'config':
        config.trainer.default_local_dir += '-changed'
    elif defect == 'input':
        (directory / 'experiment.json').write_text('{}')
    else:
        (directory / 'pytest.xml').write_text('<changed/>')
    atomic_write_json(directory / 'qualification.json', report)
    with pytest.raises(ValueError):
        runtime.run(config, qualification=directory)
    results = list((directory / 'attempts').glob('*/result.json'))
    assert len(results) == 1 and json.loads(results[0].read_text())['status'] == 'failed'
    assert not Path(config.trainer.default_local_dir).exists()


def test_wheel_only_environment_explains_checkout_requirement(tmp_path, monkeypatch):
    monkeypatch.setattr(qualification, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='full immutable source checkout'):
        admission.require_checkout()


def test_runtime_restores_upstream_actor_after_failure(admitted, monkeypatch):
    import sys

    config, directory, report = admitted
    config.ray_kwargs = {'ray_init': {'address': 'auto', 'runtime_env': {'env_vars': {}}}}
    monkeypatch.setattr(runtime, 'verify_admission', lambda *args: report)
    actor = object()

    def fail(config):
        raise RuntimeError('connection reset')

    upstream = SimpleNamespace(TaskRunnerV1=actor, main=fail)
    monkeypatch.setitem(sys.modules, 'verl', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'verl.trainer', SimpleNamespace(main_ppo=upstream))
    with pytest.raises(RuntimeError, match='connection reset'):
        runtime.run(config, qualification=directory)
    assert upstream.TaskRunnerV1 is actor
    result = json.loads(next((directory / 'attempts').glob('*/result.json')).read_text())
    assert result['failure']['retriable']


@pytest.fixture
def checkpoint(tmp_path):
    root = tmp_path / 'run'
    actor = root / 'global_step_2/actor'
    actor.mkdir(parents=True)
    (root / 'latest_checkpointed_iteration.txt').write_text('2')
    (actor.parent / 'data.pt').write_bytes(b'loader')
    for rank in range(2):
        for kind in ('model', 'optim', 'extra_state'):
            (actor / f'{kind}_world_size_2_rank_{rank}.pt').write_bytes(b'upstream-state')
    config = OmegaConf.create(dict(trainer=dict(default_local_dir=str(root), nnodes=1, n_gpus_per_node=2,
        resume_mode='disable', resume_from_path=None, v1=dict(trainer_mode='sync')),
        actor_rollout_ref=dict(actor=dict(strategy='fsdp2'))))
    return config, actor.parent


def test_resume_preserves_config_and_requires_every_rank_and_loader(checkpoint):
    config, path = checkpoint
    resumed = resume.resumed_config(config, path)
    assert config.trainer.resume_mode == 'disable'
    assert resumed.trainer.resume_from_path == str(path)
    (path / 'actor/optim_world_size_2_rank_1.pt').unlink()
    with pytest.raises(ValueError, match='optim rank 1'):
        resume.resumed_config(config, path)


@pytest.mark.parametrize('defect', ['loader', 'async', 'foreign', 'uncommitted'])
def test_resume_cannot_drop_state_or_adopt_uncommitted_checkpoints(checkpoint, tmp_path, defect):
    config, path = checkpoint
    if defect == 'loader':
        (path / 'data.pt').unlink()
    elif defect == 'async':
        config.trainer.v1.trainer_mode = 'colocate_async'
    elif defect == 'foreign':
        path = tmp_path / 'foreign/global_step_2'
    else:
        (path.parent / 'latest_checkpointed_iteration.txt').write_text('1')
    with pytest.raises(ValueError):
        resume.resumed_config(config, path)


def test_existing_checkpoint_cannot_be_adopted_by_a_new_run(checkpoint):
    config, path = checkpoint
    with pytest.raises(ValueError, match='cannot adopt'):
        resume.claim_run(Path(config.trainer.default_local_dir), {'config_sha256': 'different'}, resume=path)

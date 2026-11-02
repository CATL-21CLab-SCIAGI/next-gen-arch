import json
import subprocess
import time
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from omegaconf import OmegaConf

from archlab.artifacts import atomic_write_json
from archlab.training.supervise import Arm, ArmSpec


@pytest.fixture
def arm(tmp_path):
    qualification = tmp_path / 'qualification'
    qualification.mkdir()
    atomic_write_json(qualification / 'qualification.json', {'passed': True, 'source_root': str(tmp_path)})
    OmegaConf.save(OmegaConf.create({'trainer': {'default_local_dir': str(tmp_path / 'checkpoints'),
        'total_training_steps': 20, 'n_gpus_per_node': 8}}), qualification / 'resolved.yaml')
    return Arm(ArmSpec(arm='trial', node='gpu-worker', source=str(tmp_path),
        qualification=str(qualification), directory=str(tmp_path / 'supervision'),
        pythonpath=[str(tmp_path / 'src')], progress_timeout_seconds=60))


def failed_attempt(arm, *, retriable, attempts=1):
    attempt = arm.directory / 'attempts/previous'
    attempt.mkdir(parents=True)
    atomic_write_json(attempt / 'result.json', {'status': 'failed', 'failure': {'retriable': retriable}})
    return {'attempt': str(attempt), 'attempts_without_progress': attempts, 'step': 0,
            'last_progress_at': time.time(), 'qualification_sha256': arm.identity}


def test_pid_without_progress_becomes_stalled(arm, monkeypatch):
    state = failed_attempt(arm, retriable=True)
    state['last_progress_at'] -= 100
    monkeypatch.setattr(arm, 'alive', lambda attempt: True)
    assert arm.cycle(state) == 'stalled'
    atomic_write_json(Path(state['attempt']) / 'progress.json', {'step': 1, 'at': time.time()})
    assert arm.cycle(state) == 'running'


def test_retry_budget_survives_supervisor_restart(arm, monkeypatch):
    state = failed_attempt(arm, retriable=True, attempts=3)
    atomic_write_json(arm.state_path, state)
    restarted = Arm(arm.spec)
    monkeypatch.setattr(restarted, 'alive', lambda attempt: False)
    assert restarted.run(once=True) == 1
    assert json.loads(arm.state_path.read_text())['status'] == 'retry_limit'


def test_nonretriable_failure_stops_without_a_launch(arm, monkeypatch):
    state = failed_attempt(arm, retriable=False)
    monkeypatch.setattr(arm, 'alive', lambda attempt: False)
    assert arm.cycle(state) == 'needs_triage'


def test_ambiguous_ssh_launch_is_not_repeated(arm, monkeypatch):
    commands = []

    def timeout(command):
        if command.startswith('nvidia-smi'):
            return '0\n' * 8
        commands.append(command)
        raise subprocess.TimeoutExpired('ssh', 1)

    monkeypatch.setattr(arm, 'ssh', timeout)
    assert arm.run(once=True) == 1
    state = json.loads(arm.state_path.read_text())
    assert state['attempt'] and state['attempts_without_progress'] == 1
    monkeypatch.setattr(arm, 'alive', lambda attempt: False)
    assert arm.run(once=True) == 1
    assert len(commands) == 1
    assert json.loads(arm.state_path.read_text())['status'] == 'needs_triage'


@pytest.mark.parametrize('telemetry,status', [('2000\n' * 8, 'gpu_busy'), ('', 'probe_unavailable'),
                                             ('N/A\n' * 8, 'probe_unavailable'), ('0\n', 'probe_unavailable')])
def test_occupied_or_unverifiable_gpus_do_not_consume_attempts(arm, monkeypatch, telemetry, status):
    monkeypatch.setattr(arm, 'ssh', lambda command: telemetry)
    assert arm.run(once=True) == 1
    state = json.loads(arm.state_path.read_text())
    assert state['status'] == status and 'attempt' not in state
    assert 'attempts_without_progress' not in state


def test_idle_gpus_allow_one_launch(arm, monkeypatch):
    commands = []
    monkeypatch.setattr(arm, 'ssh', lambda command: commands.append(command) or '0\n' * 8)
    state = {}
    assert arm.cycle(state) == 'running'
    assert state['attempts_without_progress'] == 1
    assert len(commands) == 2 and 'nohup' in commands[-1]


def test_an_arm_has_only_one_supervisor(arm):
    with FileLock(arm.directory / '.supervisor.lock', timeout=0):
        with pytest.raises(Timeout):
            arm.run(once=True)


def test_ssh_has_a_bounded_timeout_and_quoted_launch_paths(arm, monkeypatch):
    captured = []

    def execute(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='')

    monkeypatch.setattr(subprocess, 'run', execute)
    arm.launch({})
    argv, options = captured[0]
    assert options['check'] and options['timeout'] == 60
    assert '--qualification' in argv[-1] and '--attempt' in argv[-1]
    assert 'ConnectTimeout=15' in argv

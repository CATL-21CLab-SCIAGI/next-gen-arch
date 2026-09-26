import json
import sys

from archlab.tracking.process_status import supervise


def test_child_failure_is_persisted_by_parent(tmp_path):
    path = tmp_path / 'status.json'
    assert supervise([sys.executable, '-c', 'raise SystemExit(7)'], path, interval=0.05) == 7
    status = json.loads(path.read_text())
    assert status['state'] == 'failed'
    assert status['exit_code'] == 7
    assert status['pid'] != status['supervisor_pid']


def test_success_is_not_reported_until_child_exits(tmp_path):
    path = tmp_path / 'status.json'
    code = 'import time; time.sleep(0.1)'
    assert supervise([sys.executable, '-c', code], path, interval=0.02) == 0
    assert json.loads(path.read_text())['state'] == 'finished'

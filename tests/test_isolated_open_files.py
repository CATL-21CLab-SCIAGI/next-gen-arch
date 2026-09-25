import json
import subprocess
import sys

import pytest

from archlab.serving.isolated_sglang_runtime import configure_open_files


def test_limit_is_inherited_by_child_without_lowering_hard_limit():
    code = '''
import json, resource, subprocess, sys
from archlab.serving.isolated_sglang_runtime import configure_open_files
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (512, hard))
configure_open_files({"open_files_soft_limit": 1024})
print(subprocess.check_output([sys.executable, "-c",
    "import json,resource; print(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))"], text=True).strip())
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (1024, hard)
configure_open_files({"open_files_soft_limit": 512})
assert resource.getrlimit(resource.RLIMIT_NOFILE) == (1024, hard)
'''
    result = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert json.loads(result)[0] == 1024


@pytest.mark.parametrize("value", [True, 0, -1, "65535"])
def test_invalid_limit_rejected(value):
    with pytest.raises(ValueError, match="positive integer"):
        configure_open_files({"open_files_soft_limit": value})


def test_limit_cannot_exceed_hard_limit(monkeypatch):
    monkeypatch.setattr("archlab.serving.isolated_sglang_runtime.resource.getrlimit", lambda _: (512, 1024))
    with pytest.raises(ValueError, match="hard limit"):
        configure_open_files({"open_files_soft_limit": 2048})

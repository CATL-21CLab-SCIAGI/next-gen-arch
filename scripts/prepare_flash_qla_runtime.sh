#!/usr/bin/env bash
set -euo pipefail

# Prepare a versioned, shared runtime without changing the base container.
FLASHQLA_COMMIT="${FLASHQLA_COMMIT:-7c7dfe16416ad21b1d03258189fc8d3b8460ae06}"
FLASHQLA_ROOT="${FLASHQLA_ROOT:-/mnt/nas/evergreen/runtime/FlashQLA-7c7dfe1}"
FLASHQLA_DEPS="${FLASHQLA_DEPS:-/mnt/nas/evergreen/runtime/flash-qla-0.1.2-py312}"
PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"

mkdir -p "$(dirname "$FLASHQLA_ROOT")" "$FLASHQLA_DEPS"
if [[ ! -d "$FLASHQLA_ROOT/.git" ]]; then
    git clone --filter=blob:none https://github.com/QwenLM/FlashQLA.git "$FLASHQLA_ROOT"
fi
git -C "$FLASHQLA_ROOT" checkout --detach "$FLASHQLA_COMMIT"
test -z "$(git -C "$FLASHQLA_ROOT" status --porcelain=v1 --untracked-files=all)"

"$PYTHON_BIN" -m pip install \
    --target "$FLASHQLA_DEPS" \
    --no-deps \
    tilelang==0.1.9 \
    apache-tvm-ffi==0.1.9

PYTHONPATH="$FLASHQLA_ROOT:$FLASHQLA_DEPS" "$PYTHON_BIN" - <<'PY'
import importlib.metadata
from pathlib import Path

import flash_qla
import tilelang

assert tilelang.__version__ == "0.1.9"
assert importlib.metadata.version("apache-tvm-ffi") == "0.1.9"
assert "FlashQLA-7c7dfe1" in str(Path(flash_qla.__file__).resolve())
print(f"validated FlashQLA source: {flash_qla.__file__}")
PY

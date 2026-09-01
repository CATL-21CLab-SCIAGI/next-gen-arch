#!/usr/bin/env bash
set -euo pipefail

target="${1:?usage: prepare_lm_eval_runtime.sh TARGET_DIR}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${NGA_PYTHON:-/opt/venv/bin/python}"

validate_runtime() {
    local site="$1"
    PYTHONPATH="$site${PYTHONPATH:+:$PYTHONPATH}" "$python_bin" -c '
import importlib.metadata as metadata
import lm_eval.models.huggingface
assert metadata.version("lm_eval") == "0.4.13"
' >/dev/null
}

if [[ -e "$target" ]]; then
    validate_runtime "$target"
    printf 'validated existing lm-eval runtime: %s\n' "$target"
    exit 0
fi

mkdir -p "$(dirname "$target")"
temporary="$(mktemp -d "${target}.incomplete.XXXXXX")"
"$python_bin" -m pip install \
    --no-deps \
    --require-hashes \
    --target "$temporary" \
    --requirement "$repo_root/requirements-eval.txt"
validate_runtime "$temporary"
cp "$repo_root/requirements-eval.txt" "$temporary/INSTALL_REQUIREMENTS.txt"
mv "$temporary" "$target"
validate_runtime "$target"
printf 'installed and validated lm-eval runtime: %s\n' "$target"

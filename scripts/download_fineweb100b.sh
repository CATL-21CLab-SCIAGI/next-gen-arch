#!/usr/bin/env bash
set -euo pipefail

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_HF="${NGA_HF:-/opt/venv/bin/hf}"
NGA_DATASET="${NGA_DATASET:-kjj0/fineweb100B-gpt2}"
NGA_DATASET_REVISION="${NGA_DATASET_REVISION:-50d1422b27e1a928440c26a8829f3f827f44ac56}"
NGA_STAGE_ROOT="${NGA_STAGE_ROOT:-/tmp/fineweb100B-${NGA_DATASET_REVISION}}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb100B}"
NGA_STATE_ROOT="${NGA_STATE_ROOT:-/mnt/nas/evergreen/next-gen-arch/data-downloads}"
NGA_DOWNLOAD_WORKERS="${NGA_DOWNLOAD_WORKERS:-32}"

mkdir -p "$NGA_STAGE_ROOT" "$NGA_DATA_ROOT" "$NGA_STATE_ROOT"
exec 9>"$NGA_STATE_ROOT/fineweb100b.lock"
if ! flock -n 9; then
    echo "another FineWeb100B downloader owns $NGA_STATE_ROOT/fineweb100b.lock" >&2
    exit 1
fi

"$NGA_HF" download "$NGA_DATASET" \
    --repo-type dataset \
    --revision "$NGA_DATASET_REVISION" \
    --local-dir "$NGA_STAGE_ROOT" \
    --max-workers "$NGA_DOWNLOAD_WORKERS"

train_shards="$(find "$NGA_STAGE_ROOT" -maxdepth 1 -name 'fineweb_train_*.bin' -type f | wc -l | tr -d ' ')"
validation_shards="$(find "$NGA_STAGE_ROOT" -maxdepth 1 -name 'fineweb_val_*.bin' -type f | wc -l | tr -d ' ')"
if [[ "$train_shards" != "1028" || "$validation_shards" != "1" ]]; then
    echo "unexpected staged inventory: train=$train_shards validation=$validation_shards" >&2
    exit 1
fi

if command -v rsync >/dev/null 2>&1; then
    rsync -a --partial --exclude='.cache/' "$NGA_STAGE_ROOT/" "$NGA_DATA_ROOT/"
else
    # PAI's NeMo image is intentionally minimal. The upstream repository is
    # flat, so copying only root files avoids its local Hugging Face cache.
    find "$NGA_STAGE_ROOT" -maxdepth 1 -type f -exec cp -p '{}' "$NGA_DATA_ROOT/" ';'
fi

NGA_DATA_ROOT="$NGA_DATA_ROOT" \
NGA_DATASET="$NGA_DATASET" \
NGA_DATASET_REVISION="$NGA_DATASET_REVISION" \
NGA_STATE_ROOT="$NGA_STATE_ROOT" \
PYTHONPATH="$NGA_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
"$NGA_PYTHON" -c '
import json
import os
from pathlib import Path
from archlab.speedrun.dataloader import inspect_fineweb_dataset

summary = inspect_fineweb_dataset(os.environ["NGA_DATA_ROOT"], expected_train_shards=1028)
summary["dataset"] = os.environ["NGA_DATASET"]
summary["revision"] = os.environ["NGA_DATASET_REVISION"]
target = Path(os.environ["NGA_STATE_ROOT"]) / "fineweb100b.COMPLETE.json"
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
'

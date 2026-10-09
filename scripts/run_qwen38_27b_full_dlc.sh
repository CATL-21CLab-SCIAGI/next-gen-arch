#!/usr/bin/env bash
set -euo pipefail

# Historical compatibility path. Model selection comes only from an explicit recipe.
: "${NGA_OUTPUT_ROOT:?set the validated controller compatibility handle}"
: "${NGA_EXPECTED_COMMIT:?set the immutable repository commit}"
: "${NGA_REPO_ROOT:?set the immutable repository root}"
: "${NGA_LAUNCH_RECIPE:?select a Flash-Next recipe explicitly; output names no longer select models}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
source "$(dirname "$0")/lib/dlc_runtime.sh"
nga_load_recipe "$NGA_LAUNCH_RECIPE"
if [[ "$NGA_LAUNCH_FAMILY" != "flash-next" ]]; then
    echo "compatibility entrypoint requires a flash-next recipe" >&2
    exit 1
fi

if [[ "${NGA_EXPECTED_NODES:-}" != "4" || "${NGA_GPUS_PER_NODE:-}" != "8" ]]; then
    echo "compatibility launch requires the 4-node, 8-GPU-per-node allocation" >&2
    exit 1
fi
if [[ "${NGA_SEQUENCE_LENGTH:-}" != "2048" ]]; then
    echo "compatibility launch requires sequence length 2048" >&2
    exit 1
fi
if [[ "${NGA_GLOBAL_BATCH_SIZE:-}" != "4096" ]]; then
    echo "compatibility launch requires global batch 4096" >&2
    exit 1
fi
if [[ "${NGA_TARGET_TRAIN_TOKENS:-}" != "100000595968" ]]; then
    echo "compatibility launch requires exactly 100000595968 effective tokens" >&2
    exit 1
fi
if [[ "${NGA_CHECKPOINT_INTERVAL_TOKENS:-}" != "100000595968" ]]; then
    echo "legacy validation interval must equal the target; the trainer saves every 1192 steps" >&2
    exit 1
fi
case "$NGA_OUTPUT_ROOT" in
    /mnt/nas/evergreen/next-gen-arch/compat-qwen38-flash-next-*|\
    /mnt/nas/evergreen/compat-qwen38-flash-next-*) ;;
    *)
        echo "invalid compatibility output handle" >&2
        exit 1
        ;;
esac

production_name="${NGA_OUTPUT_ROOT##*/compat-}"
export NGA_OUTPUT_ROOT="/mnt/oss/evergreen/next-gen-arch/$production_name"

exec bash "$(dirname "$0")/run_qwen38_flash_next_full_dlc.sh"

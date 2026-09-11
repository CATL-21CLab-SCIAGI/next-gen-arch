#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?PAI DLC must inject the node RANK}"
: "${WORLD_SIZE:?PAI DLC must inject the node WORLD_SIZE}"
: "${MASTER_ADDR:?PAI DLC must inject MASTER_ADDR}"
: "${MASTER_PORT:?PAI DLC must inject MASTER_PORT}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch-repo}"
NGA_EXPECTED_COMMIT="${NGA_EXPECTED_COMMIT:?set the immutable next-gen-arch commit}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb-edu-100BT-qwen38-de4b8e4d43b9}"
NGA_TOKENIZER="${NGA_TOKENIZER:-/mnt/oss/models/qwen38-flash-next-de4b8e4d43b9}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/qwen38-27b-quarter-bf16-fp32muon-fineweb100b-seed42}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
source "$(dirname "$0")/lib/dlc_runtime.sh"
case "$NGA_OUTPUT_ROOT" in
    */compat-qwen38-flash-next-*)
        : "${NGA_LAUNCH_RECIPE:?legacy compatibility handles now require an explicit launch recipe}"
        ;;
esac
nga_load_recipe "${NGA_LAUNCH_RECIPE:-$NGA_REPO_ROOT/recipes/launches/qwen38_dense_quarter.yaml}"
if [[ "$NGA_LAUNCH_FAMILY" == "flash-next" ]]; then
    exec bash "$(dirname "$0")/run_qwen38_27b_full_dlc.sh"
fi
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_PREFLIGHT_DATA_ROOT="${NGA_PREFLIGHT_DATA_ROOT:-$NGA_DATA_ROOT}"

SOURCE_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOKENIZER_SHA256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"

if [[ "$WORLD_SIZE" != "$NGA_EXPECTED_NODES" ]]; then
    echo "DLC injected $WORLD_SIZE nodes; expected $NGA_EXPECTED_NODES" >&2
    exit 1
fi
for value in \
    "$NGA_EXPECTED_NODES" \
    "$NGA_GPUS_PER_NODE" \
    "$NGA_SEQUENCE_LENGTH" \
    "$NGA_MICRO_BATCH_SIZE" \
    "$NGA_GLOBAL_BATCH_SIZE" \
    "$NGA_TARGET_TRAIN_TOKENS" \
    "$NGA_CHECKPOINT_INTERVAL_TOKENS"; do
    if ((value < 1)); then
        echo "positive integer required, found $value" >&2
        exit 1
    fi
done
if ((NGA_PREFLIGHT_STEPS < 1)); then
    echo "NGA_PREFLIGHT_STEPS must be positive" >&2
    exit 1
fi
if [[ "$NGA_RUNTIME_PREFLIGHT" != "0" && "$NGA_RUNTIME_PREFLIGHT" != "1" ]]; then
    echo "NGA_RUNTIME_PREFLIGHT must be 0 or 1" >&2
    exit 1
fi
if ((NGA_PROBE_STEPS < 0)); then
    echo "NGA_PROBE_STEPS must be non-negative" >&2
    exit 1
fi
if ((NGA_GLOBAL_BATCH_SIZE % (NGA_EXPECTED_NODES * NGA_GPUS_PER_NODE * NGA_MICRO_BATCH_SIZE) != 0)); then
    echo "global batch must divide by the 32-rank micro batch" >&2
    exit 1
fi
if ((NGA_CHECKPOINT_INTERVAL_TOKENS > NGA_TARGET_TRAIN_TOKENS)); then
    echo "checkpoint interval exceeds the training budget" >&2
    exit 1
fi
if ((NGA_TARGET_TRAIN_TOKENS % NGA_CHECKPOINT_INTERVAL_TOKENS != 0)); then
    echo "checkpoint interval must evenly divide target training tokens" >&2
    exit 1
fi

mountpoint -q /mnt/nas
mountpoint -q /mnt/oss
test -f "$NGA_DATA_ROOT/DATA_READY.json"
test -f "$NGA_TOKENIZER/tokenizer.json"
test "$(sha256sum "$NGA_TOKENIZER/tokenizer.json" | awk '{print $1}')" = "$TOKENIZER_SHA256"
jq -e \
    --arg tokenizer "$TOKENIZER_SHA256" \
    --argjson target "$NGA_TARGET_TRAIN_TOKENS" \
    '(.tokenizer_sha256 == $tokenizer)
     and (.train_parts | length == 32)
     and (.valid_parts | length == 1)
     and (.train_tokens >= $target)' \
    "$NGA_DATA_ROOT/DATA_READY.json" >/dev/null
PYTHONPATH="$NGA_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$NGA_PYTHON" -c '
import sys
from pathlib import Path
from archlab.megatron.indexed_data import validated_data_prefixes

train, validation = validated_data_prefixes(Path(sys.argv[1]))
if len(train) != 32 or len(validation) != 1:
    raise SystemExit(
        f"indexed-data membership must be exactly 32 train and 1 validation parts; "
        f"found {len(train)} and {len(validation)}"
    )
' "$NGA_DATA_ROOT"
nga_require_immutable_repo

nga_container_environment
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-24}"
export NGA_OUTPUT_ROOT NGA_EXPECTED_NODES NGA_GPUS_PER_NODE NGA_TOKENIZER
export NGA_SOURCE_MODEL="Qwen/Qwen3.8-27B" NGA_SOURCE_REVISION="$SOURCE_REVISION"

mkdir -p "$NGA_OUTPUT_ROOT/logs" "$NGA_OUTPUT_ROOT/checkpoints"

nga_torchrun archlab.megatron.collective_probe \
    2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/collective-node-$RANK.log"
test -f "$NGA_OUTPUT_ROOT/COLLECTIVE_VALIDATED.json"

if [[ "$NGA_RUNTIME_PREFLIGHT" = "1" && ! -f "$NGA_OUTPUT_ROOT/preflight/PROBE_COMPLETE.json" ]]; then
    nga_torchrun archlab.megatron.qwen38_27b_train \
        --data-root "$NGA_PREFLIGHT_DATA_ROOT" \
        --tokenizer "$NGA_TOKENIZER" \
        --run-dir "$NGA_OUTPUT_ROOT/preflight" \
        --model-scale "$NGA_MODEL_SCALE" \
        --sequence-length "$NGA_SEQUENCE_LENGTH" \
        --micro-batch-size "$NGA_MICRO_BATCH_SIZE" \
        --global-batch-size "$NGA_GLOBAL_BATCH_SIZE" \
        --checkpoint-interval-tokens "$NGA_CHECKPOINT_INTERVAL_TOKENS" \
        --probe-steps "$NGA_PREFLIGHT_STEPS" \
        --eval-interval "$NGA_PREFLIGHT_STEPS" \
        --eval-iters 1 \
        --log-interval 10 \
        --seed 42 \
        2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/preflight-node-$RANK.log"
fi
if [[ "$NGA_RUNTIME_PREFLIGHT" = "1" ]]; then
    test -f "$NGA_OUTPUT_ROOT/preflight/PROBE_COMPLETE.json"
fi

train_args=(
    --data-root "$NGA_DATA_ROOT"
    --tokenizer "$NGA_TOKENIZER"
    --run-dir "$NGA_OUTPUT_ROOT"
    --model-scale "$NGA_MODEL_SCALE"
    --sequence-length "$NGA_SEQUENCE_LENGTH"
    --micro-batch-size "$NGA_MICRO_BATCH_SIZE"
    --global-batch-size "$NGA_GLOBAL_BATCH_SIZE"
    --target-train-tokens "$NGA_TARGET_TRAIN_TOKENS"
    --checkpoint-interval-tokens "$NGA_CHECKPOINT_INTERVAL_TOKENS"
    --seed 42
)
if ((NGA_PROBE_STEPS > 0)); then
    train_args+=(--probe-steps "$NGA_PROBE_STEPS" --eval-interval "$NGA_PROBE_STEPS")
fi

nga_torchrun archlab.megatron.qwen38_27b_train \
    "${train_args[@]}" \
    2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/train-node-$RANK.log"

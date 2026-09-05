#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?PAI DLC must inject the node RANK}"
: "${WORLD_SIZE:?PAI DLC must inject the node WORLD_SIZE}"
: "${MASTER_ADDR:?PAI DLC must inject MASTER_ADDR}"
: "${MASTER_PORT:?PAI DLC must inject MASTER_PORT}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:?set the immutable repository root}"
NGA_EXPECTED_COMMIT="${NGA_EXPECTED_COMMIT:?set the immutable repository commit}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb-edu-100BT-qwen38-de4b8e4d43b9}"
NGA_TOKENIZER="${NGA_TOKENIZER:-/mnt/oss/models/qwen38-flash-next-de4b8e4d43b9}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:?set a fresh /mnt/oss/evergreen/next-gen-arch run directory}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_EXPECTED_NODES="${NGA_EXPECTED_NODES:-4}"
NGA_GPUS_PER_NODE="${NGA_GPUS_PER_NODE:-8}"
NGA_SEQUENCE_LENGTH="${NGA_SEQUENCE_LENGTH:-2048}"
NGA_MICRO_BATCH_SIZE="${NGA_MICRO_BATCH_SIZE:-1}"
NGA_GLOBAL_BATCH_SIZE="${NGA_GLOBAL_BATCH_SIZE:-4096}"
NGA_TARGET_TRAIN_TOKENS="${NGA_TARGET_TRAIN_TOKENS:-100000595968}"
NGA_PROBE_STEPS="${NGA_PROBE_STEPS:-0}"
NGA_PROBE_SAVE_INTERVAL="${NGA_PROBE_SAVE_INTERVAL:-0}"
NGA_FLASH_NEXT_MODEL_VARIANT="${NGA_FLASH_NEXT_MODEL_VARIANT:-full}"
# next-gen-arch itself may be an OSS symlink; use a distinct real NAS directory.
NGA_LIVE_LOG_ROOT="${NGA_LIVE_LOG_ROOT:-/mnt/nas/evergreen/arch-live-logs/${NGA_OUTPUT_ROOT##*/}}"
export NGA_EXPECTED_NODES NGA_GPUS_PER_NODE

SOURCE_CONFIG_SHA256="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
TOKENIZER_SHA256="0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"

if [[ "$WORLD_SIZE" != "$NGA_EXPECTED_NODES" ]]; then
    echo "DLC injected $WORLD_SIZE nodes; expected $NGA_EXPECTED_NODES" >&2
    exit 1
fi
if [[ "$NGA_EXPECTED_NODES" != "4" || "$NGA_GPUS_PER_NODE" != "8" ]]; then
    echo "the supported topology is exactly four nodes with eight GPUs each" >&2
    exit 1
fi
expected_micro_batch=1
if [[ "$NGA_FLASH_NEXT_MODEL_VARIANT" == "1b-depth48-no-mtp" ]]; then
    expected_micro_batch=4
fi
if [[ "$NGA_SEQUENCE_LENGTH" != "2048" || "$NGA_MICRO_BATCH_SIZE" != "$expected_micro_batch" ]]; then
    echo "this variant requires sequence 2048 and microbatch $expected_micro_batch" >&2
    exit 1
fi
if ((NGA_PROBE_STEPS == 0)); then
    if [[ "$NGA_GLOBAL_BATCH_SIZE" != "4096" ]]; then
        echo "production global batch must be exactly 4096 sequences" >&2
        exit 1
    fi
    if [[ "$NGA_TARGET_TRAIN_TOKENS" != "100000595968" ]]; then
        echo "production budget must be exactly 100000595968 tokens" >&2
        exit 1
    fi
fi
if ((NGA_PROBE_STEPS < 0 || NGA_PROBE_SAVE_INTERVAL < 0)); then
    echo "probe controls must be non-negative" >&2
    exit 1
fi
if [[ "$NGA_FLASH_NEXT_MODEL_VARIANT" != "full" && \
      "$NGA_FLASH_NEXT_MODEL_VARIANT" != "quarter-depth48-no-mtp" && \
      "$NGA_FLASH_NEXT_MODEL_VARIANT" != "1b-depth48-no-mtp" ]]; then
    echo "unsupported Flash-Next model variant: $NGA_FLASH_NEXT_MODEL_VARIANT" >&2
    exit 1
fi
case "$NGA_OUTPUT_ROOT" in
    /mnt/oss/evergreen/next-gen-arch/*) ;;
    *)
        echo "checkpoints must be under /mnt/oss/evergreen/next-gen-arch" >&2
        exit 1
        ;;
esac

mountpoint -q /mnt/nas
mountpoint -q /mnt/oss
test -f "$NGA_DATA_ROOT/DATA_READY.json"
test -f "$NGA_TOKENIZER/config.json"
test -f "$NGA_TOKENIZER/tokenizer.json"
test "$(sha256sum "$NGA_TOKENIZER/config.json" | awk '{print $1}')" = "$SOURCE_CONFIG_SHA256"
test "$(sha256sum "$NGA_TOKENIZER/tokenizer.json" | awk '{print $1}')" = "$TOKENIZER_SHA256"
jq -e \
    --arg tokenizer "$TOKENIZER_SHA256" \
    --argjson target "$NGA_TARGET_TRAIN_TOKENS" \
    '(.tokenizer_sha256 == $tokenizer)
     and (.train_parts | length == 32)
     and (.valid_parts | length == 1)
     and (.train_tokens >= $target)' \
    "$NGA_DATA_ROOT/DATA_READY.json" >/dev/null

test "$(git -C "$NGA_REPO_ROOT" rev-parse HEAD)" = "$NGA_EXPECTED_COMMIT"
repo_drift="$({ git -C "$NGA_REPO_ROOT" status --porcelain=v1 --untracked-files=all || true; } \
    | grep -Ev '^\?\? (\.LAUNCH_READY|repo-head\.txt)$' || true)"
if [[ -n "$repo_drift" ]]; then
    echo "immutable repository is not clean: $repo_drift" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
if [[ "$NGA_FLASH_NEXT_MODEL_VARIANT" == "1b-depth48-no-mtp" ]]; then
    # DP-only has no TP/SP overlap ordering requirement. Allow native TE's
    # small expert GEMMs to use independent CUDA work queues.
    export CUDA_DEVICE_MAX_CONNECTIONS=32
fi
export TOKENIZERS_PARALLELISM=true
export NVTE_ALLOW_NONDETERMINISTIC_ALGO="${NGA_ALLOW_NONDETERMINISTIC_ALGO:-1}"
unset NVTE_GROUPED_LINEAR_SINGLE_PARAM
export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-sci-agi-zhongwei-registry-vpc.cn-zhongwei.cr.aliyuncs.com/dev/nemo:26.06}"

mkdir -p "$NGA_OUTPUT_ROOT/logs" "$NGA_OUTPUT_ROOT/checkpoints" "$NGA_LIVE_LOG_ROOT"

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.collective_probe \
    2>&1 | tee -a "$NGA_LIVE_LOG_ROOT/collective-node-$RANK.log" "$NGA_OUTPUT_ROOT/logs/collective-node-$RANK.log"
test -f "$NGA_OUTPUT_ROOT/COLLECTIVE_VALIDATED.json"

train_args=(
    --data-root "$NGA_DATA_ROOT"
    --tokenizer "$NGA_TOKENIZER"
    --run-dir "$NGA_OUTPUT_ROOT"
    --model-variant "$NGA_FLASH_NEXT_MODEL_VARIANT"
    --sequence-length "$NGA_SEQUENCE_LENGTH"
    --micro-batch-size "$NGA_MICRO_BATCH_SIZE"
    --global-batch-size "$NGA_GLOBAL_BATCH_SIZE"
    --target-train-tokens "$NGA_TARGET_TRAIN_TOKENS"
    --seed 42
)
if [[ "$NGA_FLASH_NEXT_MODEL_VARIANT" == "1b-depth48-no-mtp" ]]; then
    train_args+=(--parallelism dp-only --fused-moe --fused-cross-entropy)
fi
if ((NGA_PROBE_STEPS > 0)); then
    train_args+=(
        --probe-steps "$NGA_PROBE_STEPS"
        --probe-save-interval "$NGA_PROBE_SAVE_INTERVAL"
        --eval-interval "$NGA_PROBE_STEPS"
    )
fi

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.qwen38_flash_next_full_train \
    "${train_args[@]}" \
    2>&1 | tee -a "$NGA_LIVE_LOG_ROOT/train-node-$RANK.log" "$NGA_OUTPUT_ROOT/logs/train-node-$RANK.log"

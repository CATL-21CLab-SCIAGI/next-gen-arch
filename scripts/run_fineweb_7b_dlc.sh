#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?PAI DLC must inject the node RANK}"
: "${WORLD_SIZE:?PAI DLC must inject the node WORLD_SIZE}"
: "${MASTER_ADDR:?PAI DLC must inject MASTER_ADDR}"
: "${MASTER_PORT:?PAI DLC must inject MASTER_PORT}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch-7b}"
NGA_EXPECTED_COMMIT="${NGA_EXPECTED_COMMIT:?set the immutable source commit}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb100B}"
NGA_DATA_MANIFEST="${NGA_DATA_MANIFEST:-/mnt/oss/datasets/fineweb100B.sha256}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/fineweb100b-7b-baseline-seed42}"
NGA_TOKENIZER_CACHE="${NGA_TOKENIZER_CACHE:-/mnt/oss/datasets/tokenizers/tiktoken}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_GPUS_PER_NODE="${NGA_GPUS_PER_NODE:-8}"
NGA_DATA_WAIT_SECONDS="${NGA_DATA_WAIT_SECONDS:-172800}"
NGA_BACKEND_PROFILE="${NGA_BACKEND_PROFILE:-compile-dp-overlap}"
NGA_SAVE_INTERVAL="${NGA_SAVE_INTERVAL:-10000}"

if [[ "$WORLD_SIZE" != "4" || "$NGA_GPUS_PER_NODE" != "8" ]]; then
    echo "the frozen 7B contract requires 4 nodes x 8 GPUs; got $WORLD_SIZE x $NGA_GPUS_PER_NODE" >&2
    exit 1
fi
mountpoint -q /mnt/nas
mountpoint -q /mnt/oss
test -f "$NGA_DATA_MANIFEST"
test "$(git -C "$NGA_REPO_ROOT" rev-parse HEAD)" = "$NGA_EXPECTED_COMMIT"
test -z "$(git -C "$NGA_REPO_ROOT" status --porcelain=v1 --untracked-files=all)"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="$NGA_TOKENIZER_CACHE"
export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-nemo-26.06}"

ready_marker="$NGA_OUTPUT_ROOT/DATA_READY.json"
mkdir -p "$NGA_OUTPUT_ROOT"
if [[ "$RANK" = "0" && ! -e "$ready_marker" ]]; then
    deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
    until NGA_DATA_ROOT="$NGA_DATA_ROOT" "$NGA_PYTHON" -c '
import os
from archlab.speedrun.dataloader import inspect_fineweb_dataset
inspect_fineweb_dataset(
    os.environ["NGA_DATA_ROOT"],
    expected_train_shards=1028,
    required_train_tokens=99_999_940_609,
)
'; do
        if ((SECONDS >= deadline)); then
            echo "FineWeb100B data was not ready before the wait deadline" >&2
            exit 1
        fi
        sleep 300
    done
    printf '{"dataset":"fineweb100B","required_training_tokens":99999940608}\n' > "$ready_marker"
fi

deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
while [[ ! -e "$ready_marker" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for $ready_marker" >&2
        exit 1
    fi
    sleep 60
done

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.train \
    --dataset fineweb100b \
    --data-root "$NGA_DATA_ROOT" \
    --data-manifest "$NGA_DATA_MANIFEST" \
    --data-verification metadata \
    --scale 7b \
    --variant baseline \
    --comparison-regime controlled \
    --target-train-tokens 100000000000 \
    --artifact-policy research \
    --initialization-hash shared \
    --seed 42 \
    --run-dir "$NGA_OUTPUT_ROOT/run" \
    --checkpoint-dir "$NGA_OUTPUT_ROOT/checkpoints" \
    --save-interval "$NGA_SAVE_INTERVAL" \
    --backend-profile "$NGA_BACKEND_PROFILE" \
    --optimization-recipe baseline \
    --throughput-warmup-steps 10 \
    --metrics-every 10

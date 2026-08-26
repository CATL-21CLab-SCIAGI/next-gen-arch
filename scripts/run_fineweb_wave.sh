#!/usr/bin/env bash
set -uo pipefail

: "${NGA_NODE_RANK:?set NGA_NODE_RANK to 0, 1, or 2}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 private address}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb10B}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/fineweb10b-wave-v1}"
NGA_TOKENIZER_CACHE="${NGA_TOKENIZER_CACHE:-/mnt/oss/datasets/tokenizers/tiktoken}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_MASTER_PORT="${NGA_MASTER_PORT:-29531}"
NGA_NNODES="${NGA_NNODES:-3}"
NGA_GPUS_PER_NODE="${NGA_GPUS_PER_NODE:-8}"
NGA_PROBE_STEPS="${NGA_PROBE_STEPS:-0}"
NGA_BACKEND_PROFILE="${NGA_BACKEND_PROFILE:-compile}"
NGA_SCALES="${NGA_SCALES:-1m 10m 100m 300m}"
NGA_VARIANTS="${NGA_VARIANTS:-baseline engram kda dsa attnres mhc gated-attention situ-glu inkling-relative-attention glm-mla xielu qwen-gdn inkling-sconv-kv inkling-sconv-residual partial-rope-25 kimi-k3-kda-update colu}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="$NGA_TOKENIZER_CACHE"
export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-nemo-26.06}"

mkdir -p "$NGA_OUTPUT_ROOT"

for variant in $NGA_VARIANTS; do
    for scale in $NGA_SCALES; do
        run_dir="$NGA_OUTPUT_ROOT/$scale/$variant-seed42"
        if [[ -e "$run_dir/COMPLETE.json" || -e "$run_dir/FAILED.json" ]]; then
            echo "skip terminal run: $scale/$variant"
            continue
        fi

        probe_args=()
        if [[ "$NGA_PROBE_STEPS" != "0" ]]; then
            probe_args=(--probe-steps "$NGA_PROBE_STEPS")
        fi
        echo "start: $scale/$variant node_rank=$NGA_NODE_RANK"
        "$NGA_PYTHON" -m torch.distributed.run \
            --nnodes="$NGA_NNODES" \
            --nproc-per-node="$NGA_GPUS_PER_NODE" \
            --node-rank="$NGA_NODE_RANK" \
            --master-addr="$MASTER_ADDR" \
            --master-port="$NGA_MASTER_PORT" \
            --module archlab.megatron.train \
            --dataset fineweb10b \
            --data-root "$NGA_DATA_ROOT" \
            --scale "$scale" \
            --variant "$variant" \
            --seed 42 \
            --run-dir "$run_dir" \
            --backend-profile "$NGA_BACKEND_PROFILE" \
            --optimization-recipe baseline \
            --metrics-every 10 \
            "${probe_args[@]}"
        status=$?
        echo "finish: $scale/$variant status=$status"
    done
done

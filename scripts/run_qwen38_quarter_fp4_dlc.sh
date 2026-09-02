#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?PAI DLC must inject the node RANK}"
: "${WORLD_SIZE:?PAI DLC must inject the node WORLD_SIZE}"
: "${MASTER_ADDR:?PAI DLC must inject MASTER_ADDR}"
: "${MASTER_PORT:?PAI DLC must inject MASTER_PORT}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch-qwen38-fp4}"
NGA_EXPECTED_COMMIT="${NGA_EXPECTED_COMMIT:?set the immutable next-gen-arch commit}"
NGA_SOURCE_DATA="${NGA_SOURCE_DATA:-/mnt/oss-dataset/datasets/AI-ModelScope/fineweb-edu/sample/100BT}"
NGA_SOURCE_MANIFEST="${NGA_SOURCE_MANIFEST:-/mnt/oss/datasets/fineweb-edu-100BT-qwen2p5/SOURCE_MANIFEST.json}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb-edu-100BT-qwen38-de4b8e4d43b9}"
NGA_TOKENIZER="${NGA_TOKENIZER:-/mnt/oss/models/qwen38-flash-next-de4b8e4d43b9}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/qwen38-quarter-fp4-fineweb100b-seed42}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_EXPECTED_NODES="${NGA_EXPECTED_NODES:-4}"
NGA_GPUS_PER_NODE="${NGA_GPUS_PER_NODE:-8}"
NGA_TOKENIZER_WORKERS="${NGA_TOKENIZER_WORKERS:-8}"
NGA_EXPECTED_SOURCE_SHARDS="${NGA_EXPECTED_SOURCE_SHARDS:-140}"
NGA_VALIDATION_SOURCE="${NGA_VALIDATION_SOURCE:-013_00009.parquet}"
NGA_EXPECTED_TRAIN_SHARDS="${NGA_EXPECTED_TRAIN_SHARDS:-139}"
NGA_EXPECTED_TRAIN_PARTS="${NGA_EXPECTED_TRAIN_PARTS:-$((NGA_EXPECTED_NODES * NGA_TOKENIZER_WORKERS))}"
NGA_DOCUMENT_BATCH_SIZE="${NGA_DOCUMENT_BATCH_SIZE:-512}"
NGA_DATA_WAIT_SECONDS="${NGA_DATA_WAIT_SECONDS:-86400}"
NGA_PREPARE_DATA="${NGA_PREPARE_DATA:-1}"
NGA_SEQUENCE_LENGTH="${NGA_SEQUENCE_LENGTH:-2048}"
NGA_MICRO_BATCH_SIZE="${NGA_MICRO_BATCH_SIZE:-1}"
NGA_GLOBAL_BATCH_SIZE="${NGA_GLOBAL_BATCH_SIZE:-512}"
NGA_TARGET_TRAIN_TOKENS="${NGA_TARGET_TRAIN_TOKENS:-100000000000}"
NGA_CHECKPOINT_INTERVAL_TOKENS="${NGA_CHECKPOINT_INTERVAL_TOKENS:-10000000000}"
NGA_PROBE_STEPS="${NGA_PROBE_STEPS:-0}"

if [[ "$WORLD_SIZE" != "$NGA_EXPECTED_NODES" ]]; then
    echo "DLC injected $WORLD_SIZE nodes; expected $NGA_EXPECTED_NODES" >&2
    exit 1
fi
for value in \
    "$NGA_EXPECTED_NODES" \
    "$NGA_GPUS_PER_NODE" \
    "$NGA_TOKENIZER_WORKERS" \
    "$NGA_EXPECTED_SOURCE_SHARDS" \
    "$NGA_EXPECTED_TRAIN_SHARDS" \
    "$NGA_EXPECTED_TRAIN_PARTS" \
    "$NGA_DOCUMENT_BATCH_SIZE" \
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
if [[ "$NGA_PREPARE_DATA" != "0" && "$NGA_PREPARE_DATA" != "1" ]]; then
    echo "NGA_PREPARE_DATA must be 0 or 1" >&2
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

mountpoint -q /mnt/nas
mountpoint -q /mnt/oss
mountpoint -q /mnt/oss-dataset
test -d "$NGA_SOURCE_DATA"
test -f "$NGA_SOURCE_MANIFEST"
test -f "$NGA_TOKENIZER/config.json"
test -f "$NGA_TOKENIZER/tokenizer.json"
test "$(git -C "$NGA_REPO_ROOT" rev-parse HEAD)" = "$NGA_EXPECTED_COMMIT"
repo_drift="$({ git -C "$NGA_REPO_ROOT" status --porcelain=v1 --untracked-files=all || true; } \
    | grep -Ev '^\?\? (\.LAUNCH_READY|repo-head\.txt)$' || true)"
if [[ -n "$repo_drift" ]]; then
    echo "next-gen-arch repository is not clean: $repo_drift" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-24}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-sci-agi-zhongwei-registry-vpc.cn-zhongwei.cr.aliyuncs.com/dev/nemo:26.06}"
export NGA_OUTPUT_ROOT NGA_EXPECTED_NODES NGA_GPUS_PER_NODE NGA_TOKENIZER

mkdir -p "$NGA_DATA_ROOT" "$NGA_OUTPUT_ROOT/logs" "$NGA_OUTPUT_ROOT/checkpoints"

"$NGA_PYTHON" - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ["NGA_TOKENIZER"])
config = json.loads((root / "config.json").read_text())
text = config["text_config"]
expected = {
    "model_type": "qwen4_exp_text",
    "vocab_size": 248320,
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_experts": 512,
    "num_experts_per_tok": 10,
}
drift = {key: (text.get(key), value) for key, value in expected.items() if text.get(key) != value}
if config.get("model_type") != "qwen4_exp":
    drift["outer_model_type"] = (config.get("model_type"), "qwen4_exp")
if drift:
    raise SystemExit(f"pinned Qwen3.8 config drift: {drift}")
PY

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.collective_probe \
    2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/collective-node-$RANK.log"
test -f "$NGA_OUTPUT_ROOT/COLLECTIVE_VALIDATED.json"

if [[ "$NGA_PREPARE_DATA" = "1" && ! -f "$NGA_DATA_ROOT/DATA_READY.json" ]]; then
    "$NGA_PYTHON" -m archlab.megatron.data convert-parquet \
        --source-root "$NGA_SOURCE_DATA" \
        --source-manifest "$NGA_SOURCE_MANIFEST" \
        --output-root "$NGA_DATA_ROOT" \
        --tokenizer "$NGA_TOKENIZER" \
        --split train \
        --expected-shards "$NGA_EXPECTED_TRAIN_SHARDS" \
        --nodes "$WORLD_SIZE" \
        --node-rank "$RANK" \
        --workers "$NGA_TOKENIZER_WORKERS" \
        --document-batch-size "$NGA_DOCUMENT_BATCH_SIZE" \
        --qwen-eos-id 248044

    if [[ "$RANK" = "0" ]]; then
        "$NGA_PYTHON" -m archlab.megatron.data convert-parquet \
            --source-root "$NGA_SOURCE_DATA" \
            --source-manifest "$NGA_SOURCE_MANIFEST" \
            --output-root "$NGA_DATA_ROOT" \
            --tokenizer "$NGA_TOKENIZER" \
            --split val \
            --expected-shards 1 \
            --nodes 1 \
            --node-rank 0 \
            --workers 1 \
            --document-batch-size "$NGA_DOCUMENT_BATCH_SIZE" \
            --qwen-eos-id 248044

        deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
        for ((node_rank = 0; node_rank < WORLD_SIZE; node_rank++)); do
            marker="$NGA_DATA_ROOT/train.node-$(printf '%05d' "$node_rank").json"
            while [[ ! -f "$marker" ]]; do
                if ((SECONDS >= deadline)); then
                    echo "timed out waiting for $marker" >&2
                    exit 1
                fi
                sleep 30
            done
        done
        "$NGA_PYTHON" -m archlab.megatron.data summarize \
            --output-root "$NGA_DATA_ROOT" \
            --train-parts "$NGA_EXPECTED_TRAIN_PARTS" \
            --valid-parts 1 \
            --required-train-tokens "$NGA_TARGET_TRAIN_TOKENS" \
            --output "$NGA_DATA_ROOT/DATA_READY.json"
    fi
fi

deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
while [[ ! -f "$NGA_DATA_ROOT/DATA_READY.json" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for Qwen3.8-tokenized FineWeb-Edu" >&2
        exit 1
    fi
    sleep 30
done

if [[ "$NGA_PREPARE_DATA" = "1" && "$RANK" = "0" ]]; then
    "$NGA_PYTHON" -m archlab.megatron.data validate \
        --ready "$NGA_DATA_ROOT/DATA_READY.json" \
        --source-manifest "$NGA_SOURCE_MANIFEST" \
        --tokenizer "$NGA_TOKENIZER/tokenizer.json" \
        --train-parts "$NGA_EXPECTED_TRAIN_PARTS" \
        --valid-parts 1 \
        --required-train-tokens "$NGA_TARGET_TRAIN_TOKENS" \
        --output "$NGA_OUTPUT_ROOT/DATA_VALIDATED.json"
fi

train_args=(
    --data-root "$NGA_DATA_ROOT"
    --tokenizer "$NGA_TOKENIZER"
    --run-dir "$NGA_OUTPUT_ROOT"
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

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.qwen38_train \
    "${train_args[@]}" \
    2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/train-node-$RANK.log"

#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?PAI DLC must inject the node RANK}"
: "${WORLD_SIZE:?PAI DLC must inject the node WORLD_SIZE}"
: "${MASTER_ADDR:?PAI DLC must inject MASTER_ADDR}"
: "${MASTER_PORT:?PAI DLC must inject MASTER_PORT}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch-qwen15b}"
NGA_EXPECTED_COMMIT="${NGA_EXPECTED_COMMIT:?set the immutable source commit}"
NGA_SOURCE_DATA="${NGA_SOURCE_DATA:-/mnt/oss/datasets/fineweb100B}"
NGA_SOURCE_MANIFEST="${NGA_SOURCE_MANIFEST:-/mnt/oss/datasets/fineweb100B.sha256}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb100B-qwen2p5}"
NGA_TOKENIZER="${NGA_TOKENIZER:-/mnt/oss/models/qwen2.5-1.5b-8faed761d45a}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/fineweb100b-qwen2p5-1p5b-dp32-seed42}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_GPUS_PER_NODE="${NGA_GPUS_PER_NODE:-8}"
NGA_TOKENIZER_WORKERS="${NGA_TOKENIZER_WORKERS:-8}"
NGA_DATA_WAIT_SECONDS="${NGA_DATA_WAIT_SECONDS:-21600}"
NGA_SAVE_INTERVAL="${NGA_SAVE_INTERVAL:-10000}"
NGA_MICRO_BATCH_SIZE="${NGA_MICRO_BATCH_SIZE:-32}"
NGA_GLOBAL_BATCH_SIZE="${NGA_GLOBAL_BATCH_SIZE:-1024}"
NGA_TRAIN_ITERS="${NGA_TRAIN_ITERS:-47684}"

if [[ "$WORLD_SIZE" != "4" || "$NGA_GPUS_PER_NODE" != "8" ]]; then
    echo "the Qwen2.5-1.5B contract requires 4 nodes x 8 GPUs; got $WORLD_SIZE x $NGA_GPUS_PER_NODE" >&2
    exit 1
fi
if ((NGA_GLOBAL_BATCH_SIZE % (WORLD_SIZE * NGA_GPUS_PER_NODE * NGA_MICRO_BATCH_SIZE) != 0)); then
    echo "global batch must be divisible by DP world size times micro batch" >&2
    exit 1
fi

mountpoint -q /mnt/nas
mountpoint -q /mnt/oss
test -f "$NGA_SOURCE_MANIFEST"
test -f "$NGA_TOKENIZER/config.json"
test -f "$NGA_TOKENIZER/tokenizer.json"
test "$(git -C "$NGA_REPO_ROOT" rev-parse HEAD)" = "$NGA_EXPECTED_COMMIT"
test -z "$(git -C "$NGA_REPO_ROOT" status --porcelain=v1 --untracked-files=all)"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-/mnt/oss/datasets/tokenizers/tiktoken}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-24}"
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-nemo-26.06}"
export NGA_REPO_ROOT NGA_EXPECTED_COMMIT NGA_SOURCE_DATA NGA_SOURCE_MANIFEST
export NGA_DATA_ROOT NGA_TOKENIZER NGA_OUTPUT_ROOT
export NGA_MICRO_BATCH_SIZE NGA_GLOBAL_BATCH_SIZE NGA_TRAIN_ITERS

mkdir -p "$NGA_DATA_ROOT" "$NGA_OUTPUT_ROOT/logs" "$NGA_OUTPUT_ROOT/data-cache"

"$NGA_PYTHON" - <<'PY'
import json, os
from pathlib import Path

config = json.loads((Path(os.environ["NGA_TOKENIZER"]) / "config.json").read_text())
expected = {
    "model_type": "qwen2",
    "hidden_size": 1536,
    "intermediate_size": 8960,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True,
}
drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if drift:
    raise SystemExit(f"Qwen config drift: {drift}")
PY

"$NGA_PYTHON" -m archlab.megatron.data convert \
    --source-root "$NGA_SOURCE_DATA" \
    --source-manifest "$NGA_SOURCE_MANIFEST" \
    --output-root "$NGA_DATA_ROOT" \
    --tokenizer "$NGA_TOKENIZER" \
    --split train \
    --expected-shards 1028 \
    --nodes "$WORLD_SIZE" \
    --node-rank "$RANK" \
    --workers "$NGA_TOKENIZER_WORKERS"

if [[ "$RANK" = "0" ]]; then
    "$NGA_PYTHON" -m archlab.megatron.data convert \
        --source-root "$NGA_SOURCE_DATA" \
        --source-manifest "$NGA_SOURCE_MANIFEST" \
        --output-root "$NGA_DATA_ROOT" \
        --tokenizer "$NGA_TOKENIZER" \
        --split val \
        --expected-shards 1 \
        --nodes 1 \
        --node-rank 0 \
        --workers 1

    deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
    for node_rank in 0 1 2 3; do
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
        --train-parts "$((WORLD_SIZE * NGA_TOKENIZER_WORKERS))" \
        --valid-parts 1 \
        --required-train-tokens 100000000000 \
        --output "$NGA_DATA_ROOT/DATA_READY.json"
fi

deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
while [[ ! -f "$NGA_DATA_ROOT/DATA_READY.json" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for Qwen-tokenized FineWeb" >&2
        exit 1
    fi
    sleep 30
done

if [[ "$RANK" = "0" && ! -f "$NGA_OUTPUT_ROOT/RUN_CONTRACT.json" ]]; then
    "$NGA_PYTHON" - <<'PY'
import hashlib, json, os, platform, time
from pathlib import Path
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

tokenizer = Path(os.environ["NGA_TOKENIZER"])
payload = {
    "model": "Qwen2.5-1.5B architecture, initialized from scratch",
    "model_config": json.loads((tokenizer / "config.json").read_text()),
    "model_config_sha256": sha(tokenizer / "config.json"),
    "tokenizer_sha256": sha(tokenizer / "tokenizer.json"),
    "dataset": json.loads((Path(os.environ["NGA_DATA_ROOT"]) / "DATA_READY.json").read_text()),
    "source_manifest_sha256": sha(os.environ["NGA_SOURCE_MANIFEST"]),
    "source_commit": os.environ["NGA_EXPECTED_COMMIT"],
    "seed": 42,
    "nodes": 4,
    "gpus_per_node": 8,
    "parallelism": {"data": 32, "tensor": 1, "pipeline": 1, "context": 1},
    "sequence_length": 2048,
    "micro_batch_sequences": int(os.environ["NGA_MICRO_BATCH_SIZE"]),
    "global_batch_sequences": int(os.environ["NGA_GLOBAL_BATCH_SIZE"]),
    "target_tokens": 100000000000,
    "effective_tokens": (
        int(os.environ["NGA_TRAIN_ITERS"])
        * int(os.environ["NGA_GLOBAL_BATCH_SIZE"])
        * 2048
    ),
    "container": os.environ.get("NGA_CONTAINER_DIGEST"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "python": platform.python_version(),
    "created_at_unix": time.time(),
}
path = Path(os.environ["NGA_OUTPUT_ROOT"]) / "RUN_CONTRACT.json"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
fi

mapfile -t train_prefixes < <(
    find "$NGA_DATA_ROOT/train" -maxdepth 1 -type f -name 'part-*.idx' -print \
        | sed 's/\.idx$//' | sort
)
mapfile -t valid_prefixes < <(
    find "$NGA_DATA_ROOT/val" -maxdepth 1 -type f -name 'part-*.idx' -print \
        | sed 's/\.idx$//' | sort
)
if [[ "${#train_prefixes[@]}" != "$((WORLD_SIZE * NGA_TOKENIZER_WORKERS))" ]]; then
    echo "unexpected Qwen FineWeb train part count: ${#train_prefixes[@]}" >&2
    exit 1
fi
if [[ "${#valid_prefixes[@]}" != "1" ]]; then
    echo "unexpected Qwen FineWeb validation part count: ${#valid_prefixes[@]}" >&2
    exit 1
fi

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.backend \
    --use-mcore-models \
    --num-layers 28 \
    --hidden-size 1536 \
    --ffn-hidden-size 8960 \
    --num-attention-heads 12 \
    --group-query-attention \
    --num-query-groups 2 \
    --seq-length 2048 \
    --max-position-embeddings 131072 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --disable-bias-linear \
    --add-qkv-bias \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.02 \
    --make-vocab-size-divisible-by 151936 \
    --micro-batch-size "$NGA_MICRO_BATCH_SIZE" \
    --global-batch-size "$NGA_GLOBAL_BATCH_SIZE" \
    --train-iters "$NGA_TRAIN_ITERS" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --distributed-backend nccl \
    --transformer-impl transformer_engine \
    --attention-backend flash \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$NGA_TOKENIZER" \
    --train-data-path "${train_prefixes[@]}" \
    --valid-data-path "${valid_prefixes[@]}" \
    --data-cache-path "$NGA_OUTPUT_ROOT/data-cache" \
    --num-workers 8 \
    --no-create-attention-mask-in-dataloader \
    --seed 42 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --log-interval 10 \
    --log-throughput \
    --calculate-per-token-loss \
    --save "$NGA_OUTPUT_ROOT/checkpoints" \
    --save-interval "$NGA_SAVE_INTERVAL" \
    --ckpt-format torch_dist \
    --async-save \
    --tensorboard-dir "$NGA_OUTPUT_ROOT/tensorboard" \
    2>&1 | tee "$NGA_OUTPUT_ROOT/logs/node-$RANK.log"

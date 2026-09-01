#!/usr/bin/env bash
set -euo pipefail

: "${RANK:?PAI DLC must inject the node RANK}"
: "${WORLD_SIZE:?PAI DLC must inject the node WORLD_SIZE}"
: "${MASTER_ADDR:?PAI DLC must inject MASTER_ADDR}"
: "${MASTER_PORT:?PAI DLC must inject MASTER_PORT}"

NGA_REPO_ROOT="${NGA_REPO_ROOT:-/mnt/nas/evergreen/next-gen-arch-qwen15b}"
NGA_EXPECTED_COMMIT="${NGA_EXPECTED_COMMIT:?set the immutable source commit}"
NGA_SOURCE_DATA="${NGA_SOURCE_DATA:-/mnt/oss-dataset/datasets/AI-ModelScope/fineweb-edu/sample/100BT}"
NGA_DATA_ROOT="${NGA_DATA_ROOT:-/mnt/oss/datasets/fineweb-edu-100BT-qwen2p5}"
NGA_SOURCE_MANIFEST="${NGA_SOURCE_MANIFEST:-$NGA_DATA_ROOT/SOURCE_MANIFEST.json}"
NGA_TOKENIZER="${NGA_TOKENIZER:-/mnt/oss/models/qwen2.5-1.5b-8faed761d45a}"
NGA_OUTPUT_ROOT="${NGA_OUTPUT_ROOT:-/mnt/nas/evergreen/next-gen-arch/fineweb-edu100b-qwen2p5-1p5b-mxfp8-dp32-seed42}"
NGA_MODEL_EXPORT_ROOT="${NGA_MODEL_EXPORT_ROOT:-/mnt/oss/models/${NGA_OUTPUT_ROOT##*/}}"
NGA_LM_EVAL_SITE="${NGA_LM_EVAL_SITE:-/mnt/nas/evergreen/runtime/lm-eval-0.4.13}"
NGA_PYTHON="${NGA_PYTHON:-/opt/venv/bin/python}"
NGA_MEGATRON_ROOT="${NGA_MEGATRON_ROOT:-/opt/Megatron-Bridge/3rdparty/Megatron-LM}"
NGA_GPUS_PER_NODE="${NGA_GPUS_PER_NODE:-8}"
NGA_EXPECTED_NODES="${NGA_EXPECTED_NODES:-4}"
NGA_TOKENIZER_WORKERS="${NGA_TOKENIZER_WORKERS:-8}"
NGA_EXPECTED_SOURCE_SHARDS="${NGA_EXPECTED_SOURCE_SHARDS:-140}"
NGA_VALIDATION_SOURCE="${NGA_VALIDATION_SOURCE:-013_00009.parquet}"
NGA_EXPECTED_TRAIN_SHARDS="${NGA_EXPECTED_TRAIN_SHARDS:-139}"
NGA_EXPECTED_TRAIN_PARTS="${NGA_EXPECTED_TRAIN_PARTS:-$((NGA_EXPECTED_NODES * NGA_TOKENIZER_WORKERS))}"
NGA_DATA_WAIT_SECONDS="${NGA_DATA_WAIT_SECONDS:-43200}"
NGA_DOCUMENT_BATCH_SIZE="${NGA_DOCUMENT_BATCH_SIZE:-512}"
NGA_SAVE_INTERVAL="${NGA_SAVE_INTERVAL:-4768}"
NGA_MICRO_BATCH_SIZE="${NGA_MICRO_BATCH_SIZE:-32}"
NGA_GLOBAL_BATCH_SIZE="${NGA_GLOBAL_BATCH_SIZE:-1024}"
NGA_TRAIN_ITERS="${NGA_TRAIN_ITERS:-47684}"
NGA_ENABLE_FP8="${NGA_ENABLE_FP8:-1}"
NGA_REUSE_MXFP8_GRAD_BUFFER="${NGA_REUSE_MXFP8_GRAD_BUFFER:-0}"
NGA_RUN_EVAL="${NGA_RUN_EVAL:-1}"

if [[ "$WORLD_SIZE" != "$NGA_EXPECTED_NODES" ]]; then
    echo "DLC injected $WORLD_SIZE nodes; expected $NGA_EXPECTED_NODES" >&2
    exit 1
fi
for value in \
    "$NGA_GPUS_PER_NODE" \
    "$NGA_TOKENIZER_WORKERS" \
    "$NGA_EXPECTED_SOURCE_SHARDS" \
    "$NGA_EXPECTED_TRAIN_SHARDS" \
    "$NGA_EXPECTED_TRAIN_PARTS" \
    "$NGA_DOCUMENT_BATCH_SIZE" \
    "$NGA_SAVE_INTERVAL" \
    "$NGA_MICRO_BATCH_SIZE" \
    "$NGA_GLOBAL_BATCH_SIZE" \
    "$NGA_TRAIN_ITERS"; do
    if ((value < 1)); then
        echo "positive integer required, found $value" >&2
        exit 1
    fi
done
if [[ "$NGA_ENABLE_FP8" != "0" && "$NGA_ENABLE_FP8" != "1" ]]; then
    echo "NGA_ENABLE_FP8 must be 0 or 1" >&2
    exit 1
fi
if [[ "$NGA_REUSE_MXFP8_GRAD_BUFFER" != "0" && "$NGA_REUSE_MXFP8_GRAD_BUFFER" != "1" ]]; then
    echo "NGA_REUSE_MXFP8_GRAD_BUFFER must be 0 or 1" >&2
    exit 1
fi
if [[ "$NGA_RUN_EVAL" != "0" && "$NGA_RUN_EVAL" != "1" ]]; then
    echo "NGA_RUN_EVAL must be 0 or 1" >&2
    exit 1
fi
if ((NGA_GLOBAL_BATCH_SIZE % (WORLD_SIZE * NGA_GPUS_PER_NODE * NGA_MICRO_BATCH_SIZE) != 0)); then
    echo "global batch must be divisible by DP world size times micro batch" >&2
    exit 1
fi

mountpoint -q /mnt/nas
mountpoint -q /mnt/oss
mountpoint -q /mnt/oss-dataset
test -d "$NGA_SOURCE_DATA"
test -f "$NGA_SOURCE_DATA/$NGA_VALIDATION_SOURCE"
test -f "$NGA_TOKENIZER/config.json"
test -f "$NGA_TOKENIZER/tokenizer.json"
test -f "$NGA_LM_EVAL_SITE/lm_eval/__init__.py"
test "$(git -C "$NGA_REPO_ROOT" rev-parse HEAD)" = "$NGA_EXPECTED_COMMIT"
repo_drift="$({ git -C "$NGA_REPO_ROOT" status --porcelain=v1 --untracked-files=all || true; } \
    | grep -Ev '^\?\? (\.LAUNCH_READY|repo-head\.txt)$' || true)"
if [[ -n "$repo_drift" ]]; then
    echo "repository is not clean: $repo_drift" >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$(dirname "$NGA_PYTHON"):$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-$CUDA_HOME/bin/ptxas}"
export PYTHONPATH="$NGA_REPO_ROOT/src:$NGA_MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/mnt/oss/datasets/eval-cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-24}"
export TOKENIZERS_PARALLELISM=true
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NGA_CONTAINER_DIGEST="${NGA_CONTAINER_DIGEST:-nemo-26.06}"
export NGA_REPO_ROOT NGA_EXPECTED_COMMIT NGA_SOURCE_DATA NGA_SOURCE_MANIFEST
export NGA_DATA_ROOT NGA_TOKENIZER NGA_OUTPUT_ROOT NGA_MODEL_EXPORT_ROOT
export NGA_LM_EVAL_SITE
export NGA_MICRO_BATCH_SIZE NGA_GLOBAL_BATCH_SIZE NGA_TRAIN_ITERS
export NGA_GPUS_PER_NODE NGA_EXPECTED_NODES NGA_EXPECTED_TRAIN_PARTS NGA_ENABLE_FP8
export NGA_SAVE_INTERVAL NGA_REUSE_MXFP8_GRAD_BUFFER

mkdir -p "$NGA_DATA_ROOT" "$NGA_OUTPUT_ROOT/logs" "$NGA_OUTPUT_ROOT/data-cache"

"$NGA_PYTHON" -m torch.distributed.run \
    --nnodes="$WORLD_SIZE" \
    --nproc-per-node="$NGA_GPUS_PER_NODE" \
    --node-rank="$RANK" \
    --master-addr="$MASTER_ADDR" \
    --master-port="$MASTER_PORT" \
    --module archlab.megatron.collective_probe \
    2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/collective-node-$RANK.log"
test -f "$NGA_OUTPUT_ROOT/COLLECTIVE_VALIDATED.json"

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
    "bos_token_id": 151643,
    "eos_token_id": 151643,
    "pad_token_id": None,
}
drift = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
if drift:
    raise SystemExit(f"Qwen config drift: {drift}")
PY

if [[ "$RANK" = "0" && ! -f "$NGA_SOURCE_MANIFEST" ]]; then
    "$NGA_PYTHON" -m archlab.megatron.data inventory-parquet \
        --source-root "$NGA_SOURCE_DATA" \
        --expected-shards "$NGA_EXPECTED_SOURCE_SHARDS" \
        --validation-source "$NGA_VALIDATION_SOURCE" \
        --output "$NGA_SOURCE_MANIFEST"
fi

deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
while [[ ! -f "$NGA_SOURCE_MANIFEST" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for source inventory" >&2
        exit 1
    fi
    sleep 15
done

if [[ ! -f "$NGA_DATA_ROOT/DATA_READY.json" ]]; then
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
        --qwen-eos-id 151643

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
            --qwen-eos-id 151643

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
            --required-train-tokens 100000000000 \
            --output "$NGA_DATA_ROOT/DATA_READY.json"
    fi
fi

deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
while [[ ! -f "$NGA_DATA_ROOT/DATA_READY.json" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for Qwen-tokenized FineWeb-Edu" >&2
        exit 1
    fi
    sleep 30
done

if [[ "$RANK" = "0" ]]; then
    "$NGA_PYTHON" -m archlab.megatron.data validate \
        --ready "$NGA_DATA_ROOT/DATA_READY.json" \
        --source-manifest "$NGA_SOURCE_MANIFEST" \
        --tokenizer "$NGA_TOKENIZER/tokenizer.json" \
        --train-parts "$NGA_EXPECTED_TRAIN_PARTS" \
        --valid-parts 1 \
        --required-train-tokens 100000000000 \
        --output "$NGA_OUTPUT_ROOT/DATA_VALIDATED.json"
fi

deadline="$((SECONDS + NGA_DATA_WAIT_SECONDS))"
while [[ ! -f "$NGA_OUTPUT_ROOT/DATA_VALIDATED.json" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for validated data contract" >&2
        exit 1
    fi
    sleep 5
done

if [[ "$RANK" = "0" && ! -f "$NGA_OUTPUT_ROOT/RUN_CONTRACT.json" ]]; then
    "$NGA_PYTHON" - <<'PY'
import hashlib, json, os, platform, time
from pathlib import Path
import torch

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

tokenizer = Path(os.environ["NGA_TOKENIZER"])
tokens_per_iteration = int(os.environ["NGA_GLOBAL_BATCH_SIZE"]) * 2048
payload = {
    "model": "Qwen2.5-1.5B architecture, initialized from scratch",
    "model_config": json.loads((tokenizer / "config.json").read_text()),
    "model_config_sha256": sha(tokenizer / "config.json"),
    "tokenizer_sha256": sha(tokenizer / "tokenizer.json"),
    "dataset": json.loads((Path(os.environ["NGA_DATA_ROOT"]) / "DATA_READY.json").read_text()),
    "source_manifest_sha256": sha(os.environ["NGA_SOURCE_MANIFEST"]),
    "source_commit": os.environ["NGA_EXPECTED_COMMIT"],
    "seed": 42,
    "nodes": int(os.environ["NGA_EXPECTED_NODES"]),
    "gpus_per_node": int(os.environ["NGA_GPUS_PER_NODE"]),
    "parallelism": {
        "data": int(os.environ["NGA_EXPECTED_NODES"]) * int(os.environ["NGA_GPUS_PER_NODE"]),
        "tensor": 1,
        "pipeline": 1,
        "context": 1,
    },
    "sequence_length": 2048,
    "micro_batch_sequences": int(os.environ["NGA_MICRO_BATCH_SIZE"]),
    "global_batch_sequences": int(os.environ["NGA_GLOBAL_BATCH_SIZE"]),
    "target_tokens": 100000000000,
    "effective_tokens": int(os.environ["NGA_TRAIN_ITERS"]) * tokens_per_iteration,
    "checkpoint_interval_steps": int(os.environ.get("NGA_SAVE_INTERVAL", "4768")),
    "checkpoint_interval_tokens": int(os.environ.get("NGA_SAVE_INTERVAL", "4768")) * tokens_per_iteration,
    "precision": {
        "transformer_compute": "MXFP8 hybrid" if os.environ["NGA_ENABLE_FP8"] == "1" else "BF16",
        "master_weights": "BF16",
        "optimizer_state": "FP32",
        "fp8_parameter_gather": os.environ["NGA_ENABLE_FP8"] == "1",
        "reuse_mxfp8_gradient_buffer": os.environ["NGA_REUSE_MXFP8_GRAD_BUFFER"] == "1",
    },
    "evaluation": {
        "harness": "lm_eval==0.4.13",
        "runtime_site": os.environ["NGA_LM_EVAL_SITE"],
        "tasks": ["arc_easy:25", "hellaswag:10", "piqa:0", "winogrande:5", "gsm8k:5"],
    },
    "container": os.environ.get("NGA_CONTAINER_DIGEST"),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "python": platform.python_version(),
    "created_at_unix": time.time(),
}
path = Path(os.environ["NGA_OUTPUT_ROOT"]) / "RUN_CONTRACT.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
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
if [[ "${#train_prefixes[@]}" != "$NGA_EXPECTED_TRAIN_PARTS" ]]; then
    echo "unexpected Qwen FineWeb-Edu train part count: ${#train_prefixes[@]}" >&2
    exit 1
fi
if [[ "${#valid_prefixes[@]}" != "1" ]]; then
    echo "unexpected Qwen FineWeb-Edu validation part count: ${#valid_prefixes[@]}" >&2
    exit 1
fi

fp8_args=()
if [[ "$NGA_ENABLE_FP8" = "1" ]]; then
    fp8_args=(
        --fp8-format hybrid
        --fp8-recipe mxfp8
        --fp8-param-gather
    )
    if [[ "$NGA_REUSE_MXFP8_GRAD_BUFFER" = "1" ]]; then
        fp8_args+=(--reuse-grad-buf-for-mxfp8-param-ag)
    fi
fi
load_args=()
if [[ -f "$NGA_OUTPUT_ROOT/checkpoints/latest_checkpointed_iteration.txt" ]]; then
    checkpoint_iteration="$(tr -d '[:space:]' < "$NGA_OUTPUT_ROOT/checkpoints/latest_checkpointed_iteration.txt")"
    if [[ ! "$checkpoint_iteration" =~ ^[0-9]+$ ]]; then
        echo "invalid latest checkpoint iteration: $checkpoint_iteration" >&2
        exit 1
    fi
    checkpoint_dir="$NGA_OUTPUT_ROOT/checkpoints/iter_$(printf '%07d' "$checkpoint_iteration")"
    test -f "$checkpoint_dir/.metadata"
    test -f "$checkpoint_dir/common.pt"
    load_args=(--load "$NGA_OUTPUT_ROOT/checkpoints")
fi

if [[ ! -f "$NGA_OUTPUT_ROOT/TRAINING_COMPLETE.json" ]]; then
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
        "${fp8_args[@]}" \
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
        --tensorboard-dir "$NGA_OUTPUT_ROOT/tensorboard" \
        "${load_args[@]}" \
        2>&1 | tee -a "$NGA_OUTPUT_ROOT/logs/node-$RANK.log"

    if [[ "$RANK" = "0" ]]; then
        "$NGA_PYTHON" - <<'PY'
import json, os, time
from pathlib import Path
path = Path(os.environ["NGA_OUTPUT_ROOT"]) / "TRAINING_COMPLETE.json"
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps({"completed_at_unix": time.time()}, sort_keys=True) + "\n")
temporary.replace(path)
PY
    fi
fi

deadline="$((SECONDS + 3600))"
while [[ ! -f "$NGA_OUTPUT_ROOT/TRAINING_COMPLETE.json" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for training completion marker" >&2
        exit 1
    fi
    sleep 15
done

if [[ "$RANK" = "0" ]]; then
    "$NGA_PYTHON" -m archlab.megatron.export_eval validate-checkpoints \
        --checkpoint-root "$NGA_OUTPUT_ROOT/checkpoints" \
        --train-iters "$NGA_TRAIN_ITERS" \
        --save-interval "$NGA_SAVE_INTERVAL" \
        --tokens-per-iteration "$((NGA_GLOBAL_BATCH_SIZE * 2048))" \
        --output "$NGA_OUTPUT_ROOT/CHECKPOINTS_VALIDATED.json"
    if [[ "$NGA_RUN_EVAL" = "1" && ! -f "$NGA_OUTPUT_ROOT/EVAL_COMPLETE.json" ]]; then
        "$NGA_PYTHON" -m archlab.megatron.export_eval run \
            --checkpoint-root "$NGA_OUTPUT_ROOT/checkpoints" \
            --hf-reference "$NGA_TOKENIZER" \
            --hf-output "$NGA_MODEL_EXPORT_ROOT/hf" \
            --eval-output "$NGA_OUTPUT_ROOT/eval" \
            --lm-eval-site "$NGA_LM_EVAL_SITE" \
            --completion-marker "$NGA_OUTPUT_ROOT/EVAL_COMPLETE.json"
    elif [[ "$NGA_RUN_EVAL" = "0" ]]; then
        "$NGA_PYTHON" -m archlab.megatron.export_eval mark-skipped \
            --output "$NGA_OUTPUT_ROOT/EVAL_COMPLETE.json"
    fi
fi

deadline="$((SECONDS + 21600))"
while [[ ! -f "$NGA_OUTPUT_ROOT/EVAL_COMPLETE.json" ]]; do
    if ((SECONDS >= deadline)); then
        echo "rank $RANK timed out waiting for final evaluation" >&2
        exit 1
    fi
    sleep 30
done

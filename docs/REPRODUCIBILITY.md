# Reproducibility guide

The repository separates three levels of reproduction:

- **artifact integrity:** validate the published manifest and metric table on any CPU machine;
- **functional reproduction:** run unit tests and a small forward/backward pass;
- **comparison-grade training:** reproduce a frozen run with the matching data, tokenizer, seed, geometry, and hardware-aware software stack.

## 1. Install

```bash
uv sync --extra cpu --group dev
```

Use `--extra gpu` for the CUDA 12.8 wheel index. The frozen campaign used Python 3.10.12, PyTorch 2.9.1+cu128, CUDA 12.8, BF16, and NVIDIA L20D GPUs. CPU tests are not a throughput reproduction.

## 2. Validate published artifacts

```bash
uv run next-gen-arch verify
uv run python -c "from next_gen_arch.results import verify_metrics; print(verify_metrics())"
```

The manifest must form a complete `3 sizes × 16 variants × 3 seeds = 144 runs` Cartesian grid. The result checker validates schema, unique rows, finite BPB/deltas, and seed counts.

## 3. Prepare data

Choose a writable storage directory outside the Git checkout:

```bash
export NANOCHAT_BASE_DIR=/path/to/next-gen-arch-data
uv run python -m nanochat.dataset -n 170
```

The downloader retrieves train shards 0–169 and the fixed validation shard 6542 from the public `karpathy/climbmix-400b-shuffle` dataset. The campaign data directory contained 171 parquet shards and had the following name/size fingerprint:

```text
990638db89dc3d6b235e12f5728f070d26c4290b20ed52d81298dd8788d85dcb
```

This fingerprint hashes the sorted file-name/byte-size inventory; it is an integrity check, not a content checksum for every byte. Dataset revisions or partially downloaded shards can invalidate comparison-grade reproduction.

## 4. Prepare the tokenizer

The campaign fixed one 32,768-token nanochat BPE for every run. Its recorded artifacts are:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `tokenizer.pkl` | 412,105 | `62c6425cb358409c034d039da579574c60f9af2796b5cb36d0875cd70653bc9e` |
| `token_bytes.pt` | 132,677 | `b86d28d4d4ac667c061020eeef2434c2010da0ccd69e0c5242596069fc00f05c` |
| `training_manifest.json` | 678 | `d04c935ef45f4fed033c659a5a3f9763ccad98d320f8de3741c4603ae3feca3d` |

You can train a compatible tokenizer for new experiments:

```bash
uv run python -m scripts.tok_train \
  --vocab-size 32768 \
  --max-chars 2000000000 \
  --doc-cap 10000
```

A compatible tokenizer is not necessarily byte-identical. For paired reproduction, use the same frozen tokenizer artifacts and verify their hashes.

## 5. Inspect and launch a run

```bash
uv run next-gen-arch show --size 100m --variant qwen-gdn --seed 42
uv run next-gen-arch command \
  --size 100m \
  --variant qwen-gdn \
  --seed 42 \
  --run-name reproduce-qwen-gdn-100m-s42
```

The second command prints the training invocation instead of executing it. This makes launch scripts inspectable and prevents an accidental multi-day run. Review the command, storage, GPU visibility, and experiment logger before running it.

The core frozen training values are:

| Field | Value |
| --- | --- |
| sequence length | 2,048 |
| device batch | 16 sequences |
| total batch | 393,216 tokens |
| precision | BF16 |
| warmdown | 65% of steps |
| final LR fraction | 0.05 |
| weight decay | 0.28 |
| evaluation cadence | 250 steps |
| evaluation budget | 3,932,160 tokens |
| save cadence | 1,000 steps |

Model geometry, steps, exact tokens, warmup, parameters, and variant flags differ by manifest row and are never inferred from the display label.

## 6. Preserve pairing

For a valid architecture comparison:

1. launch baseline and variant from the same manifest generation;
2. use identical seed, shard order, tokenizer, batch, and evaluation stream;
3. avoid retrying only the worse seed unless the failure is operational and documented;
4. retain the raw run summary and the exact source commit;
5. compute a seed-wise variant-minus-baseline delta before averaging.

The public manifest retains the frozen experimental source-tree hashes. The consolidated open-source code intentionally differs because it merges the Engram and mHC forks and adds fail-fast behavior. Use the source hashes to distinguish historical-result provenance from new runs made with this repository.

## 7. Numerical failure policy

`scripts/base_train.py` checks:

- validation loss whenever validation runs;
- every training microstep loss;
- gradients every `--finite-check-every` optimizer steps, defaulting to every step.

Any rank reporting a non-finite value causes all distributed ranks to raise. Record the run as failed/non-finite; do not use its last finite checkpoint as an unplanned endpoint.

## 8. Engram reference-model repair

Training constructs a smaller d12 meta-reference model for optimizer hyperparameter calibration. The historical 1B Engram launch passed full-model injection layers `7,15,23` into that d12 model and failed its bounds assertion.

The consolidated trainer maps reference layers proportionally to the target depth while leaving the actual d32 model at `7,15,23`. This is a harness repair. A rerun should preserve every other manifest field and should be labeled as a new result rather than silently replacing the dated audit.

## 9. Source and result provenance

- nanochat upstream commit recorded by the campaign: `b9f5025652d51470e2c31117100d9ff48717b911`
- frozen core tree SHA-256: `fc6bf75d17121b3321877d4aede10205fac0b8d4c33ed2e99cb426ca78feec67`
- frozen Engram fork tree SHA-256: `3ef8c02da5242dfcf3cebe69a24098d7693dcd480473202f691a22ab10ff4652`
- frozen mHC fork tree SHA-256: `c79b52cc9fc577ef61bdd61858b5bb070c5a3e6a9381d0b04157055d76b9026b`
- frozen orchestration tree SHA-256: `2ffc3950ea4679e66e1f1105a0d5fe7eab238dc712109a6fa5393bfffc5ecf00`

The exact per-file hashes remain in [`results/parameter-scale-100m-1b-v1-manifest.json`](../results/parameter-scale-100m-1b-v1-manifest.json).

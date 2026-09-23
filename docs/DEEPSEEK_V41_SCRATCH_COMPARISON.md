# DeepSeek V4.1 scratch attention comparison

This experiment trains two randomly initialized text models with every weight unfrozen. Both use the same native DeepSeek V4.1 backbone, with eight additive attention branches. The controlled difference is whether those branches use normal local attention or 2-simplicial attention; native backbone attention is retained in both variants.

| Variant | Total trainable parameters | Engram lookup parameters | Remaining parameters |
|---|---:|---:|---:|
| Normal attention | 29,055,310,008 | 24,576,731,200 | 4,478,578,808 |
| 2-simplicial | 29,055,637,816 | 24,576,731,200 | 4,478,906,616 |

Counts are logical global parameters, excluding distributed replicas. Channel dimensions are reduced by eight and depth by two: hidden width 640, 20 layers. Head counts, 384 routed experts with top-6 selection, and Engram hash-bucket counts are preserved. Consequently, Engram lookups still account for most parameters. The simplicial model has 327,808 additional parameters.

## Code review map

- [Scaling contract](../src/archlab/architectures/deepseek_v41_scratch.py): width/depth scaling and adapter-layer selection.
- [Construction](../src/archlab/automodel/deepseek_v41_scratch_construct.py): random initialization, native backbone, distributed setup, precision boundaries, and parameter accounting.
- [Normal-attention branch](../src/archlab/architectures/deepseek_v41_normal_adapter.py): deterministic local attention with shared initialization and residual-stream read/write gates.
- [2-simplicial branch](../src/archlab/architectures/deepseek_v41_adapter.py): two key/value projections, branch geometry, and the simplicial attention call.
- [Deterministic simplicial implementation](../src/archlab/architectures/simplicial_deterministic.py): forward and backward implementation.
- [Backbone insertion](../src/archlab/automodel/deepseek_v41_official_adapter.py): additive branch applied at the native attention hyperconnection expansion.
- [Trainer](../src/archlab/automodel/deepseek_v41_scratch_training.py): loss normalization, optimizer, routing updates, validation, checkpointing, and exact stopping step.
- [Portable experiment recipe](../recipes/experiments/deepseek_v41_scratch_w640_d20.yaml).

The native backbone is supplied by the external NeMo AutoModel runtime, pinned to commit `f7ccd6f7902634af34c2f31b3294ac250dc97670`. Its primary files are `nemo_automodel/components/models/deepseek_v41/model.py`, `attention.py`, and `engram.py`. Runtime libraries and upstream source are not vendored here. [Source checksums](DEEPSEEK_V41_SCRATCH_SOURCE.json) identify the implementation copied from the running experiment snapshot; that snapshot identifier is provenance, not the commit ID of this publication.

## Training contract

Each variant uses one eight-GPU node, expert parallelism 8, GPU-resident FP32-factored Adafactor with stochastic BF16 updates, and no CPU weight or optimizer offload. Sequence length is 2,048; microbatch 8 and accumulation 1 give 64 document windows per update. All branches use 8 query heads, 2 KV heads, and head dimension 16, inserted at layers 2, 4, 7, 9, 12, 14, 17, and 19.

The target is 10B supervised tokens. FineWeb-Edu examples preserve document boundaries; right padding is excluded from attention, routing, indexer objectives, and token accounting. Training includes token-weighted routing auxiliary loss, centered proportional expert-bias correction, and rolling expert coverage. The learning-rate schedule warms up for 100 updates, then decays by supervised-token progress.

Machine paths are supplied through the environment variables named in the recipe. Construction and execution require the qualified container runtime and external assets; the CPU tests do not establish distributed runtime compatibility for another container. The trainer requires a clean source checkout and `NGA_EXPECTED_COMMIT` matching that checkout, followed by qualification with the same contract.

## Checkpoints and matching

[Full-state checkpoints](../src/archlab/automodel/deepseek_v41_full_checkpoint.py) preserve model tensors, optimizer state, RNG state, and the exact data cursor across all ranks. A completion marker is written only after all rank manifests are complete.

[Newest-only retention](../src/archlab/reporting/checkpoint_retention.py) is an external service with explicit run roots. It validates the newest complete checkpoint before removing older saves, preserves incomplete writes and configured dependencies, and keeps deletion receipts. It does not alter running model code or save cadence. Scratch saves occur every 500M additional targets, on stop/final completion, and ten updates after a start or resume.

To obtain matched checkpoints, checkpoint and pause the faster variant using its `STOP_REQUEST` file. Read the resulting completed checkpoint cursor, then resume the slower variant with the trainer's `--steps TARGET_STEP` option. Matching requires identical dataset contracts and data ordering; wall-clock matching alone is insufficient. [The independent verifier](../src/archlab/reporting/v41_matched_checkpoint_watch.py) records completion only when both checkpoints have the requested token count, identical cursors, and valid shard manifests and payload lengths.

## Validation

Focused unit tests accompany the scaling contract, data cursor, adapter equations, native insertion boundary, router balancing, training normalization, deterministic attention, and retention behavior. GPU-only checks require the qualified runtime. The publication does not include model weights, training data, or private deployment configuration.

Publication validation: **109 passed, 10 GPU-only tests skipped** across the focused training, evaluation, tracking, storage, and reference-data checks, run with CUDA hidden and pytest plugin autoload disabled. No distributed/GPU qualification was rerun for this publication. The published model implementation is byte-identical to the recorded running source snapshot.

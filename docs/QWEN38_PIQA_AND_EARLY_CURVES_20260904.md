# Qwen3.8 PIQA and early-learning snapshot (2026-09-04)

## Quartered-model PIQA curve

The completed 0.95B-parameter quartered run was evaluated at every 10B-token
checkpoint on all 1,838 PIQA validation examples. The scorer reproduces the
zero-shot `lm-eval==0.4.13` prompt and character-length normalization. It loads
the Megatron distributed checkpoints directly with the historical `0fdc753`
architecture code, preserving the sigmoid attention gate used to train them.

| Training tokens | Accuracy | Normalized accuracy |
|---:|---:|---:|
| 10B | 61.53% | 61.81% |
| 20B | 63.93% | 62.35% |
| 30B | 64.58% | 63.66% |
| 40B | 65.83% | 64.64% |
| 50B | 65.56% | 64.85% |
| 60B | 66.05% | 64.47% |
| 70B | 66.38% | 65.23% |
| 80B | 66.70% | 64.53% |
| 90B | 66.70% | 64.53% |
| 100B | 66.92% | 65.07% |

Raw accuracy gained 5.39 percentage points from 10B to 100B tokens. Normalized
accuracy gained 3.26 points and peaked at 65.23% at 70B. Each point has about a
1.1-point standard error, so the small checkpoint-to-checkpoint reversals should
not be interpreted as regressions.

The 20B protocol check reproduced the reference sample likelihoods within
`3.1e-5`. Native-BF16 fused-kernel batch-shape rounding changed one raw answer
out of 1,838 and no normalized answers relative to the earlier reference run.

Artifacts:

- `results/qwen38-quarter-piqa-curve-20260904.png`
- `results/qwen38-quarter-piqa-curve-20260904.csv`
- `results/qwen38-quarter-piqa-curve-20260904.json`

## Quartered versus full early learning

The comparison uses causal cross-entropy only; it excludes the weighted MTP
contribution. Both runs use the same FineWeb-Edu data, seed, 512-sequence global
batch, 2,048-token sequence length, 1,048,576 tokens per optimizer step, Muon
recipe, and learning-rate schedule.

At the live snapshot, the full run had completed 252 logged optimizer steps
(264.2M tokens). Over the last 20 matched steps, causal cross-entropy was 8.743
for the 27.32B full model and 11.205 for the 0.95B quartered model. The full
model's 20-step-smoothed curve crossed below the quartered model at step 119
(124.8M tokens) and remained below it through the snapshot.

This token efficiency costs substantially more wall time. Median step time was
7.92 seconds for the full model and 0.489 seconds for the quartered model, a
16.21x ratio. During the same 33.1 minutes needed for the full model's 252
steps, the quartered model completed 4,089 steps (4.29B tokens) and reached a
20-step-smoothed cross-entropy of 3.529, versus 8.743 for the full model.

The result is not a pure parameter-scaling ablation. The historical quartered
run uses the `0fdc753` quarter MTP layout and sigmoid attention gate; the full
run uses the `344e678` source-faithful full MTP and SiLU gate. Microbatch size
also differs (4 quartered versus 1 full), although the global batch and tokens
per step are identical.

Artifacts:

- `results/qwen38-quarter-vs-full-early-20260904.png`
- `results/qwen38-quarter-vs-full-early-20260904.csv`
- `results/qwen38-quarter-vs-full-early-20260904.json`

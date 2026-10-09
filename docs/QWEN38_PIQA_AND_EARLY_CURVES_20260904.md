# Qwen PIQA and early-learning results — 2026-09-04

**Type:** historical snapshot.

## Quartered-model PIQA

0.95B parameters; all 1,838 validation examples; zero-shot `lm-eval==0.4.13` prompt and character normalization. Checkpoint scoring uses the historical `0fdc753` architecture.

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

Normalized accuracy gains 3.26 percentage points from 10B to 100B and peaks at 65.23% at 70B. Per-point standard error is about 1.1 points; small reversals are not established regressions.

The 20B protocol check matched sample likelihoods within 3.1e-5. Batch-shape rounding changed one raw answer and no normalized answers.

## Early quarter/full comparison

| Measurement | Full 27.32B | Quartered 0.95B |
| --- | ---: | ---: |
| Last-20-step CE at 264.2M matched tokens | 8.743 | 11.205 |
| Median update time | 7.92 s | 0.489 s |
| CE at matched 33.1-minute wall time | 8.743 | 3.529 |
| Tokens at that wall time | 264.2M | 4.29B |

This is not a pure parameter-scaling ablation: MTP layout, output-gate function and microbatch differ. Global batch and data-token schedule match.

[PIQA data](recorded-results/qwen38-quarter-piqa-curve-20260904.json) · [PIQA figure](recorded-results/qwen38-quarter-piqa-curve-20260904.png) · [Early comparison](recorded-results/qwen38-quarter-vs-full-early-20260904.json)

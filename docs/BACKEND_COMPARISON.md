# Matched 10M backend comparison

**Type:** completed historical study, 2026-08-25/26.
**Question:** does Megatron preserve the architecture signal of the frozen speedrun reference?

## Contract

16 variants × seeds 42/43/44; same ClimbMix/tokenizer, BF16, 2K context, global batch 192 and ~12 tokens/parameter. Baseline: 9,363,488 parameters, 286 steps, 112,459,776 training tokens; validation: 3,932,160 tokens.

Megatron uses the project comparison wrapper with TP=PP=CP=1. This is not qualification of every mechanism under model parallelism.

## Accepted safe-autotune results

| Variant | Megatron BPB | Megatron Δ | Megatron throughput | Speedrun BPB | Speedrun Δ | Speedrun throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 1.547758 | +0.000000 | 1.00× | 1.554980 | +0.000000 | 1.00× |
| `engram` | 1.489828 | -0.057930 | 1.00× | 1.489722 | -0.065259 | 0.99× |
| `kda` | 1.465789 | -0.081969 | 0.34× | 1.462224 | -0.092757 | 0.45× |
| `kimi-k3-kda-update` | 1.468116 | -0.079642 | 0.35× | 1.461594 | -0.093387 | 0.46× |
| `qwen-gdn` | 1.493180 | -0.054578 | 0.39× | 1.483103 | -0.071878 | 0.47× |
| `attnres` | 1.499598 | -0.048160 | 0.98× | 1.498153 | -0.056828 | 0.92× |
| `mhc` | 1.529482 | -0.018276 | 0.86× | 1.539949 | -0.015031 | 0.64× |
| `gated-attention` | 1.540339 | -0.007419 | 1.01× | 1.549218 | -0.005763 | 0.98× |
| `situ-glu` | 1.549041 | +0.001283 | 1.01× | 1.559727 | +0.004747 | 1.02× |
| `inkling-relative-attention` | 1.522713 | -0.025045 | 0.40× | 1.558444 | +0.003463 | 0.42× |
| `glm-mla` | 1.545573 | -0.002185 | 0.99× | 1.553357 | -0.001624 | 1.01× |
| `xielu` | 1.545928 | -0.001830 | 0.98× | 1.557475 | +0.002495 | 0.98× |
| `inkling-sconv-kv` | 1.523050 | -0.024708 | 0.97× | 1.547653 | -0.007328 | 0.94× |
| `inkling-sconv-residual` | 1.490894 | -0.056864 | 0.97× | 1.505017 | -0.049964 | 0.96× |
| `partial-rope-25` | 1.558349 | +0.010591 | 0.99× | 1.579620 | +0.024639 | 1.01× |
| `dsa` | 1.627023 | +0.079265 | 0.47× | 1.633543 | +0.078563 | 0.32× |

| Aggregate | Value |
| --- | ---: |
| Paired-delta correlation | 0.971361 |
| Mean absolute delta gap | 0.009241 BPB |
| Matching direction | 13/15 variants |
| Megatron median steady throughput | 1,576,266 tokens/s |
| Megatron post-warmup aggregate | 1,529,304 tokens/s |
| Speedrun historical aggregate | 1,360,228 tokens/s |
| Cold lifecycle | 567.0 s Megatron; 79.8 s speedrun |

Timing definitions differ; compare the matching aggregate when quoting the 1.124× throughput ratio.

## Corrections

| Issue | Accepted treatment |
| --- | --- |
| Max-autotune portability | 13 stable variants retain it; KDA, Kimi K3 KDA and GDN use default compilation |
| Failed/corrupted rows | Preserve original evidence; overlay nine explicitly qualified controls |
| Original DSA warmup bug | Use live iteration rather than checkpoint iteration; rerun three seeds |
| DSA implementation | Dense masked SDPA; no sparse-throughput claim |

Original regular-compile results remain a separate reference. Their architecture-delta correlation was 0.975948; baseline throughput was 752,144 tokens/s.

## Evidence and reproduction

- [Accepted safe-autotune campaign](recorded-results/megatron-10m-safe-autotune-b300/comparison.md).
- [Original aggregate](recorded-results/backend-10m-comparison.json) and [per-run ledger](recorded-results/backend-10m-runs.csv).
- [Frozen speedrun input](recorded-results/speedrun-10m-reference.csv).
- [Reproduction procedure and revisions](REPRODUCIBILITY.md).

Correction/recovery overlays must identify exact known keys and reject duplicates. Absolute cross-backend loss differences are not component effects.

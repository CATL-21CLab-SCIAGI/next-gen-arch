# Resident RL cache status — 2026-09-23

**Current status:** rejected for production. Both actual step-4537 parents failed
cache equivalence despite passing 16-rank tiny-model tests. Maximum all-vocabulary log-probability
errors were 5.037 nats (normal) and 5.876 nats (simplicial), above the unchanged 0.02
bound. Neither failed cached attempt applied an optimizer update. The cache remains
experimental and disabled in the active retained-weight fallback.

Real-model evidence: `production-{normal,simplicial}-cached-v1/rank-00-cache-equivalence.json`
under `results/deepseek-v41-math-rl-shared-20260923/`.

## Restart candidate

Recipe: `recipes/experiments/deepseek_v41_nemotron_rloo_cached.yaml`.
No CPU offloading or concurrent general evaluation. Gathered weights are retained
during sampling, then reshared before gradient replay; the existing 16 GiB driver
reserve guard remains.

The fixed replay canvas preserves FP32 mixing and RMS reduction shapes. The cache
retains the backbone state and each adapter’s last 512 input positions, executing
the original normal or simplicial adapter forward. The experimental one-query
adapter kernel is not used. Compressed-attention slot positions and indexer score
widths match padded replay. Unused full-vocabulary logits are limited to one position
through the container model’s existing `logits_to_keep` option; RL scores are computed
separately by the qualified head.

Evidence: `results/deepseek-v41-math-rl-shared-20260923/cache-qualified-v4/`.
These tiny-model checks establish numerical behavior, not production throughput
or real optimizer progress. The 0.02-nat and 0.001 relative-RMS cache gates are unchanged.

## Historical prototype

## Implementation

| Component | Path / behavior |
| --- | --- |
| Resident cache | `automodel/deepseek_v41_rl_cache.py`; request-local CSA2 and adapter state |
| Ordinary adapter | Container FlashAttention |
| Simplicial adapter | One-query form of the existing deterministic Triton forward |
| Replay | Separate actual execution shapes and padded full-prefix replay shapes |
| Recipe flag | `cache_policy`; uniform local prompt lengths required |

Cache state is discarded before optimization. Existing TileLang sparse backward is unchanged.

## Evidence

| Check | Result |
| --- | --- |
| Focused CPU suites | 95 passed |
| Small adapter kernels, lengths 1/5/17/33 | Exact tested outputs |
| Tiny CSA2 compression-boundary checks | Exact tested outputs |
| Tiny six-layer model, nonzero adapters | Simplicial exact; normal hidden max difference 0.0078125 |
| Sixteen-rank tiny normal, production runtime | Failed: all-vocabulary log-probability error 0.0820694 >0.02 |
| After padded-prefill correction | Initial prompt error zero; local incremental error still 0.0349550 >0.02 |

The first small GPU checks used bare system Python. They did not qualify the production runtime. The distributed probe stopped before any optimizer update.

## Required next

1. Resolve remaining incremental drift.
2. Qualify FSDP/EP/Engram ownership and cleanup.
3. Pass actual-parent hidden/log-probability comparison and real-rollout replay.
4. Measure throughput and memory at the intended generation budget.

Local evidence: `results/deepseek-v41-math-rl-cache-20260923/`.

[RL strategy](DEEPSEEK_V41_RL_MIMO_LESSONS.md) · [Shared-GPU restart candidate](DEEPSEEK_V41_RL_SHARED_RESTART.md)

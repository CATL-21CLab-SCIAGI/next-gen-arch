# Resident RL cache status — 2026-09-23

**Status:** experimental and disabled by default. **Production admission:** not passed.

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

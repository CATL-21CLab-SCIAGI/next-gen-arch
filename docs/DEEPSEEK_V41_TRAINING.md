# DeepSeek custom training bridge

**Type:** historical implementation record. **Status:** full-model admission failed.
**Successor:** [official AutoModel integration](DEEPSEEK_V41_OFFICIAL_TRAINING.md).

## Intended contract

| Item | Setting |
| --- | --- |
| Base | DeepSeek-V4.1-Flash, revision `df42c109f1defefcbfcedbe7d905718a12266e40` |
| Budget | Exactly 1B supervised next-token targets |
| Adapter | Eight 8Q/2KV/head128 branches; 32×512 causal pair windows |
| Trainable state | Added branches only |
| Topology | 32 ranks; node-local EP8; TP/PP/CP=1 |
| Placement | Expert/Engram sharding; replicated non-expert base; no base FSDP |
| Data | Sealed seed-2234 native math pilot; separate 1M-target validation |

This route reused AutoModel components and released equations. The then-inspected upstream did not provide the required V4.1 training path.

## Implementation map

| Module suffix under `automodel/deepseek_v41_` | Role |
| --- | --- |
| `recipe.py` | Admission and per-rank training entry |
| `execution.py`, `loading.py` | Native construction, ownership and streamed restore |
| `pytorch.py` | BF16 decoding and project numerical leaves |
| `training.py` | Loss/gradient reduction and checkpoint primitives |

## Numerical findings

| Check | Result |
| --- | --- |
| Corrected tiny native-rounding comparison | Relative logit L2 ~0.00489 |
| Removing activation rounding | ~11% relative logit change in the tiny fixture |
| FP32 simplicial core | Forward error ~1e-7; five input-gradient errors below 0.2% |
| Native FP8 quantizer | Nondeterministic NaNs from finite 2048-position inputs |
| Corrected full-v3 at 128 positions | KL 0.05435–0.29089, exceeding 0.02 |
| Full-v3 at 2048/16384 | Forward comparisons passed; later training gates not reached |

The 1B production run was not admitted through this failed path. Do not reuse passing leaf checks as full-model qualification.

## Evidence

Local root: `results/deepseek-v41-launch-20260913T165222Z/`.
Its `LAUNCH_BLOCKED.md`, `launch.json`, and `full-v3/summary.json` preserve the exact attempt.

Historical reproduction requires its own source, recipe and runtime. Current training should follow the successor's admission contract.

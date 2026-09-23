# SGLang integration preflight — 2026-09-20

**Type:** dated component evidence. This page does not establish deployment status or full-model readiness.

## Project-owned components

| Component | Role |
| --- | --- |
| `architectures/deepseek_v41_incremental.py` | FP32 bounded adapter oracle; request-local state |
| `serving/v41_checkpoint_inventory.py` | Reviewed 16-rank metadata and bounded tensor verification |
| `serving/sglang_v41_engram.py` | BF16 row-sharded Engram lookup |
| `serving/sglang/` | Plugin extending the external engine for our checkpoint/adapter contract |

SGLang itself remains external. The plugin is discovered through `SGLANG_EXTERNAL_MODEL_PACKAGE`.

## Recorded checks

18 CPU tests passed. Tests cover nonzero outputs, chunk/full equivalence, window eviction, request isolation, sharding, corrupt payloads and BF16 lookup.

| Checkpoint | Supervised tokens | Unique tensor bytes | Adapter max absolute error |
| --- | ---: | ---: | ---: |
| Normal 4537 | 756,364,650 | 1,499,017,028,096 | 2.384185791015625e-7 |
| Simplicial 3620 | 603,590,955 | 1,499,100,918,272 | 2.384185791015625e-7 |

These are layer-4 CPU FP32 adapter checks on synthetic streams. They are neither full-model parity nor a matched-token comparison. Full 1.5 TB payloads were not verified by this preflight.

## Engine boundaries requiring qualification

- Insert after attention HC expansion and before the FFN coefficient read.
- Handle fused paths and cached FFN inputs at adapted sites.
- Preserve trained BF16 Engram rows and checkpoint coverage.
- Preserve the declared MoE projection/router/reduction precision.
- Qualify cache allocation, cancellation, slot reuse and cleanup.
- Compare full logits/loss and measured prefill/decode performance.

The inspected source was `e54009240a84bf52eb7a21ec532ea49f1b9dd941`; it is not an assertion about the exact source bundled in the [Dynamo preview image](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.0-deepseek-v4.1-flash-dev.1).

Local receipt: `.runtime/sglang-v41-prototype/CHECKPOINT_ADAPTER_PROBES.json`.
Recipe: `recipes/experiments/deepseek_v41_sglang_prototype.yaml`.

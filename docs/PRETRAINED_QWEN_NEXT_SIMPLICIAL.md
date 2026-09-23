# Pretrained Qwen architecture-adapter experiment

**Type:** historical experiment record. **Audit date:** 2026-09-22. **Status:** both training lineages stopped.

## Contract

| Item | Setting |
| --- | --- |
| Parent | Released Qwen3.8-Flash-Next language backbone |
| Preserved mechanisms | QSA/indexer, GDN, MoE, PLE, residual streams, norms and gates |
| Additions | Simplicial residual branches after attention and before MoE, layers 4, 8, …, 48 |
| Trainable set | Added parameters only; original weights frozen |
| Data/objective | Existing FineWeb-Edu sample-100BT corpus; main next-token CE; 16K context |
| Execution | FSDP32, node-local EP8, expert-FSDP4; separate PLE ownership |
| Inactive components | Vision and MTP accounted for explicitly |
| Control | No separately trained ordinary-adapter arm |

The [portable proposal](../recipes/proposals/qwen38_pretrained_simplicial_fineweb.yaml) records the nonzero-init experiment. Historical resumes use their original source and recipe contracts.

## Initialization and recovery evidence

| Observation | Zero-output initialization | Tested normal-std-0.02 output initialization |
| --- | ---: | ---: |
| Initial held-out CE | 1.936017 | 12.172143 |
| Matched step-100 held-out CE | 1.881850 | 3.724166 |
| Final stopping point | Step 8089 / 4,240,965,632 targets | Step 166 / 87,031,808 targets |
| Last held-out evaluation | Step 8000: 1.862896869 | Step 100: 3.724166 |

The nonzero run intentionally perturbed the base. Its first-batch CE rose from 1.858044 with additions disabled to 12.228165 enabled. It demonstrated recovery, not a benefit over zero initialization.

The original zero-init run resumed its complete step-90 checkpoint at step 91, preserving adapters, Adam state, scheduler, RNG and cursor. The nonzero lineage was not merged into that continuation.

## Qualification

| Check | Evidence |
| --- | --- |
| Disabled nonzero adapters | Exact small-model base identity |
| Frozen backbone | Original weights preserved; no original gradients |
| EP/FSDP continuation | Tiny production entry and fresh-process restore |
| Nonzero continuation envelope | Relative L2 0.006807; maximum weight difference 2.857e-6 |
| Full-topology resume | All 32 rank-state checks passed at cursor 90 |

The runtime permits the recorded BF16/custom-kernel numerical envelope; model-state continuation is not promised bitwise.

## Interpretation

Held-out loss improved in this dataset. A simplicial-specific effect is unisolated because there is no trained ordinary-adapter control. The run stopped before the requested data pass ended.

The completed capability pilot is small and mixed; see [capability evaluation](PRETRAINED_CAPABILITY_EVALUATION.md). It establishes neither broad gains nor comprehensive retention.

## Provenance

| Lineage | Source |
| --- | --- |
| Original | `688f036` |
| Nonzero restart | `e9c31fa` |
| Pinned AutoModel | `a4ce87c003f08b74d68684d3627f6e6048bc0140` |

Local artifact roots:

- `results/pretrained-simplicial-fineweb-20260907-v1/`
- `results/pretrained-simplicial-nonzero-fineweb-20260907-v1/`
- `results/pretrained-simplicial-resume90-20260907-v1-launch.json`

[Backend selection audit](PRETRAINED_BACKEND_REUSE_AUDIT.md) · [Conclusions audit](TRAINING_CONCLUSIONS_20260922.md)

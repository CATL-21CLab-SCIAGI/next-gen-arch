# Qwen W320 source review

**Purpose:** find the implementation of the [selected model contract](QWEN38_NEXT_W320_E32_PROPOSAL.md).

| Responsibility | File under `src/archlab/` |
| --- | --- |
| Geometry, GDN, residuals and PLE | `architectures/qwen38_flash_next_full.py` |
| Native construction | `megatron/qwen38_flash_next_model.py` |
| Trainer | `megatron/qwen38_flash_next_full_train.py` |
| Optimizer/schedule/parallelism arguments | `megatron/qwen38_flash_next_config.py` |
| Deterministic data order | `megatron/token_batches.py` |
| TP1 gated projections and Muon grouping | `megatron/gated_qkv.py` |
| Geometry and numerical regression | `tests/test_qwen38_width320.py` (repository root) |

The launcher is `scripts/run_qwen38_flash_next_full_dlc.sh`; select `recipes/launches/qwen38_flash_next_w320_e32.yaml` with `NGA_LAUNCH_RECIPE`.

## Review invariants

- Emitted shapes/counts/process groups agree with the named Python factory.
- Attention gates multiply attended values before output projection.
- Q/K normalization is per head, before RoPE.
- GR/PLE/QK norm parameters are zero-centered; GDN output norm uses direct gamma.
- DP32 replicates all 32 PLE partitions; model-parallel groups are size one.
- MTP, QSA/indexer and vision remain disabled.

A geometry change needs a new named family, recipe and qualification. Preserve historical constructor semantics and checkpoint keys.

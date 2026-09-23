# DeepSeek official AutoModel integration

**Type:** frozen-backbone adapter training contract.
**Recipe:** `recipes/experiments/deepseek_v41_simplicial_math_1b_official.yaml`.
**Upstream:** `f7ccd6f7902634af34c2f31b3294ac250dc97670`.

## Preserved experiment

Released V4.1 Flash weights; eight adapters; sealed Nemotron-Math-v2 1B-target pilot; seed-2234 order; headwise Muon/AdamW grouping; 16K context.

| Boundary | Implementation |
| --- | --- |
| Construction/loading | Public AutoModel constructor, FSDP2, expert parallelism and DCP |
| Mesh | Dense FSDP32; node-local EP8; expert-FSDP4; Engram32 |
| Sparse attention | Official TileLang forward/per-query backward; private KV-gradient slots and ordered FP32 reduction |
| Hyper-connections | Released coefficient kernel; official projections/collapse/expand; explicit backward |
| MoE | Native BF16 projections, FP32 combination/reduction, private gather-backward storage |
| Simplicial branch | Existing forward; deterministic private-query gradient accumulation |
| Parameters | Decoded BF16 base with explicit FP32 exceptions; replicated FP32 adapter masters |
| Head and cache boundaries | FP32 head; released KV/index quantize/dequantize boundaries |

FSDP uses `output_dtype=None` and `cast_forward_inputs=False`. Record deterministic settings and the installed container versions; runtime frameworks remain external.

## Admission sequence

1. Qualify the actual 32-rank topology on a small six-layer model.
2. Compare the real base with the independent released reference at 128, 2048 and 16384 positions.
3. Require mean KL <0.02 and absolute CE delta <0.05.
4. Install zero-output adapters and verify base identity.
5. Run two short updates, checkpoint continuation, and a full 16K update.
6. Reset qualification state before starting the scientific budget.

Reference and training models run sequentially to avoid two full copies in memory. Source, batch and output-head identities bind their comparison.

## State and replay

| Item | Requirement |
| --- | --- |
| Serialized state | Exact adapter/optimizer/cursor/RNG restore |
| Update-vector relative L2 | Below 1%, and at most 2× measured in-memory baseline with 1e-5 floor |
| Loss replay | rtol 1e-5; atol 1e-7 |
| Completion | Exactly 1B supervised targets, with required final state |

The replay protocol is `native-update-baseline-v1`. Its motivating diagnostic found 0.063% disk-update error versus 0.051–0.065% without disk I/O; restored state and losses matched.

## Operations

`TRAINING_ADMITTED.json` records successful admission; `train.jsonl` records actual progress. `STOP_REQUEST` requests a checkpointed stop. Resume requires the same scientific contract and a fresh attempt directory.

[Normal control](DEEPSEEK_V41_NORMAL_CONTROL.md) · [Full-weight continuation](DEEPSEEK_V41_FULL_COMPARISON.md)

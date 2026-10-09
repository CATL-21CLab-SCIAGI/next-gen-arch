# DeepSeek full-weight comparison

**Type:** experiment contract and result pointer. **Parents:** matched step-307 adapters at 50,119,869 supervised tokens.
Final outcomes are in the [conclusions audit](TRAINING_CONCLUSIONS_20260922.md).

## Matched contract

| Item | Setting |
| --- | --- |
| Variants | Ordinary versus simplicial attention adapters |
| Trainable set | All constructed text parameters, including adapters |
| Per-arm mesh | Dense FSDP16, node-local EP8, expert-FSDP2, Engram16 |
| Global update | Two microbatches × one window/rank = 32 windows |
| Initialization | Each arm's own matched step-307 adapter checkpoint |
| Optimizer | Fresh FP32 factored Adafactor; BF16 stochastic rounding |
| LR | Relative LR 5e-6 → 1e-4 over 20 new-phase updates |
| Clipping/decay | Global gradient clipping 1; weight decay 0 |
| Indexer objective | Sampled KL coefficient 0.01 at 64 query positions/window |

Adafactor stores no dense FP32 masters or first moments. FSDP, expert and owner-table gradients use the recorded global-sum normalization. Replicated adapter gradients are summed once.

The indexer KL is a shared project training choice; hard top-k selection itself has no LM derivative.

## Validation and checkpointing

| Boundary | Requirement |
| --- | --- |
| Numerical | HC/head derivatives, sparse/sink gradients, optimizer and accumulation oracles |
| State | Full parameter/buffer, optimizer, RNG and cursor restore |
| Continuation | Next-update agreement on both 16-rank meshes |
| Periodic validation | Fixed 64,000 assistant targets at resume and each 10M-token boundary |
| Final comparison | Matched 1M-target math and capability subsets |

Validation preserves modes, RNG, optimizer and cursor. The 64K trend subset and 1M evaluation use separate identities and metric names.

The in-place selector optimization preserved keys, selections and candidate masks at 16K and reduced selector peak memory from about 49 to 19 GiB. This does not change the selector objective.

## Storage and interpretation

Completion markers follow verification of every rank. CPU serialization is bounded; training weights and optimizer state stay on GPU. Protect active mmap datasets from path replacement.

The final pair reaches step 4537 / 756,364,650 tokens, below the planned 1B budget. Disabling adapters after this phase does not restore original released weights.

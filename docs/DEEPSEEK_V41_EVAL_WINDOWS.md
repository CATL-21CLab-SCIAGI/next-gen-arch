# DeepSeek checkpoint evaluation service

**Type:** historical deployment record. It does not describe the current RL allocation.

## Contract

| Item | Historical setting |
| --- | --- |
| Baseline | Normal step 4537 / 756,364,650 supervised tokens |
| Baseline workers | 16 GPUs; inference-only checkpoint service |
| Secondary model | Complete simplicial checkpoint for paired evaluation and chat |
| General checks | 1,140 MMLU, 128 ARC-Challenge, 256 PIQA cases |
| Math | Sealed 1M targets; separate 64K periodic validator |
| Chat | Native non-thinking encoder; 2,048 total / 256 response tokens; uncached |
| Restore gate | Baseline 64K loss agrees with its recorded resident-training value |

The matched scientific comparison is 4537/4537. Results against other simplicial steps remain labeled unmatched.

## Lifecycle

1. Restore and qualify the pinned baseline.
2. Restore a requested complete secondary checkpoint.
3. Run paired evaluation, then expose persistent chat.
4. Refresh the secondary model for a new evaluation request.
5. Honor the explicit serving stop marker.

The service creates no optimizer. Checkpoint pins and reader leases protect models in use. Loading a new model can temporarily interrupt chat availability.

## Sources

- Entry: `archlab.automodel.deepseek_v41_checkpoint_service`.
- Policy: `recipes/experiments/deepseek_v41_eval_windows.yaml`.
- Shared protocol: `archlab.automodel.deepseek_v41_live_window`.
- Request deadlines are deployment settings, separate from service lifetime.

[Evaluation guide](wiki/Evaluation.md) · [Full comparison](DEEPSEEK_V41_FULL_COMPARISON.md)

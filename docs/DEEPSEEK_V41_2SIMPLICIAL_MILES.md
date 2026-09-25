# 2-simplicial Miles RL: useful learning per compute

This experiment corrects the September 24 normal baseline's short-response,
`debug_minimal` sampling contract. It uses the same canonical launcher and pinned
Miles driver, with `recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml`.
The sampling contract follows upstream `mode=normal`, not the script’s default
`debug_minimal`. Runtime qualification is recorded separately; changing variants and sampling
means the normal baseline's qualification does not transfer automatically.

## Recipe review and corrections

Reviewed upstream at `6c6858a42b61459467814edc1404b1d9bfa38471`:
`scripts/run_deepseek_v41.py`, `docs/models/deepseek/deepseek-v4-flash.md`,
`docs/advanced/low-precision.md`, native evaluation configurations, dynamic
filters, submission scheduler, and the unmodified training loop.

| Setting | Selected contract and reason |
| --- | --- |
| Parent | `simplicial` step 4537 / 756,364,650 supervised tokens, completion SHA `b60da9ce92b75a163a3aef3bada010eb654870bf6c126267f661661e4ec336a9`. These are trained 2-simplex adapters, not a new random branch or the normal policy's RL checkpoint. |
| Response length | Restore the upstream math-task default of 4,096 response tokens. Our previous 2,001-token override was a memory qualification setting, not the upstream default. |
| Context / train token budget | 5,120 tokens, preserving preformatted non-thinking prompts. No public-checkpoint chat template substitution. |
| Useful groups | Native `check_reward_nonzero_std` filter, which upstream enables outside `debug_minimal`. Train on 16 accepted groups × 8 samples. Filtered training accuracy is selection-conditioned and is not an evaluation score. |
| Oversampling | 512 candidate groups per submission, exactly the pinned upstream normal-mode setting. Native asynchronous completed-group aggregation and cancellation are retained. |
| Serving memory | Eight concurrent requests per engine; 40,960 KV token capacity = eight full 5,120-token contexts. Chunked prefill remains 1,024. Measure real memory and retractions before accepting this budget. |
| Numerical mismatch | Native R3 replay plus native token-level TIS, upper clip 2.0 and lower clip 0.0 (upstream defaults). Keep BF16 train / FP8 rollout. TIS diagnostics must be finite; this is not proof of exact train/serve parity. |
| Optimizer | Stock distributed Adam, lr 1e-6, betas 0.9/0.98, weight decay 0.1; BF16 stored moments, FP32 masters/arithmetic, node-local disk state. Engram tables and router gates remain frozen. |
| Aggregation | Native asynchronous generation and completed-group collection across engines. Colocated rollout/training phases remain sequential (`fully_async=False`); no custom overlap driver or stale-policy reuse. |
| Evaluation | Native math evaluation settings: 4,096 response tokens, top-p 0.7, eight samples per prompt, before training and every 20 updates. Use our disjoint 32-prompt held-out set. |
| Persistence | Request a full checkpoint after the first useful update, then every 20 updates; 24-update cap. Saves cost roughly an hour on this NAS, so include that cost in GPU-hour accounting. |

All execution remains on four nodes / 32 B300 GPUs. Optimizer disk streaming is
intentional; this is not offloading-free training. The V4 Flash document is a
reference, not a license to replace the V4.1 geometry or our trained adapters.

## Launch and qualification

From the prepared pinned runtime and four-node Ray cluster:

```bash
python -m archlab.megatron.miles_v41_stock_launch train \
  --config recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml \
  --run-root "$RUN_ROOT" --miles "$MILES_ROOT" --address "$RAY_ADDRESS" \
  --image-manifest "$IMAGE_MANIFEST"
```

The launcher checks variant and exact parent completion identity, records the
resolved contract before execution, and delegates to native `train.py`.
`RUN_ROOT` for the first qualification is
`results/deepseek-v41-2simplicial-miles-fp8-20260926`. Native evaluation paths are
resolved through the same explicit run-root binding.

Before leaving the run unattended, require complete parent import and weight
coverage, real held-out evaluation, accepted groups with reward contrast, finite
nonzero optimizer updates, finite TIS/policy-gap diagnostics, a synchronized next
rollout, and a completed native checkpoint. Inspect generation cost and cache
retractions. No positive learning claim follows merely from filtered rewards
or nonzero gradients; use fixed-set evaluation and total GPU-hours.

The normal baseline is retired only after its requested final native checkpoint
completes. Its old optimizer scratch can then be reclaimed for the new run;
its persistent checkpoints, logs, and MLflow record remain separate.

## GPU count

The pinned V4 Flash full-model launcher has no 16-GPU parallelism preset and
raises `NotImplementedError` for that layout. V4.1 accepts explicit parallelism,
but that is not qualification of our checkpoint on 16 GPUs. The baseline’s Adam
state alone measured 4.450 TB, exceeding the approximately 3.25 TB combined local
disk capacity of two retained nodes. Reducing to 16 GPUs would require a different
storage/memory arrangement and separate qualification; this run retains 32 B300s.

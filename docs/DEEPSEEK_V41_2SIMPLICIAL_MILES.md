# 2-simplicial Miles RL: useful learning per compute

This experiment corrects the September 24 normal baseline's short-response,
`debug_minimal` sampling contract. It uses the same canonical launcher and pinned
Miles driver, with `recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml`.
The current sampling contract uses native filtering and partial rollouts, with
submission and HTTP concurrency sized to the actual engines. Runtime qualification is recorded separately; changing variants and sampling
means the normal baseline's qualification does not transfer automatically.

## Recipe review and corrections

Reviewed upstream at `6c6858a42b61459467814edc1404b1d9bfa38471`:
`scripts/run_deepseek_v41.py`, `docs/models/deepseek/deepseek-v4-flash.md`,
`docs/advanced/low-precision.md`, native evaluation configurations, dynamic
filters, submission scheduler, and the unmodified training loop.

| Setting | Selected contract and reason |
| --- | --- |
| Parent | `simplicial` step 4537 / 756,364,650 supervised tokens, completion SHA `b60da9ce92b75a163a3aef3bada010eb654870bf6c126267f661661e4ec336a9`. These are trained 2-simplex adapters, not a new random branch or the normal policy's RL checkpoint. |
| Response length | 8,192 live-training response tokens. The 2,001-token baseline truncated 68.9% of responses; even the first 4K simplicial batch truncated 25%. |
| Context / train token budget | 9,216 tokens, preserving the finetuned parent's established non-thinking prompt contract explicitly. This is a checkpoint compatibility choice, not stock upstream thinking-mode math. |
| Useful groups | Native `check_reward_nonzero_std` filter, which upstream enables outside `debug_minimal`. Train on 16 accepted groups × 8 samples. Filtered training accuracy is selection-conditioned and is not an evaluation score. |
| Oversampling | Submit in 16-group increments. Bound the HTTP client to 16 requests per engine (64 total), with 32 active serving slots. Copying the upstream 512-group increment and 2,048-request pool previously overwhelmed our deployment. |
| Serving memory | Eight concurrent requests per engine; 73,728 KV token capacity = eight full 9,216-token contexts. Chunked prefill remains 1,024. This larger allocation still requires measured GPU qualification. |
| Numerical mismatch | Native R3 replay plus native token-level TIS, upper clip 2.0 and lower clip 0.0 (upstream defaults). Keep BF16 train / FP8 rollout. TIS diagnostics must be finite; this is not proof of exact train/serve parity. |
| Optimizer | Stock distributed Adam, lr 1e-6, betas 0.9/0.98, weight decay 0.1; BF16 stored moments, FP32 masters/arithmetic, node-local disk state. Engram tables and router gates remain frozen. |
| Aggregation | Native asynchronous collection and partial-rollout reuse. Colocated generation/training alternate; partial groups older than two published policy updates are dropped and their token counts recorded. Native TIS remains enabled for policy mismatch. |
| Evaluation | Fixed 4,096 response tokens, top-p 0.7, eight samples per prompt on the disjoint 32-prompt set, before training and every four updates. Preserving this budget keeps comparison with the recorded unupdated parent meaningful. |
| Signal metrics | Record integer-safe all-zero/all-one group rates, reward-contrast groups/tokens, unfiltered completed-candidate rewards, retained partials, discarded stale tokens, and observed training entropy. These token counts are not measured GPU FLOPs. |
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
`RUN_ROOT` for the corrected contract is
`results/deepseek-v41-2simplicial-miles-fp8-20260926-corrected`. Native evaluation paths are
resolved through the same explicit run-root binding.

The corrected run-local YAML adds `ARCHLAB_FIRST_ROLLOUT_MANIFEST`, pointing to
`ROLLOUT_BOOTSTRAP.json`. This consumes the saved real 4K parent batch only once,
before live 8K rollouts, and reproduces the identical parent's cached fixed-set
evaluation. It verifies model/data/source hashes, published policy version,
sample grouping and dataset order; it cannot bootstrap an RL checkpoint resume.
The first update is labeled as replayed parent data in metrics. No synthetic
gradient or fabricated training reward is used.

Run the canonical command under `python -m archlab.tracking.process_status
--status "$RUN_ROOT/DRIVER_STATUS.json" -- ...` for a detached process. The
supervisor observes actual child exit, and the MLflow sidecar closes failed runs.
Stale heartbeat means unobserved state, not evidence of ongoing training.

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

## Router capacity

The native router uses `--miles-router-max-connections 8192`, above the
64 currently admitted rollout requests, so health probes can acquire HTTP
connections while generation is queued. Every node runtime must also set
`open_files_soft_limit: 65535` before starting Ray. A 1,024-descriptor limit
failed at the first production request burst; a pool equal to rollout concurrency
subsequently starved health probes and falsely quarantined healthy engines.

The historical first attempt used 2,048 admitted requests and the earlier limits. Its one-time router-only repair
is recorded in `mlflow-evidence/ROUTER_POOL_REPAIR.json` under the run root.
The training driver, native router actor, model weights, and serving engines were
retained. The initial resolved launch remains immutable; this receipt records the
live operational override. CPU tests alone do not qualify the repaired run.

## Clean retry after disrupted collection

At 2026-09-25 22:34 UTC, the same canonical launcher started
`results/deepseek-v41-2simplicial-miles-fp8-20260926-retry1` on the retained
32 B300s. The predecessor had zero optimizer updates: after router recovery,
only one additional complete group arrived in 70 minutes while 185 groups had
partial results. Continuing that damaged request collection was not productive.
The old driver and serving processes were retired, and only their disposable
node-local optimizer scratch was reclaimed.

The retry applies the 8,192-connection pool and 65,535 file-descriptor limit from
startup. Its run-local `experiment.yaml` retains the production sampling and
numerical contract, adding only native `--skip-eval-before-train`. The completed
initial evaluation is preserved and explicitly reused for the identical,
unupdated parent; model, data, parent, and evaluation hashes are recorded in
`mlflow-evidence/INITIAL_EVALUATION_REUSE.json`. Fresh runs should use the
canonical recipe with initial evaluation enabled. Use the run-local YAML only
to reproduce this documented recovery. This retry still needs distributed
update and checkpoint qualification; launch success is not qualification.

## Determinism compatibility and second retry

The first clean retry completed its initial training rollout in 67.8 minutes,
then failed in actor log-probability computation before any optimizer update.
The recipe's `--deterministic-mode` conflicted with the checkpoint's simplicial
kernel, whose backward uses FP32 atomic accumulation. The kernel explicitly
rejects strict deterministic execution. Ray worker-death messages were secondary;
the actor traceback identifies the incompatibility.

The corrected simplicial recipe omits that flag, retains seeded execution and
all sampling, precision, and optimizer settings, and records that execution is
not bitwise deterministic. The launcher rejects a simplicial contract containing
the incompatible flag before runtime initialization. The original kernel and
container libraries remain unchanged. The second retry is recorded under
`results/deepseek-v41-2simplicial-miles-fp8-20260926-retry2`; its run-local recipe
preserves the documented reuse of the unupdated parent's initial evaluation.
CPU launcher tests and a small B300 kernel oracle check precede relaunch; real
updates, weight synchronization, and checkpoint readback still require qualification.

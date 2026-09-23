# DeepSeek math RL — original RLOO contract

**Type:** experiment record. **Lineage:** matched normal/simplicial step-4537 parents, each at 756,364,650 supervised tokens.
The changed restart contract is documented [separately](DEEPSEEK_V41_RL_SHARED_RESTART.md).

## Data and policy

| Item | Setting |
| --- | --- |
| Split | 8,192 training / 512 held-out problems |
| Exclusions | Touched prepared parts, 138,049 normalized problem hashes, 123,957 UUIDs; cross-mode deduplication |
| Input | Problem only; teacher solutions are not policy targets |
| Reference | Supported exact rational/scalar answer used by the reward verifier |
| Longest prompts | 559 training; 299 held-out tokens |
| Sampling | One distinct prompt/rank, four responses/prompt; temperature=top-p=1 |
| Parent ownership | 16 GPUs per actor; original mesh retained |

Exact exclusions do not certify absence of paraphrases or pretraining exposure.

## Objective

Online REINFORCE with leave-one-out baseline:

- Correct final answer: reward 1; incorrect, missing or unsupported answer: 0.
- Each response subtracts the mean reward of its three peers.
- Original loss: mean over trajectories of advantage-weighted token-score sums.
- Prompt/padding targets are masked; globally flat rewards skip optimization.
- All text weights train; fresh factored Adafactor at relative LR 1e-6, clipping 1.
- No reference KL, teacher-answer CE, indexer auxiliary loss or router auxiliary update.

The original bound is 128 rollout batches per arm. Applied optimizer updates and generated-token counts can differ.

## Admission and evaluation

| Gate | Evidence |
| --- | --- |
| Components | Source-bound 16-rank head/gradient and rollout/replay probe |
| Actual actor | Real-model replay/backward qualification; no synthetic optimizer step |
| Pilot | Genuine reward-driven parameter update |
| Math evaluation | Frozen 64-case greedy subset before training, after pilot, then every eight rollout batches |
| Metrics | Raw correctness, valid answer, truncation, policy loss, replay error and throughput |

The eight-case qualification evaluation is distinct. These subsets are diagnostics, not broad capability benchmarks. Original rollout execution is uncached.

## Recovery and artifacts

`STOP_REQUEST` requests a safe-boundary checkpoint and stop. Restore optimizer, RNG, prompt cursor and policy version under the same contract.

Local artifacts: `results/deepseek-v41-math-rl-20260922/`.
Tracking experiment: `deepseek-v41-nemotron-rloo-20260922`.
`ACTIVITY.json`, `RUN_CONTRACT.json`, qualification receipts and complete checkpoints determine actual run state.

[MiMo lessons](DEEPSEEK_V41_RL_MIMO_LESSONS.md) · [Cache status](DEEPSEEK_V41_RL_CACHE_STATUS.md)

# Math RL

**Scope:** verifiable math RL from matched DeepSeek V4.1 checkpoints.

## Original contract

| Item | Setting |
| --- | --- |
| Parents | Matched step-4537 supervised checkpoints |
| Data | 8,192 training / 512 held-out problems; recorded prior-exposure exclusions |
| Sampling | Four responses per prompt; temperature=top-p=1 |
| Objective | Online REINFORCE with leave-one-out baseline (RLOO) |
| Reward | Exact supported final answer; raw correctness recorded separately |
| Replay | Stored behavior log probabilities; declared 0.02-nat tolerance |
| Updates | Flat-reward batches skip optimization |

A rollout batch, an applied optimizer update, and a completed evaluation are different events.

## Lessons for the next experiment

| Observation | Required response |
| --- | --- |
| Heavy response truncation | Calibrate a useful completion budget and memory use |
| Mostly all-failure groups | Measure reward contrast before adding sophisticated ranking |
| Length growth | Examine loss weighting, entropy, and positive/negative advantage mass |
| MoE load drift | Measure router loads and test the declared trainable set |
| Inference/training mismatch | Pass actual-policy replay before deployment |

MiMo Figure 8 studies quality-based redistribution among successful code trajectories. Its separate length penalty applies to successes under a pass-rate gate. Neither creates successful math trajectories from an all-failure group.

## Restart candidate

The versioned shared-GPU proposal uses 1,489 response tokens within a 2,048-token context, prompt-group token normalization, frozen routers, and a bounded correctness-conditioned length deduction.

It also declares a 190 GiB trainer allocator cap, checkpoint-input activation offload, and a 64 GiB driver-free-memory admission requirement. The 190 GiB admission and a 198 GiB retry both failed before updates. Production remains unadmitted; the next memory change still requires distributed and full-model qualification.

The incremental cache remains a separate numerical qualification workstream.

**Source of truth:** `recipes/experiments/deepseek_v41_nemotron_rloo_shared_gpu.yaml`, its run contract and qualification receipts. See `docs/DEEPSEEK_V41_RL_SHARED_RESTART.md`.

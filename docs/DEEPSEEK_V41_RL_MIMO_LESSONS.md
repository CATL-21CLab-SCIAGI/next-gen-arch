# RL strategy after reading MiMo-V2.6 — 2026-09-23

Source: *MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement*,
especially Figure 8 and Sections 4.1, 4.3, 5.1, and 5.4. The supplied PDF matches
the [official report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf)
at SHA-256 `fb81e6e083801b3358f084ed6be953dc23b0d2e434690f4541d5eae03e01e7af`.

These are proposed changes for a new experiment contract. They have not been
applied to the running v3 actors.

## What Figure 8 establishes

The ablation compares code-only RL of MiMo-V2.6-Flash with and without online
Groupwise Advantage Redistribution (GAR), using a batch size of 128 and token-mean
loss aggregation. Evaluation is DeepSWE v1.1 avg@3. Without GAR, mean turns and
token length increase rapidly and pass-rate gains stall. With GAR, pass-rate
improvement persists through step 52, mean turns stay roughly stable, and token
length grows more slowly. Token length still increases with GAR.

GAR ranks passing solutions by quality within mixed-outcome groups, downweights
lower-quality passes, and redistributes their positive advantage to better passes.
The uncapped formula conserves total positive advantage. The practical method
caps its common scale and recenters the group. Figure 8 does not isolate the
separate group-relative length penalty introduced in Section 4.3.3.

The plots show cumulative token use across many agent interaction turns. Their
roughly 140K–210K token scale is not a prescription for a single math completion.
The experiment is evidence for MiMo code-agent training, not proof of an identical
effect in our DeepSeek math setup.

## What transfers to this experiment

1. **Completion and efficiency must be measured together.** Raising a cap alone
   can shift truncation to a later point. Calibrate both parent actors on the same
   prompts, reporting correctness, valid-answer and EOS rates, truncation, correct
   completion lengths, repetitions, memory, and verified answers per GPU-hour.
   Measure 1,024 tokens and larger feasible budgets before selecting a training
   cap. Under the present 2,048-token context, 1,489 is the largest uniform response
   budget allowed by the longest training prompt. If insufficient, qualify a
   larger context and the associated replay memory requirements.

2. **The present bottleneck in learning is missing reward contrast.** In the
   matched rollout window 73–80, normal had 123 all-fail groups out of 128;
   simplicial had 120 out of 128. Only two groups per arm contained multiple
   successes. GAR cannot create positive learning signal in all-fail groups, and
   its uncapped formula is the identity when there is only one passing trajectory.
   First obtain reliable successful completions. Then consider bounded collection
   of informative groups under a fixed policy version, while recording rejected
   groups and preserving a matched prompt schedule for the two arms. Dynamic
   filtering changes the effective task distribution and must be declared.

3. **Audit loss weighting before length growth.** Our current objective is the
   ordinary trajectory-level RLOO score-function estimator: sum response-token
   scores within each trajectory, then average trajectories. MiMo's final setup
   uses a group token-normalized surrogate followed by a prompt mean (Eq. 1 and
   Section 5.1). These are different weighting choices; our estimator is not
   invalid simply because it differs. Compare them in an explicit ablation.
   Also revisit the four-prefix replay sample budget as trajectories lengthen,
   because the time-subsampling weight and gradient variance change.

   From the stored complete trajectories in steps 73–80, negative token-weighted
   RLOO advantage mass was 1.20 times positive mass for normal and 1.31 times for
   simplicial. This calculation describes the full trajectory estimator, not the
   realized four-prefix gradient. Log entropy and both advantage masses before
   attributing length growth to that imbalance.

4. **Apply length pressure only after correctness is available.** Section 4.3.3
   penalizes only successful trajectories, relative to successful peers for the
   same prompt, and only above a group pass-rate threshold. This preserves room
   to explore difficult tasks. A bounded correctness-conditioned penalty is a
   better first math adaptation than code-specific quality rankings or rewards
   for merely emitting a box/EOS. Keep the raw correctness metric independent of
   any shaped reward and audit verifier alignment with each problem's requested
   answer format.

5. **Check router stability.** Section 5.4 shows expert-load collapse with trainable
   MiMo routers and stable loads with frozen routers. Our recipe disables router
   bias updates and auxiliary loss, but still trains router weights. A router-freeze
   pilot and load-CV/peak-load/cold-expert telemetry are warranted. Freezing routers
   changes the current all-parameters-trainable contract and must be applied and
   tested consistently across both comparison arms.

6. **Keep policy provenance exact.** MiMo's asynchronous partial rollouts preserve
   token-specific behavior probabilities, use importance correction, and rebuild
   KV caches after updates. Our synchronous on-policy RLOO does not implement that
   contract. Cache state remains local to one rollout and is discarded before
   optimization; partial trajectories must not silently continue across policy
   updates. Numerical agreement remains a prerequisite for any throughput claim.

## Current engineering gate

The first sixteen-rank tiny normal cache probe failed its padded-prefix
equivalence gate: maximum full-vocabulary log-probability difference was
**0.0820694 nats**, above **0.02**, and relative hidden RMS difference was
**0.0493285**. It stopped before the sampled-rollout gradient audit or any optimizer
update. This is stronger evidence than the earlier small single-GPU checks and
must be resolved before replacing the live sampler.

A follow-up isolated unpadded prefill as one source of drift. Preserving the
trainer's padded prefill eliminates the initial-prompt error, but a small local
decode test still exceeds the gate at 0.0349550 nats. This correction does not
establish distributed or production qualification.

Evidence is recorded under
`results/deepseek-v41-math-rl-cache-20260923/`: per-rank prefix receipts,
the normal probe logs, and the two `*-signal-audit.json` files. The existing v3
training processes continue to use their original source and recipe.

The next production run should combine qualified generation, an empirically
adequate completion budget, and a declared learning-signal strategy. Cache
acceleration alone is not evidence of improved RL sample efficiency.

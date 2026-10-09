# Limite RL health and lessons

**Evidence snapshot: 2026-10-07.** This page summarizes recorded experiments;
it does not report current node occupancy, queues, or completion estimates.
The detailed operational chronology is retained in the private archive.

## Capability outcomes

Read the [matched SFT comparison](Limite-SFT-Comparison.md) for lineage,
geometry, parameter differences, SFT data issues and the full evaluation contract.

| Continuation | AIME26 before | AIME26 after | Interpretation |
| --- | ---: | ---: | --- |
| Normal SFT10B → RL400 | 15.83% | 12.50% | No observed gain |
| Simplicial SFT10B → RL400 | 20.83% | 20.83% | No observed gain |
| Released Violetto → RL142 | 92.50% | 83.33% | Observed regression |

Each endpoint uses 30 problems × 4 seeded responses. The paired problem-bootstrap
95% interval for Violetto's change is [-18.33, -1.67] percentage points; for
simplicial minus normal at RL400 it is [0, 18.33]. This uncertainty covers
resampled evaluation problems, not variation across training seeds.

Finite gradients and valid checkpoints establish execution health, not improved
capability. In the last 100 steps of the historical paired RL runs, 84–86% of
response groups had no reward contrast.

## Distinguish three causes of poor completion

- Historical RL imposed a 16,384-token completion budget and a soft length
  penalty above 12,288 tokens. Of 111 correct released-Violetto benchmark
  responses, 69 exceeded 16K, 31 exceeded 32K and nine exceeded 64K.
- The full-context benchmark uses 131,072 minus actual prompt length. An audit
  of 959 responses found no hidden 16K/32K cutoff or budget/EOS inconsistency.
- Historical SFT dropped most document endings. High SFT native-context
  exhaustion frequently accompanies repetition; a longer limit alone does not
  repair that learned behavior.

The [native-context Violetto successor](../../recipes/limite/violetto_math_rl_native_context.yaml)
removes extra length/unfinished penalties and records each response budget.
This is a distinct protocol phase and has no claimed benchmark improvement in
this snapshot. The original direct-base pilot was different again: 8K response
cap, only one applied update in 15 steps, and 2/60 groups with reward contrast.

## Correctness and performance qualifications

| Change | Evidence and limit |
| --- | --- |
| Native grouped-query cached decoding | Whole-model likelihood/replay checks; graph cache reuse and finished-response compaction |
| Replay padding trim and flat-work skipping | Preserve loss normalization; zero-active distributed ranks remain synchronized |
| Actor/learner coordination | Drain retained actor work and use CPU rendezvous before learner GPU collectives; maximum policy lag one |
| Checkpoint staging | Preserve model, optimizer, scheduler and all-rank RNG; verify published payload hashes |
| Native full-context replay | Container cuDNN reduction with original full Q/K/V geometry, decoder checkpointing and chunked output head |
| 131,072-token admission | 31.05 s replay/update probe, 123.44 GiB peak allocated on B300 with resident actor/cache; not end-to-end RL step time |
| Distributed resume | Eight-rank fixture from step142; exact state reload and next-update equality; synthetic fixture never used as a production checkpoint |

FA4 and tiled local replay candidates did not meet the declared full-model
continuation tolerance despite close individual-layer agreement. They remain
experimental; tolerance was not relaxed to admit them. The final native path's
2053-token probe had maximum selected-log-probability error 3.8e-6 and gradient
relative L2 error 0.00561.

## Next experiments

1. Qualify a matched SFT repair retaining document endings; report overlap and
   short-document inclusion explicitly.
2. Measure training-only reward contrast and adjust curriculum before scaling
   rollout volume. Do not use benchmark answers to select training examples.
3. Batch distinct prompt groups and reduce whole-prefix repetition rescanning,
   preserving response RNG, policy lag, EOS, budgets and checkpoint semantics.
4. Validate benchmark rendering against the publisher's evaluation harness.
   Its current Base profile is few-shot; historical zero-shot Base results are
   not a reproduction of that protocol.

The publisher now provides an [evaluation harness](https://github.com/paradigma-inc/limite-violetto/tree/main/limite-evals).
Its [critic notes](https://huggingface.co/paradigma-inc/limite-1b-value-model/blob/main/MODEL_DETAILS.md)
describe SFT initialization, offline value fitting, online actor–critic training,
and informative mixed-outcome groups. They do not disclose a complete actor
recipe or establish equivalence to our GRPO/DAPO experiments.

Raw run evidence remains in team storage. Public aggregates are in
[recorded results](../recorded-results/limite-comparison-20261007.json).

## Matched restart contract, 2026-10-09

The [new matched protocol](../../recipes/limite/full_math_rl_native_context.yaml)
restarts normal and simplicial from their respective matched 10B SFT weights,
with fresh full-weight RL optimizers and identical data, seed, curriculum,
sampling and loss settings. Each variant uses eight B300 GPUs. The response
budget is the native 131,072-token context minus the actual prompt; an exhausted
budget is recorded explicitly and never relabeled as a natural EOS.

The root [`verl` submodule](https://github.com/XiaomiMiMo/verl/tree/a2ad9f6160b03ff2d47e59832bfb6b289f37c917)
pins XiaomiMiMo's code. The native Limite executor loads its repetition detector
directly and records the component hash. This is reuse of upstream components;
the complete verl Ray/FSDP execution stack is not qualified for these custom
adapters.

Admission separates exact graph/cache migration checks and independent FP32
attention oracles from BF16 differences between decoding and uncached replay.
The denominator is the actual sampled actor probability. Distribution-wide
importance-ratio clipping mass, finite gradients, fresh-Adam updates, full-context
resident memory, and distributed checkpoint/resume behavior are checked
separately. Full-context resource probes passed for both variants with about
184 GiB peak reserved memory per GPU; these disposable probes do not establish
production learning health or benchmark improvement.

For complete sampled trajectories, health uses clipping frequency, effective
sample size and mean importance weight, following the pinned upstream rollout
correction diagnostics. The recipe requires at most 5% clipped tokens, at least
95% effective sample fraction and mean weight between 0.5 and 2. Individual
ratio extrema remain reported: a single low-weight tail token is not an
importance-sampling correctness failure. Short-canary precision targets are
kept separate from these full-trajectory health criteria.

Both previous continuation queues were cancelled. New production admission
still requires the recorded math-rollout checks; live queue and learner status
are kept in private run receipts rather than inferred from this page.

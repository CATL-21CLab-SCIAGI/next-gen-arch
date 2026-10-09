# Limite: normal versus 2-simplicial attention

**Evidence snapshot: 2026-10-07.** Completed training and endpoint evaluation;
intermediate checkpoint evaluation is outside this snapshot. This page is not a
live queue or a claim that the architecture comparison is conclusive.

## Question and matched settings

Does an inserted 2-simplicial attention component improve mathematical reasoning
compared with an inserted ordinary attention component on the same pretrained model?

| Setting | Shared contract |
| --- | --- |
| Parent | `paradigma-inc/limite-1b-base`, revision `cc612bafcd4acd8445936201694c8c41aea8e479` |
| Placement | One zero-output adapter before each of 48 native blocks |
| Geometry | Model width 1280; 10 query heads; 2 KV heads; head dimension 128 |
| Attention schedule | Native local/global layout; runtime local span 1025; every fourth block global |
| Preserved mechanisms | QK RMSNorm, local RoPE/global NoPE, value embeddings, XSA, attention gates |
| Added simplicial axis | Short K/V window of 16 |
| Data | Text-only Nemotron Math-v2 with reasoning; shared tokenizer and global window order |
| Supervision | All-token next-token loss; document-contained 2048-token windows; no padding |
| Hardware | Two nodes / 16 NVIDIA B300 GPUs per variant |

| Variant | Parent parameters | Adapter parameters | Full trainable total after warmup |
| --- | ---: | ---: | ---: |
| Normal | 1,035,254,176 | 188,806,080 | 1,224,060,256 |
| 2-simplicial | 1,035,254,176 | 220,263,360 | 1,255,517,536 |

This is **geometry-matched, not parameter-matched**. Simplicial adds 31,457,280
parameters: 16.66% more adapter parameters, or about 2.57% more total parameters.
Shared adapter weights have matched initialization; zero output projections
preserve the parent logits initially.

## Training phases and authoritative recipes

1. Adapter-only warmup to the matched step-7630 checkpoints: **2,000,158,720**
   supervised targets, with the backbone frozen. These source checkpoints use
   the earlier native/Triton-CUDA path.
2. Explicit migration of both source checkpoints to TileLang, retaining adapter
   weights, AdamW state, RNG and data order. Unfreeze the full backbone and
   continue to **10B total** targets (approximately 8B additional targets).
3. Separate full-weight GRPO/DAPO comparison on Nemotron-RL-Math-v2, up to 400
   rollout-update steps. Rollout steps and applied optimizer updates differ.

Use the [full-finetuning recipe](../../recipes/limite/full_finetune_math.yaml)
for the actual matched continuation. The
[original adapter recipe](../../recipes/limite/native_adapters_math.yaml)
documents the original 10B adapter-only proposal, which was superseded at 2B.
The [later RL protocol](../../recipes/limite/full_math_rl_async.yaml)
overrides the earlier embedded 8K RL settings with a 16K completion budget and
asynchronous rollouts of at most one policy version of lag. The newer
[Violetto native-context recipe](../../recipes/limite/violetto_math_rl_native_context.yaml)
is a separate successor; it did not produce the historical comparison below.

The [fresh matched RL restart](../../recipes/limite/full_math_rl_native_context.yaml)
starts each variant from its own matched 10B SFT endpoint with a fresh optimizer,
eight GPUs per variant, and 131,072 total tokens minus each actual prompt.
It removes the historical extra length and unfinished-response penalties. This
is a new experiment; it does not replace the historical results below.

Normal and simplicial share the scientific geometry, but their qualified kernel
and graph optimizations differ. The full-finetuning recipe records these
performance contracts. Neither timing differences nor kernel-level admission
measurements should be presented as a pure FLOP-count comparison.

## Recorded results

The 10B endpoint's logged training CE is **0.809566 normal / 0.801842 simplicial**.
These are training-loss measurements, not held-out reasoning scores.

AIME26 uses all 30 problems, four fixed seeded responses per problem, temperature
0.6, top-p 0.95, a strict answer grader, and a total context of 131,072 tokens
including the actual prompt. Pass@1 below is the mean correctness across 120
responses; it is not pass@4. No artificial 16K or 32K evaluation cutoff is used.

| Model | Correct / 120 | Mean pass@1 | Reached native context limit |
| --- | ---: | ---: | ---: |
| Normal SFT10B | 19 | 15.83% | 74.17% |
| 2-simplicial SFT10B | 25 | 20.83% | 53.33% |
| Normal RL400 | 15 | 12.50% | 1.67% |
| 2-simplicial RL400 | 25 | 20.83% | 4.17% |
| Released Violetto reference | 111 | 92.50% | 3.33% |

The descriptive SFT advantage is five percentage points. A single training pair,
30 independent problems, unequal parameter counts, and the data issue below
prevent a strong causal claim about architecture. The later RL phases had
different update histories; neither improved its own SFT mean score. Lower
truncation alone did not establish better reasoning.

## Known data issue

The historical window index dropped incomplete document tails. In the inspected
sample, only **104 / 83,403** retained long documents kept their terminal EOS.
This plausibly contributes to repetitive generation and poor termination, but
causality has not been established by a matched repair experiment.

The opt-in `overlap-final-window` policy in `archlab.automodel.limite_adapter_data`
preserves long-document endings with explicit overlap accounting and a new data
contract. It does not change historical weights or automatically repair earlier
results. Short documents still require a separately qualified packing policy.

## Evidence and reproduction limits

Source modules, portable recipes and tests are included in this repository.
A compact endpoint table is in
[recorded results](../recorded-results/limite-comparison-20261007.json).
Full checkpoint payloads, raw response text, data and node receipts remain in
team storage and are not distributed here. The prepared-data and model paths
must be supplied through the recipe's environment bindings.

Historical evidence refers to pre-publication source revision `dfb0ab4` and older
commits. Their identities are preserved in a private archive, not as ancestors
of the public source snapshot. The source snapshot is not an independently
rerun experiment. See [publication policy](../PUBLICATION.md).

Related: [RL health and lessons](Limite-RL-Health-and-Performance.md),
[evaluation methodology](Evaluation.md), [results index](Results.md).

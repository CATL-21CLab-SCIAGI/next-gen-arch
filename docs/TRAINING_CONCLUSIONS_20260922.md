# Training conclusions — 2026-09-22

**Type:** dated evidence audit. **Scope:** existing Qwen/DeepSeek records and earlier campaigns.
Models were not re-evaluated and not every weight payload was rehashed.

**Conclusion:** completed controlled comparisons do not establish a simplicial-specific capability advantage. Normal attention wins the final matched DeepSeek math-loss comparison and the original scratch comparison.

## Matched outcomes

CE is in nats/token; lower is better.

| Experiment | Matched budget | Result | Limit |
| --- | --- | --- | --- |
| Qwen W320 scratch A/B/C | 3.003B tokens each | Global 3.663525; local 3.641465; simplicial 3.649627 | One seed; 131,072-token validator |
| Frozen-pretrained Qwen | Last validation at step 8000 | CE 1.936017 → 1.862897 | No trained ordinary-adapter control |
| DeepSeek frozen adapters | 50.120M tokens | Normal 0.798750; simplicial 0.800121 | Normal wins all five shared nonzero validations |
| DeepSeek early full continuation | 91.035M total tokens | Normal 0.738411; simplicial 0.737130 | Early result later reversed |
| DeepSeek final full continuation | 756.365M total tokens | Normal 0.700900; simplicial 0.701895 | One training pair |
| Original DeepSeek scratch | 240.875M tokens | Normal 4.122233; simplicial 4.160268 | Only 2.41% of planned 10B |

## DeepSeek full fine-tuning

| Item | Evidence |
| --- | --- |
| Final common checkpoint | 4537/4537; 756,364,650 supervised tokens each |
| Phases | 307 frozen-backbone adapter updates + 4,230 full-weight updates |
| Final math set | 1M targets; 200 windows from 162 problems |
| Simplicial − normal CE | +0.000994857; problem-cluster bootstrap 95% CI [+0.000745441, +0.001269267] |
| Normal throughput advantage | 24.89% in the declared frozen-adapter window; 16.27% over retrospective full-weight steps 328–4537 |
| Full-weight mean update time | 94.412 s normal; 109.774 s simplicial; excludes evaluation/checkpoint pauses |
| Adapter parameters | 146.84M normal; 167.82M simplicial |

The interval measures evaluation-sample uncertainty, not training-seed variation. This is an equal-token comparison, with different adapter parameter counts and qualified core implementations.

### Capability subsets

| Task | Cases | Normal correct | Simplicial correct | Paired exact p |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 1,140 | 995 | 996 | 1.0 |
| ARC-Challenge, raw | 128 | 77 | 78 | 1.0 |
| PIQA, raw | 256 | 219 | 211 | 0.0386 |

Raw PIQA favors normal nominally but not after three-task Bonferroni correction. Normalized PIQA is 222 versus 219, p=0.5078. These are continuation-likelihood subsets, not full benchmarks or generated math accuracy.

The 64K periodic validator favored simplicial at 22/27 shared points; correlated subset observations do not override the final larger evaluation. Neither matched arm reached its original 1B budget. Simplicial alone later stopped at 4599 / 766.749M tokens. Unmatched 4537/3554 and 4537/3620 evaluations are superseded.

## Scratch and Qwen limitations

| Lineage | Key limitation |
| --- | --- |
| Original DeepSeek scratch | Normal is 44.72% faster over final matched steps 4248–4347. MMLU selects the first option on 1,132/1,140 normal and 1,125/1,140 simplicial cases; apparent accuracy is largely label preference. |
| Scratch model size | Width 640 still yields ~29.06B parameters, including 24.58B Engram parameters. Native attention remains in all 20 layers; eight additive branches differ. |
| Qwen simplicial-only continuation | Stops at 30.342B/100B tokens; last validation is step 3576, not stopping step 3617. No matching long-run ordinary control. |
| Pretrained Qwen | Stops at 8089 / 4.241B targets; last validation at 8000. At step 100, zero-init CE 1.881850 versus tested nonzero-init 3.724166. |
| Qwen capability pilot | MMLU 53/57 → 52/57; normalized ARC 19/32 → 21/32; GSM8K 7/8 unchanged; capped AIME 0/4 unchanged. Differences are inconclusive. |

## Corrected and active studies at the audit snapshot

Snapshot: **2026-09-22 07:45:33 UTC**.

| Pair | Progress at snapshot | Conclusion |
| --- | --- | --- |
| Corrected linear RF / LinSimp | 31.565M / 30.572M tokens | Initial validation only |
| Faster TileLang normal / simplicial | 107.509M / 68.662M tokens | No matched post-training held-out result |

The original RF sampler drew chi_1 instead of chi_16 radii and omitted QR sign correction. Its 402.827M/374.290M-token runs are excluded from conclusions about the intended operator. Corrected runs start fresh.

TileLang's qualified forward and ~0.15–0.26% gradient errors support an engineering result. They do not establish a quality gain.

## Decisions

1. Keep ordinary attention as the reference for the completed DeepSeek setups.
2. Judge corrected RF/LinSimp on shared held-out checkpoints from the corrected lineage.
3. Report whole-model time separately from kernel speed.
4. Resolve first-option bias before interpreting scratch MMLU differences.
5. Require independent seeds and matched-token/compute analyses for general architecture claims.

## Evidence locations

Detailed artifacts are local/team-storage paths:

- `results/training-conclusions-audit-20260922/SOURCE_INDEX.json`: 27 primary artifact identities.
- `results/training-conclusions-audit-20260922/ACTIVE_PAIRS.json`: exact dated operational snapshot.
- `results/deepseek-v41-full-comparison-20260914/evaluation-windows/normal-004537-simplicial-004537-6da75dbb/`: final math and MC results.
- `results/deepseek-v41-scratch-w640-d20-20260915/eval-matched-240m-v2/`: original scratch comparison.
- `results/pretrained-capability-dlc-step4000-20260908-v2/summary.json`: completed Qwen capability pilot.

[Earlier campaigns](RESULTS.md) · [RF/TileLang correction](DEEPSEEK_V41_DEBUG_20260922.md)

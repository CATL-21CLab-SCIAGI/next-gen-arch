# Training conclusions audit — 2026-09-22

The completed controlled comparisons do not establish a simplicial-specific
capability advantage. Normal attention has a small final math-loss advantage
in the matched DeepSeek full-finetuning comparison and a larger loss/speed
advantage in the original DeepSeek scratch comparison. The Qwen scratch pilot
places simplicial between global and local ordinary attention. The corrected
linearized comparison is too early to judge. TileLang provides an implementation
speed improvement with qualified numerics, not a demonstrated quality gain.

This audit covers the available DeepSeek and Qwen training lineages, the older
architecture sweeps, and backend campaigns. Independent reads/reaggregations
checked matched data cursors, evaluation aggregates, benchmark predictions,
continuation records and run status. Models were not re-evaluated for this audit,
and every checkpoint payload was not rehashed. Training processes were unchanged.
The active-run snapshot is **2026-09-22 07:45:33 UTC**. Its exact values are in
[ACTIVE_PAIRS.json](../results/training-conclusions-audit-20260922/ACTIVE_PAIRS.json);
[SOURCE_INDEX.json](../results/training-conclusions-audit-20260922/SOURCE_INDEX.json)
records the paths and hashes of 27 primary artifacts.

## What the completed simplicial experiments support

CE is negative log-likelihood in nats per token; lower is better. Absolute CE
across different models, tokenizers or datasets is not a ranking.

| Experiment | Matched result | Supported conclusion |
| --- | --- | --- |
| Qwen W320 scratch pilot, 3.003B tokens per arm | Global CE 3.663525; local 3.641465; simplicial 3.649627 | Simplicial improves on global, but the local ordinary control wins. |
| Pretrained Qwen, frozen backbone + simplicial branches | CE 1.936017 at step 0 to 1.862897 at step 8000 | Adapter learning helps this dataset; no trained ordinary-adapter control isolates a simplicial contribution. |
| DeepSeek frozen-backbone adapter phase, 50.120M tokens | Normal 0.798750; simplicial 0.800121 | Normal wins all five shared nonzero validation checkpoints. |
| DeepSeek early full-weight continuation, 91.035M total tokens | Normal 0.738411; simplicial 0.737130 | Small early simplicial advantage, subsequently reversed. |
| DeepSeek final matched full-weight continuation, 756.365M total tokens | Normal 0.700900; simplicial 0.701895 | Small final advantage for normal, with higher normal throughput. |
| Original DeepSeek scratch pair, 240.875M tokens | Normal 4.122233; simplicial 4.160268 | Normal wins held-out prediction and update throughput in this early-stopped pair. |

### DeepSeek pretrained comparison

The final common checkpoint is **4537/4537**, with exactly **756,364,650**
supervised tokens each. These are 307 frozen-backbone adapter updates followed
by 4,230 full-weight updates. Calling the final experiment merely adapter-only
finetuning would be wrong. Both arms inherited their own step-307 adapter
weights, then unfroze the common pretrained text backbone.

The final 1M-target math evaluation comprises 200 windows from 162 problems.
For **simplicial minus normal**, the CE difference is **+0.000994857** with a
problem-cluster bootstrap 95% interval of **[+0.000745441, +0.001269267]**.
That is a small within-evaluation advantage for normal; the interval is not
uncertainty across independently trained seeds. There is only one training pair.

The 91M evaluation used the same math windows and MC cases and favored simplicial
by 0.001281119 nats/token. That earlier result should not be reported as the final
outcome. The 64K periodic validator also favors simplicial at 22/27 shared points,
but these are correlated observations on a smaller subset. It does not override
the final, larger matched evaluation.

Final multiple-choice counts, scored by continuation likelihood:

| Task | Cases | Normal correct | Simplicial correct | Paired exact p |
| --- | ---: | ---: | ---: | ---: |
| MMLU | 1,140 | 995 (87.28%) | 996 (87.37%) | 1.0 |
| ARC-Challenge, raw | 128 | 77 (60.16%) | 78 (60.94%) | 1.0 |
| PIQA, raw | 256 | 219 (85.55%) | 211 (82.42%) | 0.0386 |

No task establishes a simplicial win. PIQA's nominal result favors normal, but
does not survive even a three-task Bonferroni correction; character-normalized
PIQA is 222 versus 219 correct, p=0.5078. These are benchmark subsets, not full
benchmark runs or generated math solution accuracy. Exact-hash contamination
checks do not rule out paraphrases or contamination in backbone pretraining.

Recorded update-time throughput favors normal by **24.89%** in the predefined
frozen-adapter window, and **16.27%** over retrospective full-weight steps
328–4537. The latter averages 94.412 versus 109.774 seconds/update and excludes
checkpoint/evaluation downtime. This is an equal-token comparison, not an
equal-compute comparison. The adapters are not parameter matched (146.84M normal
versus 167.82M simplicial), and their core precision/backend implementations differ.

The matched comparison completed at 756M tokens; neither trajectory reached the
original 1B recipe budget. Simplicial alone later stopped at step 4599 / 766.749M
tokens. The 4537/3554 and 4537/3620 evaluations are superseded unmatched comparisons.
Older running-status summaries and failed inference-qualification attempts are
not additional scientific results.

Evidence: [final matched evaluation](../results/deepseek-v41-full-comparison-20260914/evaluation-windows/normal-004537-simplicial-004537-6da75dbb/HELDOUT_MATH.json),
[final MC](../results/deepseek-v41-full-comparison-20260914/evaluation-windows/normal-004537-simplicial-004537-6da75dbb/MULTIPLE_CHOICE.json),
[early evaluation](../results/deepseek-v41-eval91-20260915/results-v1/HELDOUT_MATH.json),
[phase starts](../results/deepseek-v41-next-phase-20260914/MATCHED_50M_CHECKPOINTS.json),
[frozen-phase throughput](../results/deepseek-v41-normal-control-20260914/THROUGHPUT_COMPARISON.json).

### Original DeepSeek scratch comparison

Both final checkpoints are step **4347 / 240,875,236** supervised tokens: only
**2.41% of the planned 10B**. Normal wins all three matched nonzero held-out
measurements. Final simplicial perplexity is 64.0887 versus 61.6969, **3.88% worse**.
The 1M-target evaluation has 1,130 windows; 814 favor normal on CE. Windows sharing
documents are not independent training replications.

Over final matched steps 4248–4347, normal delivers 8,346.32 versus 5,767.34
supervised targets per summed update-wall-second, **44.72% higher throughput**.
Median update wall times are 6.9102 and 9.9454 seconds. Pauses outside updates
are excluded. Data, LR and token cursors match in all 4,327 accepted v5 rows.

All MC differences are inconclusive. More importantly, MMLU predictions select
the first option on **1,132/1,140 normal** and **1,125/1,140 simplicial** cases.
The apparent 236-versus-237 correct difference mostly reflects answer-label
preference, not useful evidence of knowledge. High prediction agreement here
must not be interpreted as strong shared competence. Contamination overlap
for the scratch benchmark was not assessed.

All weights were randomly initialized and trainable. Native attention remains
in all 20 layers; only eight additive branches differ. Width 640 is not a claim
of a sub-billion-parameter model: the contract reports roughly 29.06B parameters,
including 24.58B Engram parameters. One seed and one shared optimizer recipe do
not establish which architecture would win after convergence or separate tuning.

The accepted lineage includes v3 steps 1–10, v4 steps 11–20 and v5 thereafter.
Failed qualification/replay attempts and the Unicode-failed first evaluation
are not independent replications. The original job was released on September 16;
startup reports saying it is still training toward 10B are stale.

Evidence: [comparison](../results/deepseek-v41-scratch-w640-d20-20260915/eval-matched-240m-v2/COMPARISON.json),
[raw normal predictions](../results/deepseek-v41-scratch-w640-d20-20260915/eval-matched-240m-v2/normal/multiple-choice.jsonl),
[raw simplicial predictions](../results/deepseek-v41-scratch-w640-d20-20260915/eval-matched-240m-v2/simplicial/multiple-choice.jsonl),
[model review](../results/deepseek-v41-scratch-w640-d20-20260915/matched-checkpoints-240m/REVIEW.md),
[release receipt](../results/deepseek-v41-scratch-w640-d20-20260915/RELEASE.json).

### Qwen simplicial lineages

The matched A/B/C pilot completed 358 updates and 3,003,121,664 tokens each.
Six of 48 layers differ; 36 GDN layers and six other global-attention layers
remain shared. Simplicial beats global CE by 0.013897, but loses to local by
0.008163. The 131,072-token validator and single seed limit generalization.
The approximately 225–227K token/s medians do not show a substantial speed win.
The later simplicial-only production run stopped at 30.342B of its planned 100B
tokens. Its last validation CE is 2.762794 at step 3576, not its stopping step
3617. It has no matching long-duration ordinary control, and production data
ordering differs from the pilot.

The frozen-pretrained Qwen branch experiment stopped at step **8089 / 4.241B**
targets before finishing its requested data pass. Its last validation was at
step 8000 on a fixed 2,097,152-target set. The zero-output initialization reached
CE 1.881850 at step 100 versus 3.724166 for normal-std-0.02 output initialization.
This supports zero initialization under that particular recovery schedule,
not a universal claim about all nonzero initializations.

The completed step-4000 capability pilot is tiny and mixed: MMLU 53/57 to 52/57,
normalized ARC 19/32 to 21/32, GSM8K 7/8 unchanged, and capped AIME 0/4 unchanged.
Its differences are not statistically clear; it proves neither a capability
gain nor comprehensive capability preservation. The incomplete DSW pilot must
not be merged with the completed DLC pilot, which uses a different runtime.

Evidence: [A/B/C pilot](../results/simplicial-dlc-pilots-20260906-5072d1d/),
[pretrained trajectory](../results/pretrained-simplicial-fineweb-20260907-v1/metrics.jsonl),
[capability pilot](../results/pretrained-capability-dlc-step4000-20260908-v2/summary.json).

## Active comparisons and the invalid RF run

Both active families retain native DeepSeek attention in all 20 layers and
compare additive adapters at zero-based indices **2,4,7,9,12,14,17,19**. All
weights are trainable from scratch. They are not fully linear-attention or
fully simplicial-attention backbones.

| Pair at snapshot | Training progress | Evaluation conclusion |
| --- | --- | --- |
| Corrected linear RF / LinSimp | 31.565M / 30.572M tokens, about 0.31% of 10B | Only initial validation exists; no trained held-out comparison yet. |
| Faster TileLang normal / simplicial | 107.509M / 68.662M tokens, 1.08% / 0.69% of 10B | Normal has a 100M validation; simplicial has not reached it. No matched post-training held-out result yet. |

Across all shared steps, both pairs have identical data cursors, target counts
and learning rates, finite observed losses/gradient norms, fresh logs and no
failure markers. Successful qualification establishes execution correctness
within its tests; it does not establish a quality advantage.

For transparency, the last 100 shared training updates give:

| Pair; shared steps | First-arm weighted train CE | Second-arm weighted train CE |
| --- | ---: | ---: |
| Linear RF / LinSimp; 442–541 | 5.879206 | 5.873033 |
| TileLang normal / simplicial; 1134–1233 | 5.105444 | 5.119275 |

These small opposite-direction gaps are retrospective training diagnostics,
not held-out evidence or independent replications. Do not compare their latest
wall-clock losses at different token budgets.

The original linear RF/LinSimp runs are excluded from conclusions about the
intended operator. Both sampled chi_1 rather than chi_16 row radii and omitted
QR sign correction. Their 402.827M/374.290M-token checkpoints are preserved, but
the corrected sampler requires the fresh pair rather than continuing those
curves. See the [debug audit](DEEPSEEK_V41_DEBUG_20260922.md).

TileLang's native-forward equality and approximately 0.15–0.26% gradient errors
against FP32 support the new backend's numerical admission. Key/value and sink
atomics preclude exact gradient replay. Observed full-update improvement is
roughly **1.2x**, while the earlier approximately **14x** gain described only an
isolated sparse-attention kernel. The new pair is not an equal-compute quality
test of backend speed.

## Earlier architecture, scaling and backend campaigns

The historical 100M/300M tables have three valid seeds for the following rows:

| Mechanism | 100M BPB delta; speed ratio | 300M BPB delta; speed ratio |
| --- | --- | --- |
| Qwen GDN | -0.013177; 0.417x | -0.008303; 0.531x |
| Engram | -0.007582; 0.948x | -0.004072; 0.973x |
| Kimi K3 KDA | -0.006573; 0.397x | -0.005633; 0.511x |
| Gated attention | -0.002151; 0.964x | -0.001217; 0.972x |

GDN has the best recorded BPB in these completed groups at substantial throughput
cost. Engram offers a strong observed quality/throughput tradeoff; gated attention
offers smaller inexpensive gains. These are **12-tokens-per-actual-parameter**
experiments, not equal-token comparisons. At nominal 100M, baseline sees 1.269B
tokens, Engram 1.424B (+12.2%), and GDN 1.327B (+4.6%). At 300M, GDN receives 7.0%
more tokens than baseline. These rankings describe the recipe, not isolated
equal-compute architecture effects. Source: [metrics](../src/archlab/data/key-metrics.csv)
and [budget manifest](../src/archlab/data/parameter-scale-100m-1b-v1-manifest.json).

mHC leads the separate fixed-token d16/d18 controls but has only two valid seeds
at 100M and fails all three at 300M. Relative attention degrades badly at 300M.
The negative DSA result uses a dense masked SDPA implementation and is not a
verdict on production sparse attention. The 1B leaderboard remains an explicitly
incomplete August 24 snapshot; its six completed arms cannot identify a final winner.

The 0.95B Qwen quartered run provides actual downstream learning evidence: PIQA
normalized accuracy increases from 61.81% at 10B to 65.07% at 100B tokens (peak
65.23% at 70B). It is not simplicial evidence. The 27.32B-versus-0.95B comparison
favors the larger model at matched early tokens but the smaller model at matched
wall time; different architecture details and a 16.21x step-time ratio prevent
treating it as a pure parameter-scaling experiment.
See [PIQA curve](recorded-results/qwen38-quarter-piqa-curve-20260904.json) and
[early size comparison](recorded-results/qwen38-quarter-vs-full-early-20260904.json).

Backend studies support engineering conclusions: safe-autotune Megatron and
speedrun paired architecture deltas correlate 0.971361, with matching directions
for 13/15 variants, but are not numerically interchangeable. The 100M multi-node
reproduction passes its curve gate; Megatron is 1.018x faster by steady-step
throughput but 0.892x by transition-inclusive aggregate. Native fusion gives
1.274x throughput in its own small-model contract; PP/TP/CP mostly trade speed
for memory there. The 37-update, 1M-model screen establishes compatibility, not
mature-scale superiority. See [backend evidence](BACKEND_COMPARISON.md).

## Decisions justified by this evidence

1. Keep normal attention as the current reference for the completed DeepSeek
   setups: the heavier variant has not earned its extra training cost there.
2. Treat the corrected RF comparison as a new experiment. Wait for shared
   post-training held-out checkpoints before judging LinSimp.
3. Treat TileLang as a validated engineering improvement, and report whole-model
   time separately from kernel time and quality.
4. Use the existing first-option bias as a reason to improve scratch capability
   evaluation before interpreting small MMLU differences.
5. Any general architecture claim still needs independent training seeds and
   explicit matched-token and matched-compute analyses. Existing evaluation
   bootstrap intervals do not substitute for those replications.

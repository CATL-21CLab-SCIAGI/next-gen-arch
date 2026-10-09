# Results

**Historical campaign review:** 2026-09-22; Limite endpoint review: 2026-10-07. Values below are historical results, not live run status.

## Main findings

| Experiment | Result | Interpretation |
| --- | --- | --- |
| Original 100M/300M sweeps | GDN had strong BPB; Engram offered a favorable quality/throughput tradeoff | Scaling recipes gave larger variants more tokens; this is not a pure equal-token effect |
| 10M backend comparison | Architecture-delta correlation 0.971361; 13/15 directions agree | Broad signal transfers, but the backends differ numerically |
| Qwen W320, 3.003B tokens | CE: global 3.663525; local 3.641465; simplicial 3.649627 | Ordinary local attention won this pilot |
| Frozen-pretrained Qwen | CE 1.936017 at initialization to 1.862897 at step 8000 | Learning occurred; no trained ordinary-adapter control isolates the mechanism |
| DeepSeek final matched full fine-tuning | CE: normal 0.700900; simplicial 0.701895 | Small normal advantage; normal was also faster |
| Original DeepSeek scratch pair | CE: normal 4.122233; simplicial 4.160268 at 240.875M tokens | Early-stopped comparison, not a converged 10B result |

## Limite endpoint comparison

At 10B total supervised targets, AIME26 mean pass@1 is 15.83% normal and 20.83% 2-simplicial (30 problems, four draws each, native full context). Subsequent historical RL gives 12.50% and 20.83%. These results do not isolate architecture: parameter counts differ, only one training pair is available, and historical SFT omitted most document endings. See the [complete SFT comparison](Limite-SFT-Comparison.md).

## Interpretation limits

- Completed comparisons do not establish a simplicial-specific capability advantage.
- The DeepSeek full comparison has one training pair. Its evaluation bootstrap does not measure variation across training seeds.
- Small capability subsets are regression screens, not proof of general equivalence.
- Corrected random-feature runs are separate experiments; the invalid sampler's curves remain excluded.
- Compare only compatible metrics, datasets, budgets and evaluation protocols.

## Evidence map

The repository's `docs/RESULTS.md` indexes the frozen campaigns.
`docs/TRAINING_CONCLUSIONS_20260922.md` audits Qwen/DeepSeek lineages and source artifacts.
Compact evidence is versioned under `docs/recorded-results/`; detailed runs may require team storage.

**Next:** [Qwen Experiments](Qwen-Experiments.md) · [DeepSeek Experiments](DeepSeek-Experiments.md)

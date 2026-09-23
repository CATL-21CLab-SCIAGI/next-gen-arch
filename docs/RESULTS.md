# Results index

**Type:** historical evidence index. **Reader summary:** [Wiki Results](wiki/Results.md).
The Qwen/DeepSeek audit is dated **2026-09-22**; the original scaling tables are frozen at **2026-08-24**.

## Find the evidence

| Study | Record | Scope |
| --- | --- | --- |
| Original component/scaling campaigns | [Key metrics](recorded-results/key-metrics.csv), [manifest](recorded-results/parameter-scale-100m-1b-v1-manifest.json) | Three sizes, 16 variants, three planned seeds |
| Matched 10M backend study | [Backend comparison](BACKEND_COMPARISON.md) | Speedrun versus Megatron |
| 100M multi-node baseline | [Recorded comparison](recorded-results/100m-multinode-b300/comparison.md) | 15-rank reproduction |
| Native parallelism | [Recorded comparison](recorded-results/100m-native-parallelism-b300-1n/README.md) | DP/TP/PP/CP/EP capacity and speed |
| FineWeb 1M screen | [Recorded comparison](recorded-results/fineweb10b-1m-b300-dsw/comparison.md) | Short compatibility screen |
| Qwen/DeepSeek lineages | [Conclusions audit](TRAINING_CONCLUSIONS_20260922.md) | Matched outcomes, invalidated runs, limitations |

## Representative 100M/300M results

Lower BPB is better. Speed is relative to the corresponding baseline.

| Variant | 100M BPB | Speed | 300M BPB | Speed |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.916171 | 1.000× | 0.808017 | 1.000× |
| Qwen GDN | 0.902994 | 0.417× | 0.799714 | 0.531× |
| Engram | 0.908589 | 0.948× | 0.803945 | 0.973× |
| Kimi K3 KDA | 0.909599 | 0.397× | 0.802384 | 0.511× |
| KDA | 0.910074 | 0.384× | 0.804987 | 0.500× |

These are tokens-per-parameter recipe outcomes. Larger variants receive more tokens; the nominal 100M Engram arm receives 12.2% more than its baseline. The full tables retain all variants, seeds and failures. The 100M mHC result is partial; all three 300M mHC runs failed.

## Backend and systems findings

| Observation | Result |
| --- | --- |
| 10M paired architecture deltas | Correlation 0.971361; matching direction for 13/15 variants |
| 100M multi-node reproduction | Speedrun passes the registered curve gate |
| Megatron versus speedrun at 100M | 1.018× steady-step throughput; 0.892× transition-inclusive aggregate |
| Native fused attention at 100M | 1.274× matched unfused throughput |
| Small-model PP/TP/CP | Lower memory, lower throughput in the tested configurations |

## Interpretation

- Compare absolute BPB/CE only within compatible data, tokenizer and budget contracts.
- Equal nominal model size does not imply equal parameters, tokens or FLOPs.
- Report cold-start and steady-state timings separately.
- Keep numerical failures and unfinished budgets visible.
- Supervised loss, task accuracy and generation efficiency are distinct outcomes.

Large artifacts referenced by reports require the declared team-storage namespace. Compact historical evidence is versioned in [recorded-results](recorded-results/README.md).

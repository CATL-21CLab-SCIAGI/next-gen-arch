# 100M-class single-node native parallelism

One B300 node ran eight sequential seed-42 Megatron jobs. DP5 and PP5 used the
same five-GPU allowlist; the factor-two TP, CP, and EP comparisons used the same
four-GPU allowlist because the six attention heads and six experts do not admit a
factor-five shard without changing the model. No topology ran concurrently with
another.

## Contract

- Dense model: 10 layers, width 384, 6 heads, FFN width 1,536, 122.6M actual
  parameters including the 128,896-entry padded vocabulary.
- MoE model: 6 experts, top-2 routing, expert FFN width 768, 158.0M total
  parameters.
- Sequence length 2,048; BF16; seed 42; global batch 180; 200 steps; 73,728,000
  training tokens per run.
- Data: the indexed FineWeb-Edu control prepared for the native Megatron harness;
  tokenizer: the matching DeepSeek-V3 tokenizer artifact.
- Stack: CUDA 13, PyTorch 2.11.0+cu130, NCCL 2.28.9, cuDNN 9.19,
  Transformer Engine 2.11, and Megatron-LM commit
  `55ac7082517c3878ae653c07c09c534b8aed49f6`.
- Throughput is `368,640 tokens / median step time` over steps 11–200. Peak memory
  is the maximum allocated value reported by any rank. Every run completed all 200
  steps with return code 0 and zero skipped or non-finite iterations.

This is a systems comparison, not a continuation of the ClimbMix/nanochat BPB
campaign. Its quality values are validation language-model loss and must not be
merged with the published BPB tables.

## Results

| Run | Attention | Median step | Steady tok/s | Relative to matched DP/EP1 | Peak allocated | Final validation-set loss | Lifecycle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DP5 | unfused deterministic | 2,045.75 ms | 180,198 | 1.000× | 5,512 MiB | 5.954379 | 479.4 s |
| PP5 | unfused deterministic | 3,314.05 ms | 111,235 | 0.617× | 3,053 MiB | 5.971462 | 761.8 s |
| DP4 | unfused deterministic | 2,510.30 ms | 146,851 | 1.000× | 5,512 MiB | 5.973920 | 580.0 s |
| TP2 + DP2 | unfused deterministic | 6,453.15 ms | 57,126 | 0.389× | 2,835 MiB | 5.939896 | 1,417.9 s |
| DP4 | TE fused nondeterministic | 1,970.25 ms | 187,103 | 1.000× | 4,028 MiB | 5.978323 | 465.7 s |
| CP2 + DP2 | TE fused nondeterministic | 5,386.85 ms | 68,433 | 0.366× | 3,114 MiB | 5.984146 | 1,210.1 s |
| MoE EP1 + DP4 | unfused deterministic | 5,316.55 ms | 69,338 | 1.000× | 6,292 MiB | 5.930279 | 1,190.4 s |
| MoE EP2 + DP2 | unfused deterministic | 5,424.35 ms | 67,960 | 0.980× | 5,843 MiB | 5.915831 | 1,252.9 s |

## Interpretation

- Five-way data parallelism scales cleanly from DP4: its per-GPU-normalized
  efficiency is 98.2%.
- PP5, TP2, and CP2 are operationally correct but lose throughput at this small,
  shallow scale. They reduce peak rank memory by 44.6%, 48.6%, and 22.7%,
  respectively, so they remain capacity tools for deeper, wider, or longer-context
  models rather than 100M defaults.
- Transformer Engine fused attention is the useful performance result: fused DP4 is
  1.274× the throughput of unfused DP4 and has a final validation loss only 0.0044
  higher in this single run. It requires the explicitly separate nondeterministic
  contract on B300.
- EP2 does not accelerate this six-expert model. It is 2.0% slower in steady state
  and 5.3% slower lifecycle-inclusive, while reducing peak allocated memory by 7.1%.
- Quality differences across PP, TP, CP, and EP paths are non-causal one-seed
  implementation-path observations. Pipeline-rank seed offsets, collective reduction
  order, and the nondeterministic fused kernel prevent bitwise initialization or
  trajectory equivalence. Use the curves to reject gross numerical failure, not to
  rank parallel topologies by model quality.

Machine-readable run statistics are in [summary.csv](summary.csv), validation points
are in [learning_curve.csv](learning_curve.csv), and [comparison.json](comparison.json)
records the frozen public contract and ratios.

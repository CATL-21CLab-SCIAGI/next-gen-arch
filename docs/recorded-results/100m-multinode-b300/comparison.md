# 100M three-node backend comparison

One `3 nodes × 5 B300 GPUs = 15 ranks` process group trained the 105,775,510-parameter
baseline once per backend with seed 42. Runs were sequential on the same physical GPU
allowlist. NVLink handled intra-node traffic; NCCL logs confirmed RoCE/GDRDMA for
inter-node collectives.

## Outcome

| Run | Final BPB | Δ vs current speedrun | Training tokens |
| --- | ---: | ---: | ---: |
| Historical single-GPU speedrun | 0.91636648 | +0.00021621 | 1,269,301,248 |
| Current 15-rank speedrun | **0.91615027** | — | 1,269,301,248 |
| Current 15-rank Megatron | 0.91756084 | +0.00141057 | 1,269,301,248 |

The current speedrun reproduces the historical curve: 14 shared points have Pearson
`0.99999787`, mean absolute error `0.00069146 BPB`, and final error
`-0.00021621 BPB`. It passes the pre-registered reproduction gate.

Megatron and the current speedrun also track closely across the 13 shared trained
checkpoints: Pearson `0.99983861`, mean absolute gap `0.00122598 BPB`, and maximum gap
`0.00237870 BPB`. Megatron is slightly better through much of warmdown, then finishes
`0.00141057 BPB` worse. With only one seed, this is implementation-path variance, not
evidence of a backend quality hierarchy.

## Exact non-divisible batch

The historical batch contains 192 sequences, which is not divisible by 15. The
comparison does not round it to 195. Every rank reconstructs the canonical historical
192-row packing stream, then consumes a contiguous slice. Ranks 0–11 train on 13 rows;
ranks 12–14 train on 12 rows plus one static-shape padding row whose targets are `-1`.
Those three rows contribute no loss, gradient normalization, BPB, or token count.

## Throughput

| Basis | speedrun | Megatron | Megatron / speedrun |
| --- | ---: | ---: | ---: |
| Sampled median steady step | 1,906,064 tok/s | 1,940,137 tok/s | 1.018× |
| Backend-native post-warm aggregate | 1,887,555 tok/s | 1,682,942 tok/s | 0.892× |

The median says Megatron's warmed training step is essentially tied and slightly
faster. The transition-inclusive aggregate says its lifecycle/tail overhead is still
material; its complete process took 782.0 seconds. The aggregate implementations do
not include exactly the same evaluation overhead, so they should not be presented as a
kernel-only benchmark.

## Scope

This validates the MCore wrapper at 100M under one real 15-way, three-node data-parallel
job with `TP=PP=CP=1`. It does not establish native tensor/pipeline/context parallelism
for all custom mechanisms. Full point values are in [learning_curve.csv](learning_curve.csv),
machine-readable statistics are in [comparison.json](comparison.json), and the frozen
public campaign record is in [campaign.json](campaign.json).

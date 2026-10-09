# DeepSeek V4.1 scratch width study

Current contract, revised 2026-09-29: the [portable recipe](../recipes/deepseek_v41/scratch_scaling.yaml)
compares ordinary local attention and exact windowed 2-simplicial attention at
widths 128, 384, 640, and 1280. Each base run has a 10B supervised-token budget.
Both variants retain the native V4.1 sparse-attention backbone and differ in
an additive branch at layers 2, 4, 7, 9, 12, 14, 17, and 19. This is an adapter
comparison, not a literal replication of replacing every fourth layer in Roy et al.

The live order is d128, then the previously checkpointed d640, then d384 and d1280.
The normal arm inserts the confirmed 222-cell [repeated-data sweep](LOOP_SCALING_RESEARCH.md)
immediately after d128. Its first cell is d128/L20/K2. Each cell repeats 100M
unique targets ten times (1B consumed within its own 10B allowance). Each arm
owns two nodes/16 B300 GPUs and advances independently. Current d128 training
finishes naturally; replacing queue supervisors does not signal GPU trainers.

Engram again uses the original width-scaled embedding budget for all widths and
repeated-data sweep cells. Bucket capacity scales with model width / 5120;
channel width follows that ratio, rounded up to a multiple of eight. This restores
the performance-v1 recipe and supersedes the temporary fixed-d128 budget.
Projection outputs, gates, convolution, and the Sinkhorn optimizer are unchanged.
The counts below include equal 16-owner row-shard rounding.

| Model width | Routed top-k | Shared experts | Active intermediate/model ratio | Engram table parameters |
|---:|---:|---:|---:|---:|
| 128 | 2 | 1 | 3.0 | 153,723,520 |
| 384 | 8 | 1 | 3.0 | 1,382,873,472 |
| 640 | 14 | 1 | 3.0 | 3,072,505,856 |
| 1280 | 29 | 1 | 3.0 | 12,289,195,008 |

Every layer has 384 routed experts plus one shared expert, intermediate width 128.
Depth is 20 for the main width sweep. Head counts remain fixed; sparse head
widths and matrix channels obey the existing alignment rules. There is no
expert-transport padding. The comparison branch has eight query heads, two KV
heads, and a 512-token ordinary window versus 512×32 simplicial windows. Branch
head dimension is 16 at d128/d384/d640 and 32 at d1280. Its additional short-axis
projections are counted explicitly; the pair is not exactly parameter matched.

The [performance-v1 contract](../recipes/deepseek_v41/performance/performance-v1.json)
is authoritative again for future runs, as it already was for running d128 and
checkpointed d640. The optional fixed-anchor implementation and its archived
contract remain available for provenance, but are not selected by this study.
No running model or saved checkpoint is resized. All four widths again belong
to the same width-scaled Engram recipe; d640 is no longer a special fixed-budget
fit exclusion. Lookup storage and active computation remain separate quantities:
the larger table count does not mean every stored parameter is active per token.

Both arms retain seed 42, the sealed FineWeb-Edu stream and tokenizer, context 2048,
64 windows/update, the existing optimizer schedules, routing/indexer objectives,
and expert parallelism 8 with expert FSDP 2. Engram has 16 owners. Model source,
upstream/container identities, actual package versions, and implementation hashes
are recorded with every contract. Current data and training objectives are unchanged.

Revised widths require fresh 16-rank qualifications. Each arm qualifies both
attention treatments sequentially on its own two nodes before admitting a
revised width, comparing common initial-weight fingerprints, the expected
parameter delta, sparse numerical oracles, checkpoint restoration, and next-update
loss replay. This avoids waiting for the other arm's 222-cell sweep. These paired
controls are short qualifications, not additional 10B production runs. Existing
qualifications remain valid only for their original geometry/source.

Five full-state evaluation checkpoints per base run are stored on OSS with NAS
symlinks at 2B, 4B, 6B, 8B, 10B targets. Sweep checkpoints are at 200M increments.
Early-stop recovery checkpoints are separate. Completion requires the exact
budget and all five links. A supervisor handoff at natural completion validates
both the real child exit status and these artifacts before advancing the queue.

MLflow places all variants under **DeepSeek V4.1 — From scratch**. Empty queued
runs are deferred until data exists. Report Engram tables, embeddings/output head,
inactive experts, active expert parameters, dense parameters, tokens, measured
throughput, and GPU-seconds separately. The study uses one seed per arm; it does
not establish seed variance or a general scaling exponent. The
[literature review and proposed reasoning evaluation](SIMPLICIAL_REASONING_RESEARCH.md)
explain why held-out language-model loss alone cannot answer whether the
simplicial mechanism improves compositional reasoning.

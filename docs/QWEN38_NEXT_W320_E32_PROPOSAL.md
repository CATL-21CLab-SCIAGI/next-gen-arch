# Width-320 / 32-expert Qwen-Next proposal

**Selected for implementation and launch.** User-specified: width 320 and
fewer routed experts. Selected expert count: **32**, half the preceding 64-expert
proposal. This supersedes width 480. Historical model families and the frozen
runtime are preserved. The implemented contract is
`recipes/models/qwen38_flash_next_w320_e32.yaml`; deployment evidence is recorded
separately from this design document.

Portable contract: `recipes/proposals/qwen38_next_width_scaled.yaml`, candidate
`next-ratio-e32-w320`, profile `next_ratio_e32`. All common restoration rules and
launch gates in that contract apply.

## Corrected properties

| Property | Current 1B run | Proposed |
|---|---:|---:|
| Model width H | 384 | 320 |
| Layers | 48 | 48 |
| Routed experts / top-k | 64 / 3 | 32 / 10 |
| Shared experts | 1 | 1 |
| Expert intermediate width | 112 | 80 = H/4 |
| Active FFN intermediate width | 448 = 1.1667H | 880 = 2.75H |
| Attention total query width | 384 = H | 768 = 2.4H |
| GDN total value width | 384 = H | 768 = 2.4H |
| Attention output gate | Absent | Sigmoid gate included |
| Attention Q/K normalization | Absent | Per-head RMSNorm included |
| Attention Q/KV heads, head dimension | 6 / 1, 64 | 24 / 2, 32 |
| GDN QK/V heads, head dimension | 4 / 12, 32 | 16 / 48, 16 |
| Residual streams | 1 | 4 |
| Residual low rank | 48 | 40 = H/8 |
| PLE hash heads | 4 | 16 |
| PLE concatenated width | 384 | 320 |
| PLE table parameters | 384,012,288 | 52,454,400 |
| Total parameters | 1,006,441,440 | 387,680,960 (analytical) |

The reduced model is **approximately 388M**, not 1B. No extra parameters are
silently added to compensate for the requested width/expert-count reduction.
Top-10 plus one shared expert preserves the source's aggregate active FFN/H
ratio. It does **not** preserve original expert-pool sparsity: 10/32=31.25 percent
of routed experts are selected, versus 10/512=1.953125 percent in the source.

Depth, head counts and expert count are fixed topology constants; feature
dimensions scale from H. Expert width 80 is intentionally H/4, matching the
source individual-expert ratio rather than a dense-MLP 4H convention.

## Principal weight shapes

Linear weights use `[out,in]`. Expert shapes are per expert; attention packing
and its logical slices describe the same parameters and must not be double-counted.

| Weight | Proposed shape |
|---|---|
| Token embedding and untied LM head, each | `[248320,320]` |
| Expert/shared gate and up, each | `[80,320]` |
| Expert/shared down | `[320,80]` |
| Router | `[32,320]` |
| Shared output gate | `[1,320]` |
| Attention query + output gate | `[1536,320]` |
| Attention key and value, each | `[64,320]` |
| Combined attention Q/gate/K/V | `[1664,320]` |
| Attention output projection | `[320,768]` |
| Attention Q/K RMSNorm, each | `[32]` |
| GDN packed QKV | `[1280,320]` |
| GDN output gate | `[768,320]` |
| GDN beta and decay, each | `[48,320]` |
| GDN output projection | `[320,768]` |
| GDN convolution | `[1280,4]` |
| GDN A_log and dt_bias, each | `[48]` |
| GDN output RMSNorm | `[16]` |
| GR norm | `[1280]` |
| GR down / up / injection | `[40,1280]` / `[1280,40]` / `[4,1280]` |
| PLE logical table | `[2622720,20]` |
| PLE physical partition, 32 per replica | `[1639200]`, logically `[81960,20]` |
| PLE key / value projection | `[1280,320]` / `[320,320]` |
| PLE key/query/convolution RMSNorm, each | `[1280]` |
| PLE convolution | `[1280,4]` |

Implementation packing: the combined attention rows above are logical, not one
physical parameter. The frozen native Muon splitter does not support gated QKV.
The adapter therefore uses separate native TE parameters Q `[768,320]`, gate
`[768,320]`, K `[64,320]`, V `[64,320]`, concatenating outputs in MCore's native
per-KV-group order. Each is a separate Muon matrix; no installed runtime code
is patched. Native SelfAttention still owns the sigmoid gate and Q/K TENorm.

## Parameter allocation

| Component | Parameters |
|---|---:|
| Embedding and head | 158,924,800 |
| Routed experts | 117,964,800 |
| Shared expert weights and gates | 3,701,760 |
| Routers | 491,520 |
| GDN | 33,734,592 |
| Attention including output gates and QK norms | 9,339,648 |
| GR and final mixer | 10,548,480 |
| PLE tables | 52,454,400 |
| PLE projections/norms/convolution | 520,960 |
| **Total** | **387,680,960** |

PLE uses the existing proposal rule: 16 successive primes >=512H, total rows
padded to a multiple of 128, branch dimension H/16, 32 local physical partitions.
The model retains source-style zero-centered normalization where specified in
the common proposal, not merely the correct normalization tensor shapes.

## Execution and validation

Retain 48 layers, MTP off, 2K dense attention (no QSA/indexer), no vision, and
DP32 with TP=PP=EP=expert-TP=CP=1 in the frozen NeMo environment. No DLC restart.
The previous microbatch 4 is only a starting candidate pending memory checks:
restoring four residual streams and projected attention widths changes activation
memory even though total parameters are lower.

Before launch, implement a separate named variant and verify actual construction
counts, attention gate execution, QK/GR/PLE normalization, forward/backward numerical
oracles, native Muon grouping of gated QKV and separate expert matrices, DP gradient
equivalence, checkpoint round-trip, memory and measured throughput. These are
pending; no speedup or quality improvement is established by these counts.

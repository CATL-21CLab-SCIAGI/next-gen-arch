# Width-scaled Qwen-Next proposals

The latest user-requested direction is width 320 with fewer routed experts.
The new **next-ratio-e32-w320** proposal uses 32 experts/top-10 and totals
387,680,960 parameters. See [the updated proposal](QWEN38_NEXT_W320_E32_PROPOSAL.md).
The width-480 recommendation below is retained as historical design context,
not the current recommended target.

Status: **proposed, not launchable**. The portable specification is
[`recipes/proposals/qwen38_next_width_scaled.yaml`](../recipes/proposals/qwen38_next_width_scaled.yaml).
No live training or frozen historical model family has been changed. The
restored mechanisms below are included in the proposed geometry and analytical
counts; native-adapter wiring, construction, numerical and distributed tests
remain outstanding. No throughput claim is made.

## One feature-width rule, explicit topology constants

Let `H` be model width. Use multiples of 160 for this candidate grid so all
listed rational dimensions are integers without silent rounding. This is an
algebraic alignment rule, not evidence that every kernel supports every shape.

| Feature dimension | Rule | H=480 |
|---|---:|---:|
| Residual stream width | H | 480 |
| Main-profile routed/shared expert intermediate width | H/4 | 120 |
| Attention head dimension | H/10 | 48 |
| Attention total query width | 24H/10 = 2.4H | 1152 |
| Attention total key or value width | 2H/10 = 0.2H | 96 |
| Attention output-gate width | 2.4H | 1152 |
| GDN key/value head dimension | H/20 | 24 |
| GDN total query or key width | 16H/20 = 0.8H | 384 |
| GDN total value / output-gate width | 48H/20 = 2.4H | 1152 |
| GDN packed QKV width | 4H | 1920 |
| Packed gated-residual width | 4H | 1920 |
| Gated-residual low rank | H/8 | 60 |
| PLE concatenated width | H | 480 |
| PLE width per hash head | H/16 | 30 |
| PLE vocabulary base per hash head | 512H | 245760 |

Architectural counts remain fixed within a profile: depth 48, 24 attention
query heads / 2 KV heads, 16 GDN QK heads / 48 V heads, four residual streams,
and 16 PLE hash heads. Expert count and top-k are also fixed within a profile.
Scaling both head count and head dimension by H would make the total projection
width quadratic in H, defeating the intended fixed-width ratios.

Tokenizer vocabulary 248320, context 2048, convolution length 4, n-gram order 3,
and physical partition count 32 are fixed contract/topology choices rather than
feature dimensions. All model-parallel degrees remain one; data parallelism is 32.

## Restored versus intentionally excluded

Included in **every proposal**:

- Sigmoid global-attention output gating, with its own projected gate channels.
- Per-head Q and K RMSNorm (not `IdentityOp`).
- Four gated-residual streams, both attention/FFN mixers and the final mixer.
- Source-style zero-centered RMSNorm parameters for QK, GR and PLE where used
  by the cached reference. GDN output RMSNorm retains its direct scale.
- Sixteen PLE hash heads: eight per n-gram order, with the source key/value,
  normalization and convolution structure.
- Source attention/GDN feature-width ratios, and explicit normalized top-k
  softmax routing with one sigmoid-gated shared expert.

MTP stays **off**, and QSA/indexer stays **off** for the existing 2K dense-attention
curriculum. Vision remains excluded. Restoring these would override earlier
experiment decisions, so they are not silently reintroduced.

The frozen MCore source exposes `attention_output_gate` and `qk_layernorm`.
Their use in our custom layer spec, gated-QKV Muon splitting, and exact source
normalization behavior still require tests. Native field availability is not
construction or numerical validation.

## Deliberate small-model deviations

The primary profile preserves source feature-width ratios and top-10, but has
**64 rather than 512 routed experts**. It is still a named small-model variant.
The PLE row budget is also deliberately reduced: base `512H`, rather than the
source base `7812.5H`. These choices prevent the stored expert/table capacity
from exhausting the roughly 1B budget before active computation is allocated.

For PLE, generate 16 successive primes greater than or equal to `512H`, sum
them, and pad to a multiple of 128. The logical table width is `H/16`. Physical
storage can retain 32 local partitions per GPU; it does not require EP.
This makes PLE approximately `512H²` parameters, with a small explicit
prime/padding correction. Never round feature dimensions implicitly.

Source router auxiliary coefficient 0.001 and z-loss coefficient 0 are proposed
explicitly, versus the current variant's 0.01 and 0.001. Keeping the current
coefficients would be a different experiment contract.

## Candidate configurations

| Name | H | Experts / top-k | Expert F | Active FFN width/H | Attention / GDN head dim | GR low rank | PLE params | Total params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| next-ratio-e64-w320 | 320 | 64 / 10 | 80 | 2.75 | 32 / 16 | 40 | 52,454,400 | 506,137,280 |
| **next-ratio-e64-w480** | **480** | **64 / 10** | **120** | **2.75** | **48 / 24** | **60** | **118,026,240** | **1,017,428,832** |
| next-ratio-e64-w640 | 640 | 64 / 10 | 160 | 2.75 | 64 / 32 | 80 | 209,786,880 | 1,700,829,184 |
| wide-expert-e16-w480 | 480 | 16 / 3 | 360 | 3.00 | 48 / 24 | 60 | 118,026,240 | 900,201,312 |

The first three form **one fixed-ratio width sweep**. The fourth is a separate,
explicit ablation profile: expert F=3H/4, E=16, top-k=3, one shared expert of
the same width. It has fewer, wider matrix problems, but 9.09% more active FFN
matrix arithmetic at the same H and fewer total parameters. It is not a
parameter- or FLOP-matched control, nor a demonstrated throughput improvement.

Recommendation for source-feature-ratio validation: start with
**next-ratio-e64-w480** after all implementation gates. Consider the wide-expert
variant as a subsequent efficiency/quality experiment, not a silent substitute.

## Recommended candidate's weight shapes

Weights use `[out,in]`. Expert rows are per expert. Native physical packing may
combine query/gate/key/value tensors; the count is independent of that packing.

| Weight | Proposed shape |
|---|---|
| Token embedding; untied LM head, each | `[248320,480]` |
| Routed/shared expert gate and up, each | `[120,480]` |
| Routed/shared expert down | `[480,120]` |
| Router | `[64,480]` |
| Shared expert output gate | `[1,480]` |
| Attention query + output gate | `[2304,480]` |
| Attention key and value, each | `[96,480]` |
| Combined native attention Q/gate/K/V | `[2496,480]` |
| Attention output projection | `[480,1152]` |
| Attention Q/K RMSNorm, each | `[48]` |
| GDN QKV | `[1920,480]` |
| GDN output gate | `[1152,480]` |
| GDN beta and decay, each | `[48,480]` |
| GDN output projection | `[480,1152]` |
| GDN depthwise convolution | `[1920,4]` |
| GDN A_log and dt_bias, each | `[48]` |
| GDN output RMSNorm | `[24]` |
| GR norm | `[1920]` |
| GR down / up / injection | `[60,1920]` / `[1920,60]` / `[4,1920]` |
| PLE table, logical total | `[3934208,30]` |
| PLE physical partition, 32 per GPU | `[3688320]`, logically `[122944,30]` |
| PLE key / value projection | `[1920,480]` / `[480,480]` |
| PLE key/query/conv norms, each | `[1920]` |
| PLE depthwise convolution | `[1920,4]` |

Logical subprojections and their combined packing are alternative descriptions,
not additional parameters. The final GR mixer has no injection weight.

## Recommended candidate's parameter allocation

| Component | Parameters |
|---|---:|
| Routed experts | 530,841,600 |
| Shared experts and output gates | 8,317,440 |
| Routers | 1,474,560 |
| Embedding and head | 238,387,200 |
| GDN | 74,930,400 |
| Global attention including output gates and QK norms | 21,013,632 |
| Gated residuals and final mixer | 23,272,320 |
| PLE tables | 118,026,240 |
| PLE projections/norms/convolution | 1,165,440 |
| **Total** | **1,017,428,832** |

Counts were independently recomputed from tensor-shape products, including
prime-derived table rows, both GR modules per layer, the final mixer, QK norms
and attention gate channels. They are **analytical**, not yet confirmed by a
constructed native model.

The current microbatch size is only a starting proposal: more projected channels
and four residual streams increase activation memory. A bounded host/GPU memory
gate must precede any production launch. Throughput and learning behavior cannot
be inferred from parameter count. Preserve the existing run until a candidate is
selected, implemented, numerically checked and benchmarked in the frozen NeMo
container. A later training-child relaunch does not require restarting DLC nodes.

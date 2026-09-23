# Qwen width-scaled design candidates

**Type:** historical proposal set. **Successor:** [selected W320/E32 design](QWEN38_NEXT_W320_E32_PROPOSAL.md).
The earlier width-480 recommendation is not the current selected target.

## Feature-width rule

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

Use H multiples of 160 for integral dimensions in this grid. This is an algebraic rule, not kernel qualification.

Depth, attention/GDN head counts, residual streams and profile topology remain explicit constants. The main profile uses top-10 routing and expert width H/4; the wide-expert profile is a separate architectural change.

## Candidate counts

| Name | H | Experts / top-k | Expert F | Active FFN width/H | Attention / GDN head dim | GR low rank | PLE params | Total params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| next-ratio-e64-w320 | 320 | 64 / 10 | 80 | 2.75 | 32 / 16 | 40 | 52,454,400 | 506,137,280 |
| **next-ratio-e64-w480** | **480** | **64 / 10** | **120** | **2.75** | **48 / 24** | **60** | **118,026,240** | **1,017,428,832** |
| next-ratio-e64-w640 | 640 | 64 / 10 | 160 | 2.75 | 64 / 32 | 80 | 209,786,880 | 1,700,829,184 |
| wide-expert-e16-w480 | 480 | 16 / 3 | 360 | 3.00 | 48 / 24 | 60 | 118,026,240 | 900,201,312 |

The subsequently selected E32/W320 candidate has 387,680,960 parameters. The first three E64 rows form a fixed-ratio width sweep; the fourth changes expert topology.

## Boundaries

- Restore the declared output gates, Q/K normalization, four residual streams and PLE.
- MTP, QSA/indexer and vision remain excluded by this proposal.
- Preserve head dimensions and native physical packing explicitly.
- Analytical counts do not prove runtime capacity or speed.
- Qualify construction, gradients, optimizer grouping and checkpoint continuation before launch.

Full shape/count definitions are in `recipes/proposals/qwen38_next_width_scaled.yaml` and the named model factories.

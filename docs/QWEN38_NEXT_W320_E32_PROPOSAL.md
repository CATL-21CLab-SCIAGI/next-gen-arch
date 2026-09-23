# Qwen W320/E32 contract

**Type:** selected model design; implementation evidence is recorded separately.
**Factory:** `Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()`.
**Recipe:** `recipes/models/qwen38_flash_next_w320_e32.yaml`.

## Geometry change

| Property | Earlier 1B configuration | W320/E32 |
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

This is approximately 388M parameters. Historical 1B and width-480 families retain their separate identities.

## Weight shapes

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

Shapes use [out, in]. Combined Q/gate/K/V rows describe logical packing; the native integration keeps separate projections for correct Muon grouping.

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

PLE uses 16 successive prime table sizes at least 512H, padded for 32 physical partitions.

## Execution and gates

48 layers, 2K context, MTP/QSA/vision off. DP32 replicates the model and PLE partitions; all model-parallel groups have size one.

Require constructor/count checks, zero-centered norm semantics, GDN and gated-attention oracles, optimizer membership, DP gradients and native checkpoint continuation.

[Source review](QWEN38_NEXT_W320_REVIEW_GUIDE.md) · [Historical width proposals](QWEN38_NEXT_WIDTH_SCALED_PROPOSALS.md)

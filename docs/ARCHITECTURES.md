# Architecture reference

**Scope:** project research implementations. Paper links identify the motivating mechanism, not equivalence to an author's complete model.

## Frozen component grid

| Manifest ID | Family | Controlled change | Primary reference |
| --- | --- | --- | --- |
| `baseline` | `sota_pool` | shared full-attention baseline | [nanochat](https://github.com/karpathy/nanochat) |
| `engram` | `engram` | hashed n-gram memory injected at selected layers | [Engram paper](https://arxiv.org/abs/2601.07372), [reference repository](https://github.com/deepseek-ai/Engram) |
| `kda` | `kimi_kda` | KDA recurrent mixer with periodic global attention | [KDA paper](https://arxiv.org/abs/2510.26692) |
| `dsa` | `deepseek_dsa` | learned top-k token selection | [DSA paper](https://arxiv.org/abs/2512.02556) |
| `attnres` | `kimi_attnres` | block attention residual routing across depth | [Attention Residuals](https://arxiv.org/abs/2603.15031) |
| `mhc` | `mhc` | multi-stream hyper-connections with Sinkhorn mixing | [mHC paper](https://arxiv.org/abs/2512.24880) |
| `gated-attention` | `sota_pool` | learned gate on attention output | [Gated Attention](https://arxiv.org/abs/2505.06708) |
| `situ-glu` | `frontier_pool` | SiTU-GLU feed-forward activation | [Kimi K3](https://arxiv.org/abs/2607.24653) |
| `inkling-relative-attention` | `frontier_pool` | learned relative-position attention term | [Inkling](https://thinkingmachines.ai/news/introducing-inkling/) |
| `glm-mla` | `frontier_pool` | MLA-style projections split for Muon | [GLM-5](https://arxiv.org/abs/2602.15763) |
| `xielu` | `sota_pool` | xIELU activation | [xIELU](https://arxiv.org/abs/2411.13010) |
| `qwen-gdn` | `frontier_pool` | gated delta-network recurrent mixer | [Qwen3.5 technical report](https://qwen.ai/blog?id=qwen3.5) |
| `inkling-sconv-kv` | `frontier_pool` | depthwise short convolution on K/V | [Inkling](https://thinkingmachines.ai/news/introducing-inkling/) |
| `inkling-sconv-residual` | `frontier_pool` | depthwise short convolution on residual stream | [Inkling](https://thinkingmachines.ai/news/introducing-inkling/) |
| `partial-rope-25` | `frontier_pool` | RoPE on one quarter of head dimensions | controlled partial-RoPE ablation |
| `kimi-k3-kda-update` | `kimi_kda` | Kimi K3 update/gating recipe | [Kimi K3](https://arxiv.org/abs/2607.24653) |

## Later model-specific work

| Family | Mechanisms | Record |
| --- | --- | --- |
| Qwen Flash-Next | GDN/global attention, MoE, residual streams and PLE | [Backbone reference](QWEN38_NEXT_BACKBONE_VARIANTS.md) |
| Simplicial attention | Joint softmax over causal key pairs; exact and linearized variants | [Qwen pilot](SIMPLICIAL_DSW_EXPERIMENTS.md), [DeepSeek design](DEEPSEEK_V41_GLOBAL_SIMPLICIAL_MATH.md) |
| Pretrained adapters | Independent branches at explicit residual boundaries | [Qwen](PRETRAINED_QWEN_NEXT_SIMPLICIAL.md), [DeepSeek control](DEEPSEEK_V41_NORMAL_CONTROL.md) |

## Implementation limits

- The historical DSA quality arm uses top-k masking over dense SDPA; it does not demonstrate sparse-kernel speed.
- Engram's trainable tables count toward parameters and scaling budgets.
- mHC numerical failures remain part of the result.
- Relative-attention results describe the tested implementation and short context.
- Throughput compares recorded research implementations, not each mechanism's best possible kernel.

Shared parameters retain matched initialization; variant-only initialization uses a private RNG. Backend construction, optimizer grouping, checkpointing and distributed behavior require their own gates.

[Code map](wiki/Code-Map.md) · [Experiment design](EXPERIMENT_CONTRACTS.md)

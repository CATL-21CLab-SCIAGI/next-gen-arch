# Architecture map

The implementations are compact research adaptations in `src/next_gen_arch/arch`. They are not claimed to be the official code from the cited projects. A paper or project link explains the mechanism that motivated each controlled arm; it does not imply exact equivalence to the authors' full system.

## Frozen 16-variant grid

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

## Shared-backbone invariant

The architecture families ultimately construct one `GPT` interface with shared token embedding, language-model head, loss evaluation, optimizer factory, data loader, and checkpoint manager. Variant-only parameters are initialized from a private RNG so adding a module does not perturb the shared backbone initialization for a matched seed.

Engram and mHC historically lived in separate campaign source trees. Their definitions are merged with the shared GPT primitives in `src/next_gen_arch/arch/base.py`; construction and checkpoint compatibility live in `src/next_gen_arch/training/models.py`. Megatron adapters are a separate capability boundary; a speedrun implementation is not automatically considered ported.

## Implementation caveats

### DSA

The current DSA backend computes selection scores and applies a top-k causal mask to dense scaled dot-product attention. This preserves functional semantics for a controlled quality experiment but cannot realize the systems benefit of a sparse gather/kernel implementation.

### Relative attention

The controlled relative-attention path is optimized neither for this short 2K context nor for production kernels. Its negative quality and throughput result is specific to this implementation and contract.

### Engram

Retrieval tables are trainable and included in parameter counts. Token-to-compressed-vocabulary mapping is configured from the campaign tokenizer. Injection layers scale with model geometry and are separately mapped for the trainer's smaller meta-reference model.

### mHC

mHC expands the residual stream, applies doubly stochastic routing derived through Sinkhorn iterations, and collapses the streams before the language-model head. It is numerically unstable in the current scaling recipe despite strong earlier fixed-token results.

### Throughput

No architecture-specific fused kernel was added solely to improve a result. Reported throughput therefore measures these research implementations, not the best achievable implementation of each method.

## Adding a variant

New variants should be small and explicit:

1. isolate the mechanism in a dedicated module or a clearly named family branch;
2. route configuration through `GPTConfig` and `model_factory.py`;
3. preserve shared-backbone initialization under the same seed;
4. add CPU forward/backward and parameter-group tests;
5. register a frozen command before launching a sweep;
6. publish baseline and all single-component controls with the same data order.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the result-reporting checklist.

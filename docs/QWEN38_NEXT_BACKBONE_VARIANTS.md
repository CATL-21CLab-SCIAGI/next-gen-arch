# Qwen Flash-Next 1B backend reference

**Type:** named-model execution reference. **Geometry:** 1,006,441,440 parameters, 48 layers, MTP off.
For the later W320/E32 family, use its [separate review map](QWEN38_NEXT_W320_REVIEW_GUIDE.md).

## Geometry and ownership

| Mechanism | 1B recipe values | Where it is implemented |
|---|---|---|
| Depth / width | 48 / 384 | Config plus native transformer block construction |
| Attention pattern | 3 GDN layers, then 1 global-attention layer | `full_attention_interval=4`; adapter `QwenFlashNextLayer` chooses per layer |
| Global attention | 6 query heads, 1 KV head, head dim 64; rotary fraction 0.25 | Native Megatron `SelfAttention`; GQA must stay explicitly enabled |
| GDN | 4 QK heads, 12 value heads, key/value dim 32, convolution width 4 | Architecture `GatedDeltaNet` and causal convolution |
| Routed/shared FFN | 64 experts, top-3, expert/shared width 112 | Native `MoELayer` and router; adapter `SplitSwiGLUExperts` / `SplitSwiGLUSharedExpert` use TE grouped/dense linears |
| Residual path | 1 stream, low-rank width 48 | Architecture `FourStreamGatedResidual`; applied around attention and FFN |
| PLE | Layer 2, four hash heads, about 1M rows/head, branch width 96 | Architecture `PLEHash`, `OwnerShardedPLEEmbedding`, `DistributedPLE` |
| MTP | Disabled | `mtp_num_layers=0`, no MTP spec or auxiliary objective |

The DP-only recipe has DP=32 and TP/PP/CP/EP/expert-TP=1. Every GPU holds all experts and PLE partitions; distributed Muon partitions optimizer work/state.

The named Python factory is executable geometry. Editing the descriptive YAML alone does not rebuild the model.

## Code map

| Owner | Module |
| --- | --- |
| Definitions/counts | `architectures/qwen38_flash_next_full.py` |
| Native model | `megatron/qwen38_flash_next_model.py` |
| Arguments | `megatron/qwen38_flash_next_config.py` |
| Training | `megatron/qwen38_flash_next_full_train.py` |
| Data order | `megatron/token_batches.py` |
| Sampling | `megatron/qwen38_flash_next_sample.py` |
| Launcher | `scripts/run_qwen38_flash_next_full_dlc.sh` |

## Runtime decisions

| Setting | Reason |
| --- | --- |
| Microbatch 4, global batch 4096 | Microbatch 16 failed host-memory admission |
| Native loss fusion | TE loss fusion rejected by this runtime's guard |
| Synchronous parameter gather | Avoids the observed repeated-gather assertion |
| Gradient reduction overlap | Retained |
| Explicit launch recipe | Output names do not select architecture |

Changed geometry uses a new named family and fresh weights. EP8-to-DP32 optimizer migration was not established by the recorded adapter work.

## Verification

Check parameter counts/shapes, optimizer tags, GDN/attention gradients, DP agreement, and native checkpoint continuation. Run contracts record actual process groups and implementation hashes.

Sampling is cache-free raw continuation. Its throughput is not a serving benchmark.

[Infrastructure](TRAINING_INFRASTRUCTURE.md) · [Recipes](../recipes/models/)

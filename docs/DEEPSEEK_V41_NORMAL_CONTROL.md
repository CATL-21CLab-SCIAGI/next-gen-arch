# DeepSeek ordinary-attention control

**Type:** matched architecture control.
**Recipe:** `recipes/experiments/deepseek_v41_normal_math_1b_official.yaml`.

## Controlled change

| Property | Ordinary control | Simplicial arm |
| --- | --- | --- |
| Attention | Softmax over Q·K | Joint softmax over Q·K1·K2 |
| Value | V | V1·V2 |
| Causal window | 512 tokens | 32×512 pairs |
| Parameters | 146,843,712 | 167,816,256 |
| Core | Container FlashAttention; BF16 projections, FP32 accumulation | Qualified FP32 ternary core |

Both retain 8Q/2KV/head128, eight insertion sites, read/write maps, RMS norms, output gate and zero-output initialization. Shared tensors match byte-for-byte; ordinary K/V inherit K2/V2 initialization.

No extra RoPE, dropout or attention bias is added. The windows include self.

## Matched axes

Base revision, data/supervision, seed-2234 order, 16K context, optimizer grouping, LR schedule, validation and checkpoint cadence match. The original frozen-adapter phase uses FSDP32/EP8/Engram32.

This is geometry-matched, not parameter-matched. Throughput compares the qualified implementations and their declared precision.

## Admission and evidence

Require fresh source-bound mesh receipts, base parity, zero-adapter identity, active adapter gradients, checkpoint continuation and a full-context update. Compare identical data windows and report both update time and supervised targets/s.

[Official integration](DEEPSEEK_V41_OFFICIAL_TRAINING.md) · [Final matched results](TRAINING_CONCLUSIONS_20260922.md)

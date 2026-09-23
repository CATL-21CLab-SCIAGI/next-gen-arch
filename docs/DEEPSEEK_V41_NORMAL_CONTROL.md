# DeepSeek V4.1 normal-attention control

The portable recipe is `recipes/experiments/deepseek_v41_normal_math_1b_official.yaml`.
This is a fresh geometry-matched 1-simplicial run on the same frozen base and
32 B300 GPUs as the paused simplicial baseline. NVML's L20D label is incorrect.

The only architectural change is the additive adapter's attention: replace
joint softmax over short/long key pairs and product values with ordinary
`softmax(q k^T / sqrt(128)) v` over the causal 512-token long window, including
the current token. Q8/KV2/head128, eight insertion layers, stream read/write,
RMS normalization, sigmoid output gate, and zero output initialization remain.
Every retained initial tensor equals the baseline tensor exactly, with normal
K/V drawn from baseline K2/V2. There is no rotary transform, dropout or bias.
The short K1/V1 projections and K1 norm are absent: 146,843,712 parameters
versus 167,816,256. This control matches geometry, not parameter count.

The implementation uses the container-owned FlashAttention API with
`deterministic=True`, causal masking, window `(511, 0)`, and BF16 Q/K/V with
FP32 score/softmax accumulation. The qualified simplicial operator retains
its FP32 ternary core. Thus throughput compares the qualified implementations,
including their operator-appropriate arithmetic, not equal-precision isolated
kernels. No backbone, library, optimizer, batching or communication optimization
is introduced into the control.

The model revision, weights, dense FSDP32/expert FSDP4/EP8/Engram32 layout,
16K context, sealed train/validation manifests, seed2234 window order, 1B target
budget, headwise Q/K Muon and matrix Muon, AdamW norms/scalars, learning-rate
schedule, gradient clipping, validation and checkpoint cadence are unchanged.
The control starts at step zero; the baseline checkpoint is preserved for
resume and is not loaded into the architecturally different control.

Admission requires new source/runtime-bound 32-rank tiny-mesh receipts,
independent full-backbone parity at 128/2048/16384, zero-adapter identity,
two real updates with all adapter gradients active, exact checkpoint restore
and numerical continuation replay, and a full 16K update. Qualification state
is reset before the sealed training run. Measure throughput on identical
training steps/data windows after initial compilation, report both supervised
targets per second and seconds per update, and keep periodic validation separate.

# Quartered Qwen3.8-27B training

The dense quarter-scale Qwen3.8-27B path consumes the Muon optimizer shipped by
the container-owned Megatron installation. It does not vendor Megatron, install a
repository-local optimizer implementation, or monkeypatch Megatron optimizer
classes.

The launcher selects Megatron's registered `muon` optimizer. Megatron routes 2-D
matrix parameters through `TensorParallelMuon` and routes embeddings, output
weights, and non-matrix parameters through its built-in Adam fallback. The Qwen
model tags only that routing boundary; it does not implement the optimizer.

The training contract uses:

- BF16 model and activation compute with FP32 optimizer master parameters;
- `--muon-fp32-matmul-prec highest`, keeping Muon state matrix products in FP32;
- native Polar Express coefficients with eight Newton-Schulz steps;
- a peak learning rate of `5e-5`, with a `5e-6` minimum;
- Megatron's distributed optimizer and gradient/parameter communication overlap;
- no optimizer CUDA graph, because this path must not patch `ChainedOptimizer`.

The DLC launcher first runs a 400-step production-shaped preflight at sequence
length 2048, micro-batch size 4, and global batch size 512. This crosses the
iteration at which the previous BF16-Muon run became non-finite. A successful
preflight proceeds directly to the long FineWeb-Edu run. The persistent DLC
controller remains alive after either training success or failure and does not
restart the allocation.

The canonical local workspace is `/Users/evergreen/inf/speedrun/next-gen-arch`.
The synchronized persistent checkout is `/mnt/nas/evergreen/next-gen-arch-repo`.
Run artifacts stay under `/mnt/nas/evergreen/next-gen-arch/` and are not mixed
with the source checkout.

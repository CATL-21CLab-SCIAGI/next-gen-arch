# DeepSeek V4.1 official AutoModel training

This migration keeps the selected DeepSeek-V4.1-Flash checkpoint, the eight
simplicial adapters, the sealed Nemotron-Math-v2 1B-target pilot, seed2234 data
order, headwise Muon/AdamW grouping, learning-rate schedule and 16K context.
The portable execution contract is
`recipes/experiments/deepseek_v41_simplicial_math_1b_official.yaml`.

The pinned upstream AutoModel source is
`f7ccd6f7902634af34c2f31b3294ac250dc97670`. Its public
`NeMoAutoModelForCausalLM.from_config` owns construction, native expert
parallelism, FSDP2 and Checkpointer/DCP loading. The original container's
PyTorch, Transformer Engine, Megatron, CUDA and NCCL packages remain in place.
The full run records the actual imported upstream source and package versions.
Launches export `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enable deterministic cuDNN
algorithms, and disable cuDNN benchmarking. Global PyTorch deterministic mode
stays disabled for compatibility with the frozen reference APIs. This variant
selects a deterministic simplicial backward explicitly, and its existing replay
envelope remains enforced. These settings are recorded
and checked against the distributed mesh receipts.
Optional uninitialized-buffer filling is disabled because the container does
not implement it for packed FP4; checkpoint and kernel buffers are populated
before use. cuBLAS/cuDNN settings and deterministic sparse-KV reduction remain enabled.

## Backend and placement

- 32 GPUs on the existing four DLC nodes, with node-local EP8.
- Dense FSDP32, expert FSDP4, and Engram row ownership across WORLD32.
- Official TileLang sparse attention, native TileLang Sinkhorn coefficients
  around the official FP32 HC projection, torch FP32 RMS normalization,
  torch BF16 dense and expert GEMMs, and official torch EP dispatch. A project
  precision boundary sums experts in ascending expert-ID order and keeps
  routed sums and shared-expert addition in FP32,
  followed by one BF16 cast, matching the released reference. Expert gate/up
  use separate native BF16 linears and eager FP32 SwiGLU to preserve projection
  rounding. Official grouped down GEMM and differentiable gather/reduction are
  retained; upstream is unchanged.
- Expert inputs use an exact FP32 representation between BF16 projections,
  so local branch gradients and EP gradient reduction accumulate in FP32
  before one cast back to the original activation dtype.
  Gather backward reduces a private copy of its incoming gradient.
- Sparse attention retains the official forward and per-query backward kernels.
  Backward gives each query compact private KV slots, then uses stable key
  sorting and segmented FP32 reduction before one BF16 cast. This removes
  races between queries updating shared KV gradients. Both128 and16K kernel
  probes reproduced outputs/query gradients exactly and KV gradients repeatably.
- The simplicial variant retains its original forward. Its backward gives each
  query private short/long gradient windows and sums contributions in fixed
  temporal order. The frozen speedrun kernel and its atomic API are unchanged.
- Released quantized weights are decoded for BF16 compute. KV/index
  quantize/dequantize boundaries remain active. The output head is FP32.
- FSDP preserves explicit module precision: `output_dtype=None` and
  `cast_forward_inputs=False`. Adapters are complete replicated FP32 masters
  installed after the base is loaded and sharded; their gradients reduce over
  WORLD32 before the unchanged optimizers.

The FSDP/Engram placement replaces the previous custom bridge's replicated
non-expert base and node-local Engram copies. It does not change the model
geometry, source weights or unique-data accounting. The HC precision boundary
calls the unchanged verified released coefficient kernel, with the existing
PyTorch-equation backward for input gradients. Official projection parameters
and collapse/expand methods are retained. The container needs no additional
TileKernels package.

## Qualification and startup

`archlab.automodel.deepseek_v41_official_mesh_probe` tests the actual 32-rank
layout using a small six-layer backbone and production adapter head/window
geometry. It checks independent data-rank inputs, gradient/parameter agreement,
frozen base storage, exact checkpoint restoration and numerical update replay.
The production entry rejects missing, failed, mismatched or stale receipts.

`archlab.automodel.deepseek_v41_official_recipe` then loads the real base and
compares it with an independent pinned released-model reference at 128, 2048
and 16384 positions on every rank. The independent EP8/Engram32 reference runs
first and retains CPU outputs; its GPU weights are released before loading the
official backbone. This matches per-expert input batches and avoids co-resident
16K memory pressure. Exact batch and output-head digests bind both phases.
Both use BF16 dense/expert compute and
retain KV/index quantization. All vocabulary logits are compared in bounded
chunks. The existing mean-KL <0.02 and absolute-CE-delta <0.05 limits remain.
The previously reproduced native FP8 activation-generation defect is handled
by the project-owned stable quantizer at the reference boundary only.

After parity, the entry installs zero-initialized adapters and verifies base
identity. Two real short updates, a checkpoint/next-update replay and a 16K
update qualify the full training mesh. Adapter, optimizer and RNG state are
then reset before the sealed token budget starts. Native GPU kernels need not
be bitwise repeatable; serialized adapter, optimizer, cursor and RNG state
must restore exactly. Numerical continuation tolerances are recorded separately.

`RUN_CONTRACT.json` contains stable experiment/source/runtime identity for
checkpoint compatibility. Invocation paths and evidence hashes live in
`RUN_PROVENANCE.json`. `TRAINING_ADMITTED.json` is published only after the
full numerical and training checks pass. `train.jsonl` records actual progress;
`TRAINING_COMPLETE.json` requires exactly 1,000,000,000 supervised targets.

Create `STOP_REQUEST` in the run directory to checkpoint and pause after the
current update. Resume into a fresh directory using the same immutable source,
recipe, inputs and `--resume-from CHECKPOINT`. The DLC nodes remain running.

The replay protocol is `native-update-baseline-v1`: disk-restored continuation
is compared with at least two exact in-memory resets. Adapter and optimizer
moment update-vector relative L2 errors must remain below 1%, and the disk
error must not exceed twice the observed in-memory maximum (with a 1e-5
relative numerical floor). Optimizer step counters and options are exact;
loss comparison retains rtol1e-5/atol1e-7. Scalar step counters are excluded
from moment-error normalization. This separates serialization integrity from
known FP32 atomic reduction ordering in sparse/simplicial backward kernels.
The diagnostic which motivated it measured disk update error 0.063% versus
0.051–0.065% without disk I/O, with identical losses and byte-identical restored
state. Componentwise near-zero-weight mismatches remain recorded diagnostics.
The full backbone forward KL and CE gates remain unchanged.

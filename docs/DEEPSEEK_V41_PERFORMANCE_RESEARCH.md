# DeepSeek V3–V4.1 training performance investigation

Research date: 2026-09-27. Production source: `cf6f577`, d640 normal and
2-simplicial, 16 B300 GPUs per variant. B300 is mislabeled L20D by NVML here.

## Measurement status

Both production variants and their queue supervisors were stopped on the user's
explicit instruction, without requesting checkpoints. The allocation is retained
for bounded diagnostic runs; production has not been restarted.

Nsight Systems 2026.2.1's default hardware CUDA tracing was blocked by CUPTI
permissions. Switching to `--trace=cuda-sw,nvtx` produced a capture without any
container changes. The user subsequently chose end-to-end wall-clock timing as
the primary speed metric. Old hardware-mode reports lacking GPU kernel events
are not used as kernel-level evidence.

The bounded probe supports fresh matched initialization, fixed data cursors,
warm-up exclusion, synchronized wall time, CUDA-event phase times, and optional
Nsight capture. It never saves training state. Ablations and numerical oracles
are under `results/deepseek-v41-performance-fixes-20260927/`; source snapshots and
hashes are recorded with the runs. Final measurements belong in the companion
performance-fix report, not in the historical observations below.

Research downloads and source hashes are under
`results/deepseek-v41-nsight-20260927/`.

## What the running code actually does

The latest 30-update sample observed during this investigation was:

| Variant | Updates | Mean update | Mean wall time | Supervised / physical token slots |
|---|---:|---:|---:|---:|
| Normal | 4047–4076 | 5.680 s | 5.704 s | 44.35% |
| 2-simplicial | 3319–3348 | 7.517 s | 7.536 s | 42.87% |

These are different data cursors, not a matched performance comparison. The
small wall/update difference suggests data loading and logging are not the
main steady-state cost in this sample. It does not identify GPU bottlenecks.

1. **Small experts executed through many separate operations.** Each layer has
   384 routed experts, EP8, 48 experts per rank, hidden width 640, expert width
   128, top-15 routing and one shared expert. `deepseek_v41_official_moe.py`
   calls `counts.tolist()`, loops over local experts, and launches separate
   BF16 gate/up linears, copies, FP32 activation operations, and weighting.
   Only the down projection uses grouped GEMM. If all 48 local experts are
   populated, 20 layers issue 1,920 gate/up GEMMs per forward; activation
   recomputation repeats them before gradient GEMMs. These are counts implied
   by source, not measured launch counts.

2. **General-purpose EP communication.** The same implementation gathers
   token hidden states, routing weights, indices and masks across EP8. It sums
   the full routed output and narrows it to local tokens; backward also sums
   gathered gradients. This differs from optimized dispatch/combine that sends
   selected token work to expert owners. At top-15 many owners may be selected,
   so bandwidth savings cannot be assumed from dispatch alone. CPU scheduling,
   metadata synchronization and communication overlap still matter.

3. **Dense work over enormous conditional-memory tables.** The d640 contract
   contains 26,689,870,264 total parameters. Engram alone accounts for
   `(384006168 + 384016682) * 32 = 24,576,731,200` parameters, approximately 92%.
   Forward lookup is sparse, but `ShardedAdafactor` explicitly requires dense
   gradients and scans every owned parameter in several passes. Its factored
   statistics, update clipping, stochastic BF16 rounding and changed-element
   accounting touch full table shards. Sparse forward access does not imply
   sparse optimizer traffic. This is a major candidate for memory-bandwidth
   cost and is absent from an active-matrix-only FLOP estimate.

4. **Frequent host synchronization and eager optimizer launches.**
   `gradient_step` evaluates a GPU finite flag in Python per parameter.
   Expert counts, EP lengths, indexer KL logging, vector optimizer statistics
   and router diagnostics also cause device-to-host scalar transfers. The
   optimizer loops through individual experts and tensor chunks. These can
   create launch gaps even when aggregate GPU busy percentage looks moderate.

5. **Sparse attention is already batched.** Production explicitly selects
   `tilelang-batched-sparse-mqa-v1`, not the older per-query Python chunk loop.
   Most layers combine 128 window slots and up to 512 compressed slots with
   invalid entries masked; 64 attention heads and head dimension 64 are used
   at d640. Shared-KV and sink gradients use FP32 atomics. The existing 14.42×
   comparison in `deepseek_v41_scratch_high_mfu.py` is an older isolated kernel
   result at B8, not an end-to-end speedup or a fresh measurement for B4.
   Sparse gather, atomic contention, masked work and index selection still
   need measurement. Replacing this with an inference-only kernel is not valid.

6. **The vocabulary head is another credible compute cost.**
   `TrainableHeadCrossEntropy` keeps a 129,280 × 640 FP32 head, processes
   valid targets in chunks of 128, and recomputes logits in backward. A claim
   that all active matrix work runs at BF16 Tensor Core peak would be wrong.
   Nsight should distinguish FP32 head GEMMs from expert BF16 GEMMs.

7. **Padding and recomputation multiply other costs.** Roughly 56–57% of
   physical positions are not targets in this sample. Masks avoid some work,
   but full-shape projections, attention kernels and activation storage remain.
   Full activation checkpointing and reshard-after-forward repeat compute and
   gathers. Low allocated memory is a reason to benchmark less recomputation,
   not evidence it can safely be disabled at all widths.

The hypothesis is therefore broader than “tiny experts and slow attention.”
The current evidence supports launch/synchronization, optimizer traffic,
communication, and head precision as additional suspects. Their ranking and
percentages remain unmeasured.

## Public repositories reviewed

These are implementations with different scopes and maturity; none is assumed
to reproduce DeepSeek's private end-to-end training system.

| Repository | What is actually available | Relevance and limits |
|---|---|---|
| [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) | Official report, weights documentation and inference demo | Architecture and precision reference; the repository does not supply the complete original pretraining engine. |
| [DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) | Official DSA model release and inference code | Explains the sparse indexer lineage; not a full training-stack replacement. |
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Official model card, technical report and inference assets on Hugging Face | Source for CSA2, shared KV/index state, CED and Engram. The guessed GitHub `deepseek-ai/DeepSeek-V4.1` URL returns 404; use verified model assets. |
| [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) | V3/V3.2/V4/V4.1 training model code; V4 pretraining and V4.1 full/LoRA fine-tuning recipes | Closest implementation to our pinned runtime. Current V4.1 recipes use HybridEP, FSDP optimizations and fused optimizer/norm options; they differ from our precision-matching orchestration. |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Grouped expert MLPs, optimized dispatch, overlap, DSA/CSA training modules | Best source for integrated high-throughput expert execution and sparse-attention forward/backward. V4 support does not establish V4.1 parity for our custom path. |
| [Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Public V3/V4 pretraining recipes, conversion and B300 performance configs | B300 V4 config uses EP8, PP4, MBS2, HybridEP settings and fused grouped MLP at 128 GPUs. Our 16-GPU d640 geometry needs its own benchmark. |
| [TorchTitan](https://github.com/pytorch/torchtitan) | Native PyTorch V3 and V4 training, grouped experts, rematerialization, compile and dispatch abstractions | Good readable training reference. Its V4 README only claims a four-GPU debug smoke test and calls for larger convergence/checkpoint validation. |
| [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) | Dense/grouped GEMMs, MoE weight-gradient kernels, indexer and HC operations, newer fused MoE functionality | Study grouped forward and backward instead of per-expert gate/up launches. Layout/alignment and precision requirements need checking at width 128. |
| [DeepEP](https://github.com/deepseek-ai/DeepEP) | Specialized EP dispatch/combine and backward communication | Current V2 uses NCCL Gin and documents new runtime requirements. Do not copy V2 into the running container; qualify a compatible runtime separately. |
| [FlashMLA](https://github.com/deepseek-ai/FlashMLA) | Official dense/sparse attention kernels and V4/V4.1 inference support | Sparse prefill API availability is not sufficient proof of a compatible training backward. Megatron's fused CSA adapter pairs FlashMLA forward with cuDNN DSA backward. |
| [TileKernels](https://github.com/deepseek-ai/TileKernels) | TileLang kernels, including Engram gate/gradient operations | Useful fused primitive references, not a pretraining loop or proof of sparse optimizer equivalence. |
| [Miles](https://github.com/radixark/miles), [slime](https://github.com/THUDM/slime) | Megatron/SGLang post-training systems and DeepSeek model integrations | Miles contains sparse-MLA training backward kernels; our AutoModel wrapper credits an earlier Miles fork/revision. RL throughput claims are not pretraining results. |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | JAX training, V3/V3.2/V4 support, Engram/mHC examples | Useful independent architecture/distributed reference; not a drop-in PyTorch/B300 migration. |
| [nanowhale](https://github.com/huggingface/nanowhale) | Small V4-inspired pretraining code at hidden width 320 | Useful small-model comparison, but only four routed experts and different attention/geometry. README has conflicting BF16 benchmark and FP32 stability guidance; its throughput is not an apples-to-apples target. |

Pinned source inspections:

- AutoModel `f21252a95fca4a2f813a386ec2c86b94cd4c7711`.
- TorchTitan `f359667137438bef39274cb869626c3798512ecd`.
- Megatron-LM `d113016bb2850c4b7d804f24ba72c055d7d4f861`.
- Megatron Bridge `c33453133cc304a661d71a784e283e1846458235`.
- Miles `23d41d711f3b80544fda655898ed4f051ed644fe`.
- TileKernels `36d9e45d38e204ebb87e6f6e833821eee0482fe5`.

Additional README snapshots have retrieval hashes in the local research
manifest. They are not commit-pinned unless listed above.

## What to test after the trace

1. Attribute CPU ranges and correlated GPU kernels to forward, backward,
   sparse attention, expert projections/dispatch, head loss, gradient checks,
   optimizer parameters and router balance. Inspect launch count, GPU idle
   gaps, synchronization APIs and NCCL overlap. Kernel summed duration and
   critical-path wall time must be reported separately.
2. Batch finite/norm/router reductions and avoid per-parameter `.item()`-style
   synchronization while retaining the same failure checks and sum semantics.
3. Fuse Adafactor's tensor passes and expert batch dimension. First preserve
   existing dense update, FP32 statistics and stochastic rounding semantics.
   Switching to sparse updates or another optimizer is a scientific change,
   not merely an implementation optimization.
4. Benchmark grouped gate/up plus down and fused activation, then a compatible
   HybridEP/DeepEP dispatcher. Verify forward, input/weight gradients, router
   weighting, empty experts, EP reduction precision and checkpoint continuation.
   Existing separate projections deliberately reproduce BF16 rounding; a
   faster GEMM cannot silently replace that contract.
5. Benchmark the FP32 head separately, including chunk-size effects. Changing
   head dtype/TF32 or loss arithmetic requires a declared numerical contract.
6. Profile sparse forward and backward individually. Only then compare a
   supported FlashMLA/cuDNN training pair or newer TileLang kernels at the
   actual head dimensions, masks, sinks, shared-KV aliases and top-k indices.
7. Benchmark selective recomputation/resharding and packing without changing
   global target count, sample order or per-document isolation. Packing must
   preserve RoPE, compression, Engram history and attention boundaries.

The historical contract above used ratio 3.2. The user subsequently approved
aligned widths 128/384/640/1280 and an exact ratio of 3.0 in every layer,
including the shared expert. Both variants must use the same new geometry and
performance contract. See the companion performance-fix report for changes and
measurements; the observations in this section describe the stopped old runs.

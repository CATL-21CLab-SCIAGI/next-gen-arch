# V4.1 aligned-width performance contract

For the subsequent optimization audit and fresh launch, see
the launch audit (historical record in the private archive). This report records the earlier
stopped-production measurement phase.

The old production jobs and queue supervisors were stopped without checkpointing.
All performance measurements below are fresh, bounded diagnostics, not resumed
production runs. GPUs are B300 despite their NVML labels.

## Geometry and scientific controls

The user approved an exact active intermediate-width / hidden-width ratio of 3.0
in every layer, including the always-active shared expert:

| Hidden width | Expert intermediate width | Routed top-k | Shared experts | Ratio |
|---:|---:|---:|---:|---:|
| 128 | 128 | 2 | 1 | 3.0 |
| 384 | 128 | 8 | 1 | 3.0 |
| 640 | 128 | 14 | 1 | 3.0 |
| 1280 | 128 | 29 | 1 | 3.0 |

These are model dimensions, with no transport-padding adapter. DeepEP rejects
unaligned model widths before model construction. Both arms use the same geometry,
seed, token order, microbatch 4, global batch 64 windows, and 10B-token allowance.
The new queue order is d640, d128, d384, d1280; the 222-cell repeated-data loop
sweep follows normal-d128 independently of the other arm. Five checkpoint
milestones and OSS/NAS-symlink storage remain in the experiment recipes.

The Figure 5 transfer keeps its recursion, weight-decay, epoch and reference-depth
grid. Model widths round upward to multiples of 128, with a minimum of 128:
128/128/128/256/256/256/256/384 at reference depths 4/6/8/10/12/14/16/18.
This is a coarser coupled width/depth grid, not a literal replication of the
paper's dense models. The d128/20-layer anchor is unchanged across loop controls.
Old source snapshots, results and campaign plans remain historical artifacts;
they must not be resumed under the new geometry.

Engram base bucket counts scale by hidden width / 5120. Per-head channels scale
by the same ratio, rounded upward to multiples of 8 (8/24/32/64). Integer channels
cannot follow the fractional ratio exactly at d128 or d384. Prime bucket counts
are regenerated in the pinned upstream order; there is no padded feature tensor.
This reduces model capacity and is a declared scientific change, not a numerically
equivalent implementation optimization.

## Implementations

- Gate/up, activation, down and token permutation use the pinned AutoModel grouped
  expert implementation. The previous separate gate/up path remains available
  for the historical numerical contract and timing control.
- The installed upstream DeepEP dispatcher is evaluated directly; no vendored
  runtime package or custom padding wrapper is used. It has upstream BF16 combine
  semantics, so it is a new numerical contract, shared by both study variants.
- The existing Adafactor groups expert matrices across the expert axis, keeps
  scaling arithmetic on the GPU, and packs independent row-owner statistics
  into one collective. Optimized gradient checks reuse computed norms;
  indexer/router metrics use batched host transfers. Failure checks remain active.
- CPU-side suffix trimming removes only positions beyond every valid target in
  a microbatch, retaining compression alignment, document isolation and data
  order. Indexer query sampling stays on the original 2048-position grid.
- Activation checkpointing and forward resharding are configurable. Retaining
  activations avoids recomputation and repeated gathers where measured memory
  permits. Production qualification still runs the actual width and microbatch.
- The optional compiled mHC backward evaluates the existing reference equations
  through PyTorch compilation. It does not substitute an inference-only kernel.

The current execution adapter is AutoModel/FSDP. Its sparse attention has TileLang
forward **and backward**; this is not a Megatron forward paired with an unrelated
TileLang backward. Reusing kernels across frameworks is normal when layouts,
precision, gradient equations and distributed reductions are validated.

## Table optimizer

[DeepSeek-V4.1 report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf),
Algorithm 1, sections 2.5, 3.1.3 and 4.2.2, supplies the table update: momentum .95,
Nesterov look-ahead, row threshold .001 of the global mean, 11 alternating L2
normalizations, epsilon 1e-20, sqrt(column-count) scaling and correction .18.
The implementation factors normalization into row/column vectors and reduces
column statistics across table owners. It is separate from mHC Sinkhorn-Knopp.

The base table LR peaks at 2.6e-4, multiplied by 5 for Engram, with no table decay.
It follows the existing warm-up/decay schedule via a .026 scale on the control's
relative LR. Applying the control's .01 directly as an absolute table LR would
be incorrect. Momentum stays FP32; BF16 storage uses stochastic rounding.
Only the tables switch optimizer: the backbone retains Adafactor, so this is not
claimed to reproduce the paper's complete Muon optimizer recipe. No reusable
compatible official PyTorch row-sharded table optimizer was found; the existing
resident-RL Sinkhorn implementation is preserved separately.

## Validation and measurements

Artifacts: `results/deepseek-v41-performance-fixes-20260927/`.

- CPU regression suite after geometry/evaluation changes: 1113 passed, 149 skipped,
  1 deselected; 205 subtests passed. CPU-only runs do not establish GPU behavior.
- Eight-rank numerical oracles validate row-sharded Sinkhorn, uneven shards,
  zero rows/columns, batched Adafactor, grouped expert forward/input/router/weight
  gradients and empty expert owners (`oracles-final/`).
- d128 upstream DeepEP forward and gradients match the grouped reference in the
  tested ragged/masked/empty-owner cases (`deepep-d128-oracle/`).
- Compiled mHC backward matches the reference at sequence lengths 17/128/256,
  maximum absolute gradient error 2.861e-6 (`hc-oracle.json`).
- A checkpoint round-trip test preserves FP32 table momentum, LR scaling, RNG and
  the next BF16 update. Full distributed production qualification remains a
  per-width admission gate.

Timing uses synchronized maximum-rank wall time on 16 GPUs, matched initialization
and windows, 12 warm-up updates and 8 measured updates. Four batches repeat for
steady-state comparisons so first-use shape compilation is excluded. Source
hashes, arguments, resolved runtime packages and every step are retained. These
short repeated-data timings do not establish convergence or scaling laws.

### d128 incremental comparison

| Cumulative stage | Median update (s) | Measured range (s) | Valid tokens/s | Throughput change from previous |
|---|---:|---:|---:|---:|
| original | 3.954 | 3.486–5.680 | 13,942 | — |
| grouped | 3.309 | 3.140–3.638 | 18,222 | +30.7% |
| scaled | 3.148 | 2.869–3.484 | 19,191 | +5.3% |
| sinkhorn | 3.135 | 2.975–3.364 | 19,299 | +0.6% |
| trimmed | 3.338 | 2.954–4.936 | 16,835 | -12.8% |
| synchronized | 2.723 | 2.393–3.852 | 21,129 | +25.5% |
| head | 2.760 | 2.341–3.640 | 21,551 | +2.0% |
| resident | 1.376 | 1.304–1.716 | 42,050 | +95.1% |
| hc | 1.332 | 1.286–2.931 | 38,438 | -8.6% |
| deepep | 1.299 | 1.203–1.380 | 47,029 | +22.4% |

All stages use the same valid targets. `scaled` changes Engram capacity;
`sinkhorn` changes its optimizer. Thus the 3.37× aggregate throughput improvement
includes declared model/optimizer changes, not only implementation acceleration.
These are short samples without confidence intervals; compare ranges as well as
medians. Changing GPU node pairs between portions of the sequence is another
source of variance; both pairs use the same allocation and B300 hardware.

Suffix trimming reduces measured physical token slots from 1,048,576 to 806,912
(23.0%) but regresses in its isolated stage. Ragged execution and shape-dependent
costs can offset removed work. It remains in the combined candidate for the
fresh-data check; it is not reported as an isolated speedup. Compiled mHC has no
clear gain here and is disabled in the selected configuration. DeepEP is selected
based on its combined full-training measurement, not merely its isolated oracle.

The selected configuration keeps microbatch 4, enables upstream grouping/DeepEP,
scaled tables, Sinkhorn, batched optimizer/metrics, suffix trimming, head chunks
of 1024 and retained activations. It disables compiled mHC backward. Wider widths
and loop cells retain separate production qualification gates; d128 throughput
must not be extrapolated to them.

No new MFU number is claimed: mixed FP32/BF16 head and expert arithmetic, sparse
attention, routing, table traffic and recomputation require a consistent FLOP
accounting convention. NVML utilization is not MFU. Wall time and valid-token
throughput above are directly measured.

### Selected configuration: fresh-data warm-up

Both variants completed 120 updates from the start of the sealed data stream,
with no repeated batches and no checkpoints. The table LR scale is .026 and
router bias rate .001; the main LR warms to .01 over 100 updates.

| Variant | Initial / final loss | Updates 100–119 median (s) | Valid tokens/s | Peak allocated GiB |
|---|---:|---:|---:|---:|
| normal | 11.7940 / 8.1871 | 1.297 | 40,457 | 11.58 |
| simplicial | 11.7940 / 8.1766 | 2.892 | 18,612 | 11.66 |

All 32 rank records pass finite loss/gradient checks. This establishes short-run
numerical stability through warm-up, not convergence. At update 120, the worst
layer still has about 19% (normal) / 21% (simplicial) unused experts over its last
20 updates; routing coverage requires continued monitoring. These fresh-data
rates are not directly comparable with the repeated-window ablation rates.

The standalone evaluator now reconstructs the recorded width, scaling/loop and
performance options and requires the checkpoint's original 8- or 16-rank mesh.
It does not silently reshard or construct the old d640 model. CPU admission and
option-reconstruction tests pass; a full GPU evaluation/restore of a new study
checkpoint remains a separate qualification, since these diagnostics saved none.

Production remains stopped. The changed scientific contract requires fresh runs,
full per-width qualification and new output directories; old queue plans are not
reused. The recipes retain independent queues, MLflow-compatible contracts and
five evaluation checkpoint milestones.

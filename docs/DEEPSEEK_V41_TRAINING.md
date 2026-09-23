# DeepSeek V4.1 frozen-backbone adapter finetuning

The requested base is **DeepSeek-V4.1-Flash**, revision
`df42c109f1defefcbfcedbe7d905718a12266e40`. Do not substitute V4 Flash.
The original checkpoint and completed Nemotron-Math-v2 tokenization are read-only
inputs to this workflow. The target is exactly **1,000,000,000 supervised next-token
targets**, not one billion padded inputs or expert-parallel copies.

## Backend boundary

The existing Automodel checkout is pinned at
`a4ce87c003f08b74d68684d3627f6e6048bc0140`. Its shipped DeepSeek recipe implements
V4, not V4.1. The installed Transformers 5.8.1 does not recognize `deepseek_v41`,
and Automodel does not register `DeepseekV41ForCausalLM`. The upstream main tree
`7d18db8146c307a8e4570bce4a64d8923b1162bb` was also checked on 2026-09-11;
no separate V4.1 recipe/model package was found.

Consequently this is a **project-owned Automodel-components recipe**, not a claim
of stock `NeMoAutoModelForCausalLM` V4.1 support. It reuses Automodel's distributed
mesh and indexed-data reader, the pinned released V4.1 model structure, existing
project adapter/optimizer primitives, and PyTorch numerical leaves. It does not
patch or upgrade installed PyTorch, Megatron, TE, CUDA, NCCL or Automodel sources.
The existing container and DLC nodes stay in place.

The Automodel onboarding and recipe-development guidance informed the explicit
architecture/config check, strict tensor mapping and numerical gates. A YAML
checkpoint-path change cannot add Engram, shared KV, CED, hierarchical indexing,
or single-pass mHC to its V4 implementation.
The parity-testing guide is applied with CPU/FP32 component oracles and the
released native GPU backbone as the full-model reference: the installed HF
runtime cannot instantiate this model. Quantized-native versus decoded-BF16
drift is reported explicitly, not called bitwise pretrained equivalence.

## Review entry points

- `recipes/experiments/deepseek_v41_simplicial_math_1b.yaml`: portable recipe.
- `src/archlab/automodel/deepseek_v41_recipe.py`: admission, construction and
  training lifecycle. Admission checks source-bound leaf/full-model receipts.
- `src/archlab/automodel/deepseek_v41_execution.py`: exact native construction,
  pre-I/O capacity check, adapter insertion and optimizer grouping.
- `src/archlab/automodel/deepseek_v41_loading.py`: strict streamed checkpoint
  loading; bounded CPU reads and explicitly placed decode chunks.
- `src/archlab/automodel/deepseek_v41_pytorch.py`: one-time frozen FP4/FP8 weight
  decoding, PyTorch linear/quantization/sparse-attention leaves and query-chunked
  indexer. Existing routing, masks, shared KV and sink semantics remain explicit.
- `src/archlab/architectures/deepseek_v41_adapter.py`: additional branches only,
  after attention hc_post and before FFN mixing at blocks 5,10,...,40.
- `src/archlab/architectures/deepseek_v41_torch.py`: query-chunked sparse joint
  softmax and native-style activation rounding, with explicit STE derivatives.
- `src/archlab/automodel/deepseek_v41_training.py`: supervised-token loss
  reduction, global adapter gradient reduction, evaluation and checksummed
  adapter/optimizer/RNG/cursor checkpoints.

All base parameters, router correction biases and non-cache training state are
frozen. Vision and speculative MTP remain in the original checkpoint but are
inactive in this text-only run. There is no replacement attention or baseline
training job. The new modules retain the reviewed 8/2 heads, head width 128 and
32-by-512 causal pair windows: exact joint softmax over those local pairs, not
global pair coverage or a delta-rule linearization.

Parallelism is DP32 with four **node-local EP8** groups, TP/PP/CP all 1. Frozen
experts and Engram rows are EP-sharded; non-expert base weights and complete
adapters are replicated. Automodel's mesh is used, but **the base is not FSDP
wrapped**. Complete adapter gradients are globally reduced before headwise Muon.
This replaces the earlier unqualified FSDP2-plus-EP8 placement candidate.

## Numerical corrections and qualification

The first PyTorch bridge failed end-to-end tiny-model parity. Its sparse core
retained FP32 exponentials, while the released attention rounds unnormalized
exponentials to BF16 within 64-key tiles before the value GEMM. The new
native-rounding path reproduces that operation and keeps stable sink handling.
The tiny logit comparison now passes the original tolerance, with relative L2
error approximately 0.00489. Native-style FP8/FP4 activation rounding matched the
three tested native quantizer formats exactly. Removing that rounding changed
tiny-model logits materially (roughly 11% relative L2 in the current fixture);
this is **not** a full-checkpoint capability/quality measurement.

The new simplicial core uses FP32 internal products (existing Triton kernel),
with BF16 projections and residual outputs. This is a scoped new-adapter choice:
BF16 intermediate products failed one cancellation-sensitive gradient element
in the 513-position production-window oracle. With FP32 core arithmetic, forward
relative error was about 1e-7 and all five input-gradient errors were under 0.2%.
The pre-existing shared kernel and older experiments are unchanged.

Additional loader fixes preserve the released model's exact BF16-to-FP32 head
and pooling-compressor promotions. Unexpected dtype conversions remain errors.
Checkpoint reads are forced to CPU despite the reference's global CUDA default;
only bounded chunks move to the destination GPU. Capacity accounting excludes
reclaimable PyTorch allocator cache from driver/NCCL memory.

Full-model qualification must compare native and PyTorch outputs at 128, 2048 and
16384 positions, report CE/KL and expert/index selection agreement, check gradient
onset, reload checkpoints, and profile forward/backward memory through 16K.
Logits are compared in 128-position chunks to avoid full-vocabulary activation
peaks. Successful receipts are bound to implementation hashes; old results do
not admit edited code. At production startup, the real 32-rank mesh is checked
again, including checkpoint next-update replay and a 16K update, then adapter,
optimizer and RNG state are restored fresh before the token budget begins.

The launch recovery in `results/deepseek-v41-launch-20260913T165222Z` found that
the released FP8 activation quantizer, executing in the existing container,
produced nondeterministic NaNs from finite first-layer inputs at 2048 positions.
The first-layer replay localizes the failure before its GEMM; quantizer scales
were finite and identical across repeats. The native reference's 128-position
outputs were also not repeatable. These failed references cannot qualify the
training path. The project bridge therefore generates FP8 activation values
and scales with PyTorch using the same quantization equations, while retaining
the native GEMMs. Its loading report records this reference correction explicitly.
The pinned reference files, runtime packages, experiment geometry, and existing
KL/CE admission thresholds remain unchanged. New qualification receipts are
required for this implementation; the earlier failed comparison is preserved.

The corrected `full-v3` run completed finite forward comparisons through 16K,
but failed admission on all eight ranks at 128 positions: KL was 0.05435–0.29089
against the unchanged 0.02 limit. The 2048/16384 forward comparisons passed;
full-model gradient, checkpoint and backward-memory qualification remained
unexecuted because admission failed. The 1B production run was not launched.
See that recovery directory's `LAUNCH_BLOCKED.md`, `launch.json`, and
`full-v3/summary.json` for the preserved results and exact source snapshot.

At the time this guide was added, leaf and CPU tests passed and full-checkpoint
qualification was still running. **This document does not claim the 1B run has
launched.** Check the run's `RUN_CONTRACT.json`, `train.jsonl` and
`TRAINING_COMPLETE.json` for actual production state.

## Launch and recovery

Populate the `ARCHLAB_*` path variables named in the recipe, plus
`NGA_CONTAINER_DIGEST` (existing image identity) and `NGA_EXPECTED_COMMIT`
(committed project source). `PYTHONPATH` must include this source and the pinned
existing Automodel checkout. Use `/opt/venv/bin/python` with the existing CUDA
toolchain. Launch the module through torchrun on the existing four nodes:

```bash
python -m archlab.automodel.deepseek_v41_recipe \
  --recipe recipes/experiments/deepseek_v41_simplicial_math_1b.yaml
```

That is the **per-rank entry**, not a single-process launch command. It requires
world size 32 and refuses missing, failed or stale qualification receipts.
Use a fresh output directory; data order is the sealed pilot's seed-2234 shuffle.
No target tokens are wrapped, silently discarded or counted twice. Validation
uses the separate 1M-target held-out pilot. Default validation/checkpoint cadence
is 10M/50M supervised tokens; final checkpointing occurs at the exact budget end.

Create `STOP_REQUEST` inside the run directory to checkpoint and pause after the
current step. This does not stop the DLC nodes. Resume into a fresh output directory
with the same contract and `--resume-from PATH/TO/CHECKPOINT`; the checkpoint
contains adapter/optimizer state, per-rank Torch RNG and the exact data cursor.

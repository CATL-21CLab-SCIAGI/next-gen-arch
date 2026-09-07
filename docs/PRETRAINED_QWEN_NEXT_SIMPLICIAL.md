# Frozen pretrained Qwen Next with additive simplicial modules

Status: the qualified production entry has been launched on the existing four
DLC nodes as `pretrained-simplicial-fineweb-20260907-v1`. It starts from the released
pretrained checkpoint with fresh additions, not any diagnostic checkpoint.

## Agreed experiment

- Start from the full released Qwen3.8-Flash-Next checkpoint, not any from-scratch
  experiment. Freeze all existing weights and retain original QSA/indexer, GDN,
  MoE, PLE, gated residual streams, norms and gates. The user now explicitly
  permits omitting MTP from the training model. Vision is inactive in this
  text-only backend. Retain the original checkpoint artifact unchanged and
  explicitly account for these unused tensor prefixes.
- Add a separate simplicial residual branch in layers 4, 8, ..., 48 after the
  original attention residual update and before the existing MoE residual read.
- Train only these additions on one pass of the existing FineWeb-Edu sample-100BT
  tokenized corpus, with 16K context and ordinary main next-token CE.
- Use one training job, no separately trained comparison/control, no automatic
  unfreezing, and no automatic restart of any prior experiment.
- Preserve installed container packages and existing DLC nodes. The user now
  permits AutoModel's in-process runtime patches and expert parallelism. Target
  FSDP2 over 32 data ranks with node-local EP8 (64 routed experts per EP rank per
  layer, additionally FSDP-sharded over the four nodes),
  TP/PP/CP/expert-TP one, subject to numerical and memory validation. EP overlays
  the data mesh: this still uses 32 GPUs. Historical curves are not a controlled
  pretrained baseline.

The portable training contract is
`recipes/proposals/qwen38_pretrained_simplicial_fineweb.yaml`.

## Production entry and launch

The source snapshot is `results/pretrained-simplicial-source-20260907-v1`, made
from project commit `688f036`. NeMo AutoModel remains pinned to
`a4ce87c003f08b74d68684d3627f6e6048bc0140`. The launch happened before git push;
no new user permission, package installation, environment, subagent or node
restart was required.

- `src/archlab/automodel/train.py`: production entry, one-pass optimizer loop,
  fixed held-out evaluation, signal-requested save/stop, and metrics.
- `src/archlab/automodel/training_config.py`: typed optimizer/schedule policy;
  the YAML `training` section is the editable experiment configuration.
- `src/archlab/automodel/execution.py`: the construction/load functions already
  exercised by the full32 probe, shared unchanged with that probe.
- `src/archlab/automodel/checkpointing.py`: adapter-only DCP, upstream PEFT/EP
  Adam-state materialization, exact per-rank state hashes, completion manifests,
  and strict cursor/runtime/data checks before fresh-process restoration.
- `src/archlab/architectures/simplicial_adapter.py`: independent added-module
  definition and shapes. `automodel/simplicial.py` owns the insertion boundary;
  the original backbone remains the pinned upstream implementation.

The launch uses microbatch one, no accumulation, 524,288 targets per step,
AdamW (betas 0.9/0.95, epsilon 1e-8, weight decay 0.1), global adapter-gradient
clipping at 1.0, LR 1e-7 rising linearly to 1e-5 over 200 steps, then cosine decay
to 1e-6 over the one-pass horizon of 191,570 steps. This deliberately does not
reuse the overshooting constant-lr=1e-4 diagnostic settings.

Validation uses the same first four global held-out batches (2,097,152 targets)
at step zero, step ten, and every 100 steps. It is a labelled fixed validation
prefix, not the whole validation split. Checkpoints are saved after step one,
warm-up, every 1,000 steps, at the final/explicit stop boundary, or after a
SIGUSR1 request to a training rank. SIGTERM/SIGINT request save-and-stop at a
completed optimizer-step boundary. Original weights are never saved over or
included in these new adapter checkpoints. `LATEST.json` points to a completed
checkpoint; incomplete saves never publish `COMPLETE.json`. Checkpoints are not
automatically deleted.

The original checkpoint remains at its supplied read-only path. This run loads
the separate SHA256-verified NAS copy; provenance retains the original location
and hashes. Source, model/config/tokenizer, data manifest, schedule, tensor names
and topology must match on resume. A new process restores the saved optimizer,
scheduler, CPU/CUDA RNG and exact next data cursor; it cannot silently load the
earlier diagnostic checkpoints or restart the data stream.

Training-entry qualification passed 19 project CPU tests, a two-GPU comparison
of FSDP gradients/global clipping norm/Adam updates against a CPU oracle, and an
actual two-process exit/restart from step two. Exact state digests passed on
both resumed ranks. Compared with uninterrupted continuation to step four,
adapter-update relative L2 error was 3.856e-5 and maximum absolute weight error
3.167e-8; scheduler, RNG and step counters matched exactly. The staged tests
followed AutoModel's recipe and parity-testing guidance. Full32 model/EP evidence
below remains distinct from this small, fresh-process training-entry test.

Run files are under `results/pretrained-simplicial-fineweb-20260907-v1`:
`metrics.jsonl`, per-attempt manifests, `LATEST.json` and `checkpoints/`.
Per-rank startup/errors are in the sibling `-logs` directory; node launcher PIDs
and source pins are in the sibling `-launch.json` record.

Initial production observations (attempt `34c05fc2e901`, through step 22):

- Exact initial 16K logits identity passed before any optimizer update.
- Fixed held-out loss decreased from 1.936017 at step zero to 1.923916 at step
  ten, on the same 2,097,152 targets. This is an early warm-up observation, not
  evidence of a converged gain or a controlled comparison with historical runs.
- All observed gradients were finite; peak allocated memory stayed at
  89,768,406,528 bytes per rank. Recent steps take about 13.3–14.1 seconds,
  approximately 37,000–39,500 target tokens/second across all 32 GPUs.
- The completed step-one checkpoint is
  `checkpoints/step-00000001-34c05fc2e901`, with cursor one, all 32 rank-state
  hashes and an 8.53 GB state payload. It contains additions and training state,
  not a duplicate of the frozen pretrained backbone.
- A full 100.44B-target pass is roughly a month at this early throughput, before
  allowing for interruptions and checkpoint/evaluation overhead. This is an
  extrapolation, not a completion-time guarantee.

## Added leaf mechanism

`src/archlab/architectures/simplicial_adapter.py` owns the independent branch. Its
interface is `[batch, sequence, 4 × hidden_size] ->` the same packed state plus a
learned update. It imports shared architecture primitives, not a trainer.

Full geometry is width 2560, 24 query heads, two KV heads, head dimension 256,
four residual streams with rank 320, windows 16 × 128, partial rotary fraction
0.25, and theta 10M. Q, K1 and K2 have independent zero-centered RMSNorms; output
has a learned sigmoid gate. The score/value operation has no KV constant bias
and uses ordinary partial RoPE, not determinant attention. There is no added FFN.

Each branch has 59,034,368 parameters; twelve branches have 708,412,416. The
output projection starts at zero, so finite inputs initially pass through
unchanged. First-step non-output parameter gradients are therefore zero by
design; after the output projection updates, gradients reach all branch
parameters. Frozen downstream layers still require input-gradient propagation.

## Evidence and limits

- Seven CPU tests cover count/shapes, RNG preservation, exact zero-output identity
  and its input gradient, causality, partial RoPE, adapter state restoration, and
  gradients through a frozen suffix without changing its weights.
- Four existing CPU attention-oracle tests pass.
- Six frozen-container GPU oracle cases pass at head dimension 256: FP32/BF16,
  both causal window boundaries, future-token perturbations, and single-pair
  softmax degeneracy, for outputs and all five input gradients.
- An isolated BF16 adapter at `[1, 16384, 10240]` passes exact initial identity and
  finite nonzero gradients for all added parameters after an output update.
  This uses a diagnostic SGD update, not a production optimizer or full model.

These leaf tests alone do not establish full-model or distributed support. The
separate integration evidence below must also pass before finetuning.

## Reuse-first backend qualification

The required order is: search maintained off-the-shelf code, inspect existing
backend implementations, reuse suitable code, and implement only uncovered gaps.
Finding an implementation is not evidence that it supports this experiment.
The pinned-source audit and qualification plan are in
[PRETRAINED_BACKEND_REUSE_AUDIT.md](PRETRAINED_BACKEND_REUSE_AUDIT.md).

NeMo AutoModel is the selected existing implementation, used from the pinned
external source checkout rather than installed into the container. Its approved
in-process patches are permitted; do not modify installed package files or add
unrelated runtime patches. No backbone equations are reimplemented.

`src/archlab/automodel/simplicial.py` extends only the existing MoE residual read
with the independent adapter. Original Parameter objects and checkpoint keys
are retained. Original attention/indexer and MoE execution remain upstream code.
`src/archlab/automodel/probe.py` is a bounded qualification entry, **not** a
finetuning launcher. The adapter is installed and independently FSDP-sharded
after base checkpoint loading but before the first model forward; this ordering
must pass the distributed tests.

## Current compatibility limits

The frozen DLC environment has Transformers 5.8.1. Its `AutoConfig` rejects the
checkpoint's `model_type=qwen4_exp`. The previously searched installed
Transformers, NeMo, Megatron Bridge and Megatron Core source locations had no
matching implementation. Native PyTorch FSDP2 is available, but that does not
supply the missing architecture. This is an installed-runtime finding, not a
claim that upstream implementations do not exist.

The old `qwen38_flash_next_full_train` adapter explicitly constructs a from-scratch
dense-attention variant. It cannot be used as a faithful pretrained loader.

The unmodified NeMo AutoModel configuration preserves all 32 audited text-config
fields. Its model, EP, FSDP and checkpoint components import in the container;
DeepEP is already available. Twenty-six selected upstream model/QSA/residual
tests and five project insertion/loading tests pass. The latter establish exact initial
logits/hidden-state identity, original-weight preservation, adapter gradients
through a frozen suffix, and strict state restoration on a small CPU model.
Nine upstream checkpoint tests also pass, including a two-rank owner-sharded
PLE save/load. Two project data-reader tests pass (42 CPU tests across these
selected suites in total).

## Integration qualification, 2026-09-07

The upstream distributed-training and parity-testing guidance informed the
FSDP2+EP selection and the staged CPU/GPU/reference/checkpoint checks. The external
AutoModel checkout remains unmodified. Container packages and nodes are unchanged.

- `src/archlab/automodel/simplicial.py` installs twelve independent modules at the
  original MoE read. It unwraps activation-checkpoint wrappers before modifying
  a decoder; assigning onto the wrapper itself registers unused parameters.
  A dedicated regression verifies that the added module actually executes.
- `src/archlab/automodel/loading.py` checks the complete global HF key map before
  loading: 1,294 active keys, 333 inactive vision keys, 31 inactive MTP keys, no
  missing or unexplained keys. It reconstructs unsaved RoPE buffers using the
  original upstream constructor, poisons floating-point destinations with NaNs,
  and requires every loaded local tensor to be finite. The original checkpoint
  is read-only and unchanged.
- The ordinary upstream initializer casts expert storage to BF16. The
  pretrained path skips random initialization, so it must explicitly reuse that
  same cast before sharding, preserving intrinsic FP32 GDN state. Otherwise
  GroupedExperts retain unnecessary FP32 frozen copies.
- `src/archlab/automodel/probe.py` performs bounded qualification only. Synthetic
  tokens, main next-token CE at every position, no MTP or router auxiliary loss.
  Zero-output identity, adapter-only gradients/AdamW, adapter/optimizer/per-rank
  RNG checkpoint restoration, and replay are tested. No diagnostic checkpoint
  is a finetuning initialization.
- Two-GPU EP2+FSDP2 with activation checkpointing passes. An unsharded combined-
  batch reference has exactly equal logits and mean loss; adapter-gradient
  relative L2 error is 0.002348 (BF16), max absolute error 7.034e-6. The reference
  reuses upstream grouped experts with the ordinary torch dispatcher.
- Adapter weights, optimizer tensors and RNG restore exactly. Subsequent GPU
  trajectories are **not bitwise deterministic**: the simplicial backward uses
  FP32 atomic sums, and computation is BF16. Replay separately bounds gradient
  relative L2 error at 3% and update relative L2 error at 1%; it does not hide
  differences by comparing relative to the much larger parameter magnitudes.
  EP2 v6 measured <=1.768e-5 gradient error and <=8.088e-5 update error; an earlier
  replay measured about 0.52% and 0.54%, respectively. Exact-state tests remain
  zero-tolerance regardless of these numerical bounds.
- A full-checkpoint EP4 load-only probe passed on every rank after ~517 seconds
  of storage reads. All floating-point destinations, including PLE and routed
  experts, were populated and finite. This initial probe retained the excessive
  FP32 expert allocation; the BF16 storage correction is in the subsequent
  full-model 16K probe. A successful load is not forward/backward qualification.
- The BF16 full-checkpoint EP8 load passed. The initial full-model 16K identity
  test failed. A repeated baseline with all adapters
  disabled also differs. Tracing at sequence length 257 shows the first
  divergence inside the first GDN's installed FLA `chunk_gated_delta_rule`,
  before PLE and before the first added module. Q/K/V/g/beta are exactly equal
  across calls; the first-rank GDN output differs by up to 0.0146484375. Its
  causal convolution is repeatable. Every zero-output adapter independently
  returns its input exactly. This forward failure is separate from the known
  simplicial backward atomic-sum nondeterminism and was not waived.
- The failure reproduces in the isolated FLA chunk-state forward kernel on our
  L20D GPUs (compute capability 10.3), matching
  [FLA issue 945](https://github.com/fla-org/flash-linear-attention/issues/945).
  `src/archlab/automodel/runtime.py` selects existing safe launch configurations
  from upstream [fix 953](https://github.com/fla-org/flash-linear-attention/pull/953)
  (forward state: two warps) and
  [fix 1000](https://github.com/fla-org/flash-linear-attention/pull/1000)
  (WY backward: two warps, four stages). This is a narrow process-local
  configuration restriction, not a package upgrade, installed-file edit or
  kernel rewrite. It clears the in-memory selection cache and disables this
  tuner's disk-selection cache; existing compiled kernel files are retained.
  The entry records original source hashes and selected configurations.
- With those settings, captured pretrained inputs and synthetic full-16K GDN
  inputs have exactly repeatable outputs and all five input gradients. Against
  the installed Transformers oracle, full-16K output relative L2 error is
  0.005586 and gradient errors are 0.00480–0.00668 (BF16). Captured pretrained
  input errors are 0.003522 forward and 0.00657–0.01050 backward. These are
  numerical comparisons, not bitwise equivalence between different algorithms.
- Full-model EP8 **exact initial identity at 16K now passes** with the safe
  settings. Two all-position main-CE diagnostic updates pass finite gradients
  for every added parameter after output warm-up; no original parameter has a
  gradient. Peak allocated memory is about 124.3 GB/rank including replay. Diagnostic AdamW uses
  lr=1e-3 on synthetic tokens; these losses are not a finetuning curve. Full-size
  checkpoint weights, optimizer and per-rank RNG restore exactly on every rank.
  Rank-zero replay gradient/update relative L2 errors are 0.000274/0.000718.
  The subsequent four-node real-data result below is separate from this
  eight-GPU synthetic-token result.
- Cross-node DeepEP EP16 initialized on two complete eight-GPU nodes but timed
  out in its first token dispatch. Its normal-mode implementation assumes
  contiguous eight-rank NVLink groups; a prior four-GPUs-per-node reduced probe
  was invalid and the entry now rejects that layout. No nodes were restarted.
- **DP16 + node-local EP8 passed** on two nodes: exact initial logits,
  adapter-only gradients, exact checkpoint weights/optimizer/RNG and bounded
  replay. Combined-batch unsharded-reference logits are equal; adapter-gradient
  relative L2 error is approximately 0.00306–0.00424. This exercises cross-node
  FSDP while keeping DeepEP token dispatch within each node. The proposed
  four-node topology is therefore DP32/EP8, subsequently exercised at full size;
  cross-node EP32 is not selected.
- The first full DP32/EP8 attempt was stopped during base loading, before any
  forward/update: all worker loaders waited in `folio_wait_bit_common` on cold
  OSS safetensor pages while the master waited for them. A sequential test read
  managed only ~252 MB in 20 seconds on a worker. This is distinct from the
  cross-node DeepEP failure. `stage_checkpoint.py` created a separate NAS
  copy using standard file-copy operations and SHA256 verification of every
  file; it does not transform weights or alter the original read-only artifact.
  The completion sentinel is written only after all indexed weight shards and
  accompanying top-level files verify. The completed cache contains all 131
  weight shards and 360,023,349,599 bytes across its files. Every source/copy
  SHA256 matched; all four nodes see the completion manifest. Its source index
  SHA256 is `99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de`.
  The four-node real-FineWeb retry uses that verified copy, not a partial cache.
  No DLC nodes were restarted.
- **Full DP32/EP8, 16K, real FineWeb passed on every rank.** All 32 ranks loaded
  finite pretrained weights, installed twelve additions, and passed exact
  initial logits identity at every position. Initial global mean next-token CE
  is 1.85804379 over 524,288 targets. All added gradients are finite and nonzero
  after output warm-up; no original parameter receives a gradient. Adapter
  parameters, optimizer moments/steps and per-rank RNG restore exactly, including
  into a newly constructed optimizer using upstream lazy-state materialization.
  Replay loss is equal; gradient relative L2 errors are 0.002165–0.003411 and
  update relative L2 errors 0.001782–0.002350. Maximum absolute replay weight
  difference is 4.992e-5, consistent with the predeclared BF16/atomic-sum bounds.
  All four launchers exited zero and released their GPUs.
- Full32 peak allocated memory is 89,768,656,384 bytes/rank (89.8 GB, 83.6 GiB).
  The verified NAS checkpoint load took approximately 13 minutes. Diagnostic
  update maximum rank times were 45.79 seconds for the first backward, followed
  by 18.65, 16.35 and 13.88 seconds (the last is replay). These are bounded probe
  timings, not a steady-state finetuning throughput claim.
- At the diagnostic constant AdamW lr=1e-4, repeated-batch global mean losses
  were **1.858044 → 2.477143 → 1.844002**, with replay exactly matching the last
  mean. The initial overshoot means this is **not** an approved production LR
  schedule. Qualify a gradual warm-up/lower starting LR, production batch size,
  held-out reporting, one-pass cursor and separate-process training-entry recovery
  before launching the corpus pass; the subsequent production-entry tests above
  cover that path. Do not interpret repeated-batch fitting as
  held-out improvement or initialize finetuning from this diagnostic checkpoint.

Saved probe evidence is under `results/automodel-ep2-probe-20260907-v6-logs`,
`results/automodel-full-ep4-load-20260907-v1-logs`,
`results/automodel-full-ep8-16k-20260907-v1-logs`, and
`results/automodel-ep16-probe-20260907-v2-logs`. Additional evidence is in
`results/automodel-dp16-ep8-probe-20260907-v2-logs` and
`results/automodel-full-ep8-identity-20260907-v4-logs`.
The safe-setting rerun is `results/automodel-full-ep8-16k-20260907-v2-logs`;
isolated evidence is `results/automodel-gdn-isolated-pretrained-safe-20260907-v1.log`
and `results/automodel-gdn-isolated-16k-safe-20260907-v1.log`. Two CPU runtime-
restriction regressions also pass (44 selected CPU tests total).
Three additional CPU tests cover exact checkpoint copying, missing/escaping
index entries, refusal to overwrite completed/unrelated artifacts, and repair
of an explicitly selected incomplete cache (47 selected CPU tests total).
Two further CPU tests cover cache provenance/index drift and exact restoration
into a fresh AdamW optimizer (49 selected CPU tests total). The latter reuses
upstream `OptimizerState` with its PEFT+EP path, including materializing lazy
Adam state before DCP loading. The full four-node retry discards the live
optimizer before checkpoint restoration as well. This is not yet a separate-
process training-entry recovery test.
Full32 real-data evidence is under
`results/automodel-full-dp32-ep8-fineweb-20260907-v2-logs`; every `probe_pass`
explicitly retains `production_qualified: false` because the production entry
and its remaining gates are distinct from the now-passed model integration.

## One-pass data contract

`src/archlab/automodel/data.py` reuses AutoModel's indexed-data reader. It adds a
manifest-ordered, contiguous fixed-window schedule with disjoint target ranges
across DP ranks, an explicit replay cursor, and no wrapping. Neither the frozen
speedrun schedule nor the old cyclic reader is changed. No padding or document-
reset masks are introduced; existing EOS tokens remain in the raw stream.

The actual FineWeb manifest and pretrained tokenizer match exactly:

- Manifest SHA256: `39f40f57d34a808c48c3babb7ca122d80f85c968613f657fe6892eefb102ab6b`.
- Tokenizer SHA256: `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
- At DP32, microbatch one, 16K: 191,570 complete global microbatches,
  100,437,852,160 consumed targets, 288,047 unused final targets, one initial
  context-only token, zero wrapped tokens. These are data accounting numbers,
  not a finalized optimizer/accumulation schedule.

Do not start a handwritten backbone port on the strength of the AutoConfig error.
First qualify reuse of upstream model code with a project-owned integration
boundary. Any necessary compatibility work must be narrowly scoped, attributed,
and tested against its upstream numerical reference. Runtime upgrades,
unapproved topology changes and silent architectural substitutions are not
fallbacks; approved AutoModel runtime hooks and EP are recorded exceptions.

Changes to the qualified model, topology or training policy must revalidate the
affected gates. Never replace the original attention with the simplicial pilot wrapper:
that wrapper swaps attention, whereas this experiment must add a new branch.

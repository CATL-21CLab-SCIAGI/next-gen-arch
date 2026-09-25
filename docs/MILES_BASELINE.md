# Miles RL: supported execution path

**Current relaunch in qualification:** [2-simplicial, 4K responses, native useful-group sampling](DEEPSEEK_V41_2SIMPLICIAL_MILES.md), using `recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml` with the same canonical launcher. The normal baseline described below is its historical predecessor; its qualification does not certify the new variant.

Use **`archlab.megatron.miles_v41_stock_launch`** with
[`deepseek_v41_2simplicial_stock_fp8.yaml`](../recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml).
It resolves explicit native arguments and calls pinned Miles `train.py`. It does
not import the upstream recipe, intercept `execute_train`, call private `_train`,
or implement a training loop. The upstream V4.1 recipe remains the provenance
for the argument contract. Paths are explicit launch inputs, not machine names
embedded in the recipe.

The current experiment identity is `deepseek-v41-2simplicial-adam-fp8-grpo-4k-filtered-v1`:
32 B300s (TP8/PP4/EP8), BF16 training, distributed Adam with BF16 stored moments
and FP32 masters/arithmetic, frozen BF16 Engram tables, FP32 adapters, native FP8
rollout, routing replay, and GRPO. Optimizer states stream to **node-local disk**;
the actor remains on GPU with no policy backup. This is not offloading-free.
Muown, BF16 rollout, NAS optimizer storage, and the historical resident driver
are different experiments and must not inherit this qualification.

## Status and evidence

This is a maintained entry page, not a live monitor. At **2026-09-25 19:10 UTC**,
the canonical launcher was running the 2-simplicial contract on all 32 B300s.
All 32 parent imports and local optimizer scratch mounts were verified; initial
held-out evaluation was generating successfully. **Optimizer-update qualification
and training performance measurements remain pending for this variant.** Its run
root is `results/deepseek-v41-2simplicial-miles-fp8-20260926`; its resolved launch
records source revision `f563957b3c986a4b6d5b7d71712c116c10a5de26`.

The normal predecessor was intentionally retired after **34 updates**, with its
final native checkpoint `iter_0000033` completed at **2026-09-25 17:28 UTC**.
The following table records that predecessor's earlier three-update qualification;
it does not qualify the current 2-simplicial run.

| Claim | Evidence / limitation |
| --- | --- |
| Canonical launch contract | CPU rendering and equivalence tests compare every effective argument against attempt 6. The canonical launcher subsequently launched the distinct 2-simplicial contract on 32 GPUs; see the current status above. |
| Operational and numerical observations | Attempt 6 completed three finite, nonzero updates, reward counts 20/10/10 out of 128, KL 0.00162–0.00202, and synchronized next rollouts. This is short-run evidence, not proof of FP8/BF16 parity or answer-quality improvement. |
| Checkpoint save/readback | Full first-update checkpoint saved; all model extents and optimizer manifests checked, four adapter tensors read through native DCP, moments sampled. Eight-GPU synthetic save/load passed. |
| Full training restoration | **Not demonstrated.** Restoring the full model, optimizer, RNG, and rollout cursor and continuing useful updates still needs a separate distributed qualification. |
| Performance measured | Third warm cycle 25.7 minutes; training 5.3–5.6 minutes; full save 53.7 minutes. Three-batch response truncation 45–78%. No throughput comparison or general performance qualification implied. |

The compact, versioned [qualification record](recorded-results/deepseek-v41-stock-fp8-20260925.json)
links the exact run and receipts. The [dated attempt history](DEEPSEEK_V41_STOCK_MILES.md)
explains failures and fixes. Large raw receipts stay under the recorded run root
on team storage. CPU tests cannot replace the missing full-resume experiment.

## Reproduce the environment and prepare a fresh run

1. Use the image digest and exact package versions in the YAML `runtime` section.
   The image alone is insufficient: the qualified installation uses a separate
   runtime overlay, described below. Do not upgrade or patch container libraries.
2. Check out Miles `6c6858a42b61459467814edc1404b1d9bfa38471` in a separate clean
   checkout. Use the same source revision on all four nodes. Do not install a
   floating `main` or edit this checkout.
3. Restore the qualified overlay artifact, with Megatron-LM
   `651dd728e3c1f763b5ba3a8de4ac0d01f70e621c` and Emerging-Optimizers
   `b309e2f01cda75dc96a6dc1a2355a7b3b64b5e16`. Its current team-storage location is
   `/mnt/nas/evergreen/runtime/miles-v41-20260924`; use its `site-packages` and
   existing compiled wheels, not a fresh unpinned dependency resolution. The
   qualified extracted image is under `.runtime/sglang-v41-deployment-20260920`.
   These are deployment artifacts, not vendored project code. Reproduction on a
   new host requires access to those artifacts; a clean-room overlay build is
   not yet documented or qualified.
4. Choose a **new** `RUN_ROOT`. Supply the prepared `model/` metadata/tokenizer
   bundle and preformatted non-thinking `train.jsonl` from the qualified data
   preparation, including the disjoint `heldout-32.jsonl` evaluation data. The model
   config must reference the finetuned parent for the selected variant (currently
   2-simplicial) via
   `archlab.full_checkpoint` and `complete_sha256`, and select
   `quantization_config={"quant_method":"fp8","activation_scheme":"dynamic",
   "fmt":"e4m3","weight_block_size":[128,128]}`. Keep its BF16 Engram buffers and
   trained adapters. Do not substitute the public model or reapply a chat template.
   The bundle points to weights; copying it does not duplicate the full parent.
5. Use the existing `archlab.serving.isolated_sglang_runtime` boundary on **each**
   node. Set its `source` to project `src`, the pinned Miles checkout, and the
   overlay's `Megatron-LM`, `site-packages`, and `Emerging-Optimizers` directories.
   Set `open_files_soft_limit: 65535` before starting Ray on every node: the
   native router inherits this limit and the 512-group request pool exceeds a
   1,024-descriptor limit. Set `private_tmp: true`, a distinct node cache, and `scratch_bind` from a named
   local directory (e.g. `/tmp/evergreen-NEW_RUN`) to `RUN_ROOT/offload`.
   Use `minimum_free_bytes: 1400000000000` for initial preparation, as in the
   qualified deployment. The destination must be empty before mounting. Ray
   workers and the driver must see these same binds; starting only the driver
   in the namespace is insufficient. Reuse the existing Ray deployment process,
   with eight GPUs per node. NVML's “L20D” label means B300 here.
6. Reserve durable storage for the parent, results, and every simultaneous full
   checkpoint: one recorded checkpoint occupies about **5.95 TB** before minor
   metadata. Local optimizer files also persist for the live run. The contract
   saves every 20 rollouts over 24 rollouts, with an initial qualification save
   requested through `SAVE_REQUEST`; budget all retained checkpoints.
   Do not delete live scratch or reuse an old run's checkpoint directory.

For example, materialize the pinned checkout and copy only the prepared inputs
into a new run directory (set these variables to your deployment paths):

```bash
git clone --no-checkout https://github.com/radixark/miles "$MILES_ROOT"
git -C "$MILES_ROOT" checkout --detach 6c6858a42b61459467814edc1404b1d9bfa38471
mkdir "$RUN_ROOT"
cp -a "$QUALIFIED_MODEL_BUNDLE" "$RUN_ROOT/model"
cp "$QUALIFIED_PROMPTS" "$RUN_ROOT/train.jsonl"
cp "$QUALIFIED_HELDOUT" "$RUN_ROOT/heldout-32.jsonl"
```

The recorded per-node namespace configuration is
`results/deepseek-v41-2simplicial-miles-fp8-20260926/runtime-NODE.json`.
Copy it into the new run and change `cache_dir`, `working_directory`, `source`,
`scratch_bind.source`, and `scratch_bind.destination` to the new deployment.
Retain the pinned image manifest/rootfs and private namespace settings. Invoke
commands through that boundary with:

```bash
PYTHONPATH=src python -m archlab.serving.isolated_sglang_runtime \
  --config "$NODE_RUNTIME_CONFIG" -- \
  -m archlab.megatron.miles_v41_stock_launch check \
  --config recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml \
  --run-root "$RUN_ROOT" --miles "$MILES_ROOT" --address "$RAY_ADDRESS" \
  --image-manifest "$IMAGE_MANIFEST"
```

Use `train` in place of `check` for the detached driver after the Ray workers
are started in their matching per-node runtime namespaces.

From the repository root in a CPU environment, inspect the exact launch without
importing GPU packages or touching the run directory:

```bash
PYTHONPATH=src python -m archlab.megatron.miles_v41_stock_launch render \
  --config recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml \
  --run-root "$RUN_ROOT" --miles "$MILES_ROOT" --address "$RAY_ADDRESS"
```

Inside the prepared runtime, validate the pinned packages, image manifest,
parent identity, local scratch mount, and upstream argument parser:

```bash
python -m archlab.megatron.miles_v41_stock_launch check \
  --config recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml \
  --run-root "$RUN_ROOT" --miles "$MILES_ROOT" --address "$RAY_ADDRESS" \
  --image-manifest "$IMAGE_MANIFEST"
```

The canonical launch command in that same runtime is:

```bash
python -m archlab.megatron.miles_v41_stock_launch train \
  --config recipes/experiments/deepseek_v41_2simplicial_stock_fp8.yaml \
  --run-root "$RUN_ROOT" --miles "$MILES_ROOT" --address "$RAY_ADDRESS" \
  --image-manifest "$IMAGE_MANIFEST"
```

For detached execution, run that command under the site's normal process
supervision. It delegates rollout, training, synchronization, checkpoint timing,
and shutdown to Miles. `--model-dir` optionally selects a prepared bundle outside
`RUN_ROOT/model`. The initial qualification save uses native Miles's sentinel:
create `RUN_ROOT/SAVE_REQUEST` before training when an early checkpoint is needed.
There is no custom admission or monitoring loop.

Before connecting to Ray, `train` exclusively creates `resolved-launch.json`
with the experiment/configuration hashes, exact argv, allowlisted environment,
project revision, actual package versions, image manifest hash, parent identity,
and prompt checksum. `launch-argv.json` remains a compatibility artifact. An
existing launch record or checkpoint directory is rejected: this contract starts
fresh Adam from the finetuned parent (`--no-load-optim --no-load-rng`), **not** a
resume from RL. Check/render do not authorize a resume or claim a healthy run.

## Necessary hook inventory

All hooks are project integrations; none edits container-owned libraries on disk.
They remain compatibility obligations against the pinned runtime.

| Hook / module under `src/archlab` | Why it remains / boundary |
| --- | --- |
| `megatron/miles_v41_model.py` | Native V4.1 model plus trained architectural adapters, BF16 row-sharded Engram, frozen tables and trainable projections. |
| `megatron/miles_v41_checkpoint.py` | Checksum-verified import of our original EP8/FSDP2 parent; installed on Miles's HF-import binding. |
| `megatron/miles_v41_stock_init.py` | Single native custom-init hook installing the compatibility hooks below. |
| `megatron/miles_v41_sync.py` | Converts names and shards, quantizes missing Engram WKV projections via upstream FP8 converter, preserves FP32 adapters/BF16 tables, frames complete weight coverage. |
| `megatron/miles_v41_weight_session.py` | In-stream transaction markers for this runtime; preserves native pause, flush, acknowledgment, versioning and resume. |
| `megatron/miles_v41_resident_policy.py` | Disables redundant backup/restore only for a single resident colocated actor; reads live GPU weights. Guards against reference/teacher/old-actor configurations. |
| `megatron/miles_v41_checkpoint_kind.py` | Prevents architectural adapters being mistaken for LoRA; retains native full-model save dispatch. |
| `megatron/miles_v41_checkpoint_writer.py` | Native DCP/MCore strategy with one tensor per file and bounded host staging; prevents demonstrated NUMA OOM, preserves native load format. |
| `serving/sglang/deepseek_v41.py` and its `sglang_v41_*` helpers | External model registration, trained adapters, BF16 Engram, padding/hash buffers, strict live-weight coverage. Pinned native model source fingerprint checked. FP8 execution remains native. |
| `serving/isolated_sglang_runtime.py` | Existing image/private-namespace and local-disk mount boundary; no RL orchestration. |

The recipe's environment controls are explicit too, including deterministic
kernels, native FP8 options, health-check bypass during dummy initialization,
and model source fingerprint. `--use-miles-router` is necessary: the tested
Rust router stripped routing replay fields. These are not optional silent defaults.

## Historical alternatives

`miles_v41_resident_launch`, `miles_v41_resident`, and `miles_v41_launch` remain
available to interpret and reproduce their historical experiments. They are not
supported launch commands for this baseline. Their Muown/BF16/resident-admission
contracts and prior parity failures do not describe this Adam/FP8 experiment.
See [resident history](DEEPSEEK_V41_RESIDENT_QUALIFICATION.md) and
[migration history](DEEPSEEK_V41_MILES_MIMO_MIGRATION.md).

When numerical settings, model variant, topology, optimizer, storage, or runtime
pins change, version a distinct recipe identity and qualify it separately.
The contract hash changes with configuration contents; it does not encode machine
paths, which are recorded separately. CPU CI checks argument equivalence and
configuration handling. Distributed qualification must record useful rewards,
nonzero updates, a synchronized next rollout, and full checkpoint restoration
before claiming the entire lifecycle is qualified.

# Track the existing training runs in shared MLflow

The project uses the existing shared MLflow service and its standard UI. A small
sidecar imports the existing JSONL ledgers and follows new rows using the official
`mlflow-skinny` client. Trainers, checkpoints, and container packages are unchanged.
There is no separate dashboard or replacement tracking server.

The configured experiments are **DeepSeek V4.1 — Full fine-tuning** (two active
16-GPU runs) and **DeepSeek V4.1 — Scratch w640 d20** (two paused 8-GPU runs). The
scratch experiment also contains validation and matched-checkpoint evaluation
metrics. Paused runs are closed in MLflow with `archlab.state=paused`; this does
not imply that their original training-token budget was reached.

## Use the shared UI

Open the colleague-provided public MLflow URL with the already-configured browser
Authorization-header extension. Select both variants in an experiment and use
MLflow's **Compare** view.

- `train/cross_entropy` uses the exact optimizer step as its x coordinate.
- `train/cross_entropy_by_tokens` uses the exact consumed supervised-token count
  as its x coordinate; use this metric with the **Step** axis for token-matched
  learning curves.
- Other metrics include token throughput, update duration, learning rate,
  gradient norm, peak allocated memory, and scratch router health.
- `eval/*` contains available validation and benchmark scores.
- `checkpoint.*` tags point to the newest complete checkpoint on persistent
  storage. Weight files are not uploaded to MLflow.

Historical records have no per-update wall-clock timestamps. Backfilled metric
timestamps represent ingestion time, not reconstructed training time. Use the
step or token axes for historical comparisons. Live updates use observation time.

## Run the sidecar

Install `mlflow-skinny==3.16.0` in an isolated environment; it matches the deployed
tracking service. Keep credentials and local service state outside version control.
The current deployment uses the ignored `.runtime/training-monitor/` directory,
with its credentials file restricted to the owner.

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m archlab.tracking.mlflow_sync \
  --config /path/to/runs.json \
  --credentials /path/to/mlflow-credentials.json \
  --state /path/to/sync-state.json \
  --watch --interval 30
```

The credentials file contains `tracking_uri` and `token`; use the internal service
URL for DSW/DLC. Never put the token in source files, shell arguments, or artifacts.
The run configuration names the experiment, variant, source directory, ordered
history files, training budget, dataset, and current lifecycle state. Scratch runs
include `prior-phase-metrics.jsonl` before the active ledger to retain ancestry.

The sidecar holds a single-writer lock, persists its import cursor after successful
batches, preserves retry timestamps, ignores partial final JSONL records, rejects
nonfinite metrics and discontinuous data cursors, and records sync failures without
writing credentials to its status output. A failed logging request never signals
or stops a training process. `SYNC_STATUS.json` reports current sync health.

Do not delete the state file to restart the service. Run identities are tagged on
the server, but local state preserves precise retry and evaluation-import cursors.

## Checkpoint references in Artifacts

Each tracked run has a `checkpoints/` artifact directory:

- `LATEST.json` points to the newest completed checkpoint reference.
- `CATALOG.json` lists attached checkpoints and whether their source completion
  marker is still present. In-progress saves are listed separately.
- `step-NNNNNN/REFERENCE.json` records the resolved source path, filesystem mount,
  optimizer step, token cursor, trained source commit, and required restore mesh.
- `step-NNNNNN/COMPLETE.json` preserves the original checkpoint contract.
- `step-NNNNNN/rank-manifests.zip` contains all original rank manifests, including
  tensor-chunk hashes and optimizer filenames.
- `METADATA_SHA256.json`, `RESTORE.md`, and `ATTACHED.json` provide metadata checksums,
  restore instructions, and an upload/readback receipt.

These are **references and metadata, not a model backup**. Model, optimizer, and
RNG payloads remain at the source path. Restoring needs a host with that source
filesystem mounted, the original payloads, and the matching runtime/source and
GPU mesh. A `file://` source URI will not download NAS weights to an unmounted Mac.

The checkpoint attachment sidecar runs independently of metric synchronization:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m archlab.tracking.mlflow_checkpoints \
  --config /path/to/runs.json \
  --credentials /path/to/mlflow-credentials.json \
  --runs-state /path/to/sync-state.json \
  --state /path/to/checkpoint-state.json \
  --watch --interval 60
```

A checkpoint is attached only after its completion marker and all rank manifests
agree on the contract and cursor. New attachments are read back from MLflow before
being marked attached. Existing payload files are never modified or pruned by
this sidecar. If checkpoint retention later removes a source checkpoint, its
metadata remains in MLflow and the catalog marks its source as unavailable.

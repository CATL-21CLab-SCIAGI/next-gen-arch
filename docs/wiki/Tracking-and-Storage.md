# Tracking and Storage

## Read a run

| Question | Inspect |
| --- | --- |
| What was launched? | Recipe, run contract, source/runtime identities |
| Is it progressing? | Fresh trainer progress and metric rows |
| Can it resume? | Complete checkpoint marker and all required payloads |
| What was evaluated? | Checkpoint identity, case IDs, prompt/scorer hashes |
| Is MLflow current? | Sidecar cursor and sync status |

A running DLC allocation does not prove that its trainer is alive.

## Metric semantics

Keep CE/BPB, policy loss, raw correctness, shaped reward, rollout count and optimizer count distinct. Compare loss at matched tokens and scores on the same evaluation set.

Historical backfilled timestamps are ingestion times unless original wall-clock timestamps were recorded. Rank-zero diagnostics are not global averages.

## MLflow sidecars

Metric ingestion and checkpoint attachment run independently of training. Preserve their single-writer locks and retry cursors.

Checkpoint artifacts in MLflow are references and metadata. Model, optimizer and RNG payloads remain on the recorded storage.

## Storage rules

- Keep active datasets and checkpoints protected from migration or retention.
- Verify copies before publishing or replacing paths.
- An unchanged file behind a new symlink can still disrupt an existing mmap reader.
- Preserve incomplete saves and reader leases until their owners are understood.
- Keep credentials in private runtime state, outside Git and public artifacts.

**References:** `docs/MLFLOW_TRACKING.md`, `docs/PROVENANCE.md`.

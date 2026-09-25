# MLflow tracking

**Reader guide:** [Tracking and Storage](wiki/Tracking-and-Storage.md).
Sidecars import trainer records into the existing shared service; they do not control training.

## Metric semantics

| Metric family | Meaning |
| --- | --- |
| Training CE | Exact optimizer-step coordinate |
| CE by tokens | Consumed supervised tokens used as the step axis |
| RL policy loss / reward | Separate from CE; raw correctness and shaped reward remain distinct |
| Rollout / optimizer counters | Sampling batches versus applied updates |
| 64K / 1M evaluation | Separate sealed samples; compare matching identities |
| Rank-zero diagnostics | Local observations, not global means |
| Throughput / memory | Recorded timing windows and peak allocation |

Historical backfilled timestamps are ingestion times. Live observations use observation time. Compare historical curves by step or tokens.

## Sidecars

| Entry | Responsibility |
| --- | --- |
| `archlab.tracking.mlflow_sync` | Supervised training metrics and lifecycle |
| `archlab.tracking.rl_mlflow` | RL metrics and progress |
| `archlab.tracking.mlflow_checkpoints` | Checkpoint metadata and references |

Use the existing compatible client/runtime. Credentials belong in private runtime files; never put tokens in source, shell arguments or artifacts.

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m archlab.tracking.mlflow_sync \
  --config /path/to/runs.json \
  --credentials /path/to/mlflow-credentials.json \
  --state /path/to/sync-state.json --watch --interval 30
```

Preserve the sidecar's single-writer lock and cursor. Partial final JSONL rows are ignored; nonfinite metrics and discontinuous cursors are rejected. Logging failures do not signal trainers.

## Checkpoint artifacts

`LATEST.json`, `CATALOG.json`, per-step `REFERENCE.json`, completion markers, manifests and restore notes describe available checkpoints.

These are metadata references, not weight backups. Restoration requires the payload storage, source/runtime identity and supported mesh. Continuations use linked run identities and preserve interrupted tails on their original runs.

A paused or failed MLflow run does not imply that its planned token budget completed.

## Native Miles logs

`archlab.tracking.miles_mlflow` attaches an already running Miles attempt without
changing the trainer. It preserves every finite scalar in native `step`, `perf`,
and `rollout` metric dictionaries, with the original UTC log timestamp and native
zero-based step. It does not remap transformed rewards to correctness or fabricate
missing evaluations. An exclusive lock and persistent byte cursor prevent two
writers and ordinary replay; crash recovery between remote acknowledgment and
cursor persistence is at-least-once.

```bash
PYTHONPATH=src python -m archlab.tracking.miles_mlflow \
  --root /path/to/run --log train-attempt6.log \
  --credentials /private/mlflow-credentials.json --watch --interval 60
```

The selected evidence includes actual argv, runtime versions, checkpoint-import
receipts, model configuration, qualification/readback receipts, and files placed
in `RUN_ROOT/mlflow-evidence/`. Individual files above 20 MiB are excluded.
`CHECKPOINT_REFERENCES.json` points to native checkpoint directories; model and
optimizer payloads are not uploaded. `MLFLOW_STATUS.json` records sync progress
or a sanitized error class. Optional `--dns-cache` uses the same exact-host,
process-local resolver as the existing sidecars. Credentials are never artifacts.

For the September 24–25 stock baseline, attempt 6 is tracked separately from the
failed attempts. Its attachment includes source files from the actual `757dcf4`
training revision, a compressed initial log snapshot, dataset provenance and
examples, and 640 generated samples from rollouts 0–4. These sample/log artifacts
are snapshots; native scalar metrics continue to sync. The historical
qualification record is not a live health check, and MLflow's RUNNING status
does not establish that the training process remains alive.

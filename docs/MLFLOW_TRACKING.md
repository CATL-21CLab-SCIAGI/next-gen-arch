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

# Local outputs and verified object-storage offloading

`results/`, its optional `result` alias, and `.runtime/` are local-only and ignored
by Git. Small historical reference reports live in `docs/recorded-results`;
reference data needed by the package lives in `src/archlab/data`. Runtime state,
credentials, weights, datasets, and migration journals must not be committed.

Some object-storage mounts cannot preserve Unix symlinks. The supported layout
keeps source repositories and small metadata directories on the original filesystem,
then moves large immutable payloads to a caller-selected destination and replaces
individual source files with symlinks. This preserves the canonical parent paths
used by strict training/checkpoint contracts. Restore hosts need both mounts.

The selector skips incomplete `checkpoints/step-*` directories and entire source
repositories containing `.git`. It selects supported payload formats of at least
4 MiB. Small files and metadata remain local. Use one migration plan/database per
destination namespace, with no other writers modifying that namespace.

```bash
PYTHONPATH=src python -m archlab.storage.results_plan \
  --source "$RESULTS_ROOT" --destination "$OSS_RESULTS_ROOT" --output plan.jsonl
PYTHONPATH=src python -m archlab.storage.bulk_offload \
  --plan plan.jsonl --database transfer.sqlite --workers 8
```

The offloader checks source identity, verifies SHA-256 readback, refuses differing
existing destinations, and atomically publishes each source symlink only after
verification. A SQLite journal supports restarting the same plan. Copy-only plans
use `"link": false`; sealed dataset plans can also provide an expected `sha256`.
Failures leave the original source available for recovery. Whole-directory model
migration is available through `archlab.storage.model_migration` for quiescent
model directories that do not contain symlinks.

`archlab.storage.offload_watch` can wait for an initial transfer and then select
newly completed payloads. It never pauses, kills, or restarts trainers. The local
journal and watcher status describe progress; launching a transfer does not mean
it has completed. Validate source consumers and destination mounts before large
moves. Git object cleanup is a separate operation and is not performed by these
storage utilities.

Checkpoint retention and offloading have distinct ownership rules. Do not run a
checkpoint-deletion service over an offloaded tree without accounting for its
symlink targets and in-flight transfers. The checkpoint-reference MLflow sidecar
uploads metadata only; source payload retention remains an operator policy.

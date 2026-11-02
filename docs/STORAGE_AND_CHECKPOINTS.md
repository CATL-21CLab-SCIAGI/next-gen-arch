# Local outputs and verified object-storage offloading

`results/`, its optional `result` alias, and `.runtime/` are local-only and ignored
by Git. Small historical reference reports live in `docs/recorded-results`;
reference data needed by the package lives in `src/archlab/data`. Runtime state,
credentials, weights, datasets, and migration journals must not be committed.

The project artifact root is `/mnt/oss/evergreen/arch/`, recorded in the private
`.runtime/storage-layout.json` environment reference. New artifacts belong under
`models/`, `datasets/`, `checkpoints/`, `runs/`, `sandbox-images/` or `archive/`
within that root. Existing cleanup archives are being relocated from the older
`/mnt/oss/evergreen/archive/` prefix; the sealed relocation receipts live under
`results/oss-arch-relocation-20261102/`. Shared canonical model copies outside
the project archive remain shared references, avoiding duplicate storage.

Some object-storage mounts cannot preserve Unix symlinks. The supported layout
keeps source repositories and small metadata directories on the original filesystem,
then moves large immutable payloads to a caller-selected destination and replaces
individual source files with symlinks. This preserves the canonical parent paths
used by strict training/checkpoint contracts. Restore hosts need both mounts.
Check that OSS is an actual mount before copying and again before replacing a NAS
file; an existing `/mnt/oss` directory alone is insufficient. Reuse an existing OSS
payload only after full source/destination SHA-256 agreement. A matching name,
size, or sample is only a candidate, not a deduplication proof.

The CPU sandbox-pool launcher requires an `oss` data source in its private site
settings, mounted at `/mnt/oss`; it requests read-only access and waits for that
mount before serving. Image archives can then remain in OSS while their original
NAS names remain links. Each image is unpacked to pod-local scratch as before.
Use a newly qualified source snapshot containing this launcher for future pools.
An older NAS-only pool must restore its selected archives to NAS or add the OSS
mount before it can read offloaded images. No running pool is changed by offloading.

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
Do not place an actively written SQLite transfer journal on NFS: the November 2
archive run encountered a database-lock failure there. The sealed cleanup runs
under `results/storage-execute-20261102` and `results/nas-to-oss-next-20261102` use
fsynced append-only hash receipts before replacing NAS names. Their manifests,
progress, exclusions, and restore notes are the authority for those specific runs.

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

# Data and artifact provenance

## Dataset identity

Filename and byte-size inventories are insufficient: different content can have the same size. New campaigns use a content manifest containing:

- immutable upstream dataset revision;
- sorted relative file paths;
- byte size and SHA-256 for every file;
- a canonical inventory identity.

Create and verify one with:

```bash
next-gen-arch data-manifest create \
  --root /datasets/fineweb \
  --dataset owner/dataset \
  --revision <commit-or-object-version> \
  --pattern 'fineweb_train_*.bin' \
  --pattern 'fineweb_val_*.bin' \
  --output /datasets/fineweb.manifest.json

next-gen-arch data-manifest verify \
  --root /datasets/fineweb \
  --manifest /datasets/fineweb.manifest.json \
  --mode full
```

`full` rehashes every byte. `metadata` checks the exact inventory and recorded sizes while trusting a previously verified transfer ledger. The result always records which mode was used. Existing GNU `sha256sum` ledgers are supported for the FineWeb100B relay.

Tokenizer identity hashes every token's decoded bytes and includes both logical and padded vocabulary sizes. It does not rely on a tokenizer filename.

## Source identity

Every research-policy speedrun and Megatron adapter run records:

- Git commit;
- clean/dirty state;
- tracked binary diff hash;
- untracked file names and aggregate content hash;
- combined worktree hash.

The patch itself is never embedded in the public result. Research-policy runs reject a dirty source tree.

## Run and attempt identity

`run_id` is a stable hash of the scientific contract: architecture, seed, budget, data, tokenizer, optimizer, source, topology, and batch. Repeating the same contract gives the same run ID.

`attempt_id` is a UUID for one process attempt. Preemption or an operational retry creates a new attempt while retaining the run identity. Raw metric rows carry both IDs.

## Initialization identity

`--initialization-hash shared` hashes names, shapes, dtypes, and raw bytes for every parameter also present in the paired baseline. Variant-exclusive parameters do not perturb this identity. `full` includes all parameters.

Research-policy runs require a shared or full initialization hash. Large runs should account for the one-time rank-0 CPU transfer in launch timing rather than steady-state throughput.

## Checkpoint contract

A reportable final checkpoint must contain:

- the exact final optimizer iteration;
- model, optimizer, and per-rank RNG payload;
- the immutable run and data identities;
- a reconstructable dataloader cursor.

FineWeb uses an iteration-derived distributed-microbatch cursor. If the restored optimizer iteration is `k` and the global batch requires `m` microbatches, the loader resumes at microbatch `k × m`. Tests compare this path against uninterrupted loading.

The speedrun backend writes model, optimizer, and RNG shards through temporary files and atomically publishes them. Rank 0 then rejects any bundle missing a model, metadata record, optimizer shard, or RNG shard before marking the attempt complete. Legacy checkpoints remain readable, but a research-policy resume requires the new complete bundle.

Raw metrics are hashed at completion and retained beside `resolved_run.json`, `result.json`, and the terminal marker. A report row should link these files and the checkpoint rather than copying only the final scalar.

## Storage

- NAS stores active repositories, state, logs, and resumable checkpoints.
- OSS stores immutable datasets and promoted model artifacts.
- Local workspaces store Git history, manifests, reports, and deliberately mirrored artifacts.

Never infer the owner of a shared NAS path from the path alone. The run manifest identifies both the storage namespace and compute attempt.

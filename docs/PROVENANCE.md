# Provenance reference

**Reader guide:** [Data and Provenance](wiki/Data-and-Provenance.md).

| Identity | Required contents |
| --- | --- |
| Dataset | Immutable revision; sorted relative paths, byte sizes and SHA-256 |
| Tokenizer | Native assets, decoded vocabulary identity, logical/padded sizes |
| Source | Commit, clean/dirty state and applicable implementation/worktree hashes |
| Run | Scientific contract; stable identity across unchanged operational retries |
| Attempt | Unique process-attempt identity |
| Initialization | Shared/full parameter identity and variant-specific initialization |
| Checkpoint | Final iteration, model/optimizer/RNG state and reconstructable cursor |

Research-policy speedrun and Megatron comparison runs reject dirty source. Other entries enforce their documented source and runtime contracts.

## Manifest API

Use the dataset-manifest functions in `archlab.provenance` from data preparation
code, and `verify_dataset_manifest(root, manifest, mode="full")` before launch.
The historical `data-manifest` CLI is retired.

`full` rehashes content. `metadata` validates inventory and sizes against a trusted prior transfer ledger; record which was used.

## Continuation

For iteration-derived FineWeb loading, restored optimizer iteration `k` and `m` microbatches per global update imply cursor `k × m`. Other loaders restore their explicit packing/window/prompt cursor.

Publish completion only after every required shard and payload is verified. Keep raw metrics beside the resolved contract and checkpoint reference.

## Storage

Use run manifests to identify storage ownership. Versioned compact evidence belongs in `docs/recorded-results/`; large payloads remain in declared artifact storage. Local artifact paths require the matching mount or a verified mirror.

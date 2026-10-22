# Data and Provenance

## Identities to preserve

| Item | Record |
| --- | --- |
| Data | Immutable revision, relative file inventory, sizes and content hashes |
| Tokenizer | Native assets, vocabulary identity and rendering implementation |
| Run | Architecture, objective, budget, data order, runtime and parent checkpoint |
| Attempt | Separate identity for each launch or operational retry |
| Initialization | Shared-parameter identity and variant-specific initialization |
| Checkpoint | Complete model/optimizer/RNG state and reconstructable cursor |

Metadata verification checks inventory and sizes against a trusted prior ledger. Full verification rehashes content. Record the chosen mode.

## Preprocessing rules

Use the model's native encoding. For DeepSeek V4.1, the official Python encoder defines message and supervision boundaries.

Preserve unfinished source conversations and their quality flags. Do not fabricate tool results, final answers, or EOS. Conversion and training decide different things: the training recipe selects supervision, packing, and target masks.

Only completed parts with verified payloads enter a training manifest. Group related examples across reasoning modes before splitting; exact-hash checks do not exclude paraphrases.

## Artifact locations

| Location | Intended contents |
| --- | --- |
| `docs/recorded-results/` | Versioned compact historical evidence |
| `src/archlab/data/` | Packaged frozen campaign data |
| Local `results/` | Detailed runs, checkpoints, receipts and logs |
| Local `.runtime/` | Private runtime and service state |
| OSS datasets root | Raw corpora, prompt-only RL datasets and shared source caches |
| OSS datasets root / `tokenized/` | Token-ID corpora, indexed shards, prepared SFT/RL token arrays and their staging archives |

Local run paths are artifact identifiers, not downloadable GitHub links. Teammates need the correct mounted storage or a verified mirror.

## Dataset naming and relocation

Repo-owned derivatives use lowercase kebab-case names that identify the source,
tokenizer or task, and existing revision/version. Token-count labels use `100b`.
Preserve established revisions; renaming storage does not create a new scientific
dataset version. Name staging archives after their dataset with a `-staging`
suffix. Put tokenized derivatives under `${DATASET_ROOT}/tokenized/`, preserving
their tokenizer/revision-specific leaf names. This layout covers user-owned data
across projects, not just this repository. Keep tokenizer assets with their
dataset when they are part of its verified contract.

Only reorganize folders with explicit ownership evidence. Shared upstream caches
and folders belonging to other teams retain their names. Never infer ownership
merely because a recipe reads a dataset.

The 2026-10-12 relocation established these leaf names. They now live under
`tokenized/` following the 2026-10-16 storage consolidation:

| Previous path | Leaf name under `tokenized/` |
| --- | --- |
| `base_data` | `nemotron-math-v2-qwen38-0997f410c57a-chat-v1-staging` |
| `evergreen/qwen38-27b-agentic-peft-20260829` | `qwen38-27b-agentic-peft-20260829` |
| `fineweb100B-qwen2p5-2f37459` | `fineweb-100b-qwen2p5-2f37459` |
| `fineweb100B-qwen2p5-f078881` | `fineweb-100b-qwen2p5-f078881` |
| `fineweb100B-qwen2p5-f5d8d57` | `fineweb-100b-qwen2p5-f5d8d57` |
| `fineweb-edu-100BT-qwen2p5-c0382117-v1` | `fineweb-edu-100b-qwen2p5-c0382117-v1` |
| `fineweb-edu-100BT-qwen38-de4b8e4d43b9` | `fineweb-edu-100b-qwen38-de4b8e4d43b9` |

The staging archive is `staging-20261012.tar.gz`; its manifest identifies the
original member paths for restoration. Dataset rows and binary payloads were
not transformed. Operational JSON indexes and maintained launch defaults follow
the canonical paths. Historical run receipts and frozen source remain unchanged;
replays must explicitly supply the relocated dataset paths. The two incomplete
FineWeb attempts (`2f37459`, `f078881`) remain incomplete and are not training-ready.

The local receipt `results/dataset-layout-20261012/MIGRATION.json` records
ownership evidence, complete file inventories, full SHA256 for files up to
16 MiB, and size plus first/middle/last chunk hashes for larger payloads. Original
metadata is preserved under ignored `.legacy/20261012/dataset-layout/`.

## Tokenized storage consolidation (2026-10-16)

All 22 identified user-owned tokenized dataset folders are grouped under
`${DATASET_ROOT}/tokenized/`: FineWeb GPT-2 and Qwen derivatives, indexed Nemotron
pretraining data, Nemotron Math variants, Limite agentic bootstrap data, the
GSM8K tokenization cache, and Qwen agentic PEFT segments. Existing dataset versions
and completion status are preserved. A folder's presence here does not make an
incomplete preparation training-ready.

Four older leaf names were normalized:

| Previous leaf | Canonical leaf |
| --- | --- |
| `fineweb100B` | `fineweb-100b-gpt2` |
| `fineweb10B` | `fineweb-10b-gpt2` |
| `Nemotron-Pretraining-Specialized-v1-indexed` | `nemotron-pretraining-specialized-v1-indexed` |
| `Nemotron-Pretraining-Specialized-v1-smoke-indexed` | `nemotron-pretraining-specialized-v1-smoke-indexed` |

The FineWeb download receipt and checksum lists now live inside their dataset
folders as `DOWNLOAD_COMPLETE.json`, `SHA256SUMS`, and `SHA256SUMS.partial-19`.
Checksum-list filenames are relative to the containing dataset directory.
The 10B partial checksum list remains partial.

Prompt-only MiMo RL Parquet files have no stored token IDs and remain outside
`tokenized/`. Raw corpora, retrieval indexes and model/tokenizer caches are also
separate. No dataset was re-tokenized or deleted.

The receipt `results/dataset-tokenized-layout-20261016/MIGRATION.json` records
the explicit move list and verification: every file's coverage and size, full
SHA256 for files up to 2 MiB and all JSON metadata, and first/middle/last 64 KiB
SHA256 samples for larger payloads. This is not a full rehash of the large
payloads. Operational shard indexes follow the new paths; immutable manifests
and historical provenance retain their original hashes. `PATH-MAP.json` maps
their old locations to canonical storage. Edited originals are backed up under
ignored `.legacy/20261016/dataset-tokenized-layout/`.

`tokenized/INDEX.json` lists the datasets and format/readiness notes, and
`tokenized/RELOCATIONS.json` publishes the old-to-new root map alongside the data.
The audit also updated 40 executable launch/preparation handles across Arch and
Engram under `/mnt/nas/evergreen`; `HANDLE-CHANGES.json` records the external and
local operational edits. Historical receipts and pinned snapshots remain intact.
The staging archive's restore location was updated separately; its archive bytes
and historical member paths remain unchanged.

**Reference:** `src/archlab/preprocessing/README.md`.

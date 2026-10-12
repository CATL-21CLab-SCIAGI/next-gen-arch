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
| OSS datasets root | Dataset payloads, indexes and descriptive staging archives |

Local run paths are artifact identifiers, not downloadable GitHub links. Teammates need the correct mounted storage or a verified mirror.

## Dataset naming and relocation

Repo-owned derivatives use lowercase kebab-case names that identify the source,
tokenizer or task, and existing revision/version. Token-count labels use `100b`.
Preserve established revisions; renaming storage does not create a new scientific
dataset version. Name staging archives after their dataset with a `-staging`
suffix. Put dataset folders directly under the configured OSS datasets root.

Only reorganize folders with explicit ownership evidence. Shared upstream caches
and folders belonging to other teams retain their names. Never infer ownership
merely because a recipe reads a dataset.

The 2026-10-12 relocation changed these paths, relative to the datasets root:

| Previous path | Canonical path |
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

**Reference:** `src/archlab/preprocessing/README.md`.

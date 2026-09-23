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

Local run paths are artifact identifiers, not downloadable GitHub links. Teammates need the correct mounted storage or a verified mirror.

**Reference:** `src/archlab/preprocessing/README.md`.

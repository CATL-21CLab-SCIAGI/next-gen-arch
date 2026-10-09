# Violetto Math v2 preparation

The model snapshot is `paradigma-inc/limite-1b-violetto` revision
`4cf321846e479e78d9ec35dbd39fee2e260cbad6`. The verified complete snapshot is at
`${MODEL_ROOT}/limite-1b-violetto-4cf321846e47`; `DOWNLOAD_VERIFIED.json` records
per-file SHA256 hashes. Hugging Face Git blob hashes or LFS SHA256 hashes were
checked for every upstream file. The older unversioned snapshot is preserved.

The dataset destination is
`${DATASET_ROOT}/nemotron-math-v2-limite-violetto-4cf321846e47-text-only-v2`.
Source: the five Parquet files from `nv-community/Nemotron-Math-v2`, comprising
7,085,839 source trajectories. JSONL mirrors are not ingested.

Run `archlab.preprocessing.nemotron_math --tokenizer-format limite` with explicit
`--source`, `--tokenizer`, `--stage`, and `--output` paths. The job uses the shared
indexed-document writer and native checkpoint chat template. It preserves
reasoning, adds no extra EOS token, and applies neither padding nor truncation.
It writes int32 `.bin`/`.idx` document shards and source metadata. This is a token
corpus, not an assistant-only loss-mask policy. Source effort modes remain
separate; validation uses the existing 1% normalized-problem hash split.

The approved text-only policy excludes entire conversations containing tool
schemas, calls, tool/function messages, or system prompts rejected by the native template. Each excluded row has its source file,
row offset, UUID, problem hash, source effort, and exclusion reason recorded in
`parts/*/excluded.metadata.jsonl.gz`. Empty optional call fields are removed before
native rendering; no content or reasoning is rewritten. Unexpected unsupported
content fails the job instead of being silently discarded.

Every completed part is checksum-verified after publication before its READY
marker is written. Retained plus excluded counts must exactly cover the source.
`progress.json` reports ongoing progress. Only `DATA_READY.json` signifies a fully
completed dataset; a running dataset must not be treated as complete.

Operational receipts, logs, smoke data, and a pinned preprocessing source copy
are in `results/limite-violetto-math-v2-20260928`. `full-stage-v2/launcher.json` records
the 16-worker detached CPU launch. No GPU training process is changed.

Validation: 22 focused tests and 3 subtests passed (7 optional-dependency tests
skipped), Ruff passed, and an end-to-end smoke run covered 80 source rows across
all five files: 64 retained, 16 excluded, 615,866 tokens. Independent readback
verified exact native tokens and disjoint, complete source-row coverage.

The initial v1 job stopped on source `high_part01`, row 116034, whose system
message is `Reasoning: high`; the native template rejects custom system messages.
The v2 policy records these exclusions without replacing the source prompt. It
imports only verified READY parts from the audited v1 implementation; source,
tokenizer, split, and runtime contracts must match. The failed v1 dataset remains
unchanged and must not be used as a complete dataset.

All four Paradigma snapshots are now verified under `${MODEL_ROOT}`:

- `limite-1b-base-cc612bafcd4a`
- `limite-1b-base-soup-31eca5ee49a2`
- `limite-1b-value-model-ce83740e0e85`
- `limite-1b-violetto-4cf321846e47`

`results/limite-violetto-math-v2-20260928/ALL_MODELS.json` records full revisions
and every file checksum. Tokenization uses the Violetto snapshot specifically.

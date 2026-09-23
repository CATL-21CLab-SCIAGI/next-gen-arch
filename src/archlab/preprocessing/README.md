# Nemotron Math preprocessing

**Entry:** `python -m archlab.preprocessing.nemotron_math --help`.
**Scope:** CPU conversion into verified indexed documents; training supervision remains a recipe decision.

## Rendering modes

| Mode | Contract |
| --- | --- |
| Qwen (default) | Native chat template; source high effort maps to xhigh |
| DeepSeek V4.1 | `--tokenizer-format deepseek-v41 --reasoning-effort 75`; pinned Python encoder and tokenizer |

DeepSeek has no Jinja chat template. Preserve native reasoning, tool schema and message boundaries; render tool arguments without executing them.

## Source and split

- Read Parquet sources, excluding duplicate JSONL mirrors.
- Keep each supplied conversation as one int32 sequence/document.
- Apply no truncation, extra EOD or trajectory deduplication.
- Default validation split: 1% of normalized problem-hash buckets.
- Normalization affects only the split key; variants of a problem stay together.
- Exact grouping does not detect near duplicates.

## DeepSeek schema 4

| Source condition | Handling |
| --- | --- |
| Consecutive reasoning-only assistant prefixes | Join only eligible fragments; preserve characters and record indices/hashes/separators |
| Completed adjacent answers or unresolved calls | Render individually in source order |
| Terminal assistant tool call | Preserve; mark `complete_answer: false` |
| Tool/user/system ending | Preserve native suffix; no invented answer/EOS or supervised trailing header |

`assistant_token_spans` are half-open indices in the unshifted sequence. The trainer shifts labels exactly once. Review incomplete-trajectory flags before selecting SFT targets.

## Output

```text
manifest.json                 immutable source/format/runtime contract
tokenizer/                    exact tokenizer assets
SOURCE_README.md               attribution and license
progress.json                  published-part progress
parts/<source-and-row-groups>/
  <split>-<tools>_text_document.bin
  <split>-<tools>_text_document.idx
  <split>-<tools>_text_document.metadata.jsonl.gz
  READY.json                  verified part coverage and checksums
DATA_READY.json                complete corpus and grouped prefix lists
```

Only verified ready parts are training inputs. Staging and output must be distinct. A single-writer lock protects publication.

## Recovery and testing

The same command verifies completed parts and rebuilds incomplete ones. Source, tokenizer, format and implementation identities must match; worker count and tokenization batch size may change.

`--reuse-completed-from` admits only the reviewed older schemas into new output/staging directories. It copies verified bytes and embeds original provenance; old outputs remain immutable.

`--smoke-rows N` writes `SMOKE_READY.json`, not a complete-corpus marker.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_nemotron_math_preprocessing.py
```

[Data/provenance guide](../../../docs/wiki/Data-and-Provenance.md) · [DeepSeek design record](../../../docs/DEEPSEEK_V41_GLOBAL_SIMPLICIAL_MATH.md)

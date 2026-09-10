# Nemotron-Math-v2 indexing

For DeepSeek V4.1 use `--tokenizer-format deepseek-v41 --reasoning-effort 75`
and the pinned checkpoint's tokenizer plus `encoding/encoding.py` assets. This
mode uses the official Python encoder (there is no Jinja template), preserves
all thinking, attaches tool schemas to the system message, and records half-open
`assistant_token_spans` in each metadata record. It uses the fixed numeric budget,
not the Qwen effort-name mapping below. See
`docs/DEEPSEEK_V41_GLOBAL_SIMPLICIAL_MATH.md` for the experiment contract and launch.
The default `--tokenizer-format qwen` retains the original behavior described below.

DeepSeek schema 3 joins consecutive assistant messages **only when every prefix
is a nonempty reasoning-only fragment**, without an answer, tool call, or other
meaningful fields. Reasoning characters are preserved with a recorded `\n\n`
separator; the final answer/call and following tool results are unchanged.
`message_repairs` records the original message indices, reasoning lengths and
SHA256 hashes. Other adjacent assistant messages, including completed answers and
unresolved tool calls, are **never merged**: the official encoder preserves them
individually in their original order, with a metadata event recording that choice.
Terminal assistant tool calls are retained and explicitly marked
`complete_answer: false`. No result or final answer is fabricated. “Complete
conversation” means the entire supplied source record, not a guarantee that its
teacher finished the problem. Training must review these quality flags before
choosing which targets to supervise. Unsupported roles/internal tasks and invalid
token boundaries still fail closed, with rendering errors identifying the row.

For the reviewed schema-2 output only, `--reuse-completed-from OLD_OUTPUT` imports
completed parts into **new, separate** stage/output directories. It checks the
pinned old implementation, identical source/tokenizer/split/runtime contracts,
coverage, READY markers and every payload checksum. Payloads are copied, not
hardlinked; the new READY records embed the original manifest and provenance.
Old output, manifests and code snapshots are never rewritten. Previously accepted
examples keep identical rendering and supervision; previously rejected assistant
sequences and terminal calls are added without deleting source messages. Progress
separates reused from newly tokenized
documents. Resume the new dataset with its same snapshotted code and arguments.

`python -m archlab.preprocessing.nemotron_math --help` describes the standalone,
CPU-only entry point. Supply source, tokenizer, output and staging directories
explicitly. The existing NeMo AutoModel checkout must be on `PYTHONPATH`; no
runtime packages are installed or modified. Staging and output must be distinct.

The converter reads all Parquet shards, **not** their duplicate JSONL mirror.
It uses the checkpoint's native chat template, preserves reasoning and tool
traces, decodes JSON tool-call arguments without executing them, and checks fast
tokenization against native Transformers tokenization for each observed variant
in every worker. Source `high` maps to the template's native `xhigh` setting.
It applies no truncation, deduplication, extra EOD token, or loss masking.

Each source conversation is one int32 indexed sequence and one document. Its
native message-end token and trailing template whitespace are preserved exactly.
Assistant-only supervision, packing, sequence lengths, blending and shuffling
are decisions for the future training recipe, not this conversion.

The default validation fraction is 1% of problem-hash buckets. NFC normalization
and collapsed whitespace are used **only for the split key**, not for tokenized
content. Identical normalized problems stay together across reasoning effort,
tool use and source shards. Near-duplicate problems are not detected.

Output layout:

```text
manifest.json                 # immutable source/format/runtime contract
tokenizer/                    # exact tokenizer assets, no model weights
SOURCE_README.md               # upstream provenance and licensing
progress.json                 # published-part progress and active work
parts/<source-and-row-groups>/
  <split>-<tools>_text_document.bin
  <split>-<tools>_text_document.idx
  <split>-<tools>_text_document.metadata.jsonl.gz
  READY.json                  # source coverage and verified file checksums
DATA_READY.json                # full corpus completion and grouped prefix lists
```

Metadata preserves original source row numbers, problem hashes, attribution and
answer fields. Full source rows remain available in the untouched input corpus.
`DATA_READY.json` lists dataset prefixes relative to the output root, grouped by
reasoning effort, split and tool availability. `length_bins` in each part's
partition summary are disjoint upper-inclusive token-length bins, not cumulative.

Incomplete files are not ready for training. Every part is built and checked on
NAS, copied to the destination, read back for checksum verification, and only
then marked ready. Rerun the same command to resume: completed parts are verified
and reused; incomplete parts are rebuilt. A staging lock prevents two writers.
Do not change source, tokenizer, conversion code or format arguments mid-run;
the recorded contract rejects mismatches. Worker count and tokenization batch
size can change on resume without changing the data contract.

`--smoke-rows N` limits each source file to its first N rows in the first part;
use separate temporary directories. It emits `SMOKE_READY.json`, never the full
corpus completion marker. The host-native standard-library test runner works:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_nemotron_math_preprocessing.py
```

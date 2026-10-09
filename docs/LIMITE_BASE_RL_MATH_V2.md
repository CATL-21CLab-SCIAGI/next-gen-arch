# Limite base RL math data

The existing source was copied, without modifying it, from
`${RAW_DATA_ROOT}/Nemotron-RL-Math-v2` to
`${DATASET_ROOT}/Nemotron-RL-Math-v2`. `COPY_VERIFIED.json` records full SHA256
readback checks for all five copied files.

The RL prompt dataset is published at
`${DATASET_ROOT}/nemotron-rl-math-v2-limite-1b-base-cc612bafcd4a`.
Consumers must require `DATA_READY.json` and its matching manifest hash.

- `train.parquet`: original training rows, native prompt tokens/masks, messages,
  UUIDs, expected answers, and verifier metadata.
- `prompts.jsonl`: policy inputs without reference-answer fields.
- `rewards.jsonl`: UUID-aligned answers and original verifier/request metadata.
- `restored.jsonl` and `reconstruction.jsonl`: restored NVIDIA records and the
  per-row reconstruction audit.
- `raw/`: exact source files, including the DAPO and Skywork math dependencies.
- `tokenizer/`: verified Limite base tokenizer and native chat template.

The source revision is `804418c1d4eceeaa453d895954887c0975e79121`. All 7,732
original training rows are retained. NVIDIA's unchanged `fill_placeholders.py`
restores 1,398 DAPO and 2,586 Skywork rows using the existing local datasets and
their recorded revisions/checksums. No validation split or deduplication is
silently introduced; duplicate-prompt counts are recorded in the manifest.

Tokenization uses `paradigma-inc/limite-1b-base` revision
`cc612bafcd4acd8445936201694c8c41aea8e479`, its native chat template, and
`add_generation_prompt=True`. Only question messages enter policy input tokens.
No answer tokens, padding, truncation, or additional EOS are appended. Each
record is checked against native template tokenization, followed by full Parquet
round-trip validation and checksum readback after OSS publication.

The original `math_with_judge` metadata is preserved. A compatible reward verifier
must be selected when launching RL; this preparation does not launch a judge or
training job. The reproducible entry point is
`python -m archlab.preprocessing.nemotron_rl_math --sources LOCAL_SOURCE_VERIFIED.json
--tokenizer MODEL_SNAPSHOT --stage NEW_STAGE --output NEW_OSS_OUTPUT`.

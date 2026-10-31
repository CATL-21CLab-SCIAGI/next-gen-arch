# Native agentic bootstrap

`qwen35_9b_teacher_sft.yaml` is a separate, **unqualified** one-pass native-9B
experiment composed directly with pinned verl's FSDP2 SFT trainer. It consumes
successful native-27B trajectories from `archlab.data.teacher_trajectories`.
Set `ARCHLAB_TEACHER_SFT_STUDENT`, `ARCHLAB_TEACHER_SFT_DATA`, and
`ARCHLAB_TEACHER_SFT_OUTPUT` explicitly. The 262,144-token cap preserves complete
episodes; preparation rejects longer episodes and never trains a silent prefix.
Actual-model long-sequence optimizer/distributed/resume admission is required
before production. Held-out 43-task paired evaluation and TB2.1 gate the later
short RL experiment; offline training loss does not qualify it.

The exporter requires a complete collection grid, an executable successful
verifier receipt, unambiguous task/seed session identity, and no capped turns.
It retains exact recorded token IDs and generated-token masks, with observations
as context and no invented stop tokens. Teacher-only audio special IDs fail
validation even if numerically inside the student's embedding table. Task IDs
determine the train/validation split before success selection, and all held-out
evaluation tasks are excluded. Manifests record selected membership and hashes.

```bash
python -m archlab.data.teacher_trajectories \
  --collection "$ARCHLAB_TEACHER_COLLECTION" \
  --student "$ARCHLAB_TEACHER_SFT_STUDENT" \
  --heldout "$ARCHLAB_HELDOUT_MANIFEST" \
  --output "$ARCHLAB_TEACHER_SFT_DATA" --max-length 262144
```

`limite_agentic_bootstrap.yaml` starts a new full-weight assistant-only CE
experiment from an immutable exported checkpoint. It composes pinned verl's
`sft_trainer_engine` configuration with Megatron, then delegates directly to
`verl.trainer.sft_trainer.run_sft`. Production qualification is separate.

Set `ARCHLAB_BOOTSTRAP_MODEL` to the exported normal control or simplicial
conversion, `ARCHLAB_BOOTSTRAP_DATA` to prepared native trajectories, and
`ARCHLAB_BOOTSTRAP_OUTPUT` to a new directory under ignored `results/`.
Preparation must record source, tokenizer, template and checkpoint lineage in
`manifest.json`, with `train.parquet` and `validation.parquet` alongside it.
The initial bootstrap caps complete trajectories at 16,384 tokens; preparation
and the resolved training configuration must agree. See
[native demonstration preparation](../../docs/agentic-sft-data.md) for the data
source, conversion and masking contract.
Teacher demonstrations may use another tokenizer because preparation renders
and tokenizes their text with the student contract. Direct token-logit KL
requires identical token meanings and vocabulary support.

Training verifies the export's conversion inventory, original-base download
receipt, tokenizer/template identities and prepared data checksums. It records
immutable resolved configuration, source hashes and asset receipt hashes in
`bootstrap-launch.json`; changing those inputs requires a new output directory.
Source hashes are checked again immediately before entering the upstream trainer.
The existing **Limite — Full fine tuning** project receives console and MLflow
metrics using the deployment's private tracking configuration.

Resolve the portable contract before launching:

```bash
python -m archlab.training.bootstrap resolve \
  recipes/sft/limite_agentic_bootstrap.yaml \
  --output "$ARCHLAB_BOOTSTRAP_OUTPUT"

torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m archlab.training.bootstrap train \
  recipes/sft/limite_agentic_bootstrap.yaml \
  --output "$ARCHLAB_BOOTSTRAP_OUTPUT"
```

For the matched control, provide its export and append
`variant=normal name=limite-normal-agentic-bootstrap-v1`. Overrides of upstream
settings use `settings.optim.lr=...` and the same literal paths as the recipe.

Data parallelism is eight, with one complete sequence per microbatch. Dataset
`pad_mode=no_padding` produces jagged tensors for verl's SFT loss; independently,
`engine.use_remove_padding=false` retains native BSHD execution. The prepared
loss mask marks assistant target tokens, including native turn EOS 151645,
before shifting. User, system and tool observations remain context. The upstream
loss performs the one-token shift and averages over supervised tokens. Sequences
over the explicit maximum fail instead of truncating. This is supervised CE;
success in an executable agent harness remains the acceptance criterion before RL.

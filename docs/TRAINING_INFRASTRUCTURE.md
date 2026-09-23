# Training infrastructure

**Type:** ownership reference with historical validation from 2026-09-11.

## Reusable boundaries

| Responsibility | Module |
|---|---|
| File SHA-256 and atomic JSON publication | `archlab.artifacts` |
| Existing JSON-content identity formats | Their original owners, including `archlab.provenance` |
| Exact DATA_READY membership checks | `archlab.megatron.indexed_data` |
| Raw-int32 ordering, cursor/repeat logic, DP prefix partitioning | `archlab.megatron.token_batches` |
| Megatron API-version dispatch and iteration/rank access | `archlab.megatron.lifecycle` |
| Legacy averaged loss vs native summed/token-count ABI | `archlab.megatron.losses` |
| Flash-Next native model/optimizer tagging | `archlab.megatron.qwen38_flash_next_model` |
| Flash-Next argument translation | `archlab.megatron.qwen38_flash_next_config` |
| Native bounded host checkpoint staging | `archlab.megatron.checkpoint_staging` |
| Flash-Next distributed qualification checks | `archlab.megatron.qwen38_flash_next_checks` |
| PIQA evaluation | `archlab.evaluation.qwen38_piqa` |
| Early curve reporting | `archlab.reporting.qwen38_early_curves` |

`invoke_pretrain` requires an explicit forward-step callback. Legacy pre-averaged loss and native summed/token-count loss remain distinct ABIs.

## Launch configuration

`recipes/launches/` supplies allowlisted configuration; machine paths are launch overrides. Output names do not choose a model.

```bash
export NGA_LAUNCH_RECIPE="$NGA_REPO_ROOT/recipes/launches/qwen38_flash_next_w320_e32.yaml"
bash "$NGA_REPO_ROOT/scripts/run_qwen38_flash_next_full_dlc.sh"
```

This requires the declared existing topology, source, data, tokenizer and output settings. It is not an allocation command.

`scripts/lib/dlc_runtime.sh` handles source checks, container setup and torchrun construction. It installs no packages.

## Compatibility

| Boundary | Preserved behavior |
| --- | --- |
| Data | Raw-int32 ordering, wrap/resume/repeat and DP prefix ownership |
| Speedrun | Frozen precision, packing and tokenizer semantics |
| Serialization | Each caller's canonical hashing and JSON policy |
| Checkpoints | Parameter keys, optimizer rules and explicit continuation gates |

Shared construction/data helpers moved to neutral modules. Frozen ClimbMix packing and tokenizer behavior still have their original owner.

## Refactor validation

At the recorded staged source: lint, compile, manifest verification and portable launch rendering passed. CPU tests: **484 passed, 50 skipped, one pre-existing failure**, plus 11 passing subtests. The failure was an unavailable AIME grading module and reproduced on the prior source.

These checks established refactor compatibility, not new distributed-model support. Later DeepSeek admission is documented [separately](DEEPSEEK_V41_OFFICIAL_TRAINING.md).

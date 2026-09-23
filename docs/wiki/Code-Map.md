# Code Map

Project Python lives in `src/archlab`. The distribution is `next-gen-arch`; imports use `archlab`.

## Ownership

| Directory | Owns |
| --- | --- |
| `architectures/` | Model definitions and independent mechanisms |
| `optimizers/` | Project optimizer extensions |
| `speedrun/` | Frozen small-model reference execution |
| `megatron/` | Megatron integration and pretraining |
| `automodel/` | Pretrained loading, FSDP/EP integration, fine-tuning and RL execution |
| `rl/` | Rollouts, rewards, policy objectives and evaluation primitives |
| `preprocessing/` | Dataset conversion and preparation |
| `evaluation/` | Dataset selection, scoring and paired statistics |
| `serving/`, `serving/sglang/` | Inference integration, export and project model plugins |
| `tracking/` | Metric ingestion and operational watchers |
| `reporting/` | Shared run readers, token matching and analysis |
| `qualification/` | Independent mechanism numerical and performance tools |
| `storage/` | Verified artifact migration and storage operations |
| `prompts/` | Versioned reusable YAML prompts |
| `data/` | Packaged frozen campaign evidence |

## Design rules

- Execution adapters depend on architecture definitions.
- Architecture modules do not import trainers or optimizers.
- Import concrete modules; keep package initializers small.
- A numerical oracle or distributed mechanism may have its own module.
- Container frameworks remain external.

`serving/sglang/` holds our model extension, not a copy of SGLang. The engine registers it through `SGLANG_EXTERNAL_MODEL_PACKAGE=archlab.serving.sglang`.

## Find an entry point

Use the recipe and model-specific report before selecting a launcher. A file named `probe` or `qualification` is a test entry, not a training run.

[Browse source](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/tree/main/src/archlab) · [Browse recipes](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/tree/main/recipes)

## Model integration layout

- `automodel/qwen/`: Qwen construction, fine-tuning, sampling and backend evaluation.
- `automodel/deepseek_v41/qualification/`: independent DeepSeek probes, kept as numerical oracles.
- `automodel/deepseek_v41_*.py`: execution and checkpoint-bound lineage. Recorded parent hashes still identify these paths; they are retained until an explicit checkpoint migration is qualified.
- `serving/sglang/`: the external SGLang model registration package. Numerical bridges and checkpoint conversion remain in `serving/`.

The [2026-09-23 cleanup record](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/blob/main/docs/REPO_CLEANUP_20260923.md) maps moved CLI entries and explains retained deployment dependencies.

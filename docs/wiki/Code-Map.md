# Code Map

Project Python lives in `src/archlab`. The distribution is `next-gen-arch`; imports use `archlab`.

| Directory | Owns |
| --- | --- |
| `training/` | Typed experiment contracts, model integration, admission and thin verl entry points |
| `architectures/` | Model definitions, attention kernels and numerical references |
| `megatron/` | Megatron/Transformer Engine integration and retained pretraining |
| `optimizers/` | Architecture-specific optimizer extensions |
| `data/`, `preprocessing/` | Reproducible indexes, conversion and provenance |
| `evaluation/` | Benchmark selection, scoring and paired statistics |
| `reporting/`, `tracking/` | Evidence readers, analysis and metric ingestion |
| `storage/`, `serving/` | Verified artifact operations, export and inference utilities |
| `qualification/` | Independent attention/decode numerical checks |
| `automodel/` | Retained numerical, loading and checkpoint references for historical models |
| `speedrun/` | Frozen small-model reference and primitives still used by Megatron |
| `prompts/` | Reusable versioned prompts |
| `third_party/verl/` | Pinned upstream trainer, rollouts, checkpoints and monitoring |

## Maintained execution

Use `archlab resolve`, `archlab qualify` and `archlab train`; see the [training guide](../TRAINING.md). verl owns RL execution. Architecture modules stay independent of trainers and optimizers. Megatron distributes training; TileLang supplies attention kernels. These are separate configuration choices, each requiring evidence for the actual model combination.

## Retirement boundary

The obsolete AutoModel Qwen/DeepSeek trainers, Limite custom warmup and math-evaluation queues, old campaign orchestration, and dedicated operational watchers were retired on 2026-10-17. Their numerical implementations and relevant tests remain; a retained reference module is not an admitted production launcher.

Historical recipes and reports preserve scientific settings and results. Their old commands require the recorded immutable source revision; they do not identify current entry points. Production snapshots under ignored `results/` remain independent of this working tree.

Checksum-verified originals of the retired files are kept locally in `.legacy/20261017-execution-retirement/`, with a manifest recording source revision and file hashes. Mixed test files were backed up before removing tests for retired executors. Maintained code never imports this archive. For historical reproduction, use an isolated checkout or snapshot instead of restoring compatibility modules into the maintained package.

[Browse source](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/tree/main/src/archlab) · [Browse recipes](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/tree/main/recipes)

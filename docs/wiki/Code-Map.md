# Code Map

Project Python lives in `src/archlab`. The distribution is `next-gen-arch`; imports use `archlab`.

| Directory | Owns |
| --- | --- |
| `training/` | Typed experiment contracts, model integration, admission and thin verl entry points |
| `architectures/` | Limite preludes, TileLang attention kernels and numerical references |
| `megatron/` | Native Limite bridge for verl's Megatron engine |
| `data/`, `preprocessing/` | Reproducible indexes, conversion and provenance |
| `evaluation/` | Benchmark selection, scoring and paired statistics |
| `reporting/`, `tracking/` | Evidence readers, analysis and metric ingestion |
| `storage/`, `serving/` | Verified artifact operations, export and inference utilities |
| `qualification/` | Independent attention/decode numerical checks |
| `prompts/` | Reusable versioned prompts |
| `third_party/verl/` | Pinned upstream trainer, rollouts, checkpoints and monitoring |

## Maintained execution

Use `archlab resolve`, `archlab qualify` and `archlab train`; see the [training guide](../TRAINING.md). verl owns RL execution. Architecture modules stay independent of trainers and optimizers. Megatron distributes training; TileLang supplies every project-owned kernel. These are separate configuration choices, each requiring evidence for the actual model combination.

## Retirement boundary

The obsolete AutoModel Qwen/DeepSeek trainers, Limite custom warmup and math-evaluation queues, old campaign orchestration, and dedicated operational watchers were retired on 2026-10-17. Their numerical implementations and relevant tests remain; a retained reference module is not an admitted production launcher.

Historical recipes and reports preserve scientific settings and results. Their old commands require the recorded immutable source revision; they do not identify current entry points. Production snapshots under ignored `results/` remain independent of this working tree.

Checksum-verified originals of the retired files are kept locally in `.legacy/20261017-execution-retirement/`, with a manifest recording source revision and file hashes. Mixed test files were backed up before removing tests for retired executors. Maintained code never imports this archive. For historical reproduction, use an isolated checkout or snapshot instead of restoring compatibility modules into the maintained package.

On 2026-11-01, the unused Qwen request-file controller, fixed matched-reasoning watcher, PIQA script shim, misnamed Flash-Next forwarding launcher and orphan Miles argv fixture were retired. Their dedicated tests were removed; PIQA identity tests now import `archlab.evaluation.qwen38_piqa` directly. Originals of every changed or deleted file are checksum-verified in ignored `.legacy/20261101-code-compact/`, with `MANIFEST.json` recording hashes and the source revision.

On 2026-11-02 the codebase became TileLang-only and was cut to the active Limite, agentic-RL and teacher-SFT lines. Retired: speedrun, Megatron scratch pretraining and its launchers, Qwen3.8 (including PIQA), DeepSeek-V4.1 scratch/AutoModel/evaluation/serving, the kimi/frontier/sota/fog/dsa/engram/linsimp/prelude-loop families, Megatron simplicial pilots, Qwen3.5 architecture-switch distillation, the orphaned AutoModel Limite trainer with its warmup optimizer and static-context cache, the FA4 normal-attention backward, and the Triton simplicial kernels (TileLang now owns training, decode and paged serving). Originals, their recipes and dedicated tests are checksum-verified in ignored `.legacy/20261102-tilelang-only/MANIFEST.json`. Running jobs continue from their immutable source snapshots.

[Browse source](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/tree/main/src/archlab) · [Browse recipes](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/tree/main/recipes)

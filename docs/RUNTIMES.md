# Runtime and scaling architecture

The repository separates an experiment contract from the code that executes it. This keeps the published speedrun path reproducible while providing a clean route to Megatron-LM scale.

## Source layout

```text
src/next_gen_arch/
├── arch/       architecture definitions only, consolidated by family
├── training/   training, data, optimizer, kernels, evaluation, and runtime
├── backends/   command renderers and backend safety checks
├── prompts/    versioned prompt assets shared by train/eval
├── spec.py     layered portable YAML composition
├── registry.py frozen campaign registry
└── results.py  published-result integrity checks
```

The Python import name uses an underscore because package identifiers cannot contain a hyphen. The distribution and repository remain `next-gen-arch`.

## Speedrun backend

`speedrun` is the direct descendant of the campaign code. Architecture definitions live only in `next_gen_arch.arch`; Muon/Adam parameter grouping, attention kernels, static-shape compilation, data-order controls, BF16 handling, and execution utilities live in `next_gen_arch.training`. This boundary changes module paths, not the frozen run arguments or model behavior.

The optimizer lineage includes the compact Muon work inherited from `KellerJordan/modded-nanogpt` at recorded commit `f411b3d346aa52d3504324ca93c230fd84c6c07f`. The historical nanochat campaign tree remains separately identified in the frozen manifest.

## Megatron backend

Megatron-LM is a real Git submodule at `third_party/Megatron-LM`, pinned to:

```text
https://github.com/NVIDIA/Megatron-LM.git
55ac7082517c3878ae653c07c09c534b8aed49f6
```

The project never edits files inside the submodule. `next_gen_arch.backends.megatron` validates that the checkout exists, matches the lock, and is clean before rendering a command. All project-specific adapters belong in `src/next_gen_arch`, following the same read-only-upstream boundary used by the `llm-arch-lab` reference design.

Release wheels and source archives intentionally exclude the Megatron-LM working tree. Use a recursive Git checkout for the Megatron backend; packaged installations contain the portable specifications and adapters, not a vendored copy of upstream code.

## Why Marin is a source, not a third backend

[Marin](https://github.com/marin-community/marin) is stronger as an end-to-end open
research program: it combines data production, experiment orchestration, Levanter/JAX
training, evaluations, reports, and reproducible experiment definitions. This project
is deliberately narrower. It needs compact PyTorch architecture definitions, a very
fast small-model screen, and a CUDA scaling path that can inherit Megatron's active
tensor/pipeline/context/expert-parallel work.

Adding Marin/Levanter as another runtime would introduce a second optimizer,
checkpoint, sharding, kernel, and cluster-control surface without replacing the B300
Megatron requirement. It would also make paired attribution harder. The project instead
imports Marin's useful research layer—experiment gates, negative-result accounting,
z-loss, clipping, PKO/partial-RoPE, residual reuse, MuonH/MuonEq hypotheses, and durable
reporting—as explicit recipes and architecture arms. The complete adoption/rejection
ledger is in [OPTIMIZATION_AUDIT.md](OPTIMIZATION_AUDIT.md). This decision can change if
a concrete Marin component beats the maintained backend under a matched contract; a
general “Marin backend” is not currently justified.

There are two intentionally distinct Megatron integration levels:

1. `next_gen_arch.backends.megatron` is the portable scaling renderer. It currently exposes only the upstream MCore baseline and can render tensor, pipeline, and context parallel topology. A custom variant remains capability-gated until it receives a native parallel adapter, checkpoint semantics, initialization alignment, and distributed numerical tests.
2. `next_gen_arch.training.megatron_train` is the controlled architecture-comparison wrapper. It keeps architecture math and the historical Muon/Adam grouping in this repository while delegating initialization, DDP accumulation, the pretrain lifecycle, scheduling, finite checks, and reporting to pinned Megatron. All 16 variants completed the approximately 10M, three-seed comparison through this path at `TP=PP=CP=1`.

The second result validates construction and training under the MCore lifecycle; it does not bypass the first layer's scale-readiness gate. In particular, it is not evidence that every custom attention or recurrent mechanism already shards correctly across tensor or context parallel ranks.

The frozen small-model campaign is defined in `training/campaigns.py`.
`training/campaign_runner.py` distributes its Cartesian run set across nodes and accepts
only an explicit physical-GPU allowlist. It records queue state durably and rechecks
each allowlisted GPU's UUID immediately before every launch, refusing to start if a
pre-existing process occupies that card. `--partition-strategy seed` preserves
simultaneous seed pairing; `variant` keeps a variant's seeds on one queue so compiler
artifacts are reused. A restarted runner reuses an existing complete result only after
checking its variant, seed, mode, backend profile, and recipe; mismatches fail closed.

## Portable configuration

Experiment YAML files compose in order:

```text
base → backend → scale → experiment
```

Committed paths must be one of:

- `env:NAME`, resolved only at launch-plan generation;
- `package:relative/path`, for versioned assets such as prompts;
- a relative path resolved from the declaring experiment file.

Absolute machine paths are rejected. An operator may override a declared path without editing YAML:

```bash
next-gen-arch render --config CONFIG --path output_dir=/mnt/run/example
```

For multi-node commands, the rendered launch plan leaves `NODE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` as explicit environment references. Scheduler-specific submission remains outside the experiment contract.

## Prompt portability

Training and evaluation load the same schema-versioned prompt set. The default is `src/next_gen_arch/prompts/smoke.yaml`; `--prompt-file` selects a different YAML file. Prompt text is no longer duplicated inside training and evaluation code, so moving a run between backends or machines does not silently change qualitative probes.

## What is and is not guaranteed

- A frozen speedrun spec reproduces the original command fields, with only the Python module path updated for the `src` layout.
- A Megatron spec guarantees pinned upstream source, explicit model/data/parallelism arguments, portable paths, and an inspectable launch plan.
- The 10M comparison wrapper guarantees a frozen 16-variant contract, three matched seeds, per-run source provenance, and single-rank MCore lifecycle coverage.
- The two backends do not promise bitwise, optimizer, throughput, or checkpoint equivalence.
- Cross-backend quality claims require a separately frozen paired contract and backend-specific validation evidence.

The accepted 96-run comparison and its DSA correction audit are published in [BACKEND_COMPARISON.md](BACKEND_COMPARISON.md).

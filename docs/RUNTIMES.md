# Runtime and scaling architecture

The repository separates an experiment contract from the code that executes it. This keeps the published speedrun path reproducible while providing a clean route to Megatron-LM scale.

## Source layout

```text
src/archlab/
├── architectures/ model and layer composition only
├── optimizers/    local optimizer extensions and recipes
├── megatron/      the sole ArchLab → Megatron adaptation boundary
├── speedrun/      frozen small-scale reference implementation
├── prompts/       versioned prompt assets shared by train/eval
├── launch.py      inspectable backend launch plans
├── spec.py        layered portable YAML composition
└── registry.py    frozen campaign and result integrity
```

The Python import name is the short, stable `archlab`; the distribution and repository
remain `next-gen-arch`.

The layout deliberately optimizes dependency direction, not a fixed file count. A
review of Marin at commit `0a0b91977c6fed61d6495ed1ac54ed1f813cebd1`
reinforced that its maintainability comes from owned subsystem boundaries and local
tests, not from mechanically merging modules. Architecture code cannot import a
trainer or optimizer, package initializers are not re-export registries, and an
independent leaf mechanism may keep its own module when it has a numerical or
distributed contract.

## Speedrun backend

`speedrun` is the direct descendant of the campaign code. Architecture definitions live
only in `archlab.architectures`; Muon/Adam parameter grouping is exposed through
`archlab.optimizers`, while the frozen attention kernels, static-shape compilation,
data-order controls, BF16 handling, and execution utilities live in `archlab.speedrun`.
This boundary changes module paths, not the frozen run arguments or model behavior.

The optimizer lineage includes the compact Muon work inherited from `KellerJordan/modded-nanogpt` at recorded commit `f411b3d346aa52d3504324ca93c230fd84c6c07f`. The historical nanochat campaign tree remains separately identified in the frozen manifest.

## Megatron backend

Megatron Core is provided by the execution environment, not by a Git submodule or a
project dependency. The validated PAI container profile is `nemo-26.06`. That profile
owns the mutually compatible PyTorch, CUDA, NCCL, Transformer Engine, NeMo, and
Megatron versions; project installation must not resolve or replace them.

`archlab.megatron.backend` locates `megatron` through Python's installed-package
mechanism and records its distribution version and package path. The portable native
renderer launches `python -m archlab.megatron.backend` under `torchrun`; that stable
module delegates to the system `pretrain_gpt.py`. A nonstandard container may point to
its entrypoint with `NGA_MEGATRON_PRETRAIN`. The controlled architecture wrapper needs
only the installed MCore package and therefore does not require that script.

Run `next-gen-arch doctor --backend megatron` inside the target container before
training. A rendered plan is intentionally inspectable on CPU systems without
Megatron installed, while execution fails closed when the package, distribution
metadata, or native entrypoint is absent. Release wheels and source archives contain
only project adapters; they never contain Megatron or NeMo source.

Historical benchmark artifacts remain tied to Megatron-LM commit
`55ac7082517c3878ae653c07c09c534b8aed49f6`. That commit is provenance for those runs,
not a source checkout in the current repository.

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

1. `archlab.megatron.backend` is the portable scaling renderer. It currently exposes only the upstream MCore baseline and can render tensor, pipeline, and context parallel topology. A custom variant remains capability-gated until it receives a native parallel adapter, checkpoint semantics, initialization alignment, and distributed numerical tests.
2. `archlab.megatron.train` is the controlled architecture-comparison wrapper. It keeps architecture math and the historical Muon/Adam grouping in this repository while delegating initialization, DDP accumulation, the pretrain lifecycle, scheduling, finite checks, and reporting to the validated system Megatron. All 16 variants completed the approximately 10M, three-seed comparison through this path at `TP=PP=CP=1`.

The 100M baseline subsequently completed one real 15-way data-parallel run over three
B300 nodes through the same wrapper, with NCCL RoCE/GDRDMA and exact 192-sequence batch
replay. This validates multi-node data parallelism for the baseline. It does not bypass
the first layer's scale-readiness gate or show that every custom attention or recurrent
mechanism shards correctly across tensor or context parallel ranks.

The native Megatron renderer also completed a single-node 100M-class topology matrix:
DP5, PP5, matched DP4/TP2, matched fused-DP4/CP2, and matched MoE EP1/EP2. All eight
200-step runs were finite. At this scale PP5, TP2, and CP2 reached only 0.617×, 0.389×,
and 0.366× their matched data-parallel throughput; EP2 reached 0.980× EP1 while saving
7.1% peak allocated memory. Transformer Engine fused attention was the useful speed
result at 1.274× unfused DP4. These are capacity capabilities, not evidence that model
parallelism should be enabled for a model that already fits comfortably in data
parallelism. The full contract and curves are in
[results/100m-native-parallelism-b300-1n](../results/100m-native-parallelism-b300-1n/).

The frozen small-model campaign is defined in `archlab.speedrun.campaigns`.
`archlab.speedrun.campaign_runner` distributes its Cartesian run set across nodes and accepts
only an explicit physical-GPU allowlist. It records queue state durably and rechecks
each allowlisted GPU's UUID immediately before every launch, refusing to start if a
pre-existing process occupies that card. `--partition-strategy seed` preserves
simultaneous seed pairing; `variant` keeps a variant's seeds on one queue so compiler
artifacts are reused. A restarted runner reuses an existing complete result only after
checking its variant, seed, mode, backend profile, and recipe; mismatches fail closed.

`compile-safe-autotune` is the measured B300 policy for this architecture grid. It uses
`max-autotune` for 13 stable variants and default `torch.compile` for KDA, Kimi K3 KDA,
and Qwen GDN. Those three overrides are numerical-safety requirements, not quality
recipes: max-autotune caused first-backward non-finite gradients or severe quality
corruption, while all nine default-compile controls were finite. Every result records
both the named profile and its resolved compile mode. The campaign aggregator keeps
failed-key recovery separate from explicit replacement, rejecting overlap in either
direction.

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

Training and evaluation load the same schema-versioned prompt set. The default is `src/archlab/prompts/smoke.yaml`; `--prompt-file` selects a different YAML file. Prompt text is no longer duplicated inside training and evaluation code, so moving a run between backends or machines does not silently change qualitative probes.

## What is and is not guaranteed

- A frozen speedrun spec reproduces the original command fields, with only the Python module path updated for the `src` layout.
- A Megatron spec guarantees pinned upstream source, explicit model/data/parallelism arguments, portable paths, and an inspectable launch plan.
- The 10M comparison wrapper guarantees a frozen 16-variant contract, three matched seeds, per-run source provenance, and single-rank MCore lifecycle coverage.
- The 100M baseline has separate evidence for 15-way, three-node MCore data parallelism with exact non-divisible-batch handling; it remains `TP=PP=CP=1`.
- The two backends do not promise bitwise, optimizer, throughput, or checkpoint equivalence.
- Cross-backend quality claims require a separately frozen paired contract and backend-specific validation evidence.

The original and safe-autotune 96-run comparisons, DSA correction, compiler recovery,
and throughput-basis audits are published in
[BACKEND_COMPARISON.md](BACKEND_COMPARISON.md).
The current backend ownership and speedrun-retirement gates are recorded in
[BACKEND_RETENTION.md](BACKEND_RETENTION.md).

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

The initial Megatron adapter deliberately supports only the upstream MCore baseline. A speedrun variant cannot be selected on Megatron until its model construction, parameter grouping, checkpoint semantics, initialization alignment, and numerical tests are implemented. This capability gate prevents false cross-backend comparisons.

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
- The two backends do not promise bitwise, optimizer, throughput, or checkpoint equivalence.
- Cross-backend quality claims require a separately frozen paired contract and backend-specific validation evidence.

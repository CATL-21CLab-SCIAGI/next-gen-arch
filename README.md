# Next-Gen Architecture Lab

Controlled experiments on attention, memory and reasoning in language models.

[中文研究愿景](https://catl-21clab-sciagi.github.io/next-gen-arch/zh/) · [Research wiki](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/wiki) · [Recorded results](docs/RESULTS.md)

## Structure

- `src/archlab/architectures`: model definitions, attention mechanisms and numerical kernels.
- `src/archlab/training`: typed experiment contracts, model integration and a thin verl launcher.
- `src/archlab/data`: reproducible dataset indexes and provenance.
- `recipes/rl`: the current agentic pilot and its data contract.
- `third_party/verl`: pinned XiaomiMiMo trainer, agent harness, rollout contracts and monitoring.

Megatron/Transformer Engine and TileLang have different responsibilities: Megatron distributes training, while TileLang implements attention kernels. Support requires model construction, numerical, optimizer, checkpoint and distributed tests for the particular combination. Selecting a configuration does not establish support.

The Limite agentic migration is under qualification. Its lossless model export and code-data index are implemented; an executable task sandbox and GPU integration admission are required before training. The new launcher rejects incomplete qualification.

## Run

Use the prepared GPU container's PyTorch, CUDA, NCCL, Transformer Engine and Megatron packages. Initialize the agent dependencies:

```bash
git submodule update --init --recursive third_party/verl
```

Apply the declared [upstream patches](docs/TRAINING.md#upstream-patches) before qualification.

Set the model export, parent checkpoint, code-data and output paths documented in [the training guide](docs/TRAINING.md), then:

```bash
archlab resolve recipes/rl/limite_code.yaml --output results/code-pilot
archlab train recipes/rl/limite_code.yaml --output results/code-pilot
```

`train` runs admission before launching. Use `qualify` instead for checks without
training; running both repeats the checks.

The launcher composes the pinned upstream configuration. verl owns training, rollouts, checkpoints, MLflow and performance monitoring. Project code preserves the experimental model and verifies its contracts.

Core model definitions, retained SFT/scratch integrations and the frozen speedrun reference remain. Superseded Miles RL, DeepSeek native RL and SGLang prototype execution stacks have been retired. Retired experiment launchers and diagnostics are stored only in the ignored local `.legacy/` archive; production snapshots and checkpoint states remain independent of the working tree.

[Apache 2.0](LICENSE) · [Attribution](docs/NOTICE.md) · [Citation](CITATION.cff)

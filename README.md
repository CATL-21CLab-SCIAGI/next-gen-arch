# Next-Gen Architecture Lab

Controlled experiments on attention, memory and reasoning in language models.

[中文研究愿景](https://catl-21clab-sciagi.github.io/next-gen-arch/zh/) · [Research wiki](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/wiki) · [Recorded results](docs/RESULTS.md)

## Structure

- `src/archlab/architectures`: model definitions, attention mechanisms and numerical kernels.
- `src/archlab/training`: typed experiment contracts, model integration and a thin verl launcher.
- `src/archlab/data`: reproducible dataset indexes and provenance.
- `recipes/`: portable contracts grouped into pretraining, distillation, SFT, RL, evaluation and frozen references.
- `tests/`: verification grouped by code domain; deployment probes stay in `tests/integration`.
- `scripts/`: operational launchers, evaluation setup and pinned upstream patches.
- `third_party/verl`: pinned XiaomiMiMo trainer, agent harness, rollout contracts and monitoring.

Megatron/Transformer Engine and TileLang have different responsibilities: Megatron distributes training, while TileLang implements attention kernels. Support requires model construction, numerical, optimizer, checkpoint and distributed tests for the particular combination. Selecting a configuration does not establish support.

Production runs use immutable source snapshots and keep their checkpoint state
independent of this checkout. Qwen3.5 recipes preserve the recorded configuration;
the changed deployment still requires native GPU admission. Limite SFT and RL
have separate experiment contracts. Muse admission remains incomplete where its
native serving/context/throughput probes are missing.

## Run

Use Python 3.12 and the prepared GPU container's PyTorch, CUDA, NCCL, Transformer Engine and Megatron packages. Initialize the agent dependencies:

```bash
git submodule update --init --recursive third_party/verl
```

Apply the declared [upstream patches](docs/TRAINING.md#upstream-patches) before qualification.

Set the model, data and runtime paths documented in [the training guide](docs/TRAINING.md), then resolve the Qwen control:

```bash
archlab resolve recipes/rl/qwen35_code.yaml \
  --overlay recipes/rl/qwen35_code_local.yaml --output results/qwen-control
```

Use `qwen35_code_efficiency.yaml` for the second arm. It also changes the batch
size and curriculum, so this is not a pure systems comparison. `qualify` checks
admission without launching; `train` requires admission before delegating to verl.
The cleaned Qwen deployment currently remains blocked on GPU qualification.

The launcher composes the pinned upstream configuration. verl owns training, rollouts, checkpoints, MLflow and performance monitoring. Project code preserves the experimental model and verifies its contracts.

Core model definitions, numerical references, Megatron integration and the frozen speedrun reference remain. Superseded trainers, queues and probes are stored only in the checksum-verified, ignored `.legacy/` archive. Production snapshots and checkpoint states remain independent of the working tree.

[Apache 2.0](LICENSE) · [Attribution](docs/NOTICE.md) · [Citation](CITATION.cff)

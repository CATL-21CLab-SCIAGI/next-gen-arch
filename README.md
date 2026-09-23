# Next-Gen Architecture Lab

**Architecture search for language models—from controlled component experiments to distributed pretraining, frontier-model adaptation, and reinforcement learning.**

[Documentation](docs/README.md) · [GitHub Wiki](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/wiki) · [Results](docs/RESULTS.md) · [Recipes](recipes/) · [Contributing](CONTRIBUTING.md)

## Research

| Track | Scope |
| --- | --- |
| Components | Attention, recurrent mixers, memory, residual routing, positional encoding |
| Models and scale | Dense and MoE Qwen families; DeepSeek scratch models |
| Pretrained adaptation | Ordinary and simplicial architecture adapters; frozen-backbone and full-weight fine-tuning |
| RL | Verifiable rewards, policy optimization, trajectory completion and generation efficiency |

We compare capability, stability, memory, and training/inference cost under explicit experiment contracts. Completed evidence and experimental proposals are labeled separately.

## Execution

| Integration | Role |
| --- | --- |
| `archlab.speedrun` | Frozen small-model reference |
| `archlab.megatron` | Distributed pretraining and scaling |
| `archlab.automodel` | Pretrained-model integration and post-training |
| `archlab.serving` | Inference and evaluation integration |

GPU work uses the validated container's existing runtime packages. Backend support is qualified per model, mechanism, and topology.

Qwen integration is under `automodel/qwen/`; independent DeepSeek diagnostics are under `automodel/deepseek_v41/qualification/`. Checkpoint-bound DeepSeek execution files retain their recorded paths. Dataset scoring lives in `evaluation/`, run analysis in `reporting/`, operational watchers in `tracking/`, and checkpoint retention in `storage/`. SGLang model registration uses `archlab.serving.sglang`.

## Explore

From a prepared environment at the repository root:

```bash
PYTHONPATH=src python -m archlab.cli verify
PYTHONPATH=src python -m archlab.cli list
```

The CLI above inspects frozen campaign evidence. Model-specific entries and recipes are mapped in the [documentation index](docs/README.md).

For paired training curves, use `python -m archlab.reporting.paired_curves --help` with explicit run directories. See the [cleanup and CLI migration record](docs/REPO_CLEANUP_20260923.md) for moved commands and retained historical paths.

Compact evidence is versioned under `docs/recorded-results/` and `src/archlab/data/`. Large run artifacts and checkpoints stay outside Git.

[MIT license](LICENSE) · [Attribution](NOTICE.md) · [Citation](CITATION.cff)

# Next-Gen Architecture Lab

[简体中文](README.zh-CN.md) · [Results](docs/RESULTS.md) · [Experiment contracts](docs/EXPERIMENT_CONTRACTS.md) · [Provenance](docs/PROVENANCE.md) · [Operations](docs/OPERATIONS.md) · [Runtimes](docs/RUNTIMES.md) · [Architectures](docs/ARCHITECTURES.md)

[![CI](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/actions/workflows/ci.yml/badge.svg)](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

A controlled architecture lab for small, rapid PyTorch screens and larger Megatron-LM scaling runs. The repository keeps architecture mechanisms, optimizer changes, data order, evaluation, and execution backends as separate experimental axes.

## Questions this repository can answer

Every new run declares one comparison regime:

| Regime | Held fixed | Question |
| --- | --- | --- |
| `controlled` | training tokens, seed, data order, optimizer, evaluation | does one component improve quality? |
| `fixed_compute` | total algorithmic model FLOPs | does it improve quality per unit of compute? |
| `scaling` | tokens per parameter | how does the architecture scale as a system? |

Results from different regimes are never merged into one causal leaderboard. Parameter overhead, executed FLOPs, throughput, memory, failures, and seed variation remain visible.

## Backends

- **`speedrun`** is the frozen nanochat/modded-nanogpt-derived comparison backend. It preserves the published Muon, compilation, packing, and data-order behavior.
- **`megatron`** uses the Megatron Core runtime supplied by the validated `nemo-26.06` container. It owns distributed execution and checkpoint lifecycle; this repository does not vendor or patch Megatron.

Architecture support is capability-gated. A speedrun implementation is not described as Megatron-native until construction, optimizer grouping, checkpointing, and distributed behavior have dedicated tests.

## Evidence

The repository contains paired three-seed controls, parameter-scaling campaigns, backend comparisons, numerical failures, and machine-readable learning curves. The important current conclusions are:

- Qwen GDN gives strong quality at completed mature scales but has a large throughput cost.
- Engram is the strongest observed quality/throughput trade-off at 100M and 300M.
- KDA-family mechanisms often improve BPB but are substantially slower in the current implementation.
- small improvements can change across backend or scale; combinations are promoted only after their components pass isolated controls.
- mHC and relative-attention failures are retained as failures, not replaced by earlier favorable checkpoints.

Exact tables, dates, caveats, and artifact locations are in [docs/RESULTS.md](docs/RESULTS.md). Frozen machine-readable evidence remains under [`results/`](results/).

## Research-grade artifact contract

A reportable new run records:

- a clean source commit plus worktree hash;
- a content-addressed dataset manifest and decoded-token vocabulary hash;
- comparison regime, discrete token/FLOP budget, paired seed, and data-order identity;
- shared-parameter initialization hash;
- raw JSONL metrics, a stable run ID, and a unique attempt ID;
- final model, optimizer, RNG, and reconstructable dataloader cursor;
- cold-inclusive wall time and a predeclared steady-state throughput window;
- an explicit failure category; only transient operational failures are retryable unchanged.

FineWeb validation always replays the same fixed token window. FineWeb resume derives the exact distributed-microbatch cursor from the restored optimizer iteration.

See [docs/EXPERIMENT_CONTRACTS.md](docs/EXPERIMENT_CONTRACTS.md) and [docs/PROVENANCE.md](docs/PROVENANCE.md).

## Quick start

```bash
git clone https://github.com/CATL-21CLab-SCIAGI/next-gen-arch.git
cd next-gen-arch
uv sync --extra cpu --group dev
uv run next-gen-arch verify
uv run pytest -m "not slow" -q
```

Create and fully verify a content manifest:

```bash
uv run next-gen-arch data-manifest create \
  --root /path/to/data \
  --dataset owner/dataset \
  --revision <immutable-revision> \
  --pattern '*.bin' \
  --output /path/to/dataset.manifest.json

uv run next-gen-arch data-manifest verify \
  --root /path/to/data \
  --manifest /path/to/dataset.manifest.json \
  --mode full
```

Inspect the frozen campaign or render a portable recipe:

```bash
uv run next-gen-arch list
uv run next-gen-arch show --size 300m --variant engram --seed 42
uv run next-gen-arch render \
  --config recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml
```

GPU runs use the container-provided PyTorch, CUDA, Transformer Engine, NCCL, and Megatron packages. Run `next-gen-arch doctor --backend megatron` inside the target container before launch.

## Repository layout

```text
src/archlab/architectures/  architecture mechanisms only
src/archlab/optimizers/     local optimizer extensions and recipes
src/archlab/megatron/       sole Megatron integration boundary
src/archlab/speedrun/       frozen small-scale reference backend
recipes/                    portable experiment specifications
results/                    immutable machine-readable evidence
docs/                       methods, operations, results, and historical context
tests/                      unit, numerical, resume, and distributed regressions
```

The project is MIT licensed. Paper names and third-party mechanisms remain attributed in [NOTICE.md](NOTICE.md).

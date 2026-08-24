# Next-Gen Architecture Lab

[简体中文](README.zh-CN.md) · [Results](docs/RESULTS.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Architecture notes](docs/ARCHITECTURES.md)

[![CI](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/actions/workflows/ci.yml/badge.svg)](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

A controlled, reproducible lab for testing next-generation language-model architecture ideas in one small [nanochat](https://github.com/karpathy/nanochat) training stack.

The project asks a deliberately narrow question:

> When data, tokenizer, initialization seed, batch order, training budget, optimizer, and evaluation are held fixed, which architectural change actually improves validation bits per byte?

It packages the code and evidence behind 16 architecture variants, three parameter scales, three matched seeds, a frozen 144-run experiment contract, and earlier fixed-token controls. The repository contains no model weights or private infrastructure configuration.

## Why this repository exists

Architecture papers are often evaluated as complete systems. That makes a gain hard to attribute: it may come from a token mixer, a residual topology, a memory module, a different optimizer, or simply a different data budget. This repository ports individual mechanisms into a shared backbone and evaluates them under paired controls.

The initial campaign studies:

- alternative token mixers: KDA, Kimi K3 KDA, Qwen-style GDN, and DSA;
- depth and residual routing: AttnRes and mHC;
- memory: Engram;
- attention modifications: gated attention, partial RoPE, relative attention, and short-convolution KV/residual paths;
- smaller component changes: xIELU, SiTU-GLU, and GLM-style MLA.

The goal is not to declare a universal architecture winner. It is to produce small, inspectable implementations and paired evidence that other researchers can reproduce, challenge, and extend.

## Headline results

Validation **BPB (bits per byte; lower is better)** is the primary quality metric. Deltas are paired against the same-scale baseline; a negative delta is an improvement. Throughput is normalized to that baseline on the campaign hardware.

### Parameter-scaling campaign: about 12 training tokens per parameter

| Scale | Variant | Mean BPB | Paired Δ BPB | Throughput | Valid seeds |
| --- | --- | ---: | ---: | ---: | ---: |
| ~100M | Qwen GDN | **0.902994** | **-0.013177** | 0.42× | 3 |
| ~100M | Engram | 0.908589 | -0.007582 | **0.95×** | 3 |
| ~100M | Kimi K3 KDA | 0.909599 | -0.006573 | 0.40× | 3 |
| ~300M | Qwen GDN | **0.799714** | **-0.008303** | 0.53× | 3 |
| ~300M | Kimi K3 KDA | 0.802384 | -0.005633 | 0.51× | 3 |
| ~300M | Engram | 0.803945 | -0.004072 | **0.97×** | 3 |

Qwen GDN has the lowest mature-scale BPB so far. Engram gives the strongest quality/throughput trade-off: its parameter count increases by about 12.2% at 100M and 7.6% at 300M while preserving roughly 95–97% of baseline throughput.

### Earlier fixed-token controls: about 1B training tokens per run

| Depth | Baseline BPB | Best observed variant | Variant BPB | Paired Δ BPB |
| --- | ---: | --- | ---: | ---: |
| d14 | 0.843770 | aligned baseline | 0.843770 | — |
| d16 | 0.830908 | mHC | **0.826490** | **-0.004419** |
| d18 | 0.820979 | mHC | **0.817117** | **-0.003862** |

These rows use a fixed training-token budget, not a fixed tokens-per-parameter budget. **Do not merge their absolute BPB values with the parameter-scaling table.**

### 1B-parameter campaign status

As of **2026-08-24 (UTC+8)**, 18 of 48 1B runs were complete, 3 had failed, 21 were still running, and 6 were pending. Among the six complete three-seed arms, AttnRes was provisionally best at `0.706122 BPB` (`-0.003726` versus baseline). Qwen GDN had not started and Kimi K3 KDA was still running, so this is **not a final 1B leaderboard**.

Known failures in that snapshot:

- all three 1B mHC runs became non-finite;
- two Inkling relative-attention runs became non-finite;
- three Engram runs hit a harness assertion because d32 injection layers were reused while constructing a d12 meta-reference model. This repository scales those reference layers proportionally and classifies the incident as a harness failure, not a model-quality result.

The full dated tables and limitations are in [docs/RESULTS.md](docs/RESULTS.md). Machine-readable values live in [results/key-metrics.csv](results/key-metrics.csv), while [results/campaign-status-2026-08-24.json](results/campaign-status-2026-08-24.json) preserves the 1B audit snapshot.

## Reproducibility contract

Every parameter-scaling comparison uses:

| Axis | Frozen value |
| --- | --- |
| Data | ClimbMix, 171-shard campaign subset |
| Tokenizer | fixed 32,768-token nanochat BPE |
| Context | 2,048 tokens |
| Precision | BF16 |
| Seeds | 42, 43, 44 |
| Data order | matched within each seed |
| Batch | 393,216 tokens total |
| Budget | approximately 12 training tokens per parameter |
| Optimizer | per-head Muon plus Adam parameter groups |
| Primary metric | validation BPB |

The public manifest records all 144 planned runs, exact command arguments, probed parameter counts, source-tree hashes, dataset fingerprint, tokenizer hashes, and software/hardware metadata. Machine-specific paths and hostnames are replaced with portable labels in the public copy.

The 100M study was repeated independently. Variant-minus-baseline deltas correlated at `0.9990`; excluding the known unstable relative-attention arm, the mean absolute difference was `0.000105 BPB`.

## Quick start

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.10 or newer.

```bash
git clone https://github.com/CATL-21CLab-SCIAGI/next-gen-arch.git
cd next-gen-arch
uv sync --extra cpu --group dev
uv run next-gen-arch verify
```

For a CUDA 12.8 environment, replace `--extra cpu` with `--extra gpu`.

Inspect the frozen experiment axes and reconstruct any exact run command:

```bash
uv run next-gen-arch list
uv run next-gen-arch show --size 300m --variant engram --seed 42
uv run next-gen-arch command --size 300m --variant engram --seed 42 --run-name my-run
```

`command` prints a portable `python -m scripts.base_train ...` invocation. Set up the data and tokenizer before executing it:

```bash
export NANOCHAT_BASE_DIR=/path/to/next-gen-arch-data
uv run python -m nanochat.dataset -n 170
uv run python -m scripts.tok_train --vocab-size 32768
```

For comparison-grade reproduction, verify the dataset and tokenizer fingerprints in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). A freshly trained tokenizer is useful for new experiments but is not automatically bit-identical to the frozen campaign artifact.

Run the local quality gate:

```bash
uv run ruff check next_gen_arch tests/test_registry.py
uv run python -m compileall -q nanochat next_gen_arch scripts
uv run pytest -m "not slow" -q
uv build
```

## Safety for long runs

The training loop now fails fast on non-finite validation loss, training loss, or gradients. The check is synchronized across distributed ranks, so one bad worker terminates the whole run instead of consuming GPU time while emitting unusable checkpoints.

Gradient scans run every step by default. They can be made less frequent with `--finite-check-every=N`, or disabled with `0`; loss and validation checks remain enabled.

## Repository layout

```text
nanochat/                 model, architecture, optimizer, and data primitives
scripts/base_train.py     unified training entry point
next_gen_arch/            manifest and result integrity tooling
results/                  frozen contract and machine-readable evidence
docs/                     architecture, result, and reproduction notes
tests/                    CPU-safe unit and integrity tests
.github/workflows/        CI and tag-based release automation
```

Engram and mHC used separate experimental forks during the live campaign. They are integrated here behind the same `GPTConfig` and model factory, so the public codebase has one training entry point and one checkpoint schema.

## What the evidence currently says

1. **Qwen GDN is the quality leader at completed 100M and 300M scales**, but its throughput cost is large.
2. **Engram is the best current quality/speed trade-off** at those scales.
3. **Kimi K3 KDA improves BPB from roughly 10M through 300M**, but is materially slower; 1B is unfinished.
4. **Short-convolution KV and gated attention are robust low-parameter changes.** The former gains more quality; the latter stays closer to baseline speed.
5. **mHC is promising but not yet reliable.** Its fixed-token results are strong, while the newer scaling setup becomes non-finite.
6. **The current DSA and relative-attention paths are poor fits for this 2K-context benchmark.** DSA uses masked dense SDPA, so this experiment does not test a true sparse-kernel speedup.

## Roadmap

- finish the 1B Qwen GDN, Kimi K3 KDA, and short-convolution KV controls;
- rerun the repaired 1B Engram harness without changing the remaining contract;
- isolate mHC instability before attempting larger runs;
- add a real sparse DSA backend and longer-context evaluations;
- test combinations only alongside the baseline and every single-component control;
- publish richer raw curves and hardware-normalized efficiency measurements.

## Scope and limitations

- The architecture modules are research adaptations, not official implementations from the cited authors.
- BPB comparisons are valid only inside a matched training regime. Hardware throughput is also environment-specific.
- Three seeds reduce noise but do not eliminate it.
- The 2,048-token context can understate the value of long-context or sparse mechanisms.
- No checkpoints, ClimbMix shards, tokenizer binaries, or private experiment-service configuration are distributed here.

## Contributing and citation

See [CONTRIBUTING.md](CONTRIBUTING.md) for the experiment contract required of new variants and [SECURITY.md](SECURITY.md) for private vulnerability reporting guidance. For academic use, cite the repository metadata in [CITATION.cff](CITATION.cff) together with the original method papers listed in [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md).

This project is based on nanochat and is released under the [MIT License](LICENSE). Architecture names and linked papers remain the property of their respective authors. See [NOTICE.md](NOTICE.md) for provenance.

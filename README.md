# Next-Gen Architecture Lab

[简体中文](README.zh-CN.md) · [Results](docs/RESULTS.md) · [10M backend comparison](docs/BACKEND_COMPARISON.md) · [Optimization audit](docs/OPTIMIZATION_AUDIT.md) · [Reproducibility](docs/REPRODUCIBILITY.md) · [Runtimes](docs/RUNTIMES.md) · [Architecture notes](docs/ARCHITECTURES.md)

[![CI](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/actions/workflows/ci.yml/badge.svg)](https://github.com/CATL-21CLab-SCIAGI/next-gen-arch/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)

A controlled, reproducible lab for testing language-model architectures in a fast single-node stack and carrying validated mechanisms into Megatron-LM for larger-scale training.

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

## Two execution backends, one experiment contract

- **`speedrun`** preserves the compact Muon, compilation, data-order, and kernel optimizations inherited from [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) and the campaign's nanochat fork. It remains the comparison-grade backend for the published 100M–1B results.
- **`megatron`** uses the system-provided MCore training lifecycle from the validated `nemo-26.06` container profile. All 16 mechanisms have passed the matched 10M three-seed comparison through the repository's model wrapper. Megatron is the scaling substrate; no upstream source tree is copied into, patched by, or shipped with this project.

The YAML contract is backend-neutral and uses layered `base + backend + scale + experiment` configuration. Paths are `env:NAME` references or CLI overrides, and sampling prompts are versioned package assets. A variant is rejected if it has not received a backend-specific adapter; the project does not claim numerical equivalence merely because two commands share an architecture name. The 10M validation used `TP=PP=CP=1`, so it validates the MCore wrapper and architecture signal, not yet native tensor/pipeline/context parallelism for every mechanism. See [docs/RUNTIMES.md](docs/RUNTIMES.md).

## Headline results

Validation **BPB (bits per byte; lower is better)** is the primary quality metric. Deltas are paired against the same-scale baseline; a negative delta is an improvement. Throughput is normalized to that baseline on the campaign hardware.

### Megatron versus speedrun at approximately 10M parameters

The B300 safe-autotune comparison contains `16 variants × 3 seeds × 2 backends = 96`
accepted runs. Across the 15 non-baseline variants, paired ΔBPB has a Pearson
correlation of `0.971361`; improvement/degradation direction agrees for `13/15`
variants, and the mean absolute delta gap is `0.009241 BPB`.

| Backend | Best variant | Mean BPB | Paired Δ BPB | Baseline tok/s |
| --- | --- | ---: | ---: | ---: |
| Megatron safe-autotune | KDA | **1.465789** | -0.081969 | 1,576,266* |
| speedrun | Kimi K3 KDA | **1.461594** | -0.093387 | 1,360,228 |

`*` Megatron reports median steady-state steps. Its post-warmup aggregate is
`1,529,304 tok/s` (`1.124×` speedrun), but fresh max-autotune caches make the one-off
process wall time `567.0 s` versus speedrun's `79.8 s`. Speedrun therefore remains the
better cold small-model screen; Megatron wins only after compilation is amortized.
Max-autotune was unsafe for KDA, Kimi K3 KDA, and Qwen GDN, so those nine accepted rows
use default `torch.compile`; all other rows use max-autotune. See the
[complete table and numerical-safety audit](docs/BACKEND_COMPARISON.md).

### 100M baseline on one three-node process group

A seed-42 baseline used one `3 nodes × 5 B300 GPUs = 15 ranks` job, with NVLink within
nodes and verified RoCE/GDRDMA between nodes. Exact replay kept 192 active sequences
despite the non-divisible world size.

| Backend | Final BPB | Curve agreement | Median steady tok/s |
| --- | ---: | ---: | ---: |
| speedrun | **0.91615027** | `r=0.99999787` vs historical | 1,906,064 |
| Megatron | 0.91756084 | `r=0.99983861` vs speedrun | **1,940,137** |

The reproduced speedrun final differs from the historical seed-42 result by only
`-0.00021621 BPB`. Megatron follows the same learning curve with mean absolute gap
`0.00122598 BPB` and finishes `+0.00141057 BPB` higher. Its median warmed step is
`1.018×` speedrun, while lifecycle-inclusive aggregate throughput is only `0.892×`;
the backends are therefore essentially tied in steady kernels but not in operational
overhead. See the [full 100M artifacts](results/100m-multinode-b300/).

### 100M-class native parallelism on one B300 node

Eight sequential seed-42 Megatron runs trained for 73.7M tokens each. DP5/PP5 used
five GPUs; factor-two TP/CP/EP used four GPUs so the six heads and six experts stayed
unchanged. Median throughput excludes the first ten steps.

| Comparison | Steady throughput ratio | Peak-memory effect | Outcome |
| --- | ---: | ---: | --- |
| PP5 / DP5 | 0.617× | -44.6% | capacity path, not a 100M default |
| TP2+DP2 / DP4 | 0.389× | -48.6% | capacity path, not a 100M default |
| fused DP4 / unfused DP4 | **1.274×** | -26.9% | useful B300 optimization |
| CP2+DP2 / fused DP4 | 0.366× | -22.7% | reserve for long context |
| MoE EP2+DP2 / EP1+DP4 | 0.980× | -7.1% | no throughput gain at six experts |

All eight runs completed 200/200 steps with zero skipped or non-finite iterations.
This harness used an indexed FineWeb-Edu control and matching DeepSeek-V3 tokenizer,
so its validation LM loss is not comparable with the ClimbMix/nanochat BPB campaign.
See the [native-parallelism report and machine-readable curves](results/100m-native-parallelism-b300-1n/).

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

The full dated tables and limitations are in [docs/RESULTS.md](docs/RESULTS.md).
Machine-readable values live in [results/key-metrics.csv](results/key-metrics.csv); the
optimized 10M comparison is in
[results/megatron-10m-safe-autotune-b300](results/megatron-10m-safe-autotune-b300/), and
[results/100m-multinode-b300](results/100m-multinode-b300/) contains the matched
three-node baseline curves, while
[results/campaign-status-2026-08-24.json](results/campaign-status-2026-08-24.json)
preserves the 1B audit snapshot.

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

The CPU extra is for artifact inspection and unit tests. For GPU training, use the
validated NeMo container, create the environment with `--system-site-packages`, and run
`next-gen-arch doctor --backend megatron` before launch. The project deliberately does
not resolve or replace the container's PyTorch, Transformer Engine, or Megatron packages.

Inspect the frozen experiment axes and reconstruct any exact run command:

```bash
uv run next-gen-arch list
uv run next-gen-arch show --size 300m --variant engram --seed 42
uv run next-gen-arch command --size 300m --variant engram --seed 42 --run-name my-run
```

`command` preserves the historical frozen-command interface. The maintained module path is `archlab.speedrun.base_train`. Set up the data and tokenizer before executing it:

```bash
export NANOCHAT_BASE_DIR=/path/to/next-gen-arch-data
uv run python -m archlab.speedrun.dataset -n 170
uv run python -m archlab.speedrun.tok_train --vocab-size 32768
```

Render the same frozen Qwen-GDN arm through the portable speedrun spec:

```bash
export NGA_DATA_ROOT=/path/to/next-gen-arch-data
uv run next-gen-arch render \
  --config recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml
```

Render an eight-rank Megatron baseline without embedding cluster paths in Git:

```bash
export NGA_TRAIN_DATA=/data/train_text_document
export NGA_VALID_DATA=/data/valid_text_document
export NGA_DATA_CACHE=/data/cache
export NGA_TOKENIZER=/models/tokenizer
export NGA_OUTPUT_DIR=/runs/megatron-baseline-1b-seed42
uv run next-gen-arch render \
  --config recipes/experiments/megatron_baseline_1b_seed42.yaml --json
```

Aggregate a matched 10M Megatron campaign against the frozen speedrun reference:

```bash
uv run python -m archlab.speedrun.campaign_compare \
  --megatron-root /path/to/megatron-results \
  --reference results/speedrun-10m-reference.csv \
  --output-dir /path/to/comparison
```

For comparison-grade reproduction, verify the dataset and tokenizer fingerprints in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). A freshly trained tokenizer is useful for new experiments but is not automatically bit-identical to the frozen campaign artifact.

Run the local quality gate:

```bash
uv run ruff check src/archlab tests
uv run python -m compileall -q src/archlab
uv run pytest -m "not slow" -q
uv build
```

## Safety for long runs

The training loop now fails fast on non-finite validation loss, training loss, or gradients. The check is synchronized across distributed ranks, so one bad worker terminates the whole run instead of consuming GPU time while emitting unusable checkpoints.

Gradient scans run every step by default. They can be made less frequent with `--finite-check-every=N`, or disabled with `0`; loss and validation checks remain enabled.

## Repository layout

```text
src/archlab/architectures/  model and layer composition only
src/archlab/optimizers/     optimizer extensions absent from system Megatron
src/archlab/megatron/       the only ArchLab → Megatron integration boundary
src/archlab/speedrun/       frozen small-scale reference backend
src/archlab/prompts/        versioned backend-neutral prompt sets
recipes/                    portable smoke and pretraining contracts
results/                    frozen contracts and machine-readable evidence
tests/                      unit, integration, and distributed regressions
```

Engram and mHC used separate experimental forks during the live campaign. They are integrated here behind the same `GPTConfig` and model factory, so the public codebase has one training entry point and one checkpoint schema.

## What the evidence currently says

1. **Qwen GDN is the quality leader at completed 100M and 300M scales**, but its throughput cost is large.
2. **Engram is the best current quality/speed trade-off** at those scales.
3. **Kimi K3 KDA improves BPB from roughly 10M through 300M**, but is materially slower; 1B is unfinished.
4. **Short-convolution KV and gated attention are robust low-parameter changes.** The former gains more quality; the latter stays closer to baseline speed.
5. **mHC is promising but not yet reliable.** Its fixed-token results are strong, while the newer scaling setup becomes non-finite.
6. **The current DSA and relative-attention paths are poor fits for this 2K-context benchmark.** DSA uses masked dense SDPA, so this experiment does not test a true sparse-kernel speedup.
7. **The 10M architecture signal transfers across backends, but backend choice still matters.** Delta correlation is high and two small effects change sign. Safe-autotune Megatron is faster in steady state; speedrun is far faster for a cold one-off run.

## Roadmap

- finish the 1B Qwen GDN, Kimi K3 KDA, and short-convolution KV controls;
- rerun the repaired 1B Engram harness without changing the remaining contract;
- isolate mHC instability before attempting larger runs;
- add a real sparse DSA backend and longer-context evaluations;
- make selected custom mechanisms native to Megatron tensor/context parallelism and validate them beyond one rank;
- test combinations only alongside the baseline and every single-component control;
- publish richer raw curves and hardware-normalized efficiency measurements.

## Scope and limitations

- The architecture modules are research adaptations, not official implementations from the cited authors.
- All 16 mechanisms have a validated single-rank MCore training wrapper at approximately 10M parameters; the 100M baseline also completed 15-way data parallelism over three nodes. This is not evidence that every mechanism already supports Megatron tensor, pipeline, expert, or context parallelism.
- BPB comparisons are valid only inside a matched training regime. Hardware throughput is also environment-specific.
- Three seeds reduce noise but do not eliminate it.
- The 2,048-token context can understate the value of long-context or sparse mechanisms.
- No checkpoints, ClimbMix shards, tokenizer binaries, or private experiment-service configuration are distributed here.

## Contributing and citation

See [CONTRIBUTING.md](CONTRIBUTING.md) for the experiment contract required of new variants and [SECURITY.md](SECURITY.md) for private vulnerability reporting guidance. For academic use, cite the repository metadata in [CITATION.cff](CITATION.cff) together with the original method papers listed in [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md).

This project is based on nanochat and is released under the [MIT License](LICENSE). Architecture names and linked papers remain the property of their respective authors. See [NOTICE.md](NOTICE.md) for provenance.

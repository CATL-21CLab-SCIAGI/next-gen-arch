# Runtime reference

**Type:** maintained integration reference. **Reader guide:** [Runtime and Backends](wiki/Runtime-and-Backends.md).

| Path | Entry or boundary | Scope |
| --- | --- | --- |
| verl | `archlab.training`, `third_party/verl` | SFT and RL execution, rollouts and monitoring |
| Megatron under verl | `archlab.megatron.limite`, `archlab.training.megatron` | Native Limite bridge for verl's Megatron engine |
| Attention kernels | `archlab.architectures.tilelang_*` | The only project-owned GPU kernels (TileLang) |
| Serving | `archlab.training.serving`, `archlab.serving` | Qualified model plugins and checkpoint export |

## Runtime ownership

The validated container owns PyTorch, CUDA, NCCL, Transformer Engine and Megatron. Record installed versions, imported paths, container identity and upstream revision. The repository owns integration code, not replacement framework installations.

Project kernels are written in TileLang only; CuTe DSL and Triton kernels were retired on 2026-11-02. Container packages that ship their own kernels (FlashAttention, Transformer Engine, cuDNN SDPA, vLLM) remain container dependencies with recorded versions.

## Qualification boundaries

- Kernel/backend selection requires numerical and restart evidence for the actual model; the CLI delegates SFT and RL execution to verl.
- The speedrun, Megatron scratch-comparison and Qwen3.8/DeepSeek-V4.1 pretraining runtimes are retired; reproduce their [recorded results](BACKEND_COMPARISON.md) from the immutable source revision they name.

## Portable configuration

Recipes use `env:NAME`, `package:relative/path`, and launch overrides. YAML proposals may document geometry without being executable builders; check the named entry.

Use the [training guide](TRAINING.md) for `resolve`, `qualify` and `train`. Historical `doctor`, `verify`, `show`, `render` commands and the Megatron catalog renderer are retired.

Reusable prompt text lives in `src/archlab/prompts/`. Marin remains a research reference, not an implemented third training backend; see the [optimization audit](OPTIMIZATION_AUDIT.md).

# Runtime reference

**Type:** maintained integration reference. **Reader guide:** [Runtime and Backends](wiki/Runtime-and-Backends.md).

| Path | Entry or boundary | Scope |
| --- | --- | --- |
| Speedrun | `archlab.speedrun` | Frozen nanochat/modded-nanogpt comparison behavior |
| Megatron renderer | `archlab.megatron.backend` | Native baseline launch plans and topology |
| Megatron comparison | `archlab.megatron.train` | Shared project architecture math under the MCore lifecycle |
| Model-specific Megatron | Qwen entries under `archlab.megatron` | Named model and checkpoint contracts |
| verl | `archlab.training`, `third_party/verl` | Current RL execution, rollouts and monitoring |
| NeMo AutoModel references | `archlab.automodel` | Retained loading and numerical references; custom trainers retired |
| Serving | `archlab.training.serving`, `archlab.serving` | Qualified model plugins and checkpoint export |

## Runtime ownership

The validated container owns PyTorch, CUDA, NCCL, Transformer Engine and Megatron. Record installed versions, imported paths, container identity and upstream revision. The repository owns integration code, not replacement framework installations.

The speedrun lineage records modded-nanogpt commit `f411b3d346aa52d3504324ca93c230fd84c6c07f`. Historical Megatron comparison provenance includes `55ac7082517c3878ae653c07c09c534b8aed49f6`; these are run identities, not vendored dependencies.

## Qualification boundaries

- A mechanism in speedrun is not automatically native to TP/PP/CP/EP.
- The 16-variant Megatron comparison used TP=PP=CP=1.
- The 100M multi-node reproduction qualified baseline data parallelism.
- Native parallelism probes establish capacity/numerics for their tested models.
- Kernel/backend selection requires numerical and restart evidence for the actual model; the current CLI delegates RL execution to verl.

[Backend evidence](BACKEND_COMPARISON.md) · [Retention decision](BACKEND_RETENTION.md)

## Portable configuration

Recipes use `env:NAME`, `package:relative/path`, and launch overrides. YAML proposals may document geometry without being executable builders; check the named entry.

Use the [training guide](TRAINING.md) for `resolve`, `qualify` and `train`. Historical `doctor`, `verify`, `show` and `render` CLI commands are retired. The retained Megatron catalog renderer is `python -m archlab.megatron.launch_config --help`.

Reusable prompt text lives in `src/archlab/prompts/`. Marin remains a research reference, not an implemented third training backend; see the [optimization audit](OPTIMIZATION_AUDIT.md).

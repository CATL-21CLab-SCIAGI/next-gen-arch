# Experiment recipes

Organize portable scientific contracts by workflow. Keep machine paths in private
bindings or launch overrides, and executable Python in `src/archlab`.

| Directory | Purpose |
| --- | --- |
| [`sft/`](sft/) | Package-backed supervised training and recorded Limite studies |
| [`rl/`](rl/) | Package-backed agentic RL; reusable harness contracts in `rl/harness/` |
| [`eval/`](eval/) | Fixed evaluation protocols and budgets |

`archlab resolve`, `archlab qualify` and `archlab train` consume the typed RL
contracts. The [SFT guide](sft/README.md) describes its separate upstream SFT
entrypoint.

Historical contracts retain their scientific settings. Their presence does not
establish current backend support: reproduce retired executors from the recorded
immutable source. The old generic `ExperimentSpec`/`LaunchPlan` renderer has been
retired; frozen `extends` profiles remain provenance, not current CLI inputs.

Pretraining (speedrun, Megatron scratch, Qwen3.8, DeepSeek-V4.1), Qwen3.5
architecture-switch distillation and their recipes were retired on 2026-11-02;
checksum-verified originals are in ignored `.legacy/20261102-tilelang-only/`.
Changes to geometry, optimizer, data order or token budgets require an explicit
experiment contract.

The [Limite study](../docs/wiki/Limite-SFT-Comparison.md) explains matched adapter
warmup and full finetuning. Its recorded contracts are under `sft/limite/`; the
sealed AIME26 protocol is under `eval/limite/`. Historical RL executors are
[retired](../docs/wiki/Limite-RL-Health-and-Performance.md).

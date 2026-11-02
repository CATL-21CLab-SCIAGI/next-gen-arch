# Experiment recipes

Organize portable scientific contracts by workflow. Keep machine paths in private
bindings or launch overrides, and executable Python in `src/archlab`.

| Directory | Purpose |
| --- | --- |
| [`pretrain/`](pretrain/) | Qwen and DeepSeek pretraining, scratch studies and launch/model catalogs |
| [`distill/`](distill/) | Qwen latent matching, forward-KL and on-policy distillation stages |
| [`sft/`](sft/) | Package-backed supervised training and recorded Limite/DeepSeek studies |
| [`rl/`](rl/) | Package-backed agentic RL; reusable harness contracts in `rl/harness/` |
| [`eval/`](eval/) | Fixed evaluation protocols and budgets |
| [`reference/`](reference/) | Frozen speedrun/Megatron comparisons and their descriptive defaults |

`archlab resolve`, `archlab qualify` and `archlab train` consume the typed RL
contracts. The [SFT guide](sft/README.md) describes its separate upstream SFT
entrypoint. Distillation uses the stage-specific modules documented in
[Latent Matching](../docs/LATENT_MATCHING.md).

Historical contracts retain their scientific settings. Their presence does not
establish current backend support: reproduce retired executors from the recorded
immutable source. The old generic `ExperimentSpec`/`LaunchPlan` renderer has been
retired; frozen `extends` profiles remain provenance, not current CLI inputs.

Select an entry in a `profiles` catalog explicitly with `FILE.yaml#PROFILE`:

```bash
PYTHONPATH=src python -m archlab.megatron.launch_config \
  --recipe 'recipes/pretrain/qwen/launches.yaml#flash_next_w320_e32' --bindings
```

This command renders allowlisted launch bindings. Admission still requires the
actual model/backend evidence. Changes to geometry, optimizer, data order or token
budgets require an explicit experiment contract.

The [Limite study](../docs/wiki/Limite-SFT-Comparison.md) explains matched adapter
warmup and full finetuning. Its recorded contracts are under `sft/limite/`; the
sealed AIME26 protocol is under `eval/limite/`. Historical RL executors are
[retired](../docs/wiki/Limite-RL-Health-and-Performance.md).

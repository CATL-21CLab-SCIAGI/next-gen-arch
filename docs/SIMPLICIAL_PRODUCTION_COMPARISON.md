# Qwen W320 simplicial continuation

**Type:** production experiment contract.
**Recipe:** `recipes/experiments/qwen38_w320_simplicial_production.yaml`.

| Item | Setting |
| --- | --- |
| Entry | `archlab.megatron.qwen38_flash_next_full_train` |
| Model / attention | `w320-e32-depth48-no-mtp` / `simplicial-rope-16x128` |
| Changed layers | 8, 16, 24, 32, 40, 48 |
| Retained | 36 GDN layers, six other global layers, MoE/PLE, four streams, gates and Q/K norms |
| Parameters | 387,926,912 versus 387,680,960 baseline |
| Execution | DP32; all model-parallel groups size one |
| Budget | Planned 100B tokens; historical ordinary control ends at 12.1803B |
| Initialization | Fresh model/optimizer/data cursor |

The 245,952 additional parameters are K2/V2 projections and K2 norms. The ordinary-RoPE variant does not reproduce a relatively invariant trilinear formulation.

The production iterator, optimizer grouping, schedule, seed, batch, evaluation and checkpoint policies are frozen. Shared initial weights are hashed and DP replicas checked.

Full-shape checkpoint probes and distributed numerical oracles gate launch. The earlier 3B A/B/C pilots use a different iterator, loop and validation window and are not exact production controls.

The later simplicial run stopped at 30.342B tokens. See [the dated audit](TRAINING_CONCLUSIONS_20260922.md) for its last validation and comparison limits.

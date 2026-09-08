# W320 simplicial production comparison

The opt-in recipe is
`recipes/experiments/qwen38_w320_simplicial_production.yaml`. It uses the existing
`archlab.megatron.qwen38_flash_next_full_train` entry with
`--model-variant w320-e32-depth48-no-mtp --parallelism dp-only
--attention-variant simplicial-rope-16x128`.

Only attention in layers 8, 16, 24, 32, 40 and 48 is replaced with the previously
qualified C mechanism. Six global-attention layers, 36 GDN layers, all MoE/PLE
components, four residual streams, output gates and Q/K norms remain. Total
parameters are 387,926,912 versus the original baseline's 387,680,960. The extra
245,952 parameters are K2/V2 projections and K2 norms. This is the named ordinary
partial-RoPE variant with causal windows 16 and 128, not an exact reproduction
of a relative-position-invariant trilinear scheme.

The production data iterator, native training loop, loss normalization, Muon/
Adam grouping, 100B-token LR horizon, seed 42, microbatch 4, global batch 4096,
2K context, evaluation window/cadence and checkpoint cadence stay unchanged.
All 32 GPUs use pure DP; TP/PP/EP/CP/expert-TP are one. No pretrained weights or
adapter-training checkpoint initialize this from-scratch run. Default global
attention behavior is unchanged, and a regression checks identical native argv.

`simplicial_production.py` applies the existing `simplicial_attention.py` wrapper
before optimizer construction, records every common initial weight hash, checks
the supplied baseline initialization reference, and verifies all DP replicas.
Production shape/parameter audits include the extra weights. Probe gradients
and replica checks include all K2/V2/K2-norm parameters. Resume refuses to switch
between global and simplicial attention even though the base config is identical.
The immutable run contract records the variant and integration/kernel hashes.

Full-shape native production save/reload probes and the two-rank numerical
oracles are required before launch. Machine paths are launch-time overrides;
the existing NeMo container and DLC nodes are retained.

The primary historical control is the original W320/E32 production run, whose
observations end at step 1452 (12.1803B tokens). The earlier A/B/C 3B pilots used
a different strided data iterator, a different loop, and a smaller held-out
window: do not present them as exact production controls. The new run targets
the original 100B budget, but no matched historical baseline observations exist
beyond 12.1803B tokens. The approximately 0.246M parameter difference remains
an explicit limitation of the comparison.

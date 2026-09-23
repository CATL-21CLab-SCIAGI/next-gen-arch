# Qwen Experiments

## Lineages

| Lineage | Purpose | Record |
| --- | --- | --- |
| Dense Qwen3.8 | Quarter/full pretraining with native Megatron Muon | `docs/QWEN38_27B_TRAINING.md` |
| Flash-Next backbones | Width, experts, GDN, attention, residual and PLE choices | `docs/QWEN38_NEXT_BACKBONE_VARIANTS.md` |
| W320/E32 | Named ~388M geometry and ordinary/simplicial controls | `docs/QWEN38_NEXT_W320_E32_PROPOSAL.md` |
| Pretrained Flash-Next | Frozen backbone plus additive simplicial branches | `docs/PRETRAINED_QWEN_NEXT_SIMPLICIAL.md` |

## What to preserve

A model recipe records geometry; the named Python constructor executes it. Check emitted model shapes, parameter counts, optimizer groups and actual process groups.

DP-only execution replicates experts and PLE partitions. It does not become expert parallelism merely because the model is an MoE.

## Findings

- At 3.003B matched tokens, the W320 ordinary local-attention control beat both global and simplicial attention on held-out CE.
- Zero-output adapter initialization recovered much faster than the tested nonzero initialization in the pretrained experiment.
- The pretrained run reduced held-out loss but had no separately trained ordinary-adapter control.
- The small step-4000 capability pilot was mixed and statistically inconclusive.

The later simplicial-only production trajectory is not a continuation of a fully matched long-duration comparison.

**Read next:** [[Experiment Design|Experiment-Design]] · [[Results]]

# Qwen simplicial A/B/C pilots

**Type:** bounded controlled experiments. DSW DP1 and DLC DP32 are distinct execution contracts.

## Shared architecture and budget

Width 320; depth 48; 36 GDN layers; four residual streams; 32 experts/top-10 plus one shared expert; PLE; Q/K normalization and attention output gates. MTP is off.

Only layers 8/16/24/32/40/48 change:

| Arm | Six selected attention slots | Total parameters |
| --- | --- | ---: |
| A | Original global gated dot-product | 387,680,960 |
| B | 128-token causal local gated dot-product | 387,680,960 |
| C | 16×128 causal gated simplicial, ordinary RoPE on Q/K1/K2 | 387,926,912 |

Arm C uses ordinary partial RoPE on Q/K1/K2. It is not a claim of trilinear relative-position invariance or an exact paper reproduction.

| Training item | Setting |
| --- | --- |
| Seed / context | 42 / 2048 |
| Microbatch / global batch | 4 / 4096 |
| Endpoint | 358 updates / 3,003,121,664 tokens |
| LR schedule | Prefix of the original 11,921-step schedule |
| Validation | Fixed 64 sequences / 131,072 unique tokens |
| Checkpoints | Model, optimizer masters/state, scheduler, RNG and cursor |

DP32 assigns global microbatch i to rank i mod 32. Replicated evaluation does not multiply the number of unique validation tokens.

## Qualification

1. Independent FP64-capable attention and all-input-gradient oracles.
2. Window, GQA, initialization and unchanged-parameter checks.
3. Native Muon/Adam grouping and DP gradient agreement.
4. Full-shape save/load and next-update continuation.
5. Fresh source/runtime-bound receipts for every arm.

Shared K/V backward atomics are not bitwise deterministic. Kernel timing does not establish whole-model throughput or convergence.

## Entries and evidence

| Entry | Purpose |
| --- | --- |
| `archlab.benchmarks.simplicial_attention` | Bounded mechanism preflight |
| `archlab.megatron.simplicial_campaign` | Sequential DSW A/B/C supervision |
| `archlab.megatron.simplicial_dlc_campaign` | DP32 probe/train phases |

Failed or incomplete arms stop advancement. Fresh pilot initialization does not reuse the paused baseline's weights.

Completed results favor ordinary local attention; see the [conclusions audit](TRAINING_CONCLUSIONS_20260922.md). The later [production continuation](SIMPLICIAL_PRODUCTION_COMPARISON.md) has a different data-order/control boundary.

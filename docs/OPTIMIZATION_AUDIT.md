# Optimization audit — 2026-08-26

**Type:** historical evidence and disposition ledger.
**Decision:** separate objective/optimizer changes from execution improvements; retain qualified per-variant compiler settings.

## Source snapshots

| Project | Revision |
| --- | --- |
| Marin | `299c7f3245e2e6998345980cadad75f45088f63f` |
| modded-nanogpt | `ecbb586296d3dac36fd206211f25d63bad4a6b35` |
| marin-speedrun | `31fe8028f8caa6e47082c02de095b1fed4f517a8` |
| Historical Megatron | `55ac7082517c3878ae653c07c09c534b8aed49f6` |

The audit classified Modded records 1–89 and 80 Marin Agent-MoE experiments. Historical B300 compilation used a compatible CUDA 13.0 assembler; current launches must use their own recorded runtime.

## Implemented candidates

| Recipe | Upstream idea | Implementation status |
|---|---|---|
| `baseline` | current Modded lineage | retained: per-head NorMuon, Polar Express, cautious Muon WD |
| `full-matrix-muon` | records 20/80 partition granularity | recipe; opposite control for per-head Muon |
| `partial-rope`, `partial-rope-25` | Marin #4849; Modded YaRN | recipe |
| `pko`, `pko-last` | Marin #4802/#4976; Modded #49 | recipe |
| `embed-std1` | Marin #5203 | recipe |
| `qk-gain` | Marin #5373 | recipe; adds one scalar per head/layer |
| `cached-attention` | Marin #4987; Modded saved activation | recipe |
| `midpoint-kv` | Marin #8196 | recipe |
| `bf16-loss` | Modded #37 | recipe |
| `asymmetric-logits` | Modded #54/current record | recipe |
| `z-loss-*` | Marin final-logit stabilization; DCLM 5e-6 setting | recipe and coefficient sweep |
| `muonh`, `muonh-lr*` | Marin #5596; Modded optimization track | recipe with mandatory nonzero matrix initialization and LR sweep |
| `muoneqh-half`, `muoneqh-quarter` | Marin #6066 | recipe |
| `adamh` | Marin leaderboard/AdamH articles | recipe with mandatory nonzero matrix initialization |
| `grad-clip-01`, `grad-clip-03` | Marin #5235 | recipe |
| `grad-clip-005`, `grad-clip-015`, `grad-clip-02` | promoted-range refinement | recipe |
| `adam-every-2` | Modded #39 | recipe; auxiliary gradients accumulate for two steps |
| `cautious-adam-wd` | Modded #50/current record | recipe |
| `marin-compound` | Marin #4999 | recipe containing only the dense, portable components |
| `compile-reduce-overhead`, `compile-max-autotune`, `compile-safe-autotune` | PyTorch compiler modes | profiles; safe policy is variant-resolved from the portability gate |
| `compile-dp-overlap*` | Megatron/Modded communication overlap | two-rank profiles; rejected at 10M, retained for scale |

A recipe changes declared training choices; a profile changes execution. Existing architecture arms remain separate controls. Hyperball recipes require nonzero matrix initialization because norm-preserving updates cannot move a zero matrix.

## Full-budget three-seed funnel

| Profile and recipe | Mean BPB | Paired Δ BPB | Mean tok/s | Throughput |
|---|---:|---:|---:|---:|
| `compile` + baseline | 1.543746 | +0.000000 | 1,323,680 | 1.000× |
| `compile` + clip 0.1 | 1.511901 | -0.031845 | 1,126,391 | 0.850× |
| `compile` + z-loss 5e-6 + clip 0.1 | **1.507486** | **-0.036260** | 1,133,348 | 0.856× |
| `max-autotune` + baseline | 1.545632 | +0.000000 | 1,665,882 | 1.000× |
| `max-autotune` + clip 0.1 | 1.511704 | -0.033929 | 1,532,709 | 0.921× |
| `max-autotune` + z-loss 5e-6 | 1.532756 | -0.012876 | 1,471,168 | 0.884× |
| `max-autotune` + z-loss 5e-6 + clip 0.05 | 1.508328 | -0.037304 | 1,528,534 | 0.918× |
| `max-autotune` + z-loss 5e-6 + clip 0.1 | **1.507245** | **-0.038387** | 1,556,884 | 0.935× |
| `max-autotune` + full-matrix Muon + clip 0.1 | 1.516369 | -0.029263 | **1,778,327** | **1.068×** |
| `max-autotune` + full-matrix Muon + z-loss + clip 0.1 | 1.512061 | -0.033572 | 1,732,109 | 1.040× |

### Portability verdict

| Candidate | Finding | Disposition |
| --- | --- | --- |
| z-loss 5e-6 + clip 0.1 | Improved baseline; failed all three Kimi K3 seeds and corrupted GDN/KDA quality | Rejected as global default |
| Recipe-free max-autotune | Five fail-fast rows plus finite-but-corrupted recurrent variants | Per-variant qualification required |
| Safe autotune | Max-autotune for 13 variants; default compile for KDA/Kimi K3/GDN | Accepted execution policy |
| Full-matrix Muon | Faster baseline Pareto point | Separate optimizer comparison |

The global z-loss/clipping study's paired-delta correlation fell to -0.320154. A baseline win did not generalize.

Cold compilation cost matters: the baseline recipe's observed break-even estimate was ~649M tokens. Short probes establish compatibility; full-budget paired seeds establish quality.

## Upstream idea inventories

| Source | Scope | Ledger |
| --- | --- | --- |
| modded-nanogpt | Records 1–89 | [Mechanism/disposition ledger](reference/MODDED_OPTIMIZATION_LEDGER.md) |
| Marin Agent-MoE | 80 tracked experiments | [Issue/disposition ledger](reference/MARIN_OPTIMIZATION_LEDGER.md) |

## Evidence and limitations

All 39 registry recipes ran at least once; the ledger contains 93 accepted probe/full-budget runs. Only matched three-seed controls enter the promotion table.

Early dirty-tree records hashed tracked diffs but omitted untracked content. The committed registry and later combined worktree identities address that provenance gap; do not overstate early source coverage.

[Per-run ledger](recorded-results/megatron-10m-optimization-runs-b300.csv) · [Promotion table](recorded-results/megatron-10m-optimization-b300.csv) · [Global-recipe rejection](recorded-results/megatron-10m-global-zclip-b300/comparison.md) · [Accepted backend comparison](BACKEND_COMPARISON.md)

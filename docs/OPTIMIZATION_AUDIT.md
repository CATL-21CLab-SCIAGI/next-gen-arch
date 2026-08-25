# Speedrun optimization audit

Updated: 2026-08-26 (UTC+8)

This is the evidence ledger for speed and sample-efficiency changes considered for the
Megatron backend. It prevents a useful upstream idea from disappearing into an
unreviewed “kitchen sink,” while also preventing benchmark-specific kernels from being
silently presented as portable improvements.

## Audited sources

The source snapshots are fixed for this audit:

- `marin-community/marin` at `299c7f3245e2e6998345980cadad75f45088f63f`
- `KellerJordan/modded-nanogpt` at `ecbb586296d3dac36fd206211f25d63bad4a6b35`
- `marin-community/marin-speedrun` at `31fe8028f8caa6e47082c02de095b1fed4f517a8`
- this repository's Megatron-LM submodule at
  `55ac7082517c3878ae653c07c09c534b8aed49f6`

The complete Modded short-track history through record 89 and the complete Marin
Agent-MoE digest (80 tracked experiments) were classified. The Marin report set was
also read in full: Agent MoE, Marin 8B retrospective, Marin 32B retrospective, Grug
archive, Markdownified Datasets, its data appendix, and the report index. Supporting
training-config, HBM, optimizer-extension, and scaling-heuristic articles were reviewed
where they change an engineering decision.

## Status vocabulary

- **retained**: already part of the comparison contract.
- **recipe**: implemented as an isolated `--optimization-recipe` and tested alone.
- **profile**: implemented as an execution-only `--backend-profile` ablation.
- **existing arm**: represented by a frozen architecture arm rather than folded into
  the backend.
- **upstream negative**: upstream evidence is consistently negative; it is documented,
  not repeated without a new hypothesis.
- **contract change**: changes tokens, data order, parameter budget, sequence length, or
  objective and therefore cannot enter the controlled 10M comparison silently.
- **scale candidate**: useful at larger models or multi-rank scale, but the 10M shape
  cannot measure its claimed benefit.
- **capability rejection**: the required kernel or shape is unavailable on the measured
  platform; the fallback is recorded.

## B300 execution baseline

The experiment hosts are NVIDIA B300 Blackwell systems. PyTorch reports CUDA capability
`(10, 3)` (`sm_103a`). The CUDA 12.8 PyTorch wheel's bundled Triton `ptxas` cannot
compile that target, so every run explicitly uses CUDA 13.0 `ptxas` at
`/usr/local/cuda-13.0/bin/ptxas`. Hardware labels emitted by `nvidia-smi` on these
nodes are known to be incorrect and are not used as provenance.

The first factorial backend probe used the same seed-42, 30-step baseline:

| Profile | Architecture compile | MCore BF16 master | tok/s | vs legacy |
|---|---:|---:|---:|---:|
| `legacy` | no | yes | 779,734 | 1.00x |
| `compile` | yes | yes | 1,258,825 | 1.61x |
| `native-master` | no | no | 702,786 | 0.90x |
| `speedrun` | yes | no | 1,107,184 | 1.42x |

The evidence therefore selects `compile`, not a union of every switch. The native
speedrun reference delivered roughly 1.33M tok/s for the same 30 training steps. Its
training completed with BPB 1.948811; the former summary-writing failure was caused by
an absent no-checkpoint output directory and is fixed.

A full-budget seed-42 `compile-max-autotune` check reached 1.709M tok/s versus 1.307M
for the matched `compile` run, with BPB 1.544228 versus 1.543209. Because compiler
algorithm selection is not bitwise invariant, the remaining seeds are measured before
promotion rather than assuming execution-only identity.

### B300 kernel capability results

The measured PyTorch SDPA fallback is not a slow math implementation: the automatic
path resolves to `aten::_scaled_dot_product_flash_attention` on B300. At the frozen
`B=16, H=7, T=2048, D=8` attention shape, ten-call forward averages were:

| Forced SDPA backend | ms/call | Result |
|---|---:|---|
| automatic | 0.200 | PyTorch flash attention |
| flash attention | 0.200 | accepted |
| efficient attention | 0.434 | accepted, slower |
| math | 6.230 | accepted, much slower |
| cuDNN attention | — | no valid execution plan |

The external `flash_attn.cute` package is absent, but adding it would not remove an SDPA
math fallback because PyTorch already dispatches its native flash kernel. The current
Modded FP8 path is shape-incompatible: `_scaled_mm(4096×56, 56×224)` rejects `K=56`
because the trailing dimension must be divisible by 16, while the aligned
`4096×64, 64×256` control succeeds. FP8 remains a larger, aligned-shape candidate for
Megatron plus Transformer Engine; it is not silently emulated at 10M.

### Two-B300 data-parallel probe

All three 30-step DP profiles use identical two-rank data sharding. Synchronous DDP is
the winner at this model size:

| Profile | Gradient buckets | Average in collective | global tok/s | BPB |
|---|---:|---:|---:|---:|
| `compile` | one synchronous reduction | no | 2,371,780 | 1.919188 |
| `compile-dp-overlap` | 4 | no | 1,951,645 | 1.919189 |
| `compile-dp-overlap-average` | 4 | yes | 2,205,077 | 1.919198 |

The plain two-card result is about 1.85× the matched single-card short-probe throughput.
At 10M, four asynchronous buckets add more launch/synchronization overhead than they
hide. Overlap remains available as an explicit scale profile, but is not the default.

## Portable recipes

The recipe registry lives in `training/optimization_recipes.py`. Each result records the
resolved recipe, backend profile, source commit/diff digest, Megatron submodule commit,
CUDA/PyTorch versions, and selected `ptxas`.

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
| `compile-reduce-overhead`, `compile-max-autotune` | PyTorch compiler modes | profiles; measured separately from model recipes |
| `compile-dp-overlap*` | Megatron/Modded communication overlap | two-rank profiles; rejected at 10M, retained for scale |

Hyperball recipes deliberately replace zero projection initialization. A
norm-preserving update cannot move a zero matrix; omitting this coupled change would
produce a plausible-looking but invalid MuonH/AdamH result.

All 39 executable recipes in the registry were run at least once on B300; the durable
ledger contains 93 accepted probe/full-budget runs, including the backend factorial,
coefficient refinements, three-seed confirmations, and two-rank DDP checks. The
[per-run ledger](../results/megatron-10m-optimization-runs-b300.csv) retains source/diff,
Megatron, toolchain, wall-time, and throughput provenance. The compact
[promotion table](../results/megatron-10m-optimization-b300.csv) contains only the
matched three-seed controls used for the final selection. A probe is evidence that a
recipe executes and has sane early numerics; it is not promoted as a quality result.

The optimization runs were launched from a dirty worktree before the recipe registry
was committed. Their recorded `source_diff_sha256` covers the tracked binary diff but,
as Git defines `diff`, not the content of untracked files. The release commit tracks the
registry and is the reproducible source snapshot. The wrapper now additionally records
the untracked-file list/digest and a combined `source_worktree_sha256` so future dirty
runs do not have that provenance gap.

## Measured optimization funnel

All rows below are full-budget three-seed baseline runs under the frozen data/token
contract. Deltas and throughput ratios use the same compiler profile and seed baseline.

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

The baseline-only promotion candidate was `compile-max-autotune` plus
`z-loss-5e-6-clip01`. Full-matrix Muon is a faster baseline Pareto point, but cannot be
applied to the all-model campaign without erasing the explicitly controlled
`per-head-muon` arm. The required all-architecture portability gate below rejected the
z-loss-plus-clipping recipe as a global default.

“Promoted” here means steady-state/scaling throughput, not shortest one-off 10M wall
time. For the z-loss-plus-clip recipe, regular `compile` averaged 122.7 seconds wall
(96.2 measured training), while `max-autotune` averaged 252.0 seconds wall
(69.7 measured training). The observed cold-start premium breaks even only after about
649M training tokens if the measured rates persist. Use regular `compile` for isolated
10M probes and cache/reuse `max-autotune` artifacts for longer campaigns.

The z-loss sweep is a useful warning against single-seed promotion. At seed 42,
coefficients from 5e-6 through 2e-5 entered a much better basin; 1e-6/2e-6 and 1e-4
were neutral. The isolated 5e-6 recipe did not reproduce that gain on seeds 43/44 and
had BPB standard deviation 0.0229. Combining 5e-6 with clipping made all three paired
seeds improve and reduced standard deviation to 0.0060.

The full seed-42 clipping refinement was monotone beyond the optimum neighborhood:
0.05/0.10 yielded 1.517251/1.517205 BPB, while 0.15/0.20/0.30 yielded
1.519837/1.521959/1.524585. The upstream 0.1 choice is therefore retained.

### All-architecture portability gate

The baseline result did not generalize. A `16 variants × 3 seeds` campaign applied
`compile-max-autotune` plus `z-loss-5e-6-clip01` without changing any other contract
field. It produced 45 finite results and three deliberate fail-fast results:

- all three `kimi-k3-kda-update` seeds developed a non-finite local gradient norm at
  iteration 2 and were stopped before the data-parallel collective;
- `qwen-gdn` regressed to mean BPB `2.939011`, or `+1.431906` relative to the matched
  recipe baseline;
- `kda` regressed to mean BPB `1.613106`, or `+0.106002`;
- over the 14 non-baseline variants with finite results in both systems, delta
  correlation with the historical speedrun reference fell to `-0.320154`.

This is strong evidence against treating a baseline-winning objective/optimizer recipe
as backend infrastructure. It remains available for isolated studies, but the optimized
backend comparison uses only the execution profile `compile-max-autotune` with the
unchanged `baseline` recipe. The partial comparison and explicit failure manifest are in
[`results/megatron-10m-global-zclip-b300`](../results/megatron-10m-global-zclip-b300/).

## Modded-NanoGPT records 1–89

The 89 records are cumulative, so they are classified by the first record that
introduced the mechanism rather than copied into 89 independent flags.

| Records | Mechanisms | Disposition here |
|---|---|---|
| 1–5 | modern GPT, RoPE, QK norm, ReLU², padding, zero projections, Muon | retained |
| 6–10 | distributed Muon, PyTorch upgrade, untied head, value/x0 paths, BF16 | retained or delegated to pinned Megatron/PyTorch |
| 11–21 | U-Net skips, attention/window tuning, value embeddings, softcap, FP8 head, fused QKV, batch tuning | retained where contract-compatible; FP8 head is a B300 capability probe |
| 22–24 | faster all-reduce, overlap, reduce-scatter | delegated to Megatron distributed optimizer/DDP; requires a multi-rank scale probe |
| 25–33 | runtime upgrade, BOS alignment, transposed MLP kernel, attention gate, FA3, layer dropping, YaRN, BF16 cleanup, async data | BOS/BF16/gates retained; FA3 is Hopper-only; shape/schedule changes stay explicit |
| 34–43 | smear, dropped layers, fused Muon comms, BF16 CE, Polar Express, Adam every two, backout, NorMuon, cautious WD | retained or isolated recipes |
| 44–50 | optimizer hooks/overlap, refined skips, batch schedule, lambda placement, Muon reshape, PKO, cautious Adam WD | Megatron/compile or recipes; batch schedule is a contract change |
| 51–57 | retie/split embeddings, scalar schedules, MTP, asymmetric logits, value gates, compiled Adam, mixed weights/interleaving | MTP/value gates/compiled Adam retained or existing arms; dynamic tying is a contract-changing parameter schedule |
| 58–61 | paired-head attention, fused ReLU², fused softcapped CE, unified/transposed optimizer layout | seven heads make exact pairing impossible; kernels are capability-probed; optimizer is already unified |
| 62–71 | bigram hash/sign precursor, untied/fused value embeddings, mimetic V/O init, Torch upgrades, kernel tuning, sparse bigram comms | Engram/smear cover the controlled lexical-memory comparison; H100/vocab-50304 kernels do not fit the 56-wide, vocab-32768 contract |
| 72–80 | sequence schedule, partitioned/simplified hyperconnections, flattened forward, CE/transpose kernels, varlen bounds, paired-head Muon | schedules are explicit contract changes; mHC is an existing arm; compile handles flattening; per-head/full controls are measured |
| 81–89 | MUDD, learnable/algebraic XSA, bigram sign, FP8 MLP, dynamic MHA, fused ReLU², prefix loss, FP8 down projection | XSA/MUDD/SConv/GDN relatives remain architecture arms; prefix loss changes the objective; FP8/fused kernels require aligned larger shapes |

The key reason not to paste the latest single-file trainer into this project is shape and
hardware specialization: its current fused CE assumes vocabulary 50,304, several
Triton kernels target `sm90`, and FP8 matmuls assume dimensions divisible by 16. The
10M contract has width 56, head dimension 8, seven heads, and vocabulary 32,768.

## Marin Agent-MoE inventory

Every tracked issue in the digest is assigned below. MoE-only experiments are not
silently generalized to the dense baseline.

| Area and issue IDs | Disposition |
|---|---|
| PKO/RoPE #4802 #4849 #4899 #4900 #4946 #4976 #5152 #5160 | portable portions are recipes; MHA is already the 10M baseline |
| attention layouts #4907 #4951 #5047 #5151 | paired heads are incompatible with seven heads; wide/MHA changes geometry; MHA already active |
| QK controls #5114 #5227 #5230 #5373 #5381 | QK gain is a recipe; known-negative removal/multiplier sweeps are not promoted; QK norm retained for scaling stability |
| value path #4986 | retained in the inherited baseline for comparison continuity despite Marin's negative MoE result |
| expert topology #4901 #5387 #5399 #5409 #5477 #5486 #5491 #5501 #5502 #5515 | MoE-only scale candidates, outside the dense 10M backend |
| router changes #5409 #5477 #5486 #5491 #5501 #5502 #5507 #5512 | upstream negative or MoE-only; no dense analogue |
| Muon family #5115 #5134 #5167 #5517 #5585 #5596 #6388 #6404 #6505 #8131 | MuonH/MuonEq recipes and LR retuning; factorized gains/row norms remain negative or higher-scale candidates |
| AdamH/numerics #5178 #5203 #5235 #5238 #5250 #5251 | AdamH, init, and clipping recipes; negative epsilon/gate/gradient-aware variants documented |
| precision/pipeline #6431 #6486 | delayed pipeline is a PP scale candidate; BF16-master negative is directly reproduced by backend profiles |
| residuals #4806 #4807 #4905 #4906 #4987 #5110 #5113 #7409 #8196 | x0/backout retained; cached and midpoint K/V are recipes; block/identity HC need deeper models; full residual is upstream negative |
| convolution #8377 | two SConv architecture arms already exist; CUDA/PyTorch backend remains independent of Pallas |
| depth/block #5002 #5154 #5423 #5938 | geometry changes or upstream negative; not folded into backend |
| head/norm #4803 #4973 #5222 #5224 #6442 | softcap retained; z-loss is a scale-stability candidate; negative init/split norms not repeated |
| activations #5407 #5460 #6519 #7255 | xIELU/SiTU arms already exist; GEGLU/SwiGLU would change the architecture comparison |
| data/objective/eval #5306 #5493 #6509 #6570 #7181 | data, sequence, serving, or evaluation questions—not backend optimizations under fixed ClimbMix |
| synthesis #4999 #5371 plus #5184 #5292 #5763 #6882 | dense portable compound is a recipe; expert count/router parts are MoE-only |

## Lessons from all Marin reports and supporting articles

- The 8B and 32B retrospectives make finite checks, deterministic/resumable data order,
  durable manifests, QK norm, and scale-matched validation non-negotiable. Tight clipping,
  update clipping, and bad-step skipping can mitigate symptoms but did not cure structural
  instability; that is why mHC/relative-attention NaNs remain failures rather than hidden
  retries.
- The Grug archive supports borrowing experiment discipline and compact recipe factories,
  not adopting a JAX/TPU runtime as a second backend for this scope.
- The Markdownified Dataset reports concern data construction. They are valuable upstream
  of ClimbMix, but changing the frozen stream would destroy paired attribution.
- The HBM article says to shard before offloading and to avoid materializing giant logits.
  The current 10M run is not memory-bound; optimizer offload/recomputation would be a
  regression here but remain scale candidates.
- The scaling-heuristic article requires optimizer-specific LR/beta/epsilon/batch retuning.
  MuonH/AdamH are therefore screened with explicit retunes before rejection.
- The optimizer tutorial favors registered, serializable configs. This directly motivated
  the compact recipe registry instead of more architecture files.

## Promotion rule

A 30-step run is a compiler/numerics screen, not a quality verdict. A candidate is promoted
only after full-budget seed-42 confirmation, then three seeds under the same token/data
contract. System-only changes must improve steady-state throughput without a material BPB
shift. Architecture/optimizer changes must report both BPB and throughput; no compound can
replace its single-component controls.

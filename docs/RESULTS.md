# Results and interpretation

Historical scaling values in this document are frozen as of 2026-08-24 (UTC+8). The
original matched 10M backend comparison completed on 2026-08-25; the B300 optimized
follow-up, 100M three-node baseline comparison, and single-node native-parallelism
matrix completed on 2026-08-26. Validation BPB is lower-is-better. `Δ` is the mean
paired difference from the same-scale, same-backend baseline over matching seeds.

## Read the regimes separately

The repository records two training-budget regimes:

1. `d14-controlled` and `scaling-clean-v1` train each model for approximately one billion tokens, regardless of parameter count.
2. `parameter-size-sweep-v1` and `parameter-scale-100m-1b-v1` train for approximately 12 tokens per parameter. The 1B-parameter model therefore sees about 12.08 billion tokens.

Absolute BPB values across these regimes are not directly comparable. Paired deltas inside a row group are the intended comparison.

## FineWeb 1M single-B300 compatibility screen

On 2026-08-27, all 17 seed-42 Megatron arms completed under clean commit `36a6830`
with FineWeb/GPT-2, BF16, 2K context, and a 192-sequence global batch accumulated from
12 microbatches of 16. The campaign took 1,231 seconds and had zero failed or
non-finite runs. Engram was best at `2.167429 BPB` (`-0.334663` versus the matched
baseline) with `0.961×` baseline steady throughput; CoLU was the only degradation at
`+0.508443 BPB`.

The baseline is materially sensitive to microbatch accumulation in this 37-step,
approximately 1M-parameter regime. Treat these values as runtime compatibility and
early-screen evidence, not as a replacement for the multi-seed mature-scale results.
The [full table and frozen provenance](../results/fineweb10b-1m-b300-dsw/) are
machine-readable.

## Optimized matched 10M Megatron versus speedrun comparison

The 16 variants completed three seeds on both backends under the same approximately
12-token-per-parameter contract. Megatron used the pinned MCore lifecycle with
`TP=PP=CP=1`; this validates the comparison wrapper, not native model parallelism for
every custom mechanism.

| Backend | Baseline BPB | Best variant | Best BPB | Best Δ | Baseline tok/s |
| --- | ---: | --- | ---: | ---: | ---: |
| Megatron safe-autotune | 1.547758 | KDA | **1.465789** | -0.081969 | 1,576,266* |
| speedrun | 1.554980 | Kimi K3 KDA | **1.461594** | -0.093387 | 1,360,228 |

`*` Megatron uses median steady-state step throughput; speedrun uses its historical
post-warmup aggregate. The comparable Megatron post-warmup aggregate is `1,529,304
tok/s`, or `1.124×` speedrun. Fresh max-autotune caches make one-off Megatron process
wall time much worse (`567.0 s` versus `79.8 s`), so speedrun remains preferable for
cold small-model screens.

Across the 15 non-baseline variants, paired deltas have Pearson correlation `0.971361`,
mean absolute gap `0.009241 BPB`, and matching improvement/degradation direction for
`13/15` variants. Megatron baseline BPB is lower by `0.007222`. The backends therefore
agree strongly on broad ranking but are not numerically interchangeable; causal
comparisons remain paired within a backend.

The initial max-autotune pass produced 43 finite and five fail-fast rows. KDA seeds
42/44 and all three Kimi K3 KDA seeds became non-finite on their first backward pass;
the finite KDA seed 43 and all three finite Qwen GDN rows were quality-corrupted. Nine
default-compile controls were all finite. The accepted set therefore uses max-autotune
for 13 variants and default compile for KDA, Kimi K3 KDA, and Qwen GDN. See the
[complete backend table and provenance audit](BACKEND_COMPARISON.md) and the
[`safe-autotune campaign artifacts`](../results/megatron-10m-safe-autotune-b300/).

The original regular-compile comparison remains published as a causal reference. It
completed 48/48 finite runs plus the three-seed DSA correction and reached correlation
`0.975948`; its Megatron baseline was materially slower (`752,144 tok/s`).

## 100M multi-node baseline reproduction

The 100M baseline ran once per backend as one `3 nodes × 5 B300 GPUs = 15 ranks`
process group. It used NVLink within each node and verified NCCL RoCE/GDRDMA between
nodes. Both backends consumed exactly 192 active sequences per optimizer step; three
ignored rows supplied the static 13-row local shape required because 192 is not
divisible by 15.

| Run | Final BPB | Δ vs current speedrun | Steady tok/s |
| --- | ---: | ---: | ---: |
| Historical single-GPU speedrun | 0.91636648 | +0.00021621 | 999,897* |
| Current 15-rank speedrun | **0.91615027** | — | 1,906,064† |
| Current 15-rank Megatron | 0.91756084 | +0.00141057 | 1,940,137† |

`*` Historical post-warmup aggregate. `†` Sampled or exact median steady step.

The speedrun reproduction passes its pre-registered gate: 14-point Pearson
`0.99999787`, curve MAE `0.00069146 BPB`, and final delta `-0.00021621 BPB`. Megatron
versus current speedrun has 13-point Pearson `0.99983861` and mean absolute gap
`0.00122598 BPB`. The single Megatron seed is slightly better through much of
warmdown, then ends `0.00141057 BPB` worse; this is not enough evidence to rank backend
quality.

Megatron's median steady step is `1.018×` speedrun, but its transition-inclusive
post-warm aggregate is only `0.892×`, and the complete process takes 782.0 seconds.
This distinction is why cold/lifecycle and warmed-step throughput remain separate
metrics. See the [full curve and provenance artifacts](../results/100m-multinode-b300/).

## 100M-class single-node native parallelism

Eight seed-42 Megatron runs used one B300 node, 2K context, global batch 180, and
73,728,000 training tokens. The native harness used an indexed FineWeb-Edu control and
matching DeepSeek-V3 tokenizer, so its validation LM loss is not BPB and is not
comparable with the ClimbMix campaign above.

| Comparison | Steady tok/s ratio | Peak-memory change | Result |
| --- | ---: | ---: | --- |
| PP5 / DP5 | 0.617× | -44.6% | slower capacity path |
| TP2+DP2 / DP4 | 0.389× | -48.6% | slower capacity path |
| fused DP4 / unfused DP4 | **1.274×** | -26.9% | useful B300 optimization |
| CP2+DP2 / fused DP4 | 0.366× | -22.7% | reserve for long context |
| MoE EP2+DP2 / EP1+DP4 | 0.980× | -7.1% | no speed gain at six experts |

All eight runs completed 200/200 steps with return code 0 and zero skipped or
non-finite iterations. DP5 scaled from DP4 at 98.2% per-GPU efficiency. PP, TP, CP,
and EP quality differences are one-seed implementation-path observations rather than
causal topology effects. The full throughput, memory, lifecycle, and validation curves
are in the [single-node artifacts](../results/100m-native-parallelism-b300-1n/).

## Completed 100M and 300M controls

| Variant | 100M BPB | 100M Δ | 100M speed | 300M BPB | 300M Δ | 300M speed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.916171 | 0.000000 | 1.000× | 0.808017 | 0.000000 | 1.000× |
| Qwen GDN | **0.902994** | **-0.013177** | 0.417× | **0.799714** | **-0.008303** | 0.531× |
| Engram | 0.908589 | -0.007582 | 0.948× | 0.803945 | -0.004072 | 0.973× |
| Kimi K3 KDA | 0.909599 | -0.006573 | 0.397× | 0.802384 | -0.005633 | 0.511× |
| KDA | 0.910074 | -0.006097 | 0.384× | 0.804987 | -0.003031 | 0.500× |
| mHC | 0.909991 | -0.006083 | 0.369× | — | — | — |
| sconv-KV | 0.913319 | -0.002852 | 0.723× | 0.806509 | -0.001508 | 0.701× |
| gated attention | 0.914021 | -0.002151 | 0.964× | 0.806801 | -0.001217 | 0.972× |
| xIELU | 0.915622 | -0.000549 | 0.949× | 0.807183 | -0.000834 | 0.942× |
| SiTU-GLU | 0.915929 | -0.000242 | 0.968× | 0.809656 | +0.001638 | 0.929× |
| AttnRes | 0.918689 | +0.002518 | 0.833× | 0.808889 | +0.000872 | 0.759× |
| GLM MLA | 0.918750 | +0.002578 | 0.965× | 0.812049 | +0.004031 | 1.021× |
| sconv-residual | 0.917915 | +0.001744 | 0.722× | 0.808924 | +0.000907 | 0.700× |
| partial RoPE 25% | 0.924229 | +0.008058 | 0.989× | 0.813246 | +0.005229 | 0.991× |
| DSA | 0.981464 | +0.065293 | 0.288× | 0.818233 | +0.010216 | 0.305× |
| relative attention | 0.976844 | +0.060673 | 0.318× | 1.251816 | +0.443799 | 0.301× |

Every complete row has three valid seeds. The 100M mHC value has only two valid seeds and is marked partial; all three 300M mHC runs failed with non-finite values.

### Quality and efficiency

Qwen GDN wins on BPB at both completed mature scales. Its speed ratio improves with scale, but remains only 0.42×/0.53× baseline. Engram gives up some of that quality gain while running at 0.95×/0.97× baseline, making it the strongest observed Pareto trade-off.

Gated attention is a smaller but inexpensive improvement. Short-convolution KV improves more, with a larger throughput cost. Kimi K3 KDA ranks second at 300M by quality but runs near half baseline speed.

### Negative results matter

Relative attention degrades sharply at 2K context and is unstable at 1B. The DSA experiment also loses quality and throughput, but its attention path uses a top-k mask over dense SDPA. It is evidence about this controlled model formulation, not evidence against a production sparse kernel.

## Fixed-token d16/d18 controls

| Variant | d16 BPB | d16 Δ | d18 BPB | d18 Δ |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.830908 | — | 0.820979 | — |
| mHC | **0.826490** | **-0.004419** | **0.817117** | **-0.003862** |
| Qwen GDN | 0.826504 | -0.004404 | 0.817408 | -0.003571 |
| sconv-KV | 0.828381 | -0.002527 | 0.818039 | -0.002940 |
| Engram | 0.827964 | -0.002945 | 0.819709 | -0.001270 |
| xIELU | 0.829311 | -0.001597 | 0.819280 | -0.001699 |
| shared MTP-3 | 0.829831 | -0.001077 | 0.819476 | -0.001504 |
| exclusive attention | 0.829391 | -0.001518 | 0.820107 | -0.000872 |
| per-head Muon | 0.830221 | -0.000688 | 0.819683 | -0.001296 |

mHC is strongest here but is not considered mature because it becomes non-finite in the later per-head-Muon scaling configuration.

## Small-model sweep

The best recorded arms at selected scales were:

| Scale | Variant | BPB | Δ BPB |
| --- | --- | ---: | ---: |
| minimal | Engram | 2.089637 | -0.023286 |
| sub-1M | AttnRes | 2.111835 | -0.048962 |
| ~10M | Kimi K3 KDA | 1.461594 | -0.093387 |

These are winners from a broader sweep, not a substitute for the full per-variant tables.

## 1B dated snapshot

Only six variants had complete three-seed results at the audit time:

| Variant | Mean BPB | Δ BPB | Speed |
| --- | ---: | ---: | ---: |
| AttnRes | **0.706122** | **-0.003726** | 0.682× |
| gated attention | 0.709346 | -0.000502 | 0.988× |
| baseline | 0.709848 | 0.000000 | 1.000× |
| KDA | 0.710721 | +0.000873 | 0.600× |
| SiTU-GLU | 0.713654 | +0.003806 | 0.934× |
| GLM MLA | 0.714892 | +0.005045 | 1.098× |

Campaign counts were 18 complete runs, 3 failed runs, 21 running runs, and 6 pending runs. These counts describe scheduler state, not 18 complete architecture arms.

The table must not be called a final leaderboard. In particular, Qwen GDN was pending and Kimi K3 KDA had not produced a complete arm.

## Metric definitions

- **BPB:** summed token negative log-likelihood divided by the number of represented UTF-8 bytes and by `ln(2)`. Unlike per-token loss, it is comparable across tokenizations in principle; this campaign nevertheless fixes the tokenizer.
- **Paired Δ BPB:** for each seed, variant final BPB minus its aligned baseline final BPB, averaged over valid paired seeds.
- **Throughput ratio:** variant training tokens per second divided by aligned baseline throughput on the campaign hardware. The optimized Megatron table uses median steady-state steps; its artifacts also retain transition-inclusive post-warmup throughput and full process wall time.
- **Parameters:** probed instantiated parameter count. Engram's retrieval tables are trainable parameters and are included.

Historical scale values, seed validity, parameter counts, and statuses are in
[`results/key-metrics.csv`](../results/key-metrics.csv). The original and optimized
10M comparisons have separate [legacy artifacts](../results/backend-10m-comparison.csv)
and [B300 safe-autotune artifacts](../results/megatron-10m-safe-autotune-b300/) so
runtime provenance is not mixed into the older campaigns.

## Interpretation rules

1. Compare deltas only within the same campaign and scale.
2. Treat incomplete or non-finite arms as missing evidence, never as a numeric loss.
3. Do not infer sparse-kernel speed from the dense masked DSA backend.
4. Keep baseline and single-component arms when testing combinations.
5. Treat throughput ratios as hardware- and implementation-specific.

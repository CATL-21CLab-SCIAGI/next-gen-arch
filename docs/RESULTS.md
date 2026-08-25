# Results and interpretation

Historical scaling values in this document are frozen as of 2026-08-24 (UTC+8). The matched 10M backend comparison was completed on 2026-08-25. Validation BPB is lower-is-better. `Δ` is the mean paired difference from the same-scale, same-backend baseline over matching seeds.

## Read the regimes separately

The repository records two training-budget regimes:

1. `d14-controlled` and `scaling-clean-v1` train each model for approximately one billion tokens, regardless of parameter count.
2. `parameter-size-sweep-v1` and `parameter-scale-100m-1b-v1` train for approximately 12 tokens per parameter. The 1B-parameter model therefore sees about 12.08 billion tokens.

Absolute BPB values across these regimes are not directly comparable. Paired deltas inside a row group are the intended comparison.

## Matched 10M Megatron versus speedrun comparison

The 16 variants completed three seeds on both backends under the same approximately
12-token-per-parameter contract. Megatron used the pinned MCore lifecycle with
`TP=PP=CP=1`; this validates the comparison wrapper, not native model parallelism for
every custom mechanism.

| Backend | Baseline BPB | Best variant | Best BPB | Best Δ | Baseline tok/s |
| --- | ---: | --- | ---: | ---: | ---: |
| Megatron | 1.533885 | KDA | **1.464074** | -0.069811 | 752,144 |
| speedrun | 1.554980 | Kimi K3 KDA | **1.461594** | -0.093387 | 1,360,228 |

Across the 15 non-baseline variants, paired deltas have Pearson correlation `0.975948`,
mean absolute gap `0.012369 BPB`, and matching improvement/degradation direction for
`12/15` variants. Megatron baseline throughput is `0.553×` speedrun, while its baseline
BPB is lower by `0.021095`. The backends therefore agree strongly on broad ranking but
are not numerically interchangeable; causal comparisons must remain paired within a
backend.

The original Megatron pass completed 48/48 finite runs. A post-run audit found that DSA
read the fixed resume iteration instead of Megatron's live `curr_iteration`, leaving its
sparsity schedule in warmup. The adapter was fixed and all three DSA seeds were rerun;
only those exact keys replace the invalid rows. The corrected DSA result is `1.625196`
BPB (`+0.091311`), consistent with the speedrun negative result. See the
[complete backend table and provenance audit](BACKEND_COMPARISON.md),
[`backend-10m-runs.csv`](../results/backend-10m-runs.csv), and
[`backend-10m-comparison.json`](../results/backend-10m-comparison.json).

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
- **Throughput ratio:** variant training tokens per second divided by aligned baseline throughput on the campaign hardware.
- **Parameters:** probed instantiated parameter count. Engram's retrieval tables are trainable parameters and are included.

Historical scale values, seed validity, parameter counts, and statuses are in [`results/key-metrics.csv`](../results/key-metrics.csv). The 10M backend comparison has separate [summary](../results/backend-10m-comparison.csv) and [per-run](../results/backend-10m-runs.csv) tables so its runtime provenance is not mixed into the older campaigns.

## Interpretation rules

1. Compare deltas only within the same campaign and scale.
2. Treat incomplete or non-finite arms as missing evidence, never as a numeric loss.
3. Do not infer sparse-kernel speed from the dense masked DSA backend.
4. Keep baseline and single-component arms when testing combinations.
5. Treat throughput ratios as hardware- and implementation-specific.

# 10M backend comparison

All deltas and throughput ratios are paired to the same backend and seed baseline.

| Backend | Variant | Seeds | Mean BPB | Paired Δ BPB | Throughput | tok/s | Basis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| megatron | kda | 3 | 1.465789 | -0.081969 | 0.34× | 543,621 | median steady-state step |
| megatron | kimi-k3-kda-update | 3 | 1.468116 | -0.079642 | 0.35× | 552,560 | median steady-state step |
| megatron | engram | 3 | 1.489828 | -0.057930 | 1.00× | 1,572,868 | median steady-state step |
| megatron | inkling-sconv-residual | 3 | 1.490894 | -0.056864 | 0.97× | 1,523,310 | median steady-state step |
| megatron | qwen-gdn | 3 | 1.493180 | -0.054578 | 0.39× | 617,318 | median steady-state step |
| megatron | attnres | 3 | 1.499598 | -0.048160 | 0.98× | 1,539,400 | median steady-state step |
| megatron | inkling-relative-attention | 3 | 1.522713 | -0.025045 | 0.40× | 620,941 | median steady-state step |
| megatron | inkling-sconv-kv | 3 | 1.523050 | -0.024708 | 0.97× | 1,528,318 | median steady-state step |
| megatron | mhc | 3 | 1.529482 | -0.018276 | 0.86× | 1,350,918 | median steady-state step |
| megatron | gated-attention | 3 | 1.540339 | -0.007419 | 1.01× | 1,600,019 | median steady-state step |
| megatron | glm-mla | 3 | 1.545573 | -0.002185 | 0.99× | 1,567,109 | median steady-state step |
| megatron | xielu | 3 | 1.545928 | -0.001830 | 0.98× | 1,544,671 | median steady-state step |
| megatron | baseline | 3 | 1.547758 | +0.000000 | 1.00× | 1,576,266 | median steady-state step |
| megatron | situ-glu | 3 | 1.549041 | +0.001283 | 1.01× | 1,599,399 | median steady-state step |
| megatron | partial-rope-25 | 3 | 1.558349 | +0.010591 | 0.99× | 1,561,553 | median steady-state step |
| megatron | dsa | 3 | 1.627023 | +0.079265 | 0.47× | 747,028 | median steady-state step |
| speedrun | kimi-k3-kda-update | 3 | 1.461594 | -0.093387 | 0.46× | 629,414 | aggregate after warmup |
| speedrun | kda | 3 | 1.462224 | -0.092757 | 0.45× | 607,291 | aggregate after warmup |
| speedrun | qwen-gdn | 3 | 1.483103 | -0.071878 | 0.47× | 645,453 | aggregate after warmup |
| speedrun | engram | 3 | 1.489722 | -0.065259 | 0.99× | 1,352,948 | aggregate after warmup |
| speedrun | attnres | 3 | 1.498153 | -0.056828 | 0.92× | 1,257,371 | aggregate after warmup |
| speedrun | inkling-sconv-residual | 3 | 1.505017 | -0.049964 | 0.96× | 1,303,316 | aggregate after warmup |
| speedrun | mhc | 3 | 1.539949 | -0.015031 | 0.64× | 871,335 | aggregate after warmup |
| speedrun | inkling-sconv-kv | 3 | 1.547653 | -0.007328 | 0.94× | 1,284,914 | aggregate after warmup |
| speedrun | gated-attention | 3 | 1.549218 | -0.005763 | 0.98× | 1,326,877 | aggregate after warmup |
| speedrun | glm-mla | 3 | 1.553357 | -0.001624 | 1.01× | 1,374,704 | aggregate after warmup |
| speedrun | baseline | 3 | 1.554980 | +0.000000 | 1.00× | 1,360,228 | aggregate after warmup |
| speedrun | xielu | 3 | 1.557475 | +0.002495 | 0.98× | 1,336,547 | aggregate after warmup |
| speedrun | inkling-relative-attention | 3 | 1.558444 | +0.003463 | 0.42× | 564,529 | aggregate after warmup |
| speedrun | situ-glu | 3 | 1.559727 | +0.004747 | 1.02× | 1,385,975 | aggregate after warmup |
| speedrun | partial-rope-25 | 3 | 1.579620 | +0.024639 | 1.01× | 1,373,968 | aggregate after warmup |
| speedrun | dsa | 3 | 1.633543 | +0.078563 | 0.32× | 434,542 | aggregate after warmup |

## Cross-backend agreement

- Compared variants: 15
- Variant-delta Pearson correlation: 0.971361
- Mean absolute paired-delta gap: 0.009241 BPB
- Improvement/degradation sign agreement: 86.7%
- Megatron minus speedrun baseline BPB: -0.007222
- Megatron/speedrun absolute baseline throughput: 1.16×

## Run provenance

- Megatron backend profile(s): `compile, compile-max-autotune`
- Megatron optimization recipe(s): `baseline`
- Full source, diff/worktree, and Megatron commit fields are retained in `runs.csv` and `comparison.json`.

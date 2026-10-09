# 10M backend comparison

All deltas and throughput ratios are paired to the same backend and seed baseline.

**Controlled-delta warning:** Megatron uses optimization recipe(s) `z-loss-5e-6-clip01` while the historical speedrun reference does not. The cross-system metrics therefore combine recipe and runtime effects; they are not a pure backend-equivalence measurement.

| Backend | Variant | Seeds | Mean BPB | Paired Δ BPB | Throughput | tok/s | Basis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| megatron | inkling-sconv-residual | 3 | 1.480435 | -0.026669 | 0.97× | 1,436,656 | aggregate after warmup |
| megatron | engram | 3 | 1.485324 | -0.021780 | 0.98× | 1,463,967 | aggregate after warmup |
| megatron | inkling-relative-attention | 3 | 1.494837 | -0.012267 | 0.40× | 597,002 | aggregate after warmup |
| megatron | inkling-sconv-kv | 3 | 1.497236 | -0.009868 | 0.97× | 1,446,822 | aggregate after warmup |
| megatron | attnres | 3 | 1.500325 | -0.006780 | 0.96× | 1,429,376 | aggregate after warmup |
| megatron | mhc | 3 | 1.500852 | -0.006252 | 0.83× | 1,240,151 | aggregate after warmup |
| megatron | glm-mla | 3 | 1.504438 | -0.002666 | 0.98× | 1,453,044 | aggregate after warmup |
| megatron | gated-attention | 3 | 1.504770 | -0.002334 | 1.00× | 1,491,347 | aggregate after warmup |
| megatron | baseline | 3 | 1.507104 | +0.000000 | 1.00× | 1,489,513 | aggregate after warmup |
| megatron | xielu | 3 | 1.510216 | +0.003112 | 0.98× | 1,467,763 | aggregate after warmup |
| megatron | situ-glu | 3 | 1.510928 | +0.003824 | 1.00× | 1,496,033 | aggregate after warmup |
| megatron | partial-rope-25 | 3 | 1.535240 | +0.028136 | 1.00× | 1,484,078 | aggregate after warmup |
| megatron | kda | 3 | 1.613106 | +0.106002 | 0.38× | 562,029 | aggregate after warmup |
| megatron | dsa | 3 | 1.615488 | +0.108384 | 0.13× | 194,981 | aggregate after warmup |
| megatron | qwen-gdn | 3 | 2.939011 | +1.431906 | 0.49× | 736,248 | aggregate after warmup |
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

- Compared variants: 14
- Variant-delta Pearson correlation: -0.320154
- Mean absolute paired-delta gap: 0.134696 BPB
- Improvement/degradation sign agreement: 78.6%
- Megatron minus speedrun baseline BPB: -0.047876
- Megatron/speedrun absolute baseline throughput: 1.10×

## Run provenance

- Megatron backend profile(s): `compile-max-autotune`
- Megatron optimization recipe(s): `z-loss-5e-6-clip01`
- Full source, diff/worktree, and Megatron commit fields are retained in `runs.csv` and `comparison.json`.

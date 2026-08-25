# Megatron versus speedrun at approximately 10M parameters

This comparison asks whether the repository's Megatron execution path preserves the
small-model architecture signal observed with the inherited speedrun backend. It is a
backend comparison, not a claim that the two implementations are numerically or
performance equivalent.

## Frozen contract

Both backends use the same 16 architecture names, seeds `42/43/44`, ClimbMix data order,
32K tokenizer, 2,048-token context, BF16, global batch of 192 sequences, approximately
12 training tokens per parameter, and a 3,932,160-token validation budget. The baseline
has 9,363,488 parameters and trains for 286 steps, or 112,459,776 tokens. Variant step
counts change only enough to retain the tokens-per-parameter contract.

The Megatron campaign ran through the MCore training lifecycle with one process per GPU
and `TP=PP=CP=1`. All 16 architecture mechanisms were constructed by this repository;
the pinned upstream Megatron-LM submodule was not modified. These runs validate the
small-scale wrapper and architecture signal. They do **not** yet demonstrate that every
custom mechanism is tensor-, pipeline-, or context-parallel native.

## B300 safe-autotune follow-up

The release comparison uses a measured compiler policy rather than applying one
compiler mode blindly. Thirteen numerically stable variants use
`compile-max-autotune`; KDA, Kimi K3 KDA, and Qwen GDN use default `torch.compile`.
The optimization recipe remains the unchanged per-head-Muon `baseline` for every row.

Each BPB delta and relative throughput value is paired against the same backend and
seed baseline. Megatron throughput below is the median steady-state step rate; the
historical speedrun reference reports its aggregate rate after warmup.

| Variant | Megatron BPB | Megatron Δ | Megatron throughput | Speedrun BPB | Speedrun Δ | Speedrun throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 1.547758 | +0.000000 | 1.00× | 1.554980 | +0.000000 | 1.00× |
| `engram` | 1.489828 | -0.057930 | 1.00× | 1.489722 | -0.065259 | 0.99× |
| `kda` | 1.465789 | -0.081969 | 0.34× | 1.462224 | -0.092757 | 0.45× |
| `kimi-k3-kda-update` | 1.468116 | -0.079642 | 0.35× | 1.461594 | -0.093387 | 0.46× |
| `qwen-gdn` | 1.493180 | -0.054578 | 0.39× | 1.483103 | -0.071878 | 0.47× |
| `attnres` | 1.499598 | -0.048160 | 0.98× | 1.498153 | -0.056828 | 0.92× |
| `mhc` | 1.529482 | -0.018276 | 0.86× | 1.539949 | -0.015031 | 0.64× |
| `gated-attention` | 1.540339 | -0.007419 | 1.01× | 1.549218 | -0.005763 | 0.98× |
| `situ-glu` | 1.549041 | +0.001283 | 1.01× | 1.559727 | +0.004747 | 1.02× |
| `inkling-relative-attention` | 1.522713 | -0.025045 | 0.40× | 1.558444 | +0.003463 | 0.42× |
| `glm-mla` | 1.545573 | -0.002185 | 0.99× | 1.553357 | -0.001624 | 1.01× |
| `xielu` | 1.545928 | -0.001830 | 0.98× | 1.557475 | +0.002495 | 0.98× |
| `inkling-sconv-kv` | 1.523050 | -0.024708 | 0.97× | 1.547653 | -0.007328 | 0.94× |
| `inkling-sconv-residual` | 1.490894 | -0.056864 | 0.97× | 1.505017 | -0.049964 | 0.96× |
| `partial-rope-25` | 1.558349 | +0.010591 | 0.99× | 1.579620 | +0.024639 | 1.01× |
| `dsa` | 1.627023 | +0.079265 | 0.47× | 1.633543 | +0.078563 | 0.32× |

Across the 15 non-baseline variants, paired-delta Pearson correlation is `0.971361`,
the mean absolute delta gap is `0.009241 BPB`, and direction agrees for `13/15`
variants (`86.7%`). KDA and Kimi K3 KDA remain the quality leaders. Engram and
short-convolution residual preserve roughly all baseline steady throughput while
improving BPB substantially.

The Megatron baseline averages `1,576,266 tok/s` by median steady step and
`1,529,304 tok/s` across measured post-warmup steps, respectively `1.159×` and
`1.124×` the historical speedrun aggregate (`1,360,228 tok/s`). This is not a claim
that Megatron wins a one-off 10M wall-clock race: fresh isolated max-autotune caches
raise mean Megatron process wall time to `567.0 s`, versus `79.8 s` for speedrun. The
steady-state gain matters when compiler caches are reused or the training budget is
large; speedrun remains the better cold-start screening backend.

### Numerical-safety recovery

The primary `max-autotune` pass terminated all 48 keys: 43 were finite and five
failed fast. KDA seeds 42/44 and all three Kimi K3 KDA seeds produced non-finite local
gradient norms on their first backward pass. KDA seed 43 was finite but corrupted;
all three Qwen GDN runs were finite but severely corrupted (mean BPB about `2.68`).

Nine isolated default-compile controls—three seeds for each sensitive variant—were
all finite and restored the expected quality signal. The accepted comparison contains
39 untouched max-autotune rows plus those nine explicit controls. The aggregation tool
distinguishes replacements from missing-key recovery and refuses silent overlap. Raw
failure/corruption evidence remains preserved; it was not converted into numeric losses.

The campaign ran on three NVIDIA B300 nodes (`sm_103a`) with GPUs `0,2,4,5,7`
explicitly allowlisted on each node. GPUs `1,3,6` were already occupied and were never
claimed.

## Original compile comparison

Each BPB delta and relative throughput value is paired against the same backend and seed
baseline. Lower BPB is better. Absolute throughput is reported in the machine-readable
artifacts and remains hardware- and runtime-specific.

| Variant | Megatron BPB | Megatron Δ | Megatron throughput | Speedrun BPB | Speedrun Δ | Speedrun throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 1.533885 | +0.000000 | 1.00× | 1.554980 | +0.000000 | 1.00× |
| `engram` | 1.488966 | -0.044920 | 0.94× | 1.489722 | -0.065259 | 0.99× |
| `kda` | 1.464074 | -0.069811 | 0.75× | 1.462224 | -0.092757 | 0.45× |
| `kimi-k3-kda-update` | 1.465779 | -0.068107 | 0.78× | 1.461594 | -0.093387 | 0.46× |
| `qwen-gdn` | 1.492595 | -0.041291 | 0.70× | 1.483103 | -0.071878 | 0.47× |
| `attnres` | 1.498124 | -0.035761 | 0.60× | 1.498153 | -0.056828 | 0.92× |
| `mhc` | 1.525564 | -0.008321 | 0.44× | 1.539949 | -0.015031 | 0.64× |
| `gated-attention` | 1.533089 | -0.000797 | 0.97× | 1.549218 | -0.005763 | 0.98× |
| `situ-glu` | 1.532925 | -0.000960 | 0.97× | 1.559727 | +0.004747 | 1.02× |
| `inkling-relative-attention` | 1.528361 | -0.005524 | 0.35× | 1.558444 | +0.003463 | 0.42× |
| `glm-mla` | 1.531783 | -0.002103 | 0.97× | 1.553357 | -0.001624 | 1.01× |
| `xielu` | 1.536003 | +0.002118 | 0.88× | 1.557475 | +0.002495 | 0.98× |
| `inkling-sconv-kv` | 1.540332 | +0.006447 | 0.93× | 1.547653 | -0.007328 | 0.94× |
| `inkling-sconv-residual` | 1.493776 | -0.040110 | 0.93× | 1.505017 | -0.049964 | 0.96× |
| `partial-rope-25` | 1.560241 | +0.026355 | 0.98× | 1.579620 | +0.024639 | 1.01× |
| `dsa` | 1.625196 | +0.091311 | 0.35× | 1.633543 | +0.078563 | 0.32× |

Across the 15 non-baseline variants:

- paired-delta Pearson correlation is `0.975948`;
- mean absolute paired-delta gap is `0.012369 BPB`;
- improvement/degradation direction agrees for `12/15` variants (`80%`);
- Megatron's baseline is `-0.021095 BPB` below speedrun's baseline;
- Megatron's absolute baseline throughput is `0.553×` speedrun's, so speedrun is about
  `1.81×` faster for this small baseline.

The three direction disagreements are SiTU-GLU, Inkling relative attention, and
short-convolution KV. They are small enough relative to the backend shift that none
should be promoted from this scale without a larger paired control.

KDA and Kimi K3 KDA are the quality leaders in both backends. Engram and
short-convolution residual retain most Megatron baseline throughput while providing a
large quality improvement. DSA remains a negative result in both backends. Its current
implementation uses masked dense SDPA, so the throughput row says nothing about a true
sparse kernel.

## DSA correction and provenance

The first 48-run Megatron pass completed, but post-run validation found that DSA read
Megatron's checkpoint iteration field, which remains fixed during training, instead of
the live `curr_iteration` field. DSA therefore never left its dense warmup schedule.
The adapter was corrected and all three DSA seeds were rerun under the unchanged
contract. The published comparison overlays exactly those three keys; the comparison
tool rejects duplicate or unknown correction rows.

- 45 accepted Megatron rows: next-gen-arch commit
  `e6d9b0b1153e74078dbb87d4c0e8b12c8d4df513`
- 3 corrected DSA rows: next-gen-arch commit
  `ed8336e5403d8da75082502a96a115f06ee17334`
- all Megatron rows: upstream commit
  `55ac7082517c3878ae653c07c09c534b8aed49f6`

The full pass completed `48/48` finite runs and the correction completed `3/3` finite
runs. It used three nodes with five explicitly allowlisted NVIDIA B300 GPUs per node. Three
pre-existing busy GPUs on each node were excluded and left untouched.

## Reproduce the aggregation

Run the primary campaign and any correction campaign into separate durable directories,
then aggregate them explicitly:

```bash
python -m next_gen_arch.training.campaign_compare \
  --megatron-root /path/to/primary-campaign \
  --override-root /path/to/dsa-correction \
  --reference results/speedrun-10m-reference.csv \
  --output-dir /path/to/comparison
```

Published artifacts:

- [`megatron-10m-safe-autotune-b300`](../results/megatron-10m-safe-autotune-b300/):
  the optimized 96-row comparison, policy manifest, and per-run provenance;
- [`backend-10m-comparison.csv`](../results/backend-10m-comparison.csv): 32 backend/variant summaries;
- [`backend-10m-runs.csv`](../results/backend-10m-runs.csv): all 96 accepted per-seed rows with source provenance;
- [`backend-10m-comparison.json`](../results/backend-10m-comparison.json): raw rows, summaries, and cross-backend metrics;
- [`speedrun-10m-reference.csv`](../results/speedrun-10m-reference.csv): frozen historical speedrun input.

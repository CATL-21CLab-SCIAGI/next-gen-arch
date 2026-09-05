# Width-320 backbone review map

The selected recipe is `recipes/models/qwen38_flash_next_w320_e32.yaml`.
It is a portable experiment contract, not a YAML-driven model constructor.
The named Python factory is the executable source of geometry; each run emits
`RUN_CONTRACT.json`, `MODEL_SHAPES.json`, `MODEL_PARAMETERS.json`, and
`PARALLELISM.json` so actual construction can be checked against the recipe.

## Files to review

- `src/archlab/architectures/qwen38_flash_next_full.py`:
  `Qwen38FlashNextFullConfig.width320_e32_depth48_no_mtp()` derives feature
  dimensions from H=320. `__post_init__` pins the selected experiment and rejects
  shape drift. `FourStreamGatedResidual`, `GroupRMSNorm`, `GatedDeltaNet`, and
  `DistributedPLE` implement the custom backbone mechanisms. The closed-form
  `parameter_count_contract()` is checked against native construction.
- `src/archlab/megatron/qwen38_flash_next_full_train.py`:
  training entry `python -m archlab.megatron.qwen38_flash_next_full_train`.
  `_megatron_argv` controls native optimizer, schedule, batching, precision and
  process groups. `_build_model_classes` wires GDN/global-attention alternation,
  residual branches, PLE injection and native MoE. `build_model` is the common
  construction boundary. `DPRankTokenBatches` defines deterministic data order.
- `src/archlab/megatron/gated_qkv.py`: four native TE projections packed for
  native gated SelfAttention. Separate Q/gate/K/V matrices prevent frozen
  Muon's ungated-only packed-QKV splitter from interpreting gate rows as K/V.
  This adapter is deliberately TP1-only, matching DP-only experiments.
- `scripts/run_qwen38_flash_next_full_dlc.sh`: frozen-container launch,
  immutable source/hash checks, 32-rank collectives, microbatch and fusion flags.
  Resident controllers enter through `scripts/run_qwen38_27b_quarter_dlc.sh`
  and the narrowly validated compatibility forwarder; nodes are not restarted.
- `tests/test_qwen38_width320.py`: count/flag contracts, zero-centered RMSNorm,
  GDN CUDA/recurrent oracle, and native gated-attention/DP-gradient oracle.

## Backbone variants

Create a new named family and recipe when changing width ratios, expert count,
top-k, GR streams/rank, GDN/global-attention cadence, or PLE dimensions. Do not
silently edit a historical family. Keep exact-count and feature assertions,
optimizer matrix grouping, numerical oracles, DP equivalence and native
checkpoint round-trip tests. Start geometry changes with fresh weights.

Native attention output gating means multiplying each head's attended values by
`sigmoid(W_gate x)` before the output projection. It controls how much contextual
information reaches the residual branch. Q/K normalization independently
RMS-normalizes each query/key head before RoPE and attention scores. Both are
enabled here, together with four residual streams and zero-centered GR/PLE/QK
norm parameters (effective scale `1 + weight`). GDN output norm uses direct gamma.

MTP, QSA/indexer and vision remain intentionally disabled. DP=32; TP=PP=EP=CP=
expert-TP=1. The 32 PLE partitions are all local on every GPU, not EP shards.

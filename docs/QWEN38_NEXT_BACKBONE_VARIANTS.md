# Reviewing and varying the Qwen3.8-Next backbone

The current experiment is a 1,006,441,440-parameter, 48-layer hybrid with MTP off.
The DP-only execution contract puts a complete model, all 64 routed experts,
and all 32 PLE table partitions on every GPU. TP, PP, EP, expert TP, and CP are
all 1; DP is 32. Native Muon distributes optimizer work/state across DP replicas;
this is not model/expert sharding. No DLC node or vendor runtime restart is needed.

## Files to review

| File | Responsibility / controls |
|---|---|
| `scripts/run_qwen38_flash_next_full_dlc.sh` | Container paths, immutable-commit checks, collective preflight, four-node torchrun, and the 1B DP/fusion switches. |
| `src/archlab/megatron/qwen38_flash_next_full_train.py` | Training entry. `_parser` owns CLI controls; `_megatron_argv` translates to the frozen native trainer; `build_model` constructs the model; `_run` binds the optimizer, data, checkpointing, and native schedule. |
| `src/archlab/architectures/qwen38_flash_next_full.py` | Model configuration, GDN, gated residuals, PLE hashing/embedding/injection, and closed-form parameter counts. No trainer imports. |
| `recipes/models/qwen38_flash_next_1b_depth48_no_mtp.yaml` | Human-readable pinned model, execution, optimizer, and data-budget contract. This is documentation/contract, **not a dynamically loaded model builder**. Editing it alone does not change training. |
| `src/archlab/megatron/qwen38_flash_next_sample.py` | Load a native checkpoint as one complete replica and generate bounded continuations without optimizer updates. Uses the same `build_model` as training. |
| `src/archlab/prompts/backbone_validation.yaml` | Versioned qualitative prompts; change/add prompts here, not inside evaluation code. |

Resident DLC controllers predate the dedicated launcher. Their admitted
`run_qwen38_27b_quarter_dlc.sh` path forwards validated `compat-qwen38-flash-next-*`
handles through `run_qwen38_27b_full_dlc.sh` to the actual Flash-Next launcher.
These names are compatibility plumbing, not the current architecture.

## Backbone controls

Start at `Qwen38FlashNextFullConfig.billion_depth48_no_mtp()` and its validation
in `__post_init__`. The named 1B family is deliberately shape-pinned: silently
changing constants under the same family is rejected. For an intentional variant,
add a named config/recipe and its parameter/shape tests, then expose the choice in
the trainer and launcher. Use a fresh run directory for changed weight geometry.

| Mechanism | Current values | Where it is implemented |
|---|---|---|
| Depth / width | 48 / 384 | Config plus native transformer block construction |
| Attention pattern | 3 GDN layers, then 1 global-attention layer | `full_attention_interval=4`; adapter `QwenFlashNextLayer` chooses per layer |
| Global attention | 6 query heads, 1 KV head, head dim 64; rotary fraction 0.25 | Native Megatron `SelfAttention`; GQA must stay explicitly enabled |
| GDN | 4 QK heads, 12 value heads, key/value dim 32, convolution width 4 | Architecture `GatedDeltaNet` and causal convolution |
| Routed/shared FFN | 64 experts, top-3, expert/shared width 112 | Native `MoELayer` and router; adapter `SplitSwiGLUExperts` / `SplitSwiGLUSharedExpert` use TE grouped/dense linears |
| Residual path | 1 stream, low-rank width 48 | Architecture `FourStreamGatedResidual`; applied around attention and FFN |
| PLE | Layer 2, four hash heads, about 1M rows/head, branch width 96 | Architecture `PLEHash`, `OwnerShardedPLEEmbedding`, `DistributedPLE` |
| MTP | Disabled | `mtp_num_layers=0`, no MTP spec or auxiliary objective |

Expert count/top-k are architecture choices even in DP-only mode: EP=1 does not
turn the MoE into a dense MLP. Attention implementation and FFN assembly live at
the Megatron integration boundary; standalone mechanisms stay in `architectures`.

## Training controls and comparison discipline

- Keep tokenizer/data order, sequence length, global batch, seed, token budget,
  LR schedule, loss normalization, precision, and evaluation windows fixed when
  comparing backbone variants. Compare CE at equal consumed tokens, and speed in
  tokens/second rather than optimizer steps/second.
- Native CLI controls include micro/global batch, LR/min-LR/warmup, clipping,
  evaluation/logging intervals, resume/load directory, parallelism, and fusion
  switches. The production launcher deliberately pins several of these values.
- `--parallelism dp-only` requires a PP1 config and checks actual runtime groups.
  `--fused-moe` enables permutation and router fusion. `--fused-cross-entropy`
  selects **native** fusion; TE loss fusion is forbidden by this container's
  stability guard. No vendor source or installed-package edits are necessary.
- `RUN_CONTRACT.json` preserves the original run contract. `contracts/attempt-*`
  and `LATEST_CONTRACT.json` record the current execution and runtime identity.
  `PARALLELISM.json` records the actual initialized process-group sizes.
- Live logs default to `/mnt/nas/evergreen/arch-live-logs/<run-name>/` and are
  mirrored into the run's OSS logs. The separate NAS directory is intentional:
  `/mnt/nas/evergreen/next-gen-arch` is an OSS compatibility symlink on this job.
- Shape/attention/loss-normalization changes must not silently resume an
  incompatible checkpoint. Execution-only changes still need numerical and
  checkpoint save/load gates; optimizer state must never be silently discarded.

## Validation entry points

Run tests in the existing NeMo container with its CUDA paths (including
`TRITON_PTXAS_PATH=$CUDA_HOME/bin/ptxas`) and the checkout's `src` plus the
container's Megatron checkout on `PYTHONPATH`:

- `tests/test_qwen38_flash_next_full.py`: shapes, parameter allocation, PLE, GDN,
  and residual mechanisms.
- `tests/test_qwen38_flash_next_full_train.py`: loss/auxiliary scaling, construction
  arguments, optimizer grouping, checkpoint staging, and DP-only guards.
- `tests/test_qwen38_flash_next_fusions.py`: CUDA fused/unfused forward/backward
  comparisons. Run under single-rank torchrun for the native loss collective.
- A bounded distributed train/save/load run must also verify finite/nonzero
  gradients, full parameter counts, DP replica equality, and resume continuity.

Sampling recomputes the entire prefix for every generated token. It avoids
assuming GDN/PLE have compatible incremental caches; the current sampler is not
an inference throughput benchmark and does not apply an instruction/chat template.

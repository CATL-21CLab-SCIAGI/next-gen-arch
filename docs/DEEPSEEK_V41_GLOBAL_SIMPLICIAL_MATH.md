# DeepSeek V4.1 + full-history softmax 2-simplicial adapters

Status, 2026-09-10: CPU tokenization launched; training is a proposal, not implemented
or launched. Model-weight downloads are deferred by the user. No subagents, package
installs, DLC stops, or node/controller restarts.

## Completed operations

- The previous Qwen pretrained adapter was already paused at step 8,089. Its
  `results/pretrained-simplicial-fineweb-20260907-v1/checkpoints/step-00008089-8d72ca55c44d/COMPLETE.json`
  remains present.
- After the user's explicit clarification, SIGTERM was sent to verified native
  Megatron rank zero, PID 285042, for the separate 0.388B run. Native distributed
  signal handling saved step **3,617**, then exited. The checkpoint marker and
  successful-save log agree. The checkpoint represents **30,341,595,136 tokens**.
- Checkpoint: `results/simplicial-production-w320-c-20260909-v1/checkpoints/iter_0003617`.
  All four nodes' training processes exited; controllers stayed alive and GPUs
  released their allocations. There is no automatic resume.
- CPU preprocessing PID **189474**, host `dlc1hig7iitpry5n-worker-0` / `22.0.254.34`.
  It uses `/opt/venv/bin/python`, 32 worker processes, and no visible CUDA devices.
- 47 tests passed, including the official encoder's complete test suite and our
  multi-turn/tool supervision checks. A five-source smoke conversion produced
  40 conversations / 360,350 tokens, with indexed readback and SHA256 verification.

## Tokenization contract

Input: `/mnt/oss-dataset/datasets/nv-community/Nemotron-Math-v2`.
Read only its five Parquet files, not their duplicate JSONL mirror:
**7,085,839 conversations**, split into 3,545 resumable row-group tasks.

The exact model is [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash),
pinned to `df42c109f1defefcbfcedbe7d905718a12266e40`. Only tokenizer, configuration,
official reference code, and small provenance/test assets have been fetched.
No `.safetensors` weight shard was downloaded. An optional weight-index download
timed out; incomplete copies are explicitly named `.incomplete` / `.partial` and
must not be treated as a validated checkpoint inventory.

V4.1 has **no Jinja chat template**. We use its unmodified
[official encoder](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/encoding/encoding.py).
It differs from V4 in spaced DSML tags, numeric effort, and system-message handling.
The encoder SHA256 is checked before importing it. `tokenizer.json` SHA256:
`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.

- Keep all source effort levels and tool/no-tool trajectories, including reasoning.
- Use native thinking mode, `drop_thinking=False`, **effort 75** (the native default)
  for all examples. This is an explicit fixed training setting, **not** a claim
  that GPT-OSS effort labels map to DeepSeek budgets. Original effort stays in metadata.
- Tools are rendered, never executed. Tool definitions enter the native system schema.
- Preserve complete conversations; no truncation, extra EOD, or trajectory deduplication.
- Store native BOS/EOS and int32 tokens with document boundaries. Compare fast
  tokenization to Transformers for every observed effort/tool combination per worker.
- Preserve assistant target spans, including reasoning, answers, calls and EOS;
  exclude prompt headers and tool results. Spans are half-open indices in the
  unshifted token sequence: the trainer must shift them exactly once with labels.
- Reserve 1% of normalized problem-hash buckets for validation. Group all variants
  of a problem together. This is exact/normalized grouping, not near-duplicate detection.
- Retain source provenance, licensing, source-row IDs and token-length histograms.

Outputs and monitoring:

```text
results/nemotron-math-v41-20260910-v1/progress.json
results/nemotron-math-v41-20260910-v1/parts/*/READY.json
results/nemotron-math-v41-20260910-v1-stage/job.log
results/nemotron-math-v41-20260910-v1-stage/launcher.json
results/nemotron-math-v41-20260910-v1-stage/parts/*/progress.json
```

Top-level progress counts only published/checksummed parts. Per-part progress also
counts in-flight work; do not add both totals. `DATA_READY.json` appears only after
all parts complete. Partial `.bin` files are not training-ready. Do not change the
conversion contract mid-run. The running code is snapshotted under
`results/nemotron-math-v41-code-20260910-v1/src`, separate from editable project code.

Runtime recorded in the manifest: torch `2.12.0a0+0291f960b6.nv26.4.48445190`,
Transformers `5.8.1`, tokenizers `0.23.0rc0`, pyarrow `25.0.0`, numpy `1.26.4`.
The existing AutoModel checkout supplies its indexed writer through `PYTHONPATH`;
no environment was created and no package was installed. Staging and published
output both live on NAS, so budget roughly twice the final token-file storage.

Resume only after verifying PID 189474 has exited, using the **same snapshot**:

```sh
export PYTHONPATH=/mnt/nas/evergreen/arch/results/nemotron-math-v41-code-20260910-v1/src:/mnt/nas/evergreen/arch/results/pretrained-backend-audit-20260907/Automodel
/opt/venv/bin/python -m archlab.preprocessing.nemotron_math \
  --source /mnt/oss-dataset/datasets/nv-community/Nemotron-Math-v2 \
  --tokenizer /mnt/nas/evergreen/arch/results/deepseek-v41-flash-assets-df42c109 \
  --tokenizer-format deepseek-v41 --reasoning-effort 75 \
  --stage /mnt/nas/evergreen/arch/results/nemotron-math-v41-20260910-v1-stage \
  --output /mnt/nas/evergreen/arch/results/nemotron-math-v41-20260910-v1 \
  --workers 32 --batch-size 16 --row-groups-per-part 1 --detach
```

This is a resume command for the existing DLC worker, not a new environment.
Post-launch import formatting in the editable source does not change the snapshot.

## Proposed model integration

The [released config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/config.json)
specifies width **5,120**, **40 layers**, **4 residual streams**, **384 routed experts**,
top-6 routing, and one shared expert. The model card reports 552B backbone parameters,
including approximately 196B Engram parameters. Packed quantized tensor counts from
the Hub API are not directly equivalent to logical model parameter counts.

Freeze **all existing parameters**. Preserve CED, CSA2, KV/index sharing, hierarchical
indexing, Single-Pass mHC, Engram, MoE, norms, gates and positional encoding. Keep
the original checkpoint intact. Vision is inactive for this text-only experiment;
DSpark/MTP has no auxiliary objective. Neither is a reason to omit Engram or replace
the text attention with a simpler model.

Initial proposal: eight additive branches after attention's residual update and
before the MoE residual read, in layers **5, 10, 15, 20, 25, 30, 35, 40** (one-based).
This samples both encoder and decoder halves. The site count is a proposal, not
inherited from Qwen's twelve full-attention blocks: V4.1 has a different architecture.

For the official reference's `Block.forward`, insertion is immediately after
attention `hc_post`, before `ffn_pre, ffn_post, ffn_comb` are calculated. Preserve
the delayed `pre_mix` / `attn_pre` flow; do not substitute Qwen's residual mixer.
With the adapter explicitly bypassed, the entire pretrained computation must match.

### Proposed branch shapes

All linear shapes below use PyTorch `[out_features, in_features]`. The backbone
width and its own head dimensions remain unchanged; only the added branch has
an explicitly smaller attention projection width.

| Added component | Shape | Parameters per site |
| --- | --- | ---: |
| Q, output gate (each) | `[1024, 5120]` | 5,242,880 |
| K1, K2, V1, V2 (each) | `[256, 5120]` | 1,310,720 |
| Output projection | `[5120, 1024]` | 5,242,880 |
| Input RMSNorm | `[5120]` | 5,120 |
| Q/K1/K2 RMSNorm (each) | `[128]` | 128 |
| Residual read/write scalar vectors | two `[4]` | 8 total |

Eight query heads, two KV heads, head dimension 128. Total **20,977,032 per site**,
**167,816,256 trainable parameters** across eight sites. No added FFN. Read a learned
softmax-weighted combination of four streams; add the projected branch back with
four learned scalar coefficients. Preserve the original stream mixing separately.

Include Q/K normalization and a learned sigmoid output gate. Proposed first variant
uses **no extra adapter RoPE**; pretrained hidden states retain their original
position encoding. This keeps the new core's algebra explicit and avoids claiming
ordinary three-way RoPE is translation invariant. Extra adapter positional schemes
are separately named experiments, not silent defaults.

Proposed initialization follows the preference for nonzero outputs: Q/K/V/gate
normal std 0.02, output normal std 0.001, effective norm scales one, stream-read
logits zero, stream-write coefficients one. Measure initial loss perturbation;
the exact pretrained control uses an explicit branch bypass. No quality improvement
is assumed from the nonzero initialization.

## Exact attention contract and feasibility

For each head, within one document:

```text
score[i,j,k] = sum_d(Q[i,d] * K1[j,d] * K2[k,d]) / sqrt(d)
P[i,:,:] = joint_softmax(score[i,:,:], over j <= i AND k <= i)
Y[i,d] = sum_{j,k} P[i,j,k] * V1[j,d] * V2[k,d]
```

Both key axes see the full causal prefix, including `j == k` and the current
position. There is no local window, pair Top-K, factorized softmax, delta rule,
or random-feature approximation. Online softmax can avoid materializing cubic
scores, but it **does not remove cubic arithmetic**.

For N tokens there are `N(N+1)(2N+1)/6` allowed triples per head:

| Context | Causal triples/head | Work relative to 2K |
| --- | ---: | ---: |
| 1,024 | 358,438,400 | 0.125x |
| 2,048 | 2,865,409,024 | 1x |
| 4,096 | 22,914,881,536 | 8x |
| 8,192 | 183,285,493,760 | 64x |
| 16,384 | 1,466,149,724,160 | 512x |

These are arithmetic ratios, not measured throughput. Begin correctness/profiling
at 128–2,048 tokens, then measure 4K/8K/16K only if feasible. Full corpus preparation
does not imply all long trajectories can be affordably trained with full cubic
attention. Before production, report length coverage and choose the actual context
policy; no silent long-example discard or truncation. A short-context diagnostic
is not the full-data run. Initial tests avoid packing; later packing must reset
attention, shared caches, positions and Engram history at document boundaries.

## Backend reuse audit and qualification order

1. **Search maintained upstream code first.** AutoModel main tree
   `f2d730b003a07559ebafea1c78368f9efe2e4569` exposes V4 support, not a dedicated V4.1
   model path. Its V4 config lacks the V4.1 CED/Engram/KV-sharing fields. Transformers
   main tree `4815a0a6a064214f2d8208c094464a5a6b76ca8d` has no dedicated V4.1 path.
   This is a bounded audit, not a claim that no implementation exists anywhere.
2. **Use the official V4.1 inference reference as a numerical oracle.** Its
   [model.py](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py)
   uses `torch.inference_mode`, mutable caches and forward-only low-precision
   kernels. Removing the decorator alone does not provide a training backend.
   Frozen weights still require dInput through later layers to train earlier adapters.
3. **Reuse existing backend infrastructure, not the wrong architecture.** Prefer
   AutoModel's FSDP/EP, loading and checkpoint infrastructure after exact-model
   support is qualified. Implement only uncovered project-owned integration gaps.
   Define the backward/surrogate policy for activation quantization and discrete
   routing while retaining the native forward. Never modify the container runtime.
4. **Reuse simplicial code where valid.** The
   [FBGEMM source](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/simplicial_attention)
   and [PyTorch blog](https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/)
   primarily optimize windowed attention. TLX installation would violate the frozen
   environment. Our stock-Triton core has forward/backward and can express full
   prefixes with both windows at least N, but that configuration is **not qualified
   for performance or full-model training**. Validate before selecting it or adapting
   tiling. A correct slow oracle is not a production performance claim.
5. **Small-to-full tests before finetuning.** FP32 forward and all five input-gradient
   oracles; causal/diagonal/full-history masks; mixed lengths; grouped heads; loss
   masking; adapter-bypass parity; gradients through frozen blocks; immutable base
   weights; distributed EP/FSDP numerical tests; fresh-process checkpoint reload.
   Verify full-sequence teacher forcing matches tokenwise next-token causality,
   especially at the encoder/decoder and shared-KV boundaries.

Candidate topology: existing 32 GPUs, FSDP2 plus node-local EP8, four cross-node
expert-FSDP shards, TP/PP/CP=1. EP overlays the 32-rank mesh. All Engram tables
need explicit sharding; they cannot be dropped for convenience. A full BF16 copy
of 552B parameters alone is about 1.1 TB, so loading/staging and node RSS matter,
not only advertised GPU memory. This topology is proposed, not V4.1-validated.

After weight availability and qualification: one adapter-only training job, assistant
CE including reasoning, initial AdamW LR 1e-5, warmup 100 steps, weight decay 0.01,
betas (0.9, 0.95), clip 1.0, microbatch 1. Choose accumulation from measured supervised
tokens and memory. Checkpoint adapters, optimizer, scheduler, RNG and exact data
cursor with base revision; never automatically unfreeze the backbone.

Observe fixed held-out assistant CE, breakdowns by effort/tool/length, branch-to-
residual norms, early-adapter gradients, tokens/s, GPU memory and node RSS. Compare
adapter enabled/disabled on **this same DeepSeek checkpoint**, including matched
GSM8K/MATH/AIME settings and contamination checks. Previous Qwen/FineWeb losses are
not a controlled baseline for a different tokenizer, backbone, dataset and loss mask.

## Reviewable files

- `recipes/proposals/deepseek_v41_global_simplicial_math.yaml`: portable proposed
  model/data/runtime contract; every machine path is injected through `env:`.
- `src/archlab/preprocessing/nemotron_math.py`: resumable inventory, worker pool,
  indexed writing, data splits, provenance and publication.
- `src/archlab/preprocessing/deepseek_v41.py`: narrow wrapper around the pinned
  official encoder, schema attachment and assistant supervision boundaries.
- `src/archlab/architectures/simplicial_attention.py`: existing mathematical oracle.
- `src/archlab/architectures/simplicial_kernels.py`: existing stock-Triton mechanism.
- `src/archlab/automodel/train.py` and `simplicial.py`: existing Qwen-specific
  execution/injection examples, **not** a working V4.1 training entry.

The new V4.1 model integration and global adapter class do not exist yet. This
document deliberately does not advertise a ready-to-launch finetuning command.

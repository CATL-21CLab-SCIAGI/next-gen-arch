# DeepSeek V4.1 + windowed softmax 2-simplicial adapters

Status, 2026-09-10, revision 2, data recovery v4: CPU tokenization repaired and
resumed with versioned provenance; training is a proposal, not
implemented or launched. Model-weight downloads are deferred by the user. No subagents, package
installs, DLC stops, or node/controller restarts.

The historical filename is retained for existing links. **Full-history cubic
attention is superseded by fixed windows**, as requested. See the complete
[report/code audit](DEEPSEEK_V41_SIMPLICIAL_AUDIT.md) for evidence, intentional
departures, and remaining qualification gates. The earlier nonzero-output/AdamW-only
proposal is superseded. The separate data recovery described below uses schema 4;
the original schema-2 output and code snapshot are preserved unchanged.

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
  It used `/opt/venv/bin/python`, 32 worker processes, and no visible CUDA devices.
  Its progress marker reports failure at **08:01:38 UTC**, after publishing
  **156 parts / 312,000 conversations / 5,396,920,359 tokens**. The process was
  verified defunct during this audit. Do not report it as running or blindly retry.
- 47 tests passed, including the official encoder's complete test suite and our
  multi-turn/tool supervision checks. A five-source smoke conversion produced
  40 conversations / 360,350 tokens, with indexed readback and SHA256 verification.
- Recovery: a post-shutdown READY inventory found **184 parts / 368,000
  conversations / 6,780,235,595 tokens**, including workers that finished after the
  old progress marker reported failure. Schema 3 imports these parts with full
  checksum and provenance checks; these are reused tokens, not new throughput.
- The first recovery (v2, PID 191680) passed the original source row but failed
  closed on another representation at file row 374002: consecutive assistant
  tool calls without an intervening result. Further inspection found trajectories
  ending in an assistant tool call. Its process exited; artifacts are preserved.
- The repaired v3 CPU job launched as PID **192780** on the same worker, using the
  frozen `/opt/venv/bin/python` and 32 workers. Before launch, **54 tests and four
  subtests passed** for the first repair; **57 tests and three subtests passed**
  for the final extension, including six actual counterexamples, exact old/new
  rendering and supervision equality on 40 previously accepted real examples,
  import-corruption rejection, and the official encoder suite.
- V3 published 31 new parts before a later source row exposed a non-assistant
  ending: `high_part00.parquet`, row group 190, row 1861, ends on a tool result.
  All **215 parts / 430,000 conversations / 7,791,512,843 tokens** are preserved.
  V3 exited; v4 resumed as PID **193616**, reusing these verified parts. Schema 4
  preserves native non-assistant endings without adding an answer or EOS, and
  excludes the trailing non-assistant text/header from supervision. Before this
  launch, 59 tests and three subtests passed, covering seven real counterexamples
  and successful-domain compatibility with both v1 and v3.

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
- Preserve entire supplied conversations, including unfinished source endings;
  no truncation, extra EOS/EOD, or trajectory deduplication.
- Store native BOS/EOS and int32 tokens with document boundaries. Compare fast
  tokenization to Transformers for every observed effort/tool combination per worker.
- Preserve assistant target spans, including reasoning, answers, calls and EOS;
  exclude prompt headers and tool results. Spans are half-open indices in the
  unshifted token sequence: the trainer must shift them exactly once with labels.
- Reserve 1% of normalized problem-hash buckets for validation. Group all variants
  of a problem together. This is exact/normalized grouping, not near-duplicate detection.
- Retain source provenance, licensing, source-row IDs and token-length histograms.
- Coalesce consecutive reasoning-only assistant prefixes into the following
  assistant message, preserving every reasoning character and its final answer or
  tool call. Record original indices, character lengths, hashes and the inserted
  paragraph separator in `message_repairs`. Never merge completed answers or
  unresolved tool calls, drop a row, or fabricate a user message.
- Other adjacent assistants are rendered individually by the official encoder,
  with their calls/reasoning/order preserved and the native-preservation policy
  recorded. Terminal assistant tool calls are retained but flagged
  `complete_answer: false`; no missing tool result or answer is synthesized.
  Native tool/user/system endings are likewise retained and flagged, including
  any native trailing assistant header, with no supervision on that suffix.
  This is full-source tokenization, not an assertion that every teacher trajectory
  is complete or suitable for SFT. Final training selection must review these flags.

Outputs and monitoring:

```text
results/nemotron-math-v41-20260910-v4/progress.json
results/nemotron-math-v41-20260910-v4/parts/*/READY.json
results/nemotron-math-v41-20260910-v4-stage/job.log
results/nemotron-math-v41-20260910-v4-stage/launcher.json
results/nemotron-math-v41-20260910-v4-stage/parts/*/progress.json
```

Top-level progress counts only published/checksummed parts. Per-part progress also
counts in-flight work; do not add both totals. `DATA_READY.json` appears only after
all parts complete. Partial `.bin` files are not training-ready. Do not change the
conversion contract mid-run. The original launch code is snapshotted under
`results/nemotron-math-v41-code-20260910-v1/src`, separate from editable project code.
The new launch code is separately snapshotted under
`results/nemotron-math-v41-code-20260910-v4/src`. Reused parts retain their original
payload bytes and embed the original READY manifest inside `reused_from`; source
and destination files are not mutable hardlinks. Progress separates reused and
newly tokenized documents. `repaired_documents` counts published documents with
encoding events (including inherited flags): reasoning merges, native adjacent
turns, terminal calls or non-assistant endings.
The failed case is file `high_part00.parquet`, row group 184, row 63 within that
group: a reasoning-only assistant fragment immediately precedes another assistant
tool-call message. The regression test now verifies its native encoding and
assistant supervision after the narrowly scoped repair.

Runtime recorded in the manifest: torch `2.12.0a0+0291f960b6.nv26.4.48445190`,
Transformers `5.8.1`, tokenizers `0.23.0rc0`, pyarrow `25.0.0`, numpy `1.26.4`.
The existing AutoModel checkout supplies its indexed writer through `PYTHONPATH`;
no environment was created and no package was installed. Staging and published
output both live on NAS, so budget roughly twice the final token-file storage.

Current launch/resume command. Keep this snapshot and all format arguments fixed;
the old snapshot still rejects the original row and is not a remedy.

```sh
export PYTHONPATH=/mnt/nas/evergreen/arch/results/nemotron-math-v41-code-20260910-v4/src:/mnt/nas/evergreen/arch/results/pretrained-backend-audit-20260907/Automodel
/opt/venv/bin/python -m archlab.preprocessing.nemotron_math \
  --source /mnt/oss-dataset/datasets/nv-community/Nemotron-Math-v2 \
  --tokenizer /mnt/nas/evergreen/arch/results/deepseek-v41-flash-assets-df42c109 \
  --tokenizer-format deepseek-v41 --reasoning-effort 75 \
  --stage /mnt/nas/evergreen/arch/results/nemotron-math-v41-20260910-v4-stage \
  --output /mnt/nas/evergreen/arch/results/nemotron-math-v41-20260910-v4 \
  --reuse-completed-from /mnt/nas/evergreen/arch/results/nemotron-math-v41-20260910-v3 \
  --workers 32 --batch-size 16 --row-groups-per-part 1 --detach
```

This command targets the existing DLC worker, not a new environment.
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

Also freeze non-optimizer state: no automatic MoE correction-bias updates, quantizer
calibration, auxiliary indexer/router losses, or checkpoint-scale changes. Ordinary
per-request KV state and native input-dependent quantization still operate during
execution. Disable speculative decoding
and approximate bounded replay initially; adding local branches changes their
cache/replay requirements. Native attention sinks and inverse output RoPE must be
preserved, even though neither is the new branch's sigmoid output gate.

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
four bounded learned coefficients `2 * sigmoid(write_logits)` (initially one).
Preserve the original stream mixing separately. This lightweight read/write is an
explicit adapter design, not an implementation of an extra Single-Pass mHC block.

Include Q/K normalization and a learned sigmoid output gate. Proposed first variant
uses **no extra adapter RoPE**; pretrained hidden states retain their original
position encoding. This keeps the new core's algebra explicit and avoids claiming
ordinary three-way RoPE is translation invariant. Extra adapter positional schemes
are separately named experiments, not silent defaults.

Proposed initialization: Q/K/V/gate normal std 0.02, **output projection zero**,
effective norm scales one, stream-read/write logits zero. The active adapter must
initially match its bypass, with finite intermediates and no hidden base-state
mutation. Zero O is an identity-preserving adapter choice, **not** a prescription
found in the V4.1 report. The first backward should give O a task gradient while
earlier branch task gradients are zero; test their onset after O's first update.
Do not initialize both V paths or the effective stream write to zero. Do not use
a norm-preserving optimizer that would trap zero-initialized O at norm zero.

## Exact attention contract and feasibility

For each head, within one document:

```text
score[i,j,k] = sum_d(Q[i,d] * K1[j,d] * K2[k,d]) / sqrt(d)
J(i) = [max(document_start, i - 32 + 1), i]
K(i) = [max(document_start, i - 512 + 1), i]
P[i,:,:] = joint_softmax(score[i,:,:], over j in J(i), k in K(i))
Y[i,d] = sum_{j,k} P[i,j,k] * V1[j,d] * V2[k,d]
```

Both windows include the current position and permit `j == k`. They contain 32 and
512 positions, not that many preceding tokens plus self. This is exact softmax over
the allowed local pairs, **not global pair attention**. There is no pair Top-K,
factorized softmax, delta rule, or random-feature approximation. No additional K/V
bias is implicit; upstream oracle comparisons must disable its optional biases.

Work is `O(B * sites * Hq * N * 32 * 512 * d)`: at most **16,384 pairs/query/head**.
Exact per-document triple count is `sum(min(t,32)*min(t,512), t=1..N)`:

| Context | Windowed triples/head | Global/windowed triple ratio |
| --- | ---: | ---: |
| 2,048 | 29,362,864 | 97.6x |
| 4,096 | 62,917,296 | 364.2x |
| 8,192 | 130,026,160 | 1,409.6x |
| 16,384 | 264,243,888 | 5,548.5x |

These are core arithmetic ratios, not throughput speedups. The windows reproduce
the paper's 512-by-32 size pair with the axes exchanged for our kernel's short-first
convention. This exchange requires swapping both K and V paths together. The smaller
GQA ratio (4 versus the paper's 64) is an intentional adapter-budget choice, not a
performance match; forward plus backward must be profiled on the actual container.

The **whole model is not globally O(N)**: original Full-mode sparse indexers still
scan the causal prefix. Frozen-block dInput, activation memory, logits and EP traffic
remain significant. Target 16K after 128–2K correctness and 4K/8K/16K profiling.
Sliding windows do not decide how >16K conversations are trained. Before production,
report token/length coverage and finalize a context policy; no silent long-example
discard, truncation, or claim that detached chunk caches give full-context gradients.
Initial tests avoid packing; later packing must isolate original attention, the
adapter, shared caches, positions and Engram history at document boundaries.

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
   environment. Our stock-Triton core has forward/backward and can express
   the selected windows, but is **not qualified for this geometry's full-model
   performance**. FBGEMM also contains backward code; the audit documents wrapper,
   window and GQA compatibility gaps rather than treating it as forward-only.
   Validate before selecting it or adapting tiling. A correct slow oracle is not a
   production performance claim.
5. **Small-to-full tests before finetuning.** FP32 forward and all five input-gradient
   oracles; causal/window-edge/diagonal masks; mixed lengths; grouped heads; loss
   masking; adapter-bypass parity; gradients through frozen blocks; immutable base
   weights; distributed EP/FSDP numerical tests; fresh-process checkpoint reload.
   Verify teacher forcing against exact tokenwise next-token execution, especially
   at the encoder/decoder and shared-KV boundaries. Do not demand equivalence to
   the report's deliberately approximate SWA bounded replay.

Candidate topology: existing 32 GPUs, FSDP2 plus node-local EP8, four cross-node
expert-FSDP shards, TP/PP/CP=1. EP overlays the 32-rank mesh. All Engram tables
need explicit sharding; they cannot be dropped for convenience. A full BF16 copy
of 552B parameters alone is about 1.1 TB, so loading/staging and node RSS matter,
not only advertised GPU memory. This topology is proposed, not V4.1-validated.

After weight availability and qualification: one adapter-only training job, assistant
CE including reasoning. Use **headwise Muon on Q/K1/K2**, ordinary full-matrix Muon
on V1/V2/O/gate, and AdamW on norms and scalar vectors. Follow the report's momentum
0.95, Nesterov and update-RMS target 0.18; matrix/norm decay 0.1, scalar decay zero;
AdamW betas (0.9, 0.95), epsilon 1e-20 with FP32 states. Inherit V4's hybrid
Newton–Schulz: eight fast iterations then two stabilization iterations; scale the
direction by `0.18 * sqrt(max(rows,cols))`, not a different post-hoc measured-RMS
normalizer. No Sinkhorn embedding optimizer is
needed because all original tables and the prediction head are frozen.

Pilot LR 1e-5, warmup 100 steps, clip 1.0, microbatch 1 are **our adapter choices**,
not the report's pretraining schedule. Set accumulation and the cosine schedule's
supervised-token horizon after profiling and finalizing length handling. Normalize
loss by global supervised-token count, not unweighted per-rank means. Headwise
orthogonalization must see each complete head gradient, not arbitrary FSDP slices.
Checkpoint adapters, optimizer, scheduler, RNG and exact data cursor with base
revision; never automatically unfreeze the backbone.

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

The new V4.1 model integration and windowed adapter class do not exist yet. This
document deliberately does not advertise a ready-to-launch finetuning command.

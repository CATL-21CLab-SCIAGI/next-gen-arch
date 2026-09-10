# DeepSeek V4.1 simplicial proposal: report and code audit

2026-09-10. This audits the **entire proposed experiment**, not just initialization
and optimizer choice. It changes the proposal, not a running model. No finetuning
has launched, no weights were downloaded, and no training environment was changed.

## Sources and scope

- [V4.1 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/DeepSeek_V41_Tech_Report.pdf):
  architecture §§2.1–2.4; optimization §2.5; infrastructure §§3.1–3.2;
  training §4.2; post-training/evaluation §§5.1, 5.3 and Appendix B.
  Downloaded PDF SHA256:
  `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.
- [Released config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/config.json)
  and [inference reference](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/df42c109f1defefcbfcedbe7d905718a12266e40/inference/model.py),
  same pinned model revision. Reference SHA256:
  `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65`.
- [Fast and Simplex](https://arxiv.org/html/2507.02754v1), §§4–8 and kernel appendices;
  [PyTorch kernel article](https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/).
- [V4 technical report, §2.4 / Algorithm 1](https://arxiv.org/html/2606.19348v1#S2.SS4),
  for the optimizer configuration inherited by V4.1.
- [FBGEMM simplicial source](https://github.com/pytorch/FBGEMM/tree/4754658081a560abdad648d5f1e5331caca439a7/fbgemm_gpu/experimental/simplicial_attention),
  inspected revision `4754658081a560abdad648d5f1e5331caca439a7`.
- Local project architecture, optimizer, integration and preprocessing code; the
  native V4.1 encoder; the dataset's supplied README; live preprocessing status.

The DeepSeek report does **not** evaluate our frozen-backbone simplicial adapters.
The simplicial paper uses interleaved layers in pretrained-from-scratch models,
not this insertion experiment. Source agreement and adapter design choices must
therefore be recorded separately; neither publication proves an improvement here.

## Full audit ledger

| Area | Finding in the previous proposal | Revised decision / required verification |
| --- | --- | --- |
| Base identity and shapes | Exact revision was pinned; a full weight index was not available. | Keep width 5120, 40 blocks, four streams, all 384 routed experts, top-6 plus shared expert. Enumerate every checkpoint tensor, scale and inactive component before claiming load coverage. |
| Attention architecture | Listing CED/CSA2 was insufficient to qualify a backend. | Preserve source/consumer ownership, compression, candidate selection, local SWA, attention sinks and output inverse RoPE. Do not equate CSA2 “Full Mode” with ordinary dense full attention. |
| Layer placement | Eight sites every five layers were proposed without source attribution. | Retain sites 5/10/15/20/25/30/35/40 as **our** balanced encoder/decoder placement, not the paper's every-fourth-layer architecture or Qwen's twelve-site pattern. |
| Residual integration | Four-stream handling was proposed, but its free write coefficients were unbounded. | Keep original delayed coefficient flow. New read is softmax over four scalars; new write is `2*sigmoid(logits)`, initially one. This is an additive adapter, not another native mHC block. |
| Output initialization | Nonzero O reused a previous Qwen experiment preference. | Propose O=0 for base identity. Other projections remain random; norms and effective writes remain nonzero. This is our adapter choice; no zero-O adapter recipe was located in the report. |
| Optimizer partition | AdamW for every adapter parameter missed the report's optimizer partition. | Headwise Muon for Q/K1/K2, full-matrix Muon for V1/V2/O/gate, AdamW for norms/scalars. Do not invent headwise V/O as a report requirement. |
| Optimizer details | One LR/decay setting obscured different update semantics. | Pin momentum, orthogonalization, update RMS, per-group decay, epsilon and global-gradient normalization. Test head blocks before FSDP sharding. Native Muon is present, but its defaults do not establish report equivalence. |
| Frozen parameters versus frozen state | `requires_grad=False` alone was treated as sufficient. | Disable router correction-bias updates, auxiliary losses and calibration mutations. No optimizer state for frozen tables/head; no need to implement Sinkhorn table updates for this experiment. |
| Attention algebra | Trilinear score, joint pair-softmax and Hadamard values were correct. | Preserve those operations exactly on the allowed pairs, with explicit `1/sqrt(d)` scaling and no factorized normalization. Test all five input gradients. |
| Locality and complexity | Full-history cubic computation was incompatible with practical long sequences. | Fixed windows 32 and 512, counting self; exact joint softmax locally. Only the branch becomes linear in N. Keep original global-access attention untouched. |
| Window boundary semantics | “Causal” alone left several choices implicit. | Explicitly include self and diagonal pairs; reset at document starts; test partial windows, odd lengths, axis swap, padding and decode cache wraparound. |
| Positional encoding | No-extra-RoPE was a named choice but could be mistaken for paper equivalence. | Retain it as a deliberate adapter choice. Ordinary RoPE on all three vectors is not relatively invariant for this trilinear form. Do not silently substitute the paper's determinant score. |
| Norms and output gate | Added Q/K norms and sigmoid output gate were borrowed design choices. | Keep and label them as adapter additions, not requirements of either source. Base norm epsilon/scales stay exact; adapter epsilon 1e-6 is separate. Head statistics are per head, with shared `[128]` affine scales. |
| Geometry and parameter budget | 8 Q / 2 KV heads imply GQA4, not the high head-sharing ratio used for published speed. | Keep the 167.816M budget proposal, explicitly unqualified for throughput. Profile forward **and backward** before increasing windows or changing heads; do not claim paper throughput. |
| Off-the-shelf kernels | “Primarily forward” was too imprecise. | Upstream backward exists, but inspected wrapper/window/GQA contracts do not establish a seamless training path. See concrete findings below. Reuse qualified pieces before implementing gaps. |
| Quantized frozen backward | Inference code was listed as an oracle without enough qualification. | Require forward parity and dInput through every later frozen component. Pin quantization/surrogate semantics; no blanket `no_grad`, inference tensors or detached shared KV after adapters. |
| CED and generation costs | Prefill savings could be mistaken for teacher-forcing savings. | Full target-token training requires the decoder. Initial evaluation uses exact execution, not bounded replay. Additional decoder adapters invalidate assumptions behind the old replay window. |
| DSpark / vision | “Preserve but inactive” was correct but not sufficient for serving. | Keep checkpoint contents; disable draft decoding initially. A frozen drafter is not automatically calibrated to the adapted target. Text-only tests cannot establish unchanged visual abilities. |
| Distributed memory | FSDP2 + EP8 was a candidate, not proof of capacity or correctness. | Preserve full Engram with explicit table sharding; qualify EP routing/backward, adapter reduction, complete-head Muon, loading RSS and re-gather traffic. No TP/PP/CP added by this proposal. |
| Data and supervision | Forty smoke rows did not cover all source-message structures. | Tokenization failed closed at an unsupported assistant sequence. Preserve valid parts, fix with versioned provenance and regression tests; do not drop rows or bypass the contract hash. |
| Lengths, weighting and packing | “Full data” and 16K were not a complete data-consumption policy. | Publish length/token coverage, deterministic data order and supervised-token weighting. Chunked targets with detached caches are not full-context backprop. Packing must isolate every original and added state path. |
| Evaluation and interpretation | Math loss alone cannot establish improvement or retention. | Matched base-enabled/disabled CE and generation, same effort, budgets, seeds, stops and tools; add non-math retention checks and benchmark-overlap checks. Prior Qwen losses are not a baseline here. |
| Restart reproducibility | Adapter-only checkpointing was necessary but not sufficient. | Save head grouping, optimizer states, scheduler/token counts, RNG, data cursor, base revision and runtime. Reload in a fresh process and compare the next update. |

## Optimizer contract: more than a name change

Report §2.5 and §4.2.2 support Q/K headwise Muon, momentum 0.95 with Nesterov,
update RMS 0.18, and AdamW for norms/non-matrix parameters. The stated decay is
0.1 for matrices and norms, zero for biases/scaling factors. AdamW uses
`betas=(0.9,0.95), eps=1e-20`. These are source facts; the proposed pilot LR 1e-5,
100-step warmup, clip 1 and microbatch 1 are **our** choices.

For each added site, Q becomes eight independent `[128,5120]` optimizer matrices;
K1 and K2 become two each. V1/V2 remain `[256,5120]`; O stays `[5120,1024]` and
the gate `[1024,5120]`. Across eight sites, Muon handles 167,772,160 parameters;
AdamW handles 44,096. Group coverage must be exhaustive and non-overlapping.

The frozen container has `torch.optim.Muon`; it accepts 2D parameters and exposes
`adjust_lr_fn`, but the inspected defaults alone do not implement our whole
grouping/0.18-RMS/distributed contract. Prefer its supported operations or the
existing backend's optimizer infrastructure, with a project-owned integration
under `src/archlab/optimizers` where needed. Do not patch PyTorch or Megatron.

`src/archlab/optimizers/speedrun.py` is **not** an interchangeable implementation:
it includes NorMuon variance reduction, Polar Express and cautious decay. Its
norm-preserving MuonH mode would also trap an exactly zero O at zero norm. Reusing
it unchanged would silently alter the experiment. The inherited V4 configuration
uses eight Newton–Schulz iterations with `(3.4445,-4.7750,2.0315)`, then two with
`(2,-1.5,0.5)`, following Frobenius input normalization. Its update multiplier is
`0.18 * sqrt(max(rows,cols))` for each logical matrix. This targets the reported
RMS; **do not replace it with exact measured-RMS renormalization**, especially for
rank-deficient or zero updates. The momentum convention is `M=0.95*M+G` followed
by orthogonalization of `0.95*M+G`. The YAML now pins these choices; implementation
and numerical equivalence remain unqualified.

Orthogonalize the **globally reduced complete head gradient**, never a rank-local
slice. Define zero-gradient behavior without dividing by zero or amplifying noise.
Use FP32 master states and test the first nonzero O update, then a second backward:
zero task gradients upstream of O on the first backward are expected, whereas
persistent missing early-adapter gradients after O changes are a bug. Decay updates
must be distinguished from task-gradient updates in those diagnostics.

## Concrete reference-code qualifications

1. **FBGEMM is not simply “missing backward.”** Its
   `simplicial/ops/pytorch/two_simplicial_attention.py` calls `triton_bwd` with
   windows before the saved output/gradient/LSE; the inspected `ops/triton/bwd.py`
   signature expects `(q,k1,k2,v1,v2,o,dO,m,w1,w2,...)`. The latter also asserts
   `w2 == 32`. Forward automatically exchanges axes when `w1>w2`, while backward
   needs consistent axis/gradient mapping. These are static findings, not a claim
   that an upstream GPU test was run during this audit.
2. **Optional bias and GQA matter.** The upstream oracle defaults to additive K2/V2
   biases unless `disable_kv_bias=True`; passing zero to wrappers with `if not bias`
   is not sufficient to disable them. The optimized forward packs query heads,
   while backward address/shape conventions need an independent GQA audit. Our
   `simplicial_attention.py` specifies a no-bias GQA oracle and the local Triton
   implementation has a checked backward, but the new 8/2/128, 32/512 configuration
   still needs its own numerical and performance qualification.
3. **The V4.1 inference reference is not a training oracle for gradients.**
   `Transformer.forward` uses inference mode and defaults to last-position logits;
   caches and quantization include in-place writes. Training needs full target
   logits, autograd-safe shared states and tested dInput. Bypass parity alone cannot
   prove any of these.
4. **CED requires a representation-level check, not just a layer-number check.**
   The report describes global decoder KV from the final encoder representation.
   In the reference, layer index 20's compressor receives that block's normalized
   attention input, after its `hc_pre`; subsequent layers reuse the published KV.
   Do not implement an arbitrary `encoder_hidden.mean(stream)` projection based
   only on the diagram. Map exact collapse/norm/projection weights and prove the
   source boundary agrees before qualifying the loader/backend. This is an open
   verification item, not evidence that the released checkpoint is broken.
5. **Approximate replay is not a parity oracle.** The report's §3.2.2 explicitly
   accepts approximate SWA states. Use exact full-prefix/tokenwise comparisons
   initially. Adding a 512-token decoder branch creates additional cache needs;
   preserving the original 128-token replay unchanged is not justified.

## Data, validation and launch gates

Observed tokenization failure at 08:01:38 UTC:
`Assistant-only/consecutive-assistant source requires explicit policy`.
PID 189474 was later verified defunct. Published progress was 156 parts,
312,000 conversations and 5,396,920,359 tokens. Those outputs and the old snapshot
are preserved. No converter repair or semantic migration was made in this audit.
The counterexample was located in `high_part00.parquet`, row group **184**, row
**63** within that group (zero-based file row **368063**). Assistant message 121
contains 1,721 reasoning characters, no answer content and no tool call; assistant
message 122 follows immediately with further reasoning and a tool call. The
converter rejects this consecutive-assistant case. It is a source representation
we did not support, not proof the mathematical example should be discarded.

Required repair: inspect the offending source messages; preserve their native
meaning and supervision without fabricating a user turn or silently deleting an
assistant. Test native formatting and span boundaries on real counterexamples.
For this case, investigate joining the reasoning-only fragment to the following
assistant message while preserving the following call and its tool result. Record
the transformation and original boundaries; do not generalize this merge to
arbitrary consecutive completed answers or unresolved tool calls.
Changing renderer bytes changes the dataset contract: use a new version or a
verified explicit migration, never rewrite old hashes merely to pass resume checks.

The corpus contains teacher-generated reasoning at multiple effort settings.
Using native effort 75 for every trajectory is an explicit conditioning choice,
not reproduction of DeepSeek's effort-training procedure. Measure retention at
several efforts. Keep original effort/tool/source metadata and assess sample
imbalance; do not equate “one pass over trajectories” with equal problem weighting.

Target 16K is not yet a promise to train all trajectories intact. A complete
long-document policy must state target-token coverage, context resets/overlap,
gradient truncation (if any), position/cache/Engram boundaries and loss weights.
Non-math retention evaluation should share decoding settings between base and
adapter; math improvements alone cannot establish general capability retention.

Launch only after: full-data preparation is qualified; exact checkpoint loading is
available; frozen-forward and two-step gradient tests pass; windowed forward/backward
and decode caches pass; optimizer grouping and scaling pass; and a 32-rank
EP/FSDP checkpoint reload reproduces the next update within defined tolerances.
Measure end-to-end step time, both processed and supervised tokens/s, peak allocated
and reserved HBM, node RSS and communication. Do not use `6 * total_params * tokens`
as adapter-training FLOPs or convert pair-count reductions into promised speedups.

The outcome is a corrected, testable proposal—not a seamless full-model backend
claim. Remaining blockers are explicit; none is permission to simplify or unfreeze
the pretrained architecture.

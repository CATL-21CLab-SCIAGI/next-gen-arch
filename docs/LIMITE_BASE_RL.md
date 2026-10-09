# Limite base math RL

Launch contract: `recipes/limite/base_nemotron_grpo.yaml`, upstream
TRL 1.4.0 with the verified Limite base `cc612bafcd4a` implementation. The existing
NeMo container's PyTorch, CUDA and Transformers are unchanged. The isolated
Python overlay also pins math-verify 0.8.0, latex2sympy2_extended 1.10.2 and
antlr4-python3-runtime 4.13.2.

Run root: `results/limite-base-rl-20260929`. One shared B300, with a supervisor
requesting graceful RL shutdown if scaling step time exceeds its measured
baseline by 30% for three minutes. Existing scaling processes are never signaled.
The native generation stopping criterion observes `STOP_REQUEST`; optimization
is skipped on stop and the native trainer checkpoints at its next boundary.

The RL split retains 5,898 unique, symbolically verifiable training problems and
128 disjoint holdout problems; exclusions record duplicates, conflicting answers
and unparseable gold. Two training prompts exceed the declared 2,048-token limit
and are excluded separately. Reference answers never enter the model prompt.
This uses a symbolic verifier, **not** NVIDIA's original judge fallback. Four
fixed holdout problems are pilot evaluation cases before training and every 32
steps; this is not a statistically strong benchmark.

Four samples per question, four questions per batch, 8,192 response tokens,
synchronous GRPO, upstream DAPO token normalization, AdamW at 3e-6, no decay,
gradient clipping at 1, and no reference KL penalty. Truncated responses cannot
earn correctness reward and are masked from loss. Zero-gradient batches do not
advance Adam state; applied updates are recorded separately from trainer steps.
Eight consecutive batches without reward contrast stop the pilot.

The model retains the checkpoint's BF16 matrices and FP32 gates/scales/MUDD
tensors. Transformer Engine's container-owned FusedAdam keeps separate FP32
masters and moments. A thin no-signal guard prevents its group-wide step counter
from advancing on flat batches. The original `oracle_exact` head mode is retained;
the alternative output-dtype matrix multiply lacks a backward in this runtime.
AMP is disabled because it changes native RMSNorm output dtype and breaks static
cache key/value dtype agreement. Model matrices remain BF16 without AMP.
Reduced-precision GEMM accumulation is disabled. All-FP32 model storage was
rejected because it changes the native activation and normalization computation.
Both native terminators are accepted: `<|im_end|>` (151645) and `<|endoftext|>`
(151643). The latter is emitted by the base-model canary after its correct answer;
ignoring it causes unrelated continuation. Actual token IDs and sampling log
probabilities are preserved. Unboxed rational answers and explicit concluding
paragraphs ending in a mathematical expression can receive correctness reward;
intermediate numbers in prose cannot.
Uncorrected cache/replay comparisons failed the original equality criterion.
Therefore the launch records actual
sampled-token probabilities and passes them as the denominator to upstream
GRPO's clipped importance ratio. No custom loss or optimization loop is used.
The corrected admission requires finite full-context gradients, all probe ratios
within [0.5,2], and at most 5% clipped by the upstream [0.8,1.2] surrogate. Raw errors and
earlier failed probes remain recorded. This is a distinct corrected numerical
contract, not a claim that cached and full-sequence computations are identical.
Generation uses native eval mode, restoring train mode for replay and gradients.
The no-AMP native-head canary answers 13 + 29 correctly and terminates at EOD;
its eval-generation/train-replay probe clips 0.83% of sampled ratios.
The standalone TE optimizer test exactly matched FP32 Adam masters and resumed
its next update from native saved state. Earlier AMP receipts are superseded.

The long-context backward probe covers 10,240 tokens. Short numerical probes and
memory tests do not establish learning; inspect `updates.jsonl`, reward contrast,
heldout metrics and checkpoints for production evidence. Native checkpoints are
saved to NAS after step 1 and every 32 steps, retaining five, then copied and
checksum-verified on OSS before creating the run's published checkpoint symlink.
Direct safetensors saves to OSS fail with unsupported filesystem operations.
Each save gets complete file checksums and the unchanged
upstream Python model files. Full interrupted-run restoration remains unqualified.

MLflow group: **Limite — RL**. Run name:
`limite-1b-base-nemotron-math-v2-grpo-v1`.

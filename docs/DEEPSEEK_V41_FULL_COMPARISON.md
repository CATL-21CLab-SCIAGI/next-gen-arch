# Full-weight comparison from the 50M adapter checkpoints

Both variants continue their step307 checkpoints (50,119,869 supervised tokens),
with a fresh common full-weight optimizer and the same next data windows. Each
receives 16 B300 GPUs: dense FSDP16, node-local EP8, expert FSDP2, and Engram
ownership over all 16 ranks. Two accumulated microbatches of one window per rank
preserve the previous 32-window global updates, context lengths and target masks.

All constructed text-model parameters are unfrozen, including routed experts,
Engram tables, HC coefficients, attention sinks, indexers, the vocabulary head,
and adapters. The previous text-only model excludes the inactive vision/draft
modules. Setting requires_grad happens before FSDP's first lazy initialization.

The optimizer uses FP32 Adafactor row/column statistics, per-matrix parameter
scaling, RMS update clipping, and stochastic rounding of updated BF16 weights.
It keeps no dense FP32 masters or first moments. Dense matrices reduce column
statistics over their row-owner group. Expert banks retain per-expert factors
and reduce across expert-FSDP row shards. FP32 parameters retain FP32 storage.
The relative learning rate warms from 5e-6 to 1e-4 across 20 new-phase updates.
Global gradient clipping is 1; weight decay is 0. This is a new optimizer contract,
chosen to fit full training on 16 GPUs per variant.

FSDP gradient collectives sum. Each CE backward is normalized by the complete
32-window target count. Expert communication and Engram owner routing already
sum their contributions. Replicated adapter gradients are summed once across
the 16 ranks. The global norm counts each uniquely owned parameter once.

The released hard-top-k indexer has no LM derivative. An explicit auxiliary KL
trains its score projections and key projection/norm against detached current
compressed-attention probabilities at 64 sampled positions per window. The
coefficient is 0.01, averaged over indexers, ranks and accumulation passes.
Selection and model forward values remain unchanged. This objective is shared
by both variants; it is a project experiment contract, not a claimed released
DeepSeek full-training recipe.

Full checkpoints save model parameters/buffers, optimizer factors/counters/options,
data cursor, implementation/runtime identity, and per-rank CPU/CUDA RNG. Payloads
use bounded CPU serialization chunks; no weights or optimizer state are offloaded
to CPU during training. A COMPLETE marker is published only after all ranks finish.
Same-mesh restoration validates chunk hashes. Existing adapter checkpoints remain
untouched. The first full checkpoint is scheduled after 5 new updates.

Qualification covers independent HC/head derivatives, masked indexer gradients,
stochastic-rounding unbiasedness and RNG replay, distributed optimizer agreement,
native sparse-forward and sink-gradient agreement, full parameter gradient
coverage, two-pass gradient accumulation, exact full-state restoration, and
next-update checkpoint continuation on both 16-rank meshes. Production health
still requires measured memory, finite gradients/loss and stable sustained updates.

The production selector eliminates redundant score copies using in-place ReLU
and multiplication on fresh, no-gradient score storage. At 16K, independent
Full/CSA-source/Reindex tests preserved every key, selection and candidate mask
exactly and reduced selector peak memory from about 49 GiB to 19 GiB. Both
production starts subsequently matched their original step 308 losses to
float64 reporting precision. The runtime uses expandable allocator segments.

## Periodic held-out validation

The periodic-evaluation continuation evaluates the same sealed 64,000 assistant
targets on resident GPU weights at resume and after each 10M-token boundary.
The subset follows the validation pilot's fixed shuffled order and caps only the
last window's labels. Its digest is recorded in `VALIDATION_PLAN.json` and the
run contract. The data audit found no problem-ID overlap between the training
and validation pilots. This small sample is a trend diagnostic; benchmark scores
and the previous 1M-target validation are recorded separately.

`deepseek_v41_full_validation.py` computes FP32 full-vocabulary cross entropy,
perplexity, top-1/top-5 token accuracy, and predictive entropy in bounded chunks.
It restores module modes and Python, NumPy, CPU Torch, and current-device CUDA
RNG state. Evaluation does not change weights, optimizer state, or the training
data cursor. Both variants must pass a 16-rank qualification that fingerprints
state before/after validation and compares the next update with a no-eval replay.

The production continuation retains the exact original `gradient_step` source
and model/optimizer/runtime hashes. Admission from the recorded older checkpoint
allows only the trainer and new evaluation module to differ; all other contract
fields and protected implementation hashes must match. The immutable production
snapshot preserves the original update, while the working-tree function also
retains its existing optional scratch-training arguments.

Keep active mmap datasets protected with `.archlab-storage-in-use.json` until
all readers have exited. An atomic NAS-file-to-OSS-symlink replacement can
invalidate a different NFS client's existing mapping even when every byte was
copied and verified. The bulk offloader rejects link replacement under a guarded
tree, and the result planner skips it.

# Debugging the two scratch comparisons, 2026-09-22

The inspected runs are the linear RF versus LinSimp comparison under
`results/deepseek-v41-scratch-linsimp-w640-d20-20260921` and the faster TileLang
normal versus simplicial comparison under
`results/deepseek-v41-scratch-highmfu-w640-d20-20260922`. All four trainers were
alive when inspected. The separate full-finetuning 4537/4537 evaluation has a
complete, matched-token receipt; it is not either of these scratch runs.

## Confirmed random-feature bug

The original `orthogonal_feature_bank` sampled each orthogonal row's radius as
`abs(N(0,1))`. A D-dimensional standard Gaussian requires a chi_D radius.
At the production head dimension D=16, the original bank's expected squared
radius was therefore 1 rather than 16. It also omitted the QR diagonal-sign
correction needed for Haar-distributed directions.

A CPU probe using eight heads, rank 4096 and seed 123 measured mean squared
radius **1.00289**. For unit x=y, the estimated unnormalized exponential kernel
was **0.405–0.421**, versus the intended **exp(1)=2.71828**. This is a sampler
defect, not finite-rank noise. Both arms used it, so their results describe a
different feature kernel from the intended one. The numerical attention
reduction can be internally consistent while implementing the wrong kernel.

The corrected bank uses sign-corrected QR directions and independent chi_D
radii. Query features also use a common causal maximum across features and,
for LinSimp, all valid anchors. The shared scaling cancels in the output ratio
and avoids underflow at high learned temperatures. Padded anchors are excluded
from this maximum. No per-anchor normalization is introduced.

Reference: [Linearized 2-Simplicial Attention, Appendix B](https://arxiv.org/html/2608.09307v1).
This remains an adapter experiment with its existing shared temperature and
gating design, rather than a reproduction of the paper's full architecture.

Regression coverage in `tests/test_linsimp_attention.py` checks Gaussian
marginals, orthogonality, kernel expectation, explicit causal pair sums, all
five input gradients and temperature gradients, grouped query attention,
window boundaries, causal prefix invariance, and high-temperature behavior.
All three test methods passed in the existing container on CPU.

## Recovery and provenance

The original runs were gracefully checkpointed and stopped:

| Variant | Preserved step | Preserved supervised tokens |
| --- | ---: | ---: |
| Linear RF | 7,385 | 402,827,317 |
| LinSimp | 6,854 | 374,289,814 |

Original snapshots, weights and curves are retained. Each original output has
`DEBUG_DISPOSITION.json`. The corrected source is `source-rf-corrected-v2`,
commit `9b62513a5ddccdbec90d59c6d7d8ce5c28b3d0be`, branch
`codex/fix-linsimp-gaussian-features`. The launch manifest is
`RF_CORRECTION_V2.json`; scripts are in `launch-rf-corrected-v2`.

The correction requires a fresh matched pair from seed 42 and data cursor 0.
Restoring old checkpoint buffers would restore the defective feature bank;
replacing those buffers under trained weights would create a mixed-operator
learning curve. Each new arm must pass eight-rank qualification before training.
New outputs and MLflow identities are separate from the invalid original runs.
The completion-time state is recorded in
`results/deepseek-v41-debug-20260922/RECOVERY.json`.

The new trainer hashes the feature implementation and adapter dispatch in its
contract and records the actual loaded TileLang/TVM module versions and paths.
This also removes the prior asymmetry where the original linear arm's process
was still executing an older commit than the snapshot on disk.

## Faster TileLang backend

The bounded GPU probe in
`src/archlab/automodel/deepseek_v41_scratch_sparse_probe.py` checked 17, 65 and
512 selected-key slots, H=64, D=64, masked rows, duplicate keys, single-key rows,
and gradients for queries, shared keys/values and attention sinks. Forward
outputs matched the existing native path exactly. Relative L2 errors against
an independent FP32 softmax reference were approximately **0.15–0.26%**.
The production-slot case also passed. This is kernel-level evidence at short
query lengths, not an independent end-to-end gradient proof at context 2048.

Repeated forward outputs and query gradients were identical in these tests;
key/value and sink gradients can differ because both use atomic accumulation.
The working adapter metadata now explicitly records sink nondeterminism too.
This is compatible with bounded numerical error and does not justify requiring
bitwise-identical optimizer updates from the faster backend.

The actual loaded TileLang and TVM FFI versions are **0.1.9**. Distribution
metadata alone reported TileLang 0.1.8 from another package location. Both
metadata and resolved modules are recorded in
`results/deepseek-v41-debug-20260922/tilelang-parity-with-512-slots.json`.
No container-owned libraries were edited or installed.

The initial diagnosis left TileLang running. After the user explicitly asked
to improve and restart both pairs, it was gracefully checkpointed at normal
step **1,387** (76,778,385 tokens) and simplicial step **872** (48,924,137 tokens).
The restart source `source-qualified-v3` adds an independent FP32 gradient
oracle to every rank's qualification, records resolved kernel versions, and
admits this maintenance-only continuation while checking the model, data,
optimizer, runtime semantics, and protected implementation hashes. Its
`SOURCE_TRANSITION.json` records preserve the existing MLflow curve lineage.
Model weights, optimizer state, RNG and data cursor are restored from the
preserved full checkpoints; the training operator is unchanged.

A 50-update comparison at matching step
ranges measured approximately 6.87 to 5.61 seconds for normal, and 10.37 to
8.96 seconds for simplicial. These observations are not a fresh controlled
timing benchmark. The earlier approximately 14x claim was for the isolated
sparse-attention kernel; it is not the full-model training speedup.

## Restart verification

All four variants passed eight-rank qualification and produced finite training
losses and gradient norms after restart. At the recorded completion snapshot,
corrected linear RF was at step 120 and LinSimp at step 116; both had passed
the router-health checks. TileLang normal advanced from step 1,387 to 1,397
and was saving its next checkpoint; simplicial advanced from 872 to 879.
Both restored all eight ranks at exactly their preserved data cursors.
MLflow showed the corrected RF runs under new identities and continued the
existing TileLang identities with source-transition records and intact history.
The monitor reported no synchronization errors. These are startup and recovery
checks, not evidence of a new capability or validation-loss advantage.

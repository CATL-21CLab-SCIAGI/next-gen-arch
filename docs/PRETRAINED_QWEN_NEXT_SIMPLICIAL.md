# Frozen pretrained Qwen Next with additive simplicial modules

Status: the added leaf module is implemented and tested. Full-model training is
**not implemented or launched**; faithful pretrained-model integration is blocked.

## Agreed experiment

- Start from the full released Qwen3.8-Flash-Next checkpoint, not any from-scratch
  experiment. Freeze all existing weights and retain original QSA/indexer, GDN,
  MoE, PLE, gated residual streams, norms and gates. Retain vision/MTP checkpoint
  weights; do not silently discard them during loading.
- Add a separate simplicial residual branch in layers 4, 8, ..., 48 after the
  original attention residual update and before the existing MoE residual read.
- Train only these additions on one pass of the existing FineWeb-Edu sample-100BT
  tokenized corpus, with 16K context and ordinary main next-token CE.
- Use one training job, no separately trained comparison/control, no automatic
  unfreezing, and no automatic restart of any prior experiment.
- Preserve the frozen NeMo runtime and existing DLC nodes. Target native fully
  sharded DP, with TP/PP/EP/CP/expert-TP all one, subject to numerical and memory
  validation. Existing historical curves are not a controlled pretrained baseline.

The portable, non-launchable contract is
`recipes/proposals/qwen38_pretrained_simplicial_fineweb.yaml`.

## Added leaf mechanism

`src/archlab/architectures/simplicial_adapter.py` owns the independent branch. Its
interface is `[batch, sequence, 4 × hidden_size] ->` the same packed state plus a
learned update. It imports shared architecture primitives, not a trainer.

Full geometry is width 2560, 24 query heads, two KV heads, head dimension 256,
four residual streams with rank 320, windows 16 × 128, partial rotary fraction
0.25, and theta 10M. Q, K1 and K2 have independent zero-centered RMSNorms; output
has a learned sigmoid gate. The score/value operation has no KV constant bias
and uses ordinary partial RoPE, not determinant attention. There is no added FFN.

Each branch has 59,034,368 parameters; twelve branches have 708,412,416. The
output projection starts at zero, so finite inputs initially pass through
unchanged. First-step non-output parameter gradients are therefore zero by
design; after the output projection updates, gradients reach all branch
parameters. Frozen downstream layers still require input-gradient propagation.

## Evidence and limits

- Seven CPU tests cover count/shapes, RNG preservation, exact zero-output identity
  and its input gradient, causality, partial RoPE, adapter state restoration, and
  gradients through a frozen suffix without changing its weights.
- Four existing CPU attention-oracle tests pass.
- Six frozen-container GPU oracle cases pass at head dimension 256: FP32/BF16,
  both causal window boundaries, future-token perturbations, and single-pair
  softmax degeneracy, for outputs and all five input gradients.
- An isolated BF16 adapter at `[1, 16384, 10240]` passes exact initial identity and
  finite nonzero gradients for all added parameters after an output update.
  This uses a diagnostic SGD update, not a production optimizer or full model.

No full-model parity, checkpoint import, native distributed optimizer,
fully-sharded-DP, or pretrained training support is claimed by these tests.

## Compatibility blocker

The frozen DLC environment has Transformers 5.8.1. Its `AutoConfig` rejects the
checkpoint's `model_type=qwen4_exp`; installed Transformers, NeMo, Megatron Bridge
and Megatron Core sources contain no matching model implementation. Native
PyTorch FSDP2 is available, but that does not supply the missing architecture.

The old `qwen38_flash_next_full_train` adapter explicitly constructs a from-scratch
dense-attention variant. It cannot be used as a faithful pretrained loader.

The upstream Qwen4Exp implementation is available as a reference at
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py.
It is not installed or copied into this repository. Finishing within the frozen
runtime requires a separate project-local compatibility implementation and a
complete pretrained tensor/forward audit. Runtime upgrades, monkey patches and
silent substitutions are not fallbacks.

Do not launch the portable proposal until every listed gate has dedicated
evidence. Never replace the original attention with the simplicial pilot wrapper:
that wrapper swaps attention, whereas this experiment must add a new branch.

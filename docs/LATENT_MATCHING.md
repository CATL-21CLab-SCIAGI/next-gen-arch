# Qwen3.5 layerwise architecture switching

The maintained entry point is `python -m archlab.training.latent_match run.json`.
Resolve `recipes/patching/qwen35_4b_stage1.yaml` with OmegaConf and explicit
`ARCHLAB_TEACHER`, `ARCHLAB_OUTPUT`, `ARCHLAB_VARIANT`, and `ARCHLAB_LATENT_DATA`
environment references. Run each variant from the same qualified immutable source
snapshot. Use `torchrun --standalone --nproc_per_node=4` for one variant; place the
two variants on disjoint GPU sets and distinct rendezvous ports.

This adapts [OPAL stage 1](https://hunchings.github.io/late-stage-patching/) to a
Qwen3.5-4B teacher that already contains 24 Gated DeltaNet and 8 full-attention
layers. The two interventions are intentionally different:

| Variant | Replacement sites | Mechanism |
|---|---|---|
| `simplicial` | All 8 full-attention mixers | Full causal signed determinant 2-simplicial attention |
| `triadic` | All 24 GDN mixers | Triadic GDN with 4 feature slices |

Both retain every unchanged layer. Stage 1 trains all parameters of each
replacement mixer, including independent copies of inherited weights. It feeds
each mixer the corresponding frozen teacher input and targets that teacher
mixer's output. The objective sums per-layer mean squared errors; the mean is
over batch, sequence and hidden channels, not the number of replaced layers.
Teacher parameters never participate in the optimizer. End-to-end evaluation
composes all replacements and reports next-token KL and cross-entropy alongside
per-layer raw and normalized errors. These are reconstruction metrics, not
evidence of improved agentic capability.

The determinant variant follows the signed 3-coordinate chunk form in
[Fast and Simplex, section 5](https://arxiv.org/html/2507.02754v1#S5). Q, K1 and K2
receive common SO(3) rotations within each chunk: rotate coordinates 0 and 1,
leave coordinate 2 fixed. The unique frequencies of the teacher's partial RoPE
are assigned to the leading chunks; remaining chunks are unrotated. This changes
the teacher's coordinate pairing. For head dimension 256, 255 coordinates enter
scores; all 256 value coordinates remain. Sarrus expansion uses a padded score
width of 512. The explicit temperature is `256**-0.5`; the paper's determinant
equation itself is unscaled. Both key axes span the entire causal prefix,
including self and diagonal pairs. Complexity is cubic in sequence length.
The old elementwise-trilinear softmax identity does not apply to this variant.

Triadic GDN uses the pinned, unmodified `cute-triadic-gdn` package and validates
its source/runtime identity. Zero feature-projection weights produce uniform
unit-norm slices and reduce to ordinary GDN, up to backend numerical error.
Experiments initialize the new projections with standard deviation 0.02 so this
is not an identity-only exercise. The container must expose the qualified
CUTLASS DSL runtime and CUDA assembler explicitly; no silent reference fallback
is allowed. Both variants support unpadded, uncached full-sequence training only.

The recipe uses matched seeds, document sampling, sequence length 512, 32K global
batch tokens and 25,001,984 processed tokens. The retained teacher and replacement
site counts differ, so this is not a parameter- or compute-matched comparison.
Report trained parameter counts, wall time and measured throughput with results.
The available corpus mix is 60% Wiki-Rewrite, 20% InfiniByte-Reasoning and 20%
STEM-SFT, rather than the article's 70% FineWeb-Edu, 20% Python, 10% open-web-math.
This change of source distributions is part of the experiment contract.

Each source requires an `archlab-latent-documents-v1` manifest: exact teacher
tokenizer file hashes, checksum-pinned one-dimensional integer `tokens.npy`,
document boundaries in `offsets.npy`, and stable group IDs in `groups.json`.
Group hashes choose the train/evaluation partition consistently across sources;
windows never cross document boundaries. Content-hash grouping certifies exact
document separation, not semantic task-family disjointness. Sampling is with
replacement; processed tokens are not unique tokens.

New runs refuse existing output directories. `--stop-after N` runs only through
the specified absolute optimizer step while preserving the full budget's learning
rate schedule, then evaluates and saves. Resume explicitly with
`--resume <same-output>/checkpoints/step-XXXXXXXX.pt`. Resume checks model weights,
tokenizer, data, source, runtime, world size and configuration, and restores the
optimizer plus each rank's random state. It rejects another contract or an older
checkpoint. A run lock prevents two writers; checkpoints are written atomically.
Attempts have separate metrics files. Do not edit a deployed snapshot to repair
a failed run; preserve the failed receipt and qualify a new snapshot.


## Oriented sliding-window bounds (`window_1`, `window_2`)

Full causal determinant 2-simplicial attention (`window_1=None, window_2=None`) scales as
`O(L^3)` over sequence length `L`. For longer-context Stage-2 and Stage-3 training (`L >= 2048`),
`SwitchConfig` exposes optional causal window bounds `window_1` (for the inherited `(k1, v1)`
stream) and `window_2` (for the new auxiliary `(k2, v2)` stream). `SimplicialSwitch` dispatches
through `oriented_determinant_simplicial_attention`, which exploits exact `3x3` determinant
antisymmetry `det([-q, k2, k1]) = det([q, k1, k2])` and `v2 * v1 = v1 * v2` whenever
`window_1 > window_2` so the shorter window always occupies Triton's scalar outer loop and the
longer window always occupies Triton's Tensor Core `T=16` inner loop. Setting
`(window_1=None, window_2=64)` preserves unbounded long-range causal access on the inherited
stream while reducing per-token cost to `O(L * window_2)`.

## Stage 2: End-to-end off-policy forward-KL distillation

The maintained Stage-2 entry point is `python -m archlab.training.stage2_distill run.json`,
paired with `recipes/patching/qwen35_4b_stage2.yaml` (`ARCHLAB_TEACHER`,
`ARCHLAB_STAGE1_CHECKPOINT`, `ARCHLAB_OUTPUT`, `ARCHLAB_VARIANT`, `ARCHLAB_LATENT_DATA`).
Stage 2 composes all replacement switches on an independent student copy initialized from a
Stage-1 `archlab-latent-checkpoint-v1` checkpoint, keeps the original teacher frozen in BF16,
retains FP32 master weights on the active causal-LM student parameters, and minimizes token-level
forward KL `KL(p_teacher || p_student)` over packed windows (`L = 2048`) using vocabulary-chunked
autograd (`kl_chunk_tokens`, avoiding `[B, L, V]` logit allocation) and two-timescale AdamW
learning rates (`1.0x` peak LR on replacement switches, `backbone_lr_ratio` [default `0.1x`] on
retained causal-LM layers; any non-text multimodal submodules remain frozen). Checkpoints use
atomic `archlab-stage2-checkpoint-v1` saves with full contract, optimizer, and per-rank RNG
verification on `--resume`.

## Stage 3: On-policy reverse-KL distillation

The maintained Stage-3 entry point is `python -m archlab.training.stage3_onpolicy run.json`,
paired with `recipes/patching/qwen35_4b_stage3.yaml` (`ARCHLAB_TEACHER`,
`ARCHLAB_INIT_CHECKPOINT`, `ARCHLAB_OUTPUT`, `ARCHLAB_VARIANT`, `ARCHLAB_LATENT_DATA`).
Stage 3 initializes from either a Stage-1 (`archlab-latent-checkpoint-v1`), Stage-2
(`archlab-stage2-checkpoint-v1`), or prior Stage-3 (`archlab-stage3-checkpoint-v1`) checkpoint,
samples prompts of length `prompt_length` from checksum-pinned manifests, rolls out student
continuations on-policy at temperature `temperature` (default `1.0`) without KV-cache assumptions,
and minimizes vocabulary-chunked reverse KL `KL(p_student || p_teacher)` strictly over generated
positions (`prompt_length` context tokens receive zero direct loss). Learning rates follow a
token-parameterized Warmup-Stable-Decay (`wsd_lr_at`) schedule, and `CurriculumState` doubles
`gen_tokens` from `min_gen_tokens` up to `max_gen_tokens` whenever EMA-smoothed reverse KL fails
to improve by `improvement_rel` (default `2%`) for `plateau_steps` (default `30`) steps.

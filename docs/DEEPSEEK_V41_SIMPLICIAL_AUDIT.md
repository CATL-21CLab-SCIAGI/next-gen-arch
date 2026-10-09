# DeepSeek simplicial design audit

**Type:** historical report/code audit, 2026-09-10.
**Disposition:** corrected proposal; production admission belongs to the [official integration](DEEPSEEK_V41_OFFICIAL_TRAINING.md).

## Decisions

| Boundary | Correct contract |
| --- | --- |
| Backbone | Preserve CED/CSA2 ownership, hierarchical selection, sinks, inverse RoPE, Engram and mHC |
| Placement | Eight project-chosen sites; retain delayed stream coefficients |
| Read/write | Softmax stream read; bounded `2*sigmoid` write |
| Initialization | Zero output for initial base identity; test upstream gradient onset after its first update |
| Attention | Exact joint pair-softmax, trilinear scores, product values; causal 32/512 windows including self |
| Position encoding | No extra adapter RoPE; ordinary three-way RoPE is not relatively invariant |
| Frozen state | Freeze correction-bias/calibration/auxiliary mutations as well as parameters |
| Quantized backward | Preserve activation-rounding boundaries and test gradients through frozen suffixes |
| Long documents | Declare target coverage, resets/overlap, gradient truncation and weighting |
| Evaluation | Match parent identity, effort, decoding budget, stops, prompts and scorer |

## Optimizer contract

| Parameter group | Optimizer |
| --- | --- |
| Q/K1/K2 | Headwise Muon on complete globally reduced head gradients |
| V1/V2/O/output gate | Full-matrix Muon |
| Norms and scalar vectors | AdamW |

Across eight sites: 167,772,160 Muon parameters and 44,096 AdamW parameters.

The audited source specifies momentum 0.95 with Nesterov, update-RMS target 0.18, matrix/norm decay 0.1, no bias/scaling decay, and AdamW betas (0.9, 0.95), epsilon 1e-20. Pilot LR 1e-5, 100-step warmup, clip 1 and microbatch 1 are project choices.

The recorded Newton–Schulz sequence is eight iterations with (3.4445, -4.7750, 2.0315), then two with (2, -1.5, 0.5), after Frobenius normalization. Apply `0.18 * sqrt(max(rows, cols))`; do not substitute measured-RMS renormalization.

The speedrun optimizer adds different NorMuon/Polar Express/decay semantics. Norm-preserving MuonH would trap an exactly zero output matrix at zero norm.

## Reference-code findings

| Finding | Consequence |
| --- | --- |
| Inspected FBGEMM backward wrapper argument order disagreed with its Triton signature; backward asserted long window 32 | Static inspection did not qualify the desired 32/512 path |
| Optional K/V biases and GQA conventions differed | Disable biases explicitly and test all five input gradients |
| Released forward uses inference mode and caches | Serving forward parity is insufficient for training gradients |
| CED KV source depends on the exact normalized/collapsed representation | Verify the source boundary, not only its layer number |
| Released bounded replay accepts approximation | New decoder branches need their own exact cache/replay qualification |

## Data repair

The original failure involved a reasoning-only assistant fragment followed by another assistant message. The scoped repair joins only eligible reasoning fragments and records original boundaries/hashes.

Later schemas preserve unresolved adjacent calls, terminal calls and non-assistant endings without inventing results, answers or EOS. Schema 4 reused 215 completed parts unchanged and passed 59 preprocessing/native tests plus three subtests. Retained incomplete trajectories still require an explicit supervision policy.

## Sources and evidence

- [Pinned model assets](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/df42c109f1defefcbfcedbe7d905718a12266e40).
- [Windowed proposal and tensor contract](DEEPSEEK_V41_GLOBAL_SIMPLICIAL_MATH.md).
- Numerical/runtime identities belong to the source-bound qualification receipts.
- This audit did not establish full-model throughput or capability gains.

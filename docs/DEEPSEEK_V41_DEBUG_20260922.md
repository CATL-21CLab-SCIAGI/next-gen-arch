# DeepSeek scratch corrections — 2026-09-22

**Type:** dated debugging and recovery record.
**Studies:** linear RF/LinSimp and faster TileLang normal/simplicial scratch comparisons.

## Confirmed feature-bank defect

| Item | Original | Corrected |
| --- | --- | --- |
| Orthogonal row radius | abs(N(0,1)); chi_1 | Independent chi_D |
| QR orientation | Missing diagonal-sign correction | Sign-corrected Haar directions |
| Expected squared radius at D=16 | 1 | 16 |
| Observed kernel at unit x=y | 0.405–0.421 | Target exp(1)=2.71828 |

The measured original mean squared radius was 1.00289 (eight heads, rank 4096, seed 123). This was a different feature kernel, not finite-rank noise.

The corrected query feature scale uses a common causal maximum across features and valid anchors. It cancels in the output ratio; no per-anchor normalization is introduced.

## Recovery

| Original arm | Preserved step | Supervised tokens |
| --- | ---: | ---: |
| Linear RF | 7,385 | 402,827,317 |
| LinSimp | 6,854 | 374,289,814 |

Corrected runs start fresh from seed 42 and cursor zero. Restoring defective feature buffers, or replacing them under trained weights, would mix operators in one curve.

Corrected source: `9b62513a5ddccdbec90d59c6d7d8ce5c28b3d0be`.
The original records remain preserved and excluded from conclusions about the intended operator.

## Qualification

- Feature-bank marginals, orthogonality and kernel expectation.
- Explicit causal pair-sum oracle, all five input gradients and temperature gradients.
- GQA, window boundaries, prefix invariance and high-temperature behavior.
- Fresh eight-rank admission for each corrected arm.

TileLang sparse-forward/backward qualification is separate. Its numerical and speed results do not establish a trained-model quality advantage.

## Evidence

Local roots:

- `results/deepseek-v41-scratch-linsimp-w640-d20-20260921/`
- `results/deepseek-v41-scratch-highmfu-w640-d20-20260922/`
- `results/deepseek-v41-debug-20260922/RECOVERY.json`

[Linearized 2-Simplicial Attention, Appendix B](https://arxiv.org/html/2608.09307v1) · [Dated outcome audit](TRAINING_CONCLUSIONS_20260922.md)

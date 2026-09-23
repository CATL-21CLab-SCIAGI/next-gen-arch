# MiMo lessons for DeepSeek math RL — 2026-09-23

**Type:** design rationale, not a claim of reproduced MiMo results.
[Source report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf), Figure 8 and §§4.1, 4.3, 5.1, 5.4.
PDF SHA-256: `fb81e6e083801b3358f084ed6be953dc23b0d2e434690f4541d5eae03e01e7af`.

## Figure 8

Code-only MiMo-V2.6-Flash RL, batch 128, token-mean loss; evaluation is DeepSWE avg@3.

GAR ranks passing solutions and redistributes positive advantage toward higher-quality passes. The experiment shows sustained pass-rate improvement, stable turn counts and slower token growth. Token length still increases. The separate success-conditioned length penalty is not the intervention isolated by Figure 8.

Its 140K–210K token scale spans many agent turns; it is not a math-completion budget recommendation.

## Our signal audit

Matched rollouts 73–80:

| Arm | All-failure groups | Multiple-success groups | Negative/positive token-advantage mass |
| --- | ---: | ---: | ---: |
| Normal | 123/128 | 2/128 | 1.20 |
| Simplicial | 120/128 | 2/128 | 1.31 |

These masses describe full trajectories, not the realized four-prefix gradient estimate. GAR cannot create positive signal in all-failure groups; uncapped GAR is the identity for a single passing response.

## Adaptation decisions

| Issue | Proposed response |
| --- | --- |
| Truncation | Calibrate completion, correctness, cost and replay memory at larger budgets |
| Sparse reward contrast | Measure useful groups before adding a quality grader |
| Loss weighting | Compare original trajectory-sum RLOO with prompt-group token normalization |
| Longer trajectories | Revisit replay-prefix sampling and gradient variance |
| Excessive length | Penalize only correct responses under a success-rate gate |
| Router drift | Test router freezing and record expert-load health |
| Asynchronous reuse | Requires behavior probabilities and importance correction; absent from current synchronous RLOO |
| Cache precision | Preserve the numerical gate before deployment |

The current 2,048-token context permits a uniform response budget of at most 1,489 for the longest training prompt.

## Status

The [shared-GPU restart recipe](DEEPSEEK_V41_RL_SHARED_RESTART.md) records selected changes as a new contract. The [cache](DEEPSEEK_V41_RL_CACHE_STATUS.md) remains unqualified.

Local evidence: `results/deepseek-v41-math-rl-cache-20260923/*-signal-audit.json` and per-rank prefix receipts. Faster generation alone does not establish better RL sample efficiency.

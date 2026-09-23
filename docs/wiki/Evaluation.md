# Evaluation

## Match the claim to the measurement

CE is cross-entropy in nats/token; BPB is bits/byte. Keep data, tokenization and scoring scope explicit.

| Measurement | Supports | Does not establish |
| --- | --- | --- |
| Held-out CE/BPB | Predictive quality on the fixed token set | Generated-solution accuracy |
| Capability subset | A bounded regression screen | Broad capability preservation |
| Kernel oracle | Tested numerical behavior | Whole-model speed or quality |
| Throughput | Efficiency under the recorded workload | Architecture quality by itself |

## Comparative protocol

Use the same example IDs, prompt construction, scoring code, decoding settings, token budgets, and model identities. Report truncation and malformed answers separately.

Use the released weights for an original-model baseline. Disabling an adapter restores that baseline only when the backbone remained unchanged.

For paired results, report gains, regressions and uncertainty. Bootstrap at the independent example/problem unit; correlated windows and matched twins are not independent samples.

## Existing lightweight checks

| Task | Protocol used in existing comparisons |
| --- | --- |
| MMLU | Answer-label continuation likelihood |
| ARC-Challenge | Answer-text likelihood; distinguish raw and normalized scores |
| PIQA | Choice likelihood; distinguish raw and normalized scores |
| Math | Native rendering and declared final-answer verifier |

Preserve exact task-specific protocols when reusing recorded results.

## Concurrent evaluation

Run against an immutable checkpoint or an explicitly recorded resident-policy state. Use separate output and communication namespaces. Shared-GPU evaluation needs measured memory headroom and must report contention.

**Next:** [[Results]] · [[Math RL|Math-RL]]

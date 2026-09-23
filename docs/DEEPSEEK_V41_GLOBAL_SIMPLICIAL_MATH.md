# DeepSeek windowed simplicial adapter proposal

**Type:** historical design, 2026-09-10; data recovery schema 4.
**Successor:** [official AutoModel training](DEEPSEEK_V41_OFFICIAL_TRAINING.md).
The filename is retained for existing links; the design uses fixed windows.

## Model and insertion

| Item | Contract |
| --- | --- |
| Base | DeepSeek-V4.1-Flash revision `df42c109f1defefcbfcedbe7d905718a12266e40` |
| Backbone | Width 5120; 40 layers; four residual streams; 384 routed experts, top-6, one shared expert |
| Initial trainable set | Added branches only; original weights and mutable training state frozen |
| Sites | Layers 5, 10, …, 40, after attention expansion and before FFN mixing |
| Adapter geometry | Eight query / two KV heads; head dimension 128 |
| Parameters | 20,977,032/site; 167,816,256 total |
| Initialization | Q/K/V/gate std 0.02; output zero; effective norms/writes one |
| Position scheme | No extra adapter RoPE |
| Inactive components | Vision and speculative MTP; original checkpoint retained |

Read the four streams with softmax weights; write with bounded `2 * sigmoid(logits)` coefficients. Preserve the backbone's delayed coefficient flow.

## Attention equation

```text
score[i,j,k] = sum_d(Q[i,d] * K1[j,d] * K2[k,d]) / sqrt(d)
J(i) = [max(document_start, i-31), i]
K(i) = [max(document_start, i-511), i]
P[i,:,:] = joint_softmax(score[i,:,:], over J(i) × K(i))
Y[i,d] = sum_{j,k} P[i,j,k] * V1[j,d] * V2[k,d]
```

Self and diagonal pairs are included. The branch evaluates at most 16,384 pairs/query/head. There is no implicit K/V bias or factorized normalization.

## Adapter tensors

| Tensor | Shape |
| --- | --- |
| Q and output gate, each | [1024, 5120] |
| K1/K2/V1/V2, each | [256, 5120] |
| Output | [5120, 1024] |
| Input norm | [5120] |
| Q/K1/K2 norms, each | [128] |
| Read/write logits | Two [4] vectors |

## Data contract

Five Parquet sources contain 7,085,839 conversations; duplicate JSONL mirrors are excluded. The native Python encoder uses effort 75 and preserves reasoning, tool traces and unfinished endings.

Assistant spans are half-open indices in the unshifted token sequence. Training shifts labels once. Split by normalized problem hash (1% validation); this does not detect near duplicates.

| Recovery | Preserved output |
| --- | --- |
| Schema-2 inventory | 184 parts; 368,000 conversations; 6,780,235,595 tokens |
| Schema-3 inventory | 215 parts; 430,000 conversations; 7,791,512,843 tokens |
| Schema 4 | Preserves native non-assistant endings; no invented answer/EOS or supervised trailing header |

Reused parts are verified copies with original provenance. Original outputs and snapshots stay immutable. See [preprocessing](../src/archlab/preprocessing/README.md).

## Admission

Require complete checkpoint mapping, base identity, gradient onset after zero-O warm start, exact local-window oracles, optimizer grouping, distributed continuation and full-context memory measurements. Zero output, placement and optimizer pilot settings are project choices—not paper-prescribed defaults.

[Report/code audit](DEEPSEEK_V41_SIMPLICIAL_AUDIT.md) · [Portable proposal](../recipes/proposals/deepseek_v41_global_simplicial_math.yaml)

# Native Limite attention adapter comparison

The consolidated [Limite SFT comparison](wiki/Limite-SFT-Comparison.md) records
geometry, parameter counts, the actual 2B warmup + 8B full-finetuning sequence,
endpoint evaluations and limitations.

The [full-finetuning recipe](../recipes/limite/full_finetune_math.yaml)
is the executable contract. The original optimization chronology is preserved
in the private history archive; measured speedups were workload-specific and
must not be treated as architecture-independent performance guarantees.

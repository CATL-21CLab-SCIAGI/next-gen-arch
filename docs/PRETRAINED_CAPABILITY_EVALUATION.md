# Pretrained Qwen capability pilot

**Type:** historical evaluation protocol. **Scope:** unchanged released backbone versus the same backbone with trained additions.

## Frozen pilot

| Task | Cases | Protocol |
| --- | ---: | --- |
| MMLU | 57; one per subject | Zero-shot answer-label likelihood |
| ARC-Challenge | 32 | Character-normalized answer-text likelihood |
| GSM8K | 8 | Native chat; final numeric exact match |
| AIME 2024/2025 | 2 each | Native chat; boxed final-answer match |

Math uses greedy decoding, medium reasoning, at most 1,024 response and 2,048 total tokens. Identical prompts, budgets and scorers apply to both modes. Unfinished reasoning numbers are not final answers.

## Implementations

| Entry | Purpose |
| --- | --- |
| `archlab.automodel.evaluate` | Single-GPU DSW pilot |
| `archlab.automodel.evaluate_distributed` | Separate concurrent FSDP32/EP8 evaluation |
| `archlab.benchmarks.capability` | Selection, scoring and paired statistics |

The distributed entry evaluates immutable step-4000 additions, not live trainer weights. It uses a separate rendezvous port, no optimizer, frozen parameters and before/after hashes. Each allocator is capped at 30%; startup requires 96 GiB free per device.

Use `torch.no_grad()`: this runtime's FSDP all-gathers require tensor version counters and are incompatible with `inference_mode()`.

## Results and limits

Completed DLC pilot: MMLU 53/57 → 52/57; normalized ARC 19/32 → 21/32; GSM8K 7/8 unchanged; capped AIME 0/4 unchanged.

These differences are inconclusive. Paired intervals and exact McNemar tests are exploratory and not corrected across tasks. The short AIME budget is a material limitation. Do not merge the 90 completed DSW pairs into the different-runtime DLC pilot.

## Evidence

- Contract: `recipes/evaluations/qwen38_simplicial_regression_pilot.yaml`.
- Prompts: `src/archlab/prompts/capability_regression.yaml`.
- Completed result: `results/pretrained-capability-dlc-step4000-20260908-v2/summary.json`.
- Per-example predictions and run identities accompany the summary.

[General evaluation guide](wiki/Evaluation.md) · [Training lineage](PRETRAINED_QWEN_NEXT_SIMPLICIAL.md)

# Pretrained capability regression evaluation

The DSW pilot compares the **unchanged Qwen3.8-Flash-Next language backbone**
with **the same backbone plus the trained step-200 simplicial additions**.
It is not a comparison against published leaderboard numbers or another model.
FineWeb training loss and qualitative samples do not establish capability gains.

## Current run

- Entry: `src/archlab/automodel/evaluate.py`.
- Contract: `recipes/evaluations/qwen38_simplicial_regression_pilot.yaml`.
- Prompts: `src/archlab/prompts/capability_regression.yaml`.
- Dataset/grading/statistics: `src/archlab/benchmarks/capability.py`.
- Output: `results/pretrained-capability-pilot-step200-20260907-v2/`.
- Checkpoint: `results/pretrained-simplicial-fineweb-20260907-v1/checkpoints/step-00000200-96a408d584bc`.

`contract.json` is immutable and records dataset/source hashes, selected IDs,
environment versions, pretrained checkpoint identity, grading source hashes,
chat template, and EOS tokens. `pairs.jsonl` saves per-example paired predictions,
scores, generated answers, truncation state, and timings after both modes finish.
`summary.json` updates after every completed pair; `complete` distinguishes a
finished pilot from partial output. Do not interpret unfinished tasks as zero
accuracy. The log is the output directory name plus `.log`.

| Task | Pilot pairs | Primary metric | Protocol |
|---|---:|---|---|
| MMLU | 57, one per subject | Accuracy | Zero-shot answer-label likelihood |
| ARC-Challenge | 32 | Character-normalized accuracy | Zero-shot answer-text likelihood |
| GSM8K | 8 | Final numeric exact match, flexible extraction | Native chat, medium reasoning, greedy |
| AIME 2024 | 2 | Boxed final-answer exact match | Native chat, medium reasoning, greedy |
| AIME 2025 | 2 | Boxed final-answer exact match | Native chat, medium reasoning, greedy |

Both modes get the same examples, prompts, token budgets, and scorer. Mode order
alternates by example. Disabling only `AdditiveMoERead.adapter_enabled` restores
the original pretrained residual update; no weights are swapped or modified.
The loader restores only the adapter subset of the distributed checkpoint.

The existing pinned lm-eval 0.4.13 source supplies `RegexFilter` and the AIME
grader. Its full task manager requires an unavailable `sacrebleu` dependency, so
this adapter does **not** claim to run the full harness. It directly computes
causal choice likelihoods and reproduces the GSM8K YAML's exact-match
normalization. Primary references: [lm-eval model interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md),
[GSM8K task documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/README.md),
and [AIME scoring](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/aime/utils.py).
Run provenance hashes the actual local pinned source, not these mutable links.

## Interpretation and limits

This is a **101-pair bounded pilot**, not a full benchmark result. Math uses at
most 1,024 new tokens and 2,048 total tokens. The qualified upstream model has no
generation cache. AIME commonly requires substantially longer reasoning; its
pilot outcomes must be read alongside token-cap and unfinished-thinking counts.
Numbers inside an unfinished think block do not count as final answers. Both
checkpoint EOS IDs are honored. Prompts are never silently truncated or replaced
when they exceed the declared budget.

The summary reports paired gains, regressions, accuracy deltas, approximate
95% intervals, and exact two-sided McNemar p-values. Intervals combine 97.5%
Wilson gain/loss intervals with a Bonferroni bound; they remain nonzero even when
there are no observed changes. These per-task exploratory intervals/tests are
not corrected across tasks and are not a preregistered non-inferiority test.
This small sample cannot establish that abilities are preserved. A larger
held-out evaluation with adequate reasoning budgets is needed before any such
claim. Do not tune the model or prompts against these held-out answers.

## Qualification and environment

The 19-test CPU/GPU suite passed in the approved existing DSW environment.
The FlexAttention comparison was then extended to 2,048 tokens and passed
separately at the real 24-query-head / 2-KV-head / 256-head-width geometry.

```bash
OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:tests:results/pretrained-backend-audit-20260907/Automodel:/mnt/nas/evergreen/runtime/lm-eval-0.4.13 \
/mnt/nas/evergreen/env/venv/bin/python -m unittest \
  test_capability_evaluation test_automodel_sampling test_automodel_simplicial
```

Checks include exact CPU FP32 hidden-state/logit equality between the original
small upstream model and the installed-but-disabled **nonzero** additions;
frozen-parameter identity/value preservation; exception-safe mode restoration;
continuation likelihoods against a tokenwise oracle; EOS and truncation handling;
final-answer extraction; selection; paired statistics; and the existing sampler
GPU checks. They do not prove full-model cross-backend numerical equivalence.
The backend recipe/parity guidance informed the immutable contract and the
nonzero bypass oracle.

The entry runs by filename with the original immutable training source snapshot
on `PYTHONPATH`; it fails if that source or pinned AutoModel revision has changed.
It reuses the qualified BF16 loader and native CPU PLE lookup hook. No package
installation, new environment, upstream source edit, or DLC interruption is
needed. Existing FLA autotuner restrictions are reused and recorded. The normal
FlexAttention unfused fallback may occur at new sequence lengths; the unsafe
debug bypass is never enabled.

```bash
OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=results/pretrained-simplicial-source-20260907-v1/src:results/pretrained-backend-audit-20260907/Automodel:/mnt/nas/evergreen/runtime/lm-eval-0.4.13 \
/mnt/nas/evergreen/env/venv/bin/python -u src/archlab/automodel/evaluate.py \
  --base results/pretrained-cache-20260907/Qwen3.8-Flash-Next \
  --checkpoint results/pretrained-simplicial-fineweb-20260907-v1/checkpoints/step-00000200-96a408d584bc \
  --data results/pretrained-capability-data-20260907-v1 \
  --recipe recipes/evaluations/qwen38_simplicial_regression_pilot.yaml \
  --prompts src/archlab/prompts/capability_regression.yaml \
  --harness /mnt/nas/evergreen/runtime/lm-eval-0.4.13 \
  --output results/pretrained-capability-pilot-step200-20260907-v2
```

Add `--preflight-only` to validate all selected prompts without model allocation.
Use `--resume` only after the prior writer has exited; a file lock prevents two
writers, and the complete contract must match. Do not run a second full-model
process while the pilot is using the DSW GPU.

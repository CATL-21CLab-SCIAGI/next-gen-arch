# Shared-GPU RL restart candidate — 2026-09-23

Portable contract: `recipes/experiments/deepseek_v41_nemotron_rloo_shared_gpu.yaml`.
Both variants start from their matched step-4537 SFT parents. The changed reward,
normalization, trainable set, and generation budget make this a new experiment;
it does not resume a v3 optimizer checkpoint.

## Learning strategy

- Four samples per prompt, the existing fixed prompt order and seed, and on-policy
  temperature/top-p of one are retained.
- The response budget is 1,489 tokens inside the current 2,048-token context,
  covering the longest 559-token training prompt without truncating prompts.
- RLOO advantages use leave-one-out baselines. Each prompt group's loss is divided
  by that group's response-token count, then prompt groups receive equal weight.
  Eight uniformly sampled replay times estimate that objective before clipping.
- Router parameters are frozen; other text parameters remain trainable.
- A length deduction applies only to correct responses in groups with at least
  two successes and a pass rate above 0.25. Its reference is the median successful
  length, with 10% tolerated excess and a maximum deduction of 0.1. Correct
  responses remain above failed responses. Raw correctness stays separate from
  the shaped training reward.
- Policy entropy, EOS probability, token-weighted advantage masses, and clipping
  remain visible in the metrics. No external grader, GAR ranking, or asynchronous
  policy reuse is introduced.

## Memory and numerical admission

The candidate caps each trainer's PyTorch allocator at 190 GiB, disables retention
of gathered weights between forwards, and moves only saved checkpoint input
activations to pinned CPU memory. Weights and gradients keep their GPU ownership.
CUDA contexts and external library allocations are additional to the allocator
cap, so the cap alone is not proof of evaluation headroom.

Before training admission, each actual actor runs a maximum-context sampled-token
forward/backward audit without an optimizer update. A background observation of
driver-free memory must show at least 64 GiB free on every rank. The ordinary
0.02-nat replay gate also applies. Failure stops admission; neither the numerical
bound nor the memory reserve is silently weakened.

The unqualified incremental cache is disabled. Initial held-out evaluation is
deferred so production reaches its first training updates before evaluation.
The existing post-pilot and periodic math evaluation remain in the loop.

## Deferred lightweight general evaluation

After the restarted actors are admitted and a real update is verified, evaluate
the released original DS4.1 Flash weights, with no adapter, on a small fixed subset
of the existing MMLU/ARC/PIQA cases. Reuse the existing scoring protocol and compare
the exact same case IDs against recorded adapter-finetuned results. A small sample
can screen for large regressions; it does not prove equivalence on every task.

Evaluation shares the current 32 GPUs and receives an explicit memory budget.
There is no additional evaluation node. The user's evaluation pipeline was cloned
to `/mnt/nas/evergreen/eval_pipeline` at commit
`f3c8822a29047bbff7ea4e4c758856089a336a04`; a full 16-dataset campaign is outside
this restart's scope.

All launches use the existing DLC image's `/opt/venv/bin/python` and installed
Megatron/PyTorch/CUDA runtime. No training environment is created or upgraded.

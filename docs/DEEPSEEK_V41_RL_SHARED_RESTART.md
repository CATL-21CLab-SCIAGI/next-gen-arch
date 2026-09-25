# Shared-GPU RL restart candidate — 2026-09-23

> Historical experiment record. For the supported command and current qualification scope, use [Miles baseline](MILES_BASELINE.md). Statements below refer to their dated attempts, not live health.

**Superseded:** the user withdrew concurrent general evaluation as a requirement.
See [the training-first restart](DEEPSEEK_V41_RL_TRAINING_PRIORITY.md).
The failed candidates below are retained as experiment history.

**Status (2026-09-23):** the 190 GiB and 198 GiB admission attempts failed at maximum-context backward before optimizer updates. Production has not been admitted. The completed v3 recovery checkpoints remain intact.
**Recipe:** `recipes/experiments/deepseek_v41_nemotron_rloo_shared_gpu.yaml`.
**Parents:** matched step-4537 SFT pair; fresh RL state because the scientific contract changes.

## Learning settings

| Item | Candidate |
| --- | --- |
| Sampling | Four responses/prompt; fixed paired order and seed; temperature=top-p=1 |
| Response/context | 1,489 / 2,048 tokens; longest training prompt 559 |
| Advantage | RLOO leave-one-out baseline |
| Loss | Normalize by each prompt group's response tokens, then average prompts |
| Replay | Eight uniformly sampled times |
| Trainable set | Routers frozen; other text parameters trainable |
| Length deduction | Correct responses only; at least two successes and pass rate >0.25 |
| Length reference | Median successful length; 10% tolerated excess; maximum deduction 0.1 |
| Metrics | Raw correctness, shaped reward, entropy, EOS probability, advantage masses and clipping |

No GAR grader or asynchronous policy reuse is introduced. Correct responses retain higher reward than failures.

## Memory and admission

| Boundary | Requirement |
| --- | --- |
| Trainer allocator | 190 GiB original candidate; 198 GiB retry also failed |
| Gathered weights | No between-forward retention |
| Activation placement | Saved checkpoint inputs may move to pinned CPU memory |
| Weight/gradient placement | Existing GPU ownership preserved |
| Memory gate | At least 64 GiB driver-free memory on every rank in the maximum-context audit |
| Numerical gate | 0.02-nat replay bound |
| Qualification update | No optimizer step |

External-library and CUDA-context allocations are additional to the allocator cap. A configured cap alone does not prove headroom. The unqualified cache is disabled.

## Evaluation order

Reach admitted training and a verified real update before evaluation. Existing post-pilot and periodic math checks remain.

Then evaluate the released original model without adapters on a small fixed subset of existing MMLU/ARC/PIQA cases, using the same case IDs and scoring protocol as recorded fine-tuned results.

Evaluation shares the existing 32 GPUs under an explicit memory budget. A small subset screens for large regressions; it does not prove broad equivalence.

The optional `eval_pipeline` reference was inspected at `f3c8822a29047bbff7ea4e4c758856089a336a04`. Full multi-dataset evaluation is outside this restart contract. Execution uses the existing container runtime.

## Latest memory findings

The actual actor uses full activation checkpointing. A proposed selective-GEMM cache change was rejected by the distributed probe because that wrapper policy is absent; it is not enabled. An opt-in in-place ordered MoE sum passed exact component output/gradient tests and awaits distributed/full-model admission. The 64 GiB reserve and replay tolerance remain unchanged.

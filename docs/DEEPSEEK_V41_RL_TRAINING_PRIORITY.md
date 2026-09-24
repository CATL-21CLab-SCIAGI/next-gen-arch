# DeepSeek V4.1 training-first RL restart

**Retired by user request (2026-09-24):** the replacement is Miles using the
MiMo report's RL recipe, starting from the matched step-4537 SFT parents.
See [migration status and blockers](DEEPSEEK_V41_MILES_MIMO_MIGRATION.md).
The fallback and admission notes below describe the historical AutoModel run.

The scientific comparison is matched normal versus simplicial (three-body) attention.

**Active fallback candidate:** `recipes/experiments/deepseek_v41_nemotron_rloo_retained.yaml`.
It retains gathered weights during sampling, disables the rejected KV cache, and avoids
unused full-vocabulary projections. No CPU offloading or concurrent general evaluation.
Actual replay admission and real optimizer updates remain required.
Concurrent general evaluation is deferred to a later phase. No extra GPUs are allocated.

The initial uncached attempts were stopped before their first optimizer update after
more than 40 minutes of generation. The successor candidate is
`recipes/experiments/deepseek_v41_nemotron_rloo_cached.yaml`; see the
[cache qualification record](DEEPSEEK_V41_RL_CACHE_STATUS.md). The execution contract
below records the initial uncached attempt.

## Execution contract

- Recipe: `recipes/experiments/deepseek_v41_nemotron_rloo_training_priority.yaml`.
- Matched step-4537 SFT parents, fresh RL optimizer state, 16 B300 GPUs per actor.
- Context 2,560; response budget 2,001; group size four; fixed paired prompt order and seed.
- Prompt-token normalization, frozen routers, eight sampled replay times, and success-only
  length deduction retained from the [MiMo-informed strategy](DEEPSEEK_V41_RL_MIMO_LESSONS.md).
- Training allocator cap 250 GiB; no evaluation reservation; no CPU weight, gradient,
  checkpoint-input, or HC activation offloading.
- Exact ordered expert sum, expert activation recomputation, bounded pointwise temporaries,
  early release of concatenated inputs, and compute-stream FSDP gathers.
- Existing container runtime only. KV cache remains disabled pending numerical qualification.

## Memory evidence and limits

Both 16-rank full-geometry synthetic tests passed two accumulated backward passes at
context 2,560, including a placeholder for future optimizer state. Normal peaked at
219.58 GiB and simplicial at 219.84 GiB; minimum driver-free memory was 34.22 and
33.89 GiB respectively. These tests force routing to one expert owner to stress memory.
They establish resource feasibility, not model quality or production admission.

Production still requires actual-model numerical replay within the unchanged 0.02-nat
bound. Health means real reward-driven updates, finite metrics, and a completed pilot
checkpoint. Held-out math cases remain fixed; no positive research result is assumed.

Run artifacts: `results/deepseek-v41-math-rl-shared-20260923/production-{normal,simplicial}-training-priority-v1`.
Memory evidence: `component-training-priority-v1` under the same root.
Compare accuracy changes from each arm's SFT baseline, completion rates, and GPU time
per useful update. A memory improvement alone is not a generation-throughput result.

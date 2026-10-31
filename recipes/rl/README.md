# Agentic RL contracts

The current Qwen3.5-4B configuration is `qwen35_code.yaml` plus exactly one
placement overlay: `qwen35_code_local.yaml` (v3c) or
`qwen35_code_efficiency.yaml` (v4e). They replace the sequential Python config
builders. Use `archlab resolve ... --overlay ... --output ...`; deployment paths
remain environment bindings. The efficiency arm also changes batch size and
selects an 80-task curriculum, so it is not a pure systems comparison.

Both migrated configurations reproduce their live resolved values exactly.
Production still runs immutable snapshots: the new Qwen qualification gate
remains pending until model/cache, optimizer/restart, real verifier and throughput
checks execute on the deployment. See [Training](../../docs/TRAINING.md).

The Limite and Muse contracts below remain historical candidates; they do not
describe the current Qwen jobs.

`limite_conversion_code.yaml` is the explicit original-base attention replacement
experiment after full-weight agentic SFT. Set `ARCHLAB_POLICY` to the finalized
package SFT Hugging Face export and `ARCHLAB_PARENT_CHECKPOINT` to its completed
`global_step_N` checkpoint. Admission verifies the package SFT receipt, original
publisher, source data/tokenizer/template, actual completed sample count and
recorded token upper bound. It starts a fresh RL optimizer and scheduler.

The candidate uses pinned verl's MiMo code runner, GRPO and bounded async
execution with at most one policy version of lag and token importance weights
capped at two. The converted serving path requires eager FA2, one request per
replica, TP=PP=1, no prefix cache and no chunked prefill. The learner uses DP8 and
native BSHD microbatches of one. These settings are a contract, not evidence of
GPU admission; generation/replay, restart, native context, throughput and an
executable nonzero-reward update must still pass.

The initial curriculum uses two prompts with four responses each (eight
trajectories), the standard MiMo 500-step limit and an explicit 8,192-token turn
cap. Set `ARCHLAB_HARNESS_SPEC` to `harness/conversion_single_bash.yaml`. The full
131,072-token model context remains available across turns. Length caps stay
visible as truncated agent runs; the package's unfinished-episode mask excludes
them from policy loss. Admission requires eight completed verifiers across at
least two prompt groups, at least one real tool call and observation token, zero
truncated trajectories, at least one group with mixed rewards and a finite
positive gradient. These are minimum execution/learning-signal checks, not a
claim of sustained improvement. A capped or all-zero pilot fails admission.

The v2 candidate explicitly samples training and validation at temperature 0.6
with full support (`top_p=1`, `top_k=-1`). Its single harness must use the same
distribution as the learner, and serving records processed log probabilities.
Admission separately compares those sampled-response probabilities against
temperature-scaled native replay; raw prompt/cache parity is insufficient.
Top-p/k filtering is excluded because the current learner does not recompute
its truncated-distribution normalizer. Historical v1 receipts remain unchanged.

Other deployment inputs remain `ARCHLAB_CODE_DATA`, `ARCHLAB_HARNESS_SPEC`,
`ARCHLAB_OUTPUT` and private sandbox/tracking configuration. Resolve without
launching:

```bash
archlab resolve recipes/rl/limite_conversion_code.yaml --output "$ARCHLAB_OUTPUT"
```

`limite_code.yaml` records the historical inserted-normal-attention 2B adapter
SFT control. Its unchanged contract and existing callers remain available for
reproducing the failed agentic pilot; it does not describe the replacement model.
The legacy active math training source is separate and remains untouched.

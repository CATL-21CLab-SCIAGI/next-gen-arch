# Experiment Design

Every run needs a stated question, comparison, budget, and promotion criterion.

## Choose the comparison

| Regime | Budget constraint | Interpretation |
| --- | --- | --- |
| Controlled | Matched effective training tokens | Component effect under the matched recipe |
| Fixed compute | Matched algorithmic model FLOPs | Quality at the same model-compute budget |
| Scaling | Declared tokens per parameter | Behavior as model size and training budget grow |

Disclose parameter overhead, executed fallback work, memory, and wall time.

## Pair the arms

Hold seed, shared initialization, data order, tokenizer, context, batch, optimizer, precision, and evaluation fixed unless they are the declared variable.

For pretrained work, also pin parent weights, adapter placement, initialization, and trainable parameters. For RL, record behavior-policy identity, reward implementation, replay rules, generated tokens, and applied versus skipped updates.

## Qualification and promotion

1. Check construction, forward/backward numerics, optimizer grouping, and checkpoint continuation.
2. Run paired small-scale controls; use multiple seeds for confirmation.
3. Review quality, stability, throughput, and memory together.
4. Promote to a larger scale under a new recorded contract.
5. Combine mechanisms only with their constituent controls retained.

A passing probe establishes its tested property. It does not establish a capability gain.

## Report honestly

- Keep failed seeds and unfinished budgets visible.
- Separate training-seed variation from evaluation-sample uncertainty.
- Compare checkpoints at matched budgets.
- Preserve earlier findings when a later result reverses them.
- Give numerical, capacity, and contract failures new diagnoses rather than silently retrying changed settings.

**Repository reference:** `docs/EXPERIMENT_CONTRACTS.md`.

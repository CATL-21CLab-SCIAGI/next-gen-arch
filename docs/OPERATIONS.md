# Operations reference

**Reader guide:** [Runbook](wiki/Runbook.md). Launch settings come from the selected recipe and run contract.

## Launch admission

1. Verify compute identity, mounts, ownership, locks and memory.
2. Verify source, data, tokenizer, runtime and parent checkpoint identities.
3. Resolve the exact command, comparison budget and fresh output namespace.
4. Pass the required forward/backward, replay, memory and distributed probes.
5. Start the declared attempt and verify actual trainer progress.

Exclusive allocation is the default. Shared-GPU execution requires explicit authorization, a recorded sharing policy and measured headroom.

## Launch interfaces

Use `archlab resolve`, `archlab qualify` and `archlab train` with a package-backed contract from `recipes/rl/`; the [training guide](TRAINING.md) lists model and environment bindings. The launcher delegates execution to verl.

Historical speedrun/Megatron contracts retain their packing, optimizer and data-order semantics. Reproduce a dated experiment from its recorded source revision. Retired AutoModel and custom math-RL commands must not be used against the current working tree.

## Recovery

| Situation | Action |
| --- | --- |
| Operational failure | Resume a complete checkpoint with a new attempt identity |
| Numerical failure | Preserve evidence; diagnose before changing the contract |
| Capacity failure | Requalify changed batching, memory policy or topology |
| Contract mismatch | Repair source/data/runtime identity before launch |
| Unknown failure | Triage before retry |

Resume restores model, optimizer, scheduler, RNG and data cursor. A new objective or geometry is a new experiment, not an implicit resume.

## Monitoring

Record lifecycle time, steady-state time, generated/supervised tokens as appropriate, peak memory, finite gradients and complete checkpoints. A running allocation or stale metric file is insufficient evidence of progress.

Protect live source and outputs. A Git push does not change imported code or authorize restarting another process. Development location and launch synchronization follow the active task's agreed workflow.

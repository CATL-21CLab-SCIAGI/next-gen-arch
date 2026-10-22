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

## Agentic RL on external Ray heads

Long agentic runs use a GPU node as the Ray head, CPU DLC agent pods for runner tasks, and the CPU DLC sandbox pool.
Each handle takes a JSON spec or plan that keeps machine paths outside the code:

| Handle | Role |
| --- | --- |
| `python -m archlab.training.supervise <arm.json>` | the only launcher: resume from the latest checkpoint, resubmit evicted agent pods, switch data or source snapshots at the next saved checkpoint |
| `python -m archlab.training.runtime <resolved.yaml>` | the launch it performs: upstream verl, TaskRunner pinned to the driver node |
| `python -m archlab.training.dlc_jobs {sandbox-pool,agent-pods,status}` | CPU DLC jobs from one job body; site settings in `.runtime/dlc-site.json`; CPU pods only on GPU-full nodes |
| `python -m archlab.evaluation.ladder heldout <plan.json>` | export every tenth checkpoint and run the frozen held-out curve |
| `python -m archlab.evaluation.tb21 {oracle,run} <plan.json>` | Terminal-Bench 2.1 ladder with official budgets and a mounted verifier toolchain |

Controllers stay on the GPU head (uni-agent patch 0007 and `runtime`), so a pod eviction costs runner tasks only.
Snapshots are git worktrees of a commit plus the pinned third-party checkouts with the recorded patch series.
A data change resumes the actor only, with a fresh dataloader; a source change resumes the full checkpoint.
Both are recorded in `LAUNCHES.jsonl` and the run contract.

## Monitoring

Record lifecycle time, steady-state time, generated/supervised tokens as appropriate, peak memory, finite gradients and complete checkpoints. A running allocation or stale metric file is insufficient evidence of progress.

Protect live source and outputs. A Git push does not change imported code or authorize restarting another process. Development location and launch synchronization follow the active task's agreed workflow.

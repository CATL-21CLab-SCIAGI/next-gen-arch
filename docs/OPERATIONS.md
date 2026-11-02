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
| `python -m archlab.training.supervise <arm.json>` | exclusive arm ownership, durable bounded retries and progress monitoring for one qualified experiment |
| `python -m archlab.training.runtime <resolved.yaml> --qualification <admission-dir>` | shared admission gate; upstream verl, TaskRunner pinned to the driver node |
| `python -m archlab.training.dlc_jobs {sandbox-pool,agent-pods,status}` | CPU DLC jobs from one job body; site settings in `.runtime/dlc-site.json`; CPU pods only on GPU-full nodes |
| `python -m archlab.evaluation.ladder heldout <plan.json>` | export every tenth checkpoint and run the frozen held-out curve |
| `python -m archlab.evaluation.tb21 {oracle,run} <plan.json>` | Terminal-Bench 2.1 ladder with official budgets and a mounted verifier toolchain |

Controllers stay on the GPU head (uni-agent patch 0007 and `runtime`), so a pod eviction costs runner tasks only.
Snapshots are git worktrees of a commit plus the pinned third-party checkouts with the recorded patch series.
Every launch rechecks a passing v2 qualification receipt against the configuration,
source, upstream, runtime and model/data bytes. Changed experiments need new
qualification and run directories. Old v1 receipts cannot admit the new source.

`ArmSpec` in `training/supervise.py` defines the machine settings: `arm`, `node`,
`source`, `qualification`, `directory`, and `pythonpath` are required. Paths are
absolute; the first import path is the immutable snapshot's `src`. Optional
settings include `python`, `mlflow_config`, polling/SSH/progress timeouts and the
retry limit. Each attempt has its own directory, log and result receipt. Retry
counts survive supervisor restarts. Unknown exits or ambiguous SSH launches
require triage; a stale live PID stops supervision without killing the trainer.
CPU pod provisioning remains an explicit `dlc_jobs` operation.
Before starting an attempt, the supervisor waits for complete GPU telemetry and
less than 8,000 MiB total allocated memory on the selected node. Waiting does
not consume retries or interrupt the current GPU workload; exclusive allocation
must still be established by the operator.

Same-run continuation requires the run identity, latest committed checkpoint,
all actor shards and dataloader state. Numerical/capacity/contract failures never
retry unchanged. Synchronous continuation is supported by the wrapper contract;
asynchronous trajectory-queue continuation remains blocked pending its own
admission evidence. A new data snapshot is a new experiment.

Evaluation writers for the same MLflow experiment share a NAS `run_registry`.
Their ledgers serialize writers, preserve pending event timestamps and reconcile
remote metric history after a crash. This is recovery reconciliation, not a
cross-service transaction.

## Monitoring

Record lifecycle time, steady-state time, generated/supervised tokens as appropriate, peak memory, finite gradients and complete checkpoints. A running allocation or stale metric file is insufficient evidence of progress.

Protect live source and outputs. A Git push does not change imported code or authorize restarting another process. Development location and launch synchronization follow the active task's agreed workflow.

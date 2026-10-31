# Runbook

Current package-backed RL workflow: [training guide](../TRAINING.md). The [Miles experiment](../MILES_BASELINE.md) is a retired historical integration.

## Before launch

| Check | Required evidence |
| --- | --- |
| Source | Recorded commit, implementation hashes, clean launch tree where required |
| Runtime | Container identity and resolved package versions |
| Data | Complete manifest, tokenizer identity, approved split and cursor |
| Model | Named recipe, exact parent checkpoint, trainable set |
| Compute | Live allocation identity, mounts, GPU ownership and memory budget |
| Numerics | Relevant load/forward/backward/replay and distributed checks |
| Output | Fresh attempt namespace; no conflicting writer |

Exclusive execution requires an explicitly available GPU allocation. Shared-GPU execution requires a declared sharing policy and measured memory admission.

## Inspect before executing

```bash
archlab resolve recipes/rl/limite_code.yaml --output results/code-pilot
archlab qualify recipes/rl/limite_code.yaml --output results/code-pilot
```

Select the contract for the intended model and supply its documented environment bindings. `resolve` writes the composed upstream configuration; `qualify` checks admission. `train` repeats the required admission checks before launching, so a normal launch can use `resolve` followed directly by `train`.

Historical AutoModel, custom math-RL and campaign commands in dated reports belong to their recorded source snapshots. They are not current launch instructions; see the [retirement boundary](Code-Map.md#retirement-boundary).

## Resume or restart

| Operation | State handling |
| --- | --- |
| Resume same contract | Restore model, optimizer, scheduler, RNG and data cursor |
| New scientific contract | Record the changed axes and initialization explicitly |
| Graceful stop | Use the trainer's documented stop protocol; verify a complete checkpoint |
| Operational retry | New attempt identity, same scientific run identity |

A source update does not change modules already imported by a live process. A running allocation is not proof of a healthy trainer.

## During a run

Watch fresh progress, finite losses/gradients, memory peaks, data continuity and complete checkpoint markers. Compare throughput using the declared measurement window; report shared-resource contention.

**Next:** [Data and Provenance](Data-and-Provenance.md) · [Tracking and Storage](Tracking-and-Storage.md)

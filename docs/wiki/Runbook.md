# Runbook

Current DeepSeek RL launch and qualification entry: [Miles baseline](../MILES_BASELINE.md).

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
PYTHONPATH=src python -m archlab.cli verify
PYTHONPATH=src python -m archlab.cli render \
  --config recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml \
  --path data_root=/path/to/prepared/data
```

Rendering prints a command. It does not launch training.

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

**Next:** [[Data and Provenance|Data-and-Provenance]] · [[Tracking and Storage|Tracking-and-Storage]]

# Repository cleanup — 2026-09-22

**Type:** dated refactor record.

## Changes

| Area | Change | Preserved |
| --- | --- | --- |
| JSON publication | Shared `atomic_write_json`; explicit caller policies | Byte format, nonfinite handling, directory policy and locks |
| Imports | Remove unused Qwen re-exports; use `archlab.fineweb` directly | Existing function/data-order semantics |
| PLE checkpoints | Move runtime-aware subclasses to `archlab.megatron.ple_checkpoint` | Method ASTs, keys, offsets, replica IDs and metadata |
| Architecture boundary | Reject direct and method-local Megatron imports | Model mechanisms stay independent |

CPU numerical checks preserve PLE initialization/RNG, names, values, optimizer tags, forward output and gradients.

## Validation

| Suite | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| Artifacts and serialization compatibility | 15 | 0 | 0 |
| MLflow synchronization | 14 | 0 | 0 |
| MLflow checkpoint tracking | 5 | 0 | 0 |
| RL MLflow tracking | 9 | 0 | 0 |
| Checkpoint retention | 9 | 0 | 0 |
| DLC launch configuration | 15 | 0 | 0 |
| Source layout | 7 | 0 | 0 |
| New PLE boundary tests | 1 | 0 | 19 |
| Existing Flash-Next full architecture | 16 | 0 | 1 |
| Existing Flash-Next full trainer | 43 | 0 | 3 |
| Existing width-320 tests | 5 | 0 | 3 |
| **Total** | **139** | **0** | **26** |

Native/distributed PLE checkpoint parity remained unverified. The unavailable checks were 20 native PLE cases, four other Megatron cases, one FLA CUDA case and one torchrun model case.

Runtime: Python 3.12.3 and container PyTorch 2.12.0a0, CUDA build 13.2. No runtime packages or running trainers were changed by the cleanup.

Local evidence: `results/repo-cleanup-20260922/SUMMARY.json` and `tests.xml`.

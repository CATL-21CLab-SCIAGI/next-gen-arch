# Scoped repository cleanup — 2026-09-22

Changes were limited to the requested helpers, imports, PLE checkpoint boundary
and focused tests. Existing training processes, frozen snapshots, checkpoint
payloads, monitor configuration, git staging and container dependencies were not
changed. Concurrent RL evaluation-metric edits in `rl_mlflow.py` were preserved.

## Serialization and imports

`artifacts.atomic_write_json` now accepts `sort_keys`, defaulting to **True** as
before. MLflow synchronization, checkpoint tracking and RL tracking use the shared
implementation with `sort_keys=False`, `allow_nan=False`, `create_parents=False`.
Retention and the matched-checkpoint watcher use `sort_keys=False`, `allow_nan=True`,
`create_parents=False`. Existing `atomic_json` names remain thin policy bindings;
dependent modules import the implementation from `archlab.artifacts` directly.
All existing `fcntl.flock` calls were retained.

Tests compare serialized bytes, including nested insertion order, Unicode escapes,
trailing LF, nonfinite-value policies, missing-parent failures and preservation of
an old file after failed serialization. Existing atomic replacement/concurrency
and distinct canonical-hash tests also pass. No canonical JSON hashing policy
was unified, and the capability evaluator's historical-snapshot hash helper was
not changed.

Unused Qwen re-exports were removed from `architectures/__init__.py`. The two
FineWeb shell scripts now import `inspect_fineweb_dataset` from `archlab.fineweb`;
the former dataloader import was a re-export of that same function. Frozen
ordering, tokenizer/runtime extraction and ClimbMix work remain deferred.

## PLE checkpoint integration

`archlab.megatron.ple_checkpoint` provides checkpoint-aware PLE subclasses. The
architecture PLE class has an overridable embedding type; it still owns table
creation, routing, lookup and arithmetic. The Megatron model and probe import the
checkpoint-aware classes. The architecture import guard now rejects both direct
`megatron` imports and `archlab.megatron` dependencies, including method-local ones.

Both moved `sharded_state_dict` method bodies are AST-identical to their previous
implementations. Local/global keys, flattened partition offsets, prepended axes,
replica IDs and metadata delegation are retained in the source. A CPU numerical
test confirms identical initialization/RNG state, tensor names and values,
optimizer tags, forward output, input gradients and parameter gradients against
the architecture class.

**Native and distributed checkpoint parity remains unverified in this session.**
The native descriptor/recursion tests were collected but skipped because Megatron
Core is not importable in the available local container runtime. No mock checkpoint
implementation was substituted and no other task's GPUs were used.

## Validation

Executed locally on Python **3.12.3**, container PyTorch
**2.12.0a0+0291f960b6.nv26.04.48445190**, CUDA build **13.2**. CUDA was disabled for
the CPU tests and was not usable in this sandbox. Existing read-only pytest
**9.1.1** and MLflow packages were appended to the import path; the container's
Torch and NumPy retained precedence. Nothing was installed or patched.

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

Unavailable checks: **20 native PLE checkpoint cases**, **4 other Megatron cases**,
**1 frozen FLA CUDA case**, and **1 torchrun native distributed-model case**.
The skips and two runtime warnings are recorded in the JUnit report. These CPU
and structural results do not establish multi-rank checkpoint save/load parity.

Both modified shell scripts pass `bash -n`; whitespace checks pass. The native
checkpoint tests should be run in the frozen Megatron/CUDA runtime before using
this new PLE boundary for a distributed checkpoint campaign.

Evidence: [summary](../results/repo-cleanup-20260922/SUMMARY.json) and
[JUnit results](../results/repo-cleanup-20260922/tests.xml).

# Training infrastructure boundaries

The training entries orchestrate execution. Model construction, indexed-data
validation, loss callback ABIs and checkpoint staging are independently reusable.
No runtime framework is vendored, upgraded or patched by this refactor.

| Responsibility | Module |
|---|---|
| File SHA-256 and atomic JSON publication | `archlab.artifacts` |
| Existing JSON-content identity formats | Their original owners, including `archlab.provenance` |
| Exact DATA_READY membership checks | `archlab.megatron.indexed_data` |
| Raw-int32 ordering, cursor/repeat logic, DP prefix partitioning | `archlab.megatron.token_batches` |
| Megatron API-version dispatch and iteration/rank access | `archlab.megatron.lifecycle` |
| Legacy averaged loss vs native summed/token-count ABI | `archlab.megatron.losses` |
| Flash-Next native model/optimizer tagging | `archlab.megatron.qwen38_flash_next_model` |
| Flash-Next argument translation | `archlab.megatron.qwen38_flash_next_config` |
| Native bounded host checkpoint staging | `archlab.megatron.checkpoint_staging` |
| Flash-Next distributed qualification checks | `archlab.megatron.qwen38_flash_next_checks` |
| PIQA evaluation | `archlab.evaluation.qwen38_piqa` |
| Early curve reporting | `archlab.reporting.qwen38_early_curves` |

`invoke_pretrain(..., forward_step=...)` requires the caller's callback. A
trainer cannot accidentally bind another trainer's forward implementation.
Loss ABIs are deliberately distinct: merging a legacy pre-averaged callback
with native token-count scaling would change MoE/MTP gradient normalization.

The two raw-int32 iterators now share one implementation. The legacy
`DPRankTokenBatches` name is retained, as is `_source_index`, which strided
simplicial pilots override. Membership validation consistently rejects invalid
JSON objects, empty/duplicate parts, and unexpected or missing artifacts.

## Explicit launch recipes

Launch defaults live in `recipes/launches/`. Machine paths remain environment
overrides; recipe files contain no machine-specific paths. For example:

```bash
export NGA_LAUNCH_RECIPE="$NGA_REPO_ROOT/recipes/launches/qwen38_flash_next_w320_e32.yaml"
bash "$NGA_REPO_ROOT/scripts/run_qwen38_flash_next_full_dlc.sh"
```

This assumes the existing DLC topology, source-commit, data, tokenizer and output
environment variables have been set. It does not allocate or restart nodes.
`python -m archlab.megatron.launch_config --recipe PATH` renders resolved values
without launching anything. Numeric environment overrides remain supported;
model selection must agree with the named recipe. Renaming an output directory
cannot change model scale, preflight length or probe-save behavior.

The named dense-quarter and Flash-Next-full entrypoints have fixed default
recipes. The historical compatibility forwarder requires an explicit recipe
and only translates/validates the output location. It no longer chooses variants
from labels. Old resident-controller allowlists are not changed in memory:
use the dedicated launcher on the existing nodes when the old controller cannot
accept `NGA_LAUNCH_RECIPE`. Historical runs remain reproducible at their pinned
commits; do not silently replay an old name-based request under new source.

`scripts/lib/dlc_runtime.sh` owns shared immutable-source checks, frozen-container
environment setup and torchrun construction. It never installs packages. Recipe
bindings are allowlisted data, not evaluated shell code. Training contracts
record recipe identity and hashes of newly extracted implementation modules.

## Frozen reference compatibility

Shared campaign definitions, model factory, FineWeb binary reading, dataset path
resolution and environment topology now live in `archlab.campaigns`,
`archlab.model_factory`, `archlab.fineweb`, `archlab.dataset_paths` and
`archlab.distributed`. Speedrun retains compatibility imports/wrappers and binds
its original dtype, attention operations and logging explicitly. Megatron uses
neutral construction with that same explicitly supplied reference runtime.

The ClimbMix best-fit packing algorithm and tokenizer implementation remain in
the frozen backend. Moving them is not needed for this cleanup; their replay and
byte-accounting semantics need a separate extraction boundary. We do not hide
those remaining dependencies behind a supposedly neutral forwarding module.

Atomic publication preserves each caller's sorted, indented JSON and trailing
newline. Canonical JSON hashes are **not** unified: escaped-ASCII identities in
the dense trainer/PIQA and UTF-8 identities in provenance remain distinct.
Source-file hashes and evaluator paths naturally change when code moves; old
cached evaluations are not relabelled as results of the new implementation.
Checkpoint keys, architecture geometry, optimizer rules and frozen data ordering
are not intentionally changed. Private trainer aliases remain temporarily for
old analysis imports; project consumers use the owning modules directly.

## DeepSeek launch gate

The separate PyTorch-first V4.1 implementation is still work in progress. The
1B-supervised-token data selection is preserved, and no production finetune was
launched during this refactor. Resume full-pretrained forward/quantization-drift,
adapter-gradient, memory and distributed checkpoint qualification before launch.
This structural refactor does not itself establish V4.1 training correctness.

## Refactor validation (2026-09-11)

The exact staged source tree `f32ca30a2e2db450461a6b62fe8c8ae06ded464a`
was exported into an isolated directory on the existing DLC worker-2, excluding
unfinished DeepSeek code and untracked experiment artifacts. No packages were
installed or nodes restarted. Only this validation note was added afterward.

- Repository-wide Ruff and Python compilation passed.
- Frozen manifest verification passed: 144 runs; 60 metric rows in four campaigns.
- Both CI portable launch examples rendered successfully; training, sampling,
  PIQA and curve-report CLI help worked without launching a job.
- `pytest -m "not slow" -q`: **484 passed, 50 skipped, 1 failed**, plus
  11 passing subtests. The failure is the unchanged benchmark-grading test:
  the installed `lm_eval` lacks `tasks/aime/utils.py`. The identical failure was
  reproduced from the original `f822c4b` source in a separate clean snapshot.
- New regression coverage includes token wrap/resume/repeat/strided ordering,
  callback dispatch for both Megatron APIs, JSON serialization and atomic-write
  failures/concurrency, launch-name independence, and shell argument validation.
- Four structural fingerprints match the original model construction, optimizer
  tagging and checkpoint-staging function ASTs. These are extraction checks,
  not distributed numerical qualification or a new GPU throughput measurement.

Validation used `/opt/venv/bin/python` in the existing image reference
`sci-agi-zhongwei-registry-vpc.cn-zhongwei.cr.aliyuncs.com/dev/nemo:26.06`
(an image tag, not a newly verified immutable image digest): Python 3.12.3,
PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, NCCL 2.29.7;
package metadata reports Megatron Core 0.18.2, Transformer Engine
`2.16.0+b9d690e0`, and Triton `3.6.0+git5d72932fc5.nv26.4`.

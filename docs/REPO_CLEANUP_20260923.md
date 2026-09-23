# Repository cleanup — 2026-09-23

## Scope and ownership

This cleanup traces Python imports, module CLI commands, recipes, tests, checkpoint source inventories, and recorded deployment launchers. Import counts alone are not a deletion rule. Architecture operators and numerical oracles are retained.

| Previous owner | Current owner |
| --- | --- |
| `benchmarks/capability.py` | `evaluation/capability.py` |
| `benchmarks/learning_curves.py` | `reporting/learning_curves.py` |
| `benchmarks/simplicial_attention.py` | `qualification/simplicial_attention.py` |
| Checkpoint retention and checkpoint verification reports | `storage/` |
| Health, resume, checkpoint, and evaluation watchers | `tracking/` |
| Qwen AutoModel integration | `automodel/qwen/` |
| 16 independent DeepSeek probes | `automodel/deepseek_v41/qualification/` |
| SGLang external model registration | `serving/sglang/` |

## Reporting consolidation

Four experiment-version-specific curve/report commands (`v41_full_curves`, `v41_full_matched_curves`, `v41_matched_learning_curves`, and `v41_scratch_report`) are retired. Their historical generated evidence remains available; Git history preserves their exact implementation. New comparisons use explicit run directories:

```bash
PYTHONPATH=src python -m archlab.reporting.paired_curves \
  --run normal /path/to/normal-run \
  --run simplicial /path/to/simplicial-run \
  --output /path/to/new-comparison
```

`reporting/runs.py` owns LF-delimited ledger reading, partial-write handling, source snapshots, contiguous token-cursor checks, matching data/schedules, and token-weighted smoothing. The shared reader preserves Unicode separators inside JSON strings. Curves use only matched updates; they do not infer held-out evaluation results or current service status. Operational health checks retain their experiment-specific numerical invariants under `tracking/`.

## AutoModel lineage review

| Lineage | Disposition |
| --- | --- |
| Qwen frozen-backbone adaptation | Maintained, grouped under `automodel/qwen/`; new entry points use that prefix |
| DeepSeek official/native construction, full checkpoints and RL | Maintained at checkpoint-recorded paths |
| DeepSeek scratch training and its sparse oracle | Retained: distinct model geometry and distributed qualification |
| Early DeepSeek PyTorch/native execution | Retained: still imported by current data, numerical, and checkpoint paths |
| Independent DeepSeek diagnostics | Moved; numerical behavior is not replaced by import-count heuristics |
| Full-model and RL qualification probes named in source receipts | Retained at their recorded paths |
| Resident RL cache | Experimental and disabled; the numerical gate has not passed |

The matched SFT parents name 39 and 45 project source files. Nineteen needed style cleanup; each migration records old/new byte SHA-256 values and an identical Python AST (including import order and string values). Checkpoint admission verifies this explicit record before applying the original source-change rules. Unlisted behavioral changes remain rejected. Existing numerical thresholds and parameter ownership are unchanged.

## Serving dependencies retained

The recorded MLflow deployment invokes `serving/mlflow_playground_overlay.py` from this checkout, and its hook serves the adjacent JavaScript. Removing these files would break that deployment on restart. They remain until its launcher and UI hook are migrated. The authenticated pair proxy is also retained with its existing tests; no service or gateway is removed by repository cleanup.

Checkpoint conversion, direct loading, numerical qualification, and the OpenAI-compatible inference interface remain. Engine registration now uses `SGLANG_EXTERNAL_MODEL_PACKAGE=archlab.serving.sglang`. The plugin is project integration code; no SGLang runtime is vendored.

## CLI path migration

Use the new module names for future launches. Historical immutable source snapshots and their recorded commands remain historical evidence.

| Old module | New module |
| --- | --- |
| `archlab.automodel.checkpointing` | `archlab.automodel.qwen.checkpointing` |
| `archlab.automodel.data` | `archlab.automodel.qwen.data` |
| `archlab.automodel.deepseek_v41_attention_probe` | `archlab.automodel.deepseek_v41.qualification.attention_probe` |
| `archlab.automodel.deepseek_v41_ep_probe` | `archlab.automodel.deepseek_v41.qualification.ep_probe` |
| `archlab.automodel.deepseek_v41_fp4_probe` | `archlab.automodel.deepseek_v41.qualification.fp4_probe` |
| `archlab.automodel.deepseek_v41_kernel_probe` | `archlab.automodel.deepseek_v41.qualification.kernel_probe` |
| `archlab.automodel.deepseek_v41_local_leaf_probe` | `archlab.automodel.deepseek_v41.qualification.local_leaf_probe` |
| `archlab.automodel.deepseek_v41_local_probe` | `archlab.automodel.deepseek_v41.qualification.local_probe` |
| `archlab.automodel.deepseek_v41_moe_precision_probe` | `archlab.automodel.deepseek_v41.qualification.moe_precision_probe` |
| `archlab.automodel.deepseek_v41_moe_rounding_probe` | `archlab.automodel.deepseek_v41.qualification.moe_rounding_probe` |
| `archlab.automodel.deepseek_v41_normal_probe` | `archlab.automodel.deepseek_v41.qualification.normal_probe` |
| `archlab.automodel.deepseek_v41_official_prefix_probe` | `archlab.automodel.deepseek_v41.qualification.official_prefix_probe` |
| `archlab.automodel.deepseek_v41_probe` | `archlab.automodel.deepseek_v41.qualification.probe` |
| `archlab.automodel.deepseek_v41_replay_trace_probe` | `archlab.automodel.deepseek_v41.qualification.replay_trace_probe` |
| `archlab.automodel.deepseek_v41_scratch_sparse_probe` | `archlab.automodel.deepseek_v41.qualification.scratch_sparse_probe` |
| `archlab.automodel.deepseek_v41_simplicial_repeat_probe` | `archlab.automodel.deepseek_v41.qualification.simplicial_repeat_probe` |
| `archlab.automodel.deepseek_v41_sparse_probe` | `archlab.automodel.deepseek_v41.qualification.sparse_probe` |
| `archlab.automodel.deepseek_v41_sparse_repeat_probe` | `archlab.automodel.deepseek_v41.qualification.sparse_repeat_probe` |
| `archlab.automodel.evaluate` | `archlab.automodel.qwen.evaluate` |
| `archlab.automodel.evaluate_distributed` | `archlab.automodel.qwen.evaluate_distributed` |
| `archlab.automodel.execution` | `archlab.automodel.qwen.execution` |
| `archlab.automodel.gdn_probe` | `archlab.automodel.qwen.gdn_probe` |
| `archlab.automodel.loading` | `archlab.automodel.qwen.loading` |
| `archlab.automodel.numerics` | `archlab.automodel.qwen.numerics` |
| `archlab.automodel.probe` | `archlab.automodel.qwen.probe` |
| `archlab.automodel.runtime` | `archlab.automodel.qwen.runtime` |
| `archlab.automodel.sample` | `archlab.automodel.qwen.sample` |
| `archlab.automodel.simplicial` | `archlab.automodel.qwen.simplicial` |
| `archlab.automodel.stage_checkpoint` | `archlab.automodel.qwen.stage_checkpoint` |
| `archlab.automodel.train` | `archlab.automodel.qwen.train` |
| `archlab.automodel.training_config` | `archlab.automodel.qwen.training_config` |
| `archlab.benchmarks.capability` | `archlab.evaluation.capability` |
| `archlab.benchmarks.learning_curves` | `archlab.reporting.learning_curves` |
| `archlab.benchmarks.simplicial_attention` | `archlab.qualification.simplicial_attention` |
| `archlab.reporting.checkpoint_retention` | `archlab.storage.checkpoint_retention` |
| `archlab.reporting.v41_eval_progress` | `archlab.tracking.v41_eval_progress` |
| `archlab.reporting.v41_full_checkpoint_receipt` | `archlab.storage.v41_checkpoint_receipt` |
| `archlab.reporting.v41_full_health` | `archlab.tracking.v41_full_health` |
| `archlab.reporting.v41_matched_checkpoint_watch` | `archlab.tracking.v41_matched_checkpoint_watch` |
| `archlab.reporting.v41_resume_health` | `archlab.tracking.v41_resume_health` |
| `archlab.reporting.v41_scratch_monitor` | `archlab.tracking.v41_scratch_monitor` |
| `archlab.sglang_models` | `archlab.serving.sglang` |
| `archlab.reporting.v41_training_audit` | `archlab.tracking.v41_training_audit` |

## CI and dependency review

- Keep GPU frameworks owned by the validated container; the CPU extra is an explicit test dependency.
- Include the upstream pinned `tracking` extra in CPU CI so tracking tests have their declared dependency.
- Preserve the pinned optional evaluation harness; an arbitrary installed `lm_eval` module is not its numerical oracle.
- Run the existing lint, source compilation, frozen-artifact, launch-plan, CPU-test and distribution-build gates.
- Require the same CI workflow to succeed before a tag can publish release artifacts.
- Build and inspect distributions for relocated modules, prompts, source-compatibility records, and absence of vendored runtime frameworks.

Production RL is not admitted by repository cleanup or CPU tests. It still requires current-source distributed qualification, actual-model replay checks, and the measured shared-GPU memory reserve. General evaluation remains deferred until both actors have verified real updates.

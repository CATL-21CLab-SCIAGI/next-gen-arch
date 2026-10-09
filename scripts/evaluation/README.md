# Evaluation integrations

`eval_pipeline/` contains the pinned patch and integration notes for the external
runner. The upstream repository remains external; do not vendor it here.

`requirements-lm-eval.txt` pins the hash-verified, dependency-free lm-eval overlay.
Install it with `scripts/prepare_lm_eval_runtime.sh TARGET_DIR` from the repository
root. Runtime dependencies remain owned by the validated GPU container.

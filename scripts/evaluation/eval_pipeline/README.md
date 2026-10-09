# eval_pipeline integration

`eval_pipeline_aime26.patch` applies to the sibling evaluation repository at
`f3c8822a29047bbff7ea4e4c758856089a336a04`. It adds optional runner/dataset
injection to its AIME26 evaluator and forwards sampling parameters to injected
HF/custom runners. Existing default entry points remain unchanged.

The run launcher snapshots the upstream Python files, applies this patch, and
pins every resulting file digest in the experiment plan. The sibling checkout
is unchanged. Project-owned inference and validation live in
`src/archlab/automodel/limite_pipeline_benchmark.py` and
`src/archlab/evaluation/limite_pipeline.py`.

Production always injects the sealed real 30-question exam, bypassing upstream's
toy-data fallback. The original evaluator's answer extraction and aggregation
are retained and reported alongside strict final-answer accuracy. All raw
responses, token IDs, EOS/length reasons, prompt lengths, effective budgets, and
seeds are retained separately from upstream's shortened result previews.

# eval_pipeline integration

`aime26.patch` applies to the sibling evaluation repository at
`f3c8822a29047bbff7ea4e4c758856089a336a04`. It adds optional runner/dataset
injection to its AIME26 evaluator and forwards sampling parameters to injected
HF/custom runners. Existing default entry points remain unchanged.

The historical launcher snapshotted the upstream Python files, applied this
patch, and pinned resulting file digests in the experiment plan. That launcher
is retired. `src/archlab/evaluation/limite_pipeline.py` retains plan validation
and scoring contracts; use the recorded immutable source to reproduce old runs.

Production always injects the sealed real 30-question exam, bypassing upstream's
toy-data fallback. The original evaluator's answer extraction and aggregation
are retained and reported alongside strict final-answer accuracy. All raw
responses, token IDs, EOS/length reasons, prompt lengths, effective budgets, and
seeds are retained separately from upstream's shortened result previews.

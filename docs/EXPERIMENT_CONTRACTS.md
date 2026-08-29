# Experiment contracts

This document defines which conclusions a run may support. It supersedes informal use of model-size labels as experimental contracts.

## Comparison regimes

### Controlled component comparison

Use `--comparison-regime controlled --target-train-tokens N`.

Baseline and variant must match on:

- seed and shared-parameter initialization;
- dataset manifest, tokenizer vocabulary, and logical data order;
- sequence length, global batch tokens, optimizer, schedule, precision, and evaluation;
- effective training tokens after discrete-step rounding.

Parameter and FLOP overhead are disclosed but need not be removed. If parameter matching is required, it is a separate experiment and must state which capacity was removed.

### Fixed-compute comparison

Use `--comparison-regime fixed_compute --target-model-flops F`.

The trainer resolves the largest whole number of optimizer steps that does not exceed the requested algorithmic model FLOPs. Executed fallback FLOPs and hardware throughput remain separate metrics. A dense fallback for a sparse algorithm cannot claim a realized systems speedup.

### Scaling comparison

Use `--comparison-regime scaling --tokens-per-parameter R`.

Model parameters and training tokens may both change. These runs support scaling and total-compute Pareto analysis, not a fixed-token causal claim.

## Pairing and statistics

- Use the same seeds for baseline and variant. The default confirmation set is 42–44.
- Report paired per-seed deltas, mean, and sample standard deviation.
- Label a ±1 SD band as variation across seeds, not a confidence interval.
- Do not replace a failed seed with an earlier favorable checkpoint.
- Retry an identical run only after an operational failure. Numerical, capacity, and contract failures require diagnosis or a newly named contract.

`next-gen-arch pair-check` rejects drift in the primary frozen axes:

```bash
next-gen-arch pair-check \
  --baseline /runs/baseline/result.json \
  --variant /runs/variant/result.json
```

## Promotion ladder

1. Run a short construction/forward/backward/checkpoint-resume smoke test.
2. Run three paired seeds at two small scales.
3. Promote only mechanisms that improve the predeclared metric at both scales without unacceptable stability or efficiency regressions.
4. Test an intermediate scale.
5. Keep the largest scale out of scaling-law fitting and use it as a prediction check.
6. Form combinations only from components that passed isolated controls; retain every constituent control in the report.

For expensive mechanisms, an optional isoflop sweep should establish the compute-optimal token/model allocation instead of assuming the baseline allocation transfers.

## Evaluation

Validation BPB is the primary language-model metric. New campaigns should additionally pin:

- a fixed validation content manifest and exact token window;
- multi-domain language-model evaluation;
- at least one decontaminated or hard-to-game sanity suite;
- task-specific evaluations needed by the claimed use case.

FineWeb training in this repository resets the validation loader after each exact evaluation window. Absolute results from ClimbMix, FineWeb, and indexed FineWeb-Edu are not directly comparable.

## Historical evidence

The published 100M–1B manifest reproduces the original tokens-per-parameter campaign, including its `--no-save-final-checkpoint` flag. It is intentionally frozen. Reconstructing that command is historical reproduction, not the recommended artifact policy for new work.

The earlier fixed-token d14/d16/d18 controls and the later parameter-scaling campaign answer different questions. Their absolute BPB values must remain in separate tables.

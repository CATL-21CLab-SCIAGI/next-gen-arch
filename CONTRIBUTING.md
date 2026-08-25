# Contributing

Contributions are welcome when they keep the experiment interpretable and the training path compact.

## Development setup

```bash
uv sync --extra cpu --group dev
uv run pytest -m "not slow" -q
```

Before opening a pull request, run:

```bash
uv run ruff check src/next_gen_arch tests/test_registry.py tests/test_portable_runtime.py
uv run python -m compileall -q src/next_gen_arch
uv run next-gen-arch verify
uv run next-gen-arch doctor --backend megatron
uv run python -c "from next_gen_arch.results import verify_metrics; print(verify_metrics())"
uv build
```

## Architecture changes

A new architecture pull request should include:

- the motivating paper or primary technical source;
- a concise statement of the isolated mechanism;
- configuration routed through the shared model factory;
- a CPU forward/backward test and parameter-count check;
- confirmation that shared-backbone initialization remains aligned at the same seed;
- a proposed frozen baseline/variant contract before any large result is added.

Avoid combining several mechanisms in the first experiment. If a combination is later tested, retain the baseline and each single-component control.

## Publishing results

Do not edit an old dated result in place. Add a new campaign or dated snapshot and record:

- source commit and dirty-tree status;
- data and tokenizer fingerprints;
- complete command arguments;
- model parameter count and training-token budget;
- every planned seed, including failures;
- BPB paired against the aligned baseline;
- hardware and throughput measurement method;
- NaN, OOM, scheduler, or harness failures separately from quality results.

Large binaries, checkpoints, datasets, credentials, internal hostnames, and machine-specific paths must not be committed.

## Code style

Prefer a small readable implementation over a framework abstraction used by only one variant. Type and test the maintained manifest/result tooling. Keep `src/next_gen_arch/arch` limited to architecture definitions and place training, data, optimizer, kernel, evaluation, and runtime code in `src/next_gen_arch/training`. Never edit `third_party/Megatron-LM`; backend-specific adaptations belong in `src/next_gen_arch/backends` or a dedicated local architecture adapter.

By contributing, you agree that your contribution is licensed under the repository's MIT License.

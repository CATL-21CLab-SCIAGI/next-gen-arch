# Contributing

Contributions are welcome when they keep the experiment interpretable and the training path compact.

## Development setup

```bash
uv sync --extra cpu --group dev
uv run pytest -m "not slow" -q
```

Before opening a pull request, run:

```bash
uv run ruff check src/archlab tests/test_registry.py tests/test_portable_runtime.py
uv run python -m compileall -q src/archlab
uv run next-gen-arch verify
uv run next-gen-arch doctor --backend megatron
uv run python -c "from archlab.results import verify_metrics; print(verify_metrics())"
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

Prefer a small readable implementation over a framework abstraction used by only one variant. Keep model math in `src/archlab/architectures`, local optimizer extensions in `src/archlab/optimizers`, and all Megatron-specific adaptation in `src/archlab/megatron`. The frozen reference runner lives in `src/archlab/speedrun`. Consolidate shared primitives, but do not merge unrelated mechanisms merely to reduce the file count. Import concrete submodules directly rather than building re-export registries in package initializers. Megatron is container-owned: never vendor or patch it in this repository.

By contributing, you agree that your contribution is licensed under the repository's MIT License.

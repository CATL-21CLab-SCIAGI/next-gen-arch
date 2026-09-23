# Contributing

Start with the [Wiki contributor guide](docs/wiki/Contributing.md) and [code map](docs/wiki/Code-Map.md).

## Change requirements

| Change | Include |
| --- | --- |
| Architecture | Mechanism reference, geometry, initialization, parameter count and matched control |
| Backend | Construction, optimizer grouping, numerical and checkpoint-continuation tests |
| Training/RL | Explicit objective, trainable set, data/policy identity and regression coverage |
| Evaluation | Fixed cases, prompts, scorer, budgets and interpretation limits |
| Documentation | Status/date, concise facts, evidence and verified links |

Keep architecture definitions independent of trainers and optimizers. Use concrete imports and preserve the frozen reference behavior. Container-owned frameworks remain external.

## Checks

In a prepared development environment with the dependencies from `pyproject.toml`:

```bash
PYTHONPATH=src python -m archlab.cli verify
python -m ruff check src/archlab tests
python -m pytest -m "not slow" -q
python -m build
```

Run the numerical/distributed gates relevant to the change in the validated GPU runtime. CPU tests do not establish distributed correctness.

## Results and artifacts

Preserve source/data/tokenizer identities, complete checkpoint state, paired seeds, budgets, metric definitions, timing windows, and failures. Add a dated record for changed conclusions; retain superseded evidence.

Keep credentials, large payloads, private deployment settings, and mutable service state outside Git.

## Documentation

Reader guides live in `docs/wiki/`; versioned technical records remain in `docs/`. Use tables for comparisons and numbered steps for procedures. See [publishing instructions](docs/README.md#publish-the-wiki).

Contributions are licensed under the repository's MIT license.

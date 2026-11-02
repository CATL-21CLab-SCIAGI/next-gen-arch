# Contributing

## Choose a change

| Change | Include |
| --- | --- |
| Architecture | Mechanism reference, equation, geometry, initialization and ordinary control |
| Backend | Supported model/topology, numerical oracle, checkpoint continuation |
| Data | Rendering/split policy, content identities, coverage and exclusions |
| RL | Reward, advantage/loss rules, policy provenance and replay tests |
| Evaluation | Fixed cases, prompts, scorer, budgets and interpretation limits |
| Documentation | Status, scope, sources, decisions and working navigation |

## Review checklist

1. Keep ownership boundaries clear; import concrete modules.
2. Preserve historical contracts and frozen reference behavior.
3. Test the behavior that changed.
4. Report numerical failures, resource limits and incomplete budgets.
5. Distinguish proposed, implemented, qualified and measured results.
6. Link the implementation and evidence needed for review.

Do not include credentials, checkpoint payloads, private deployment settings or mutable “latest” pointers in public documentation.

## Documentation style

Use a short purpose statement, tables for settings/comparisons, and ordered steps for procedures. Give each fact one home. Keep chronological debugging detail in dated records or run artifacts.

A useful experiment page answers: **What changed? Against what control? What was measured? What follows from it?**

## Development checks

Use an isolated Python 3.12 environment with PyTorch supplied by the chosen
CPU/GPU container. Install project development tools with standard pip:

```bash
python -m pip install -e '.[dev,tracking]'
python -m ruff check src/archlab tests
NANOCHAT_ATTENTION_BACKEND=sdpa python -m pytest -m "not slow" -q \
  --basetemp=results/local-checks/pytest
```

Automatic CI and project uv settings are removed. Pages deployment is manual.
Verification is local and scoped to the change. Record the actual package/container versions with results.
`CITATION.cff` remains the citation metadata; Python support is declared in
`pyproject.toml` and the runtime container.

For a manual release, build into a fresh directory under `results/`, run
`scripts/check_distribution.py` on that directory, install its wheel in an
isolated environment, and check imports/CLI help from outside the checkout.
Qualification and training additionally require the full immutable source
snapshot, its tests, pinned upstream checkout and deployment assets.

CPU checks do not qualify GPU or distributed behavior. Record the container,
resolved packages, model/data identities, topology and numerical comparisons
for backend changes. Report failed or incomplete qualification explicitly.

Contributions use Apache 2.0; retain applicable upstream notices in
[Attribution](../NOTICE.md). Report vulnerabilities privately to maintainers,
without publishing credentials or exploit details in public issues.

See [Documentation Maintenance](Documentation-Maintenance.md) and
[Publication policy](../PUBLICATION.md).

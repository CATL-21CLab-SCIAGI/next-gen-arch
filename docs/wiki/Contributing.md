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

```bash
uv sync --locked --extra cpu --extra tracking --group dev
uv run --no-sync ruff check src/archlab tests
NANOCHAT_ATTENTION_BACKEND=sdpa uv run --no-sync pytest -m "not slow" -q
```

CPU checks do not qualify GPU or distributed behavior. Record the container,
resolved packages, model/data identities, topology and numerical comparisons
for backend changes. Report failed or incomplete qualification explicitly.

Contributions use Apache 2.0; retain applicable upstream notices in
[Attribution](../NOTICE.md). Report vulnerabilities privately to maintainers,
without publishing credentials or exploit details in public issues.

See [Documentation Maintenance](Documentation-Maintenance.md) and
[Publication policy](../PUBLICATION.md).

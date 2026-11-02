# Verification

Tests follow the maintained code domains: `architectures`, `automodel`, `training`,
`megatron`, `evaluation`, `data`, `storage`, `tracking`, `reporting`, and shared
`contracts`. `reference` preserves speedrun behavior. `integration` contains the
actual GPU/deployment admission probes; `site` covers the research site.

Run focused tests for a change, then its relevant regression suite. The full CPU
suite is `pytest -m 'not slow'`. Some tests open loopback sockets or use two CPU
processes; use an environment that permits both. Keep test outputs under ignored
`results/`. Tests import shared fixtures by their full `tests.<domain>` path.
Live sandbox tests unpack task images and run namespace isolation checks; select
them explicitly with `pytest tests/training/test_dlc_sandbox.py -m slow`.

Keep numerical, gradient, packing, truncation and checkpoint-resume evidence.
Retire tests only with the obsolete behavior they exclusively cover. Fewer files
is not a correctness metric. Tests stay separate from scientific recipe inputs.

CPU passes and skipped GPU tests never qualify production. Admission rejects
missing, skipped or failed deployment probes and binds their reports to the
exact source, configuration, upstream runtime, model and data identities.

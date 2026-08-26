# Repository working agreement

- Keep project-owned Python under `src/archlab`. `architectures` defines models,
  `optimizers` owns local optimizer extensions, `megatron` is the only Megatron
  integration boundary, and `speedrun` is the frozen small-scale reference backend.
- Treat `third_party/Megatron-LM` as a read-only submodule. Pin upgrades explicitly and
  record the commit in tests and runtime documentation; never patch upstream in place.
- Put portable experiment contracts under `recipes`. Inject machine paths through
  `env:NAME`, `package:relative/path`, or launch-time overrides.
- Preserve frozen speedrun arguments and data-order semantics. Intentional behavior
  changes require an explicit experiment contract and regression coverage.
- Do not claim Megatron support for a speedrun mechanism until construction, optimizer
  grouping, checkpointing, and distributed numerical behavior have dedicated tests.
- Put reusable qualitative prompts in versioned YAML under `src/archlab/prompts`; do
  not duplicate prompt text in train or evaluation code.

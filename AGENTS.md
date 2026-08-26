# Repository working agreement

- Keep project-owned Python under `src/archlab`. `architectures` defines models,
  `optimizers` owns local optimizer extensions, `megatron` is the only Megatron
  integration boundary, and `speedrun` is the frozen small-scale reference backend.
- Keep dependencies pointed from execution adapters toward architecture definitions,
  never from `architectures` back into a trainer or optimizer. Import concrete modules
  directly; package `__init__.py` files should not become public registries.
- Consolidate genuinely shared model primitives, but do not optimize for a fixed file
  count. A leaf mechanism deserves its own module when it has an independent interface,
  numerical oracle, or distributed implementation.
- Treat Megatron Core, PyTorch, Transformer Engine, CUDA, and NCCL as a container-owned
  runtime contract. Never vendor or patch them in this repository; record the container
  identity and resolved package versions with every run.
- Put portable experiment contracts under `recipes`. Inject machine paths through
  `env:NAME`, `package:relative/path`, or launch-time overrides.
- Preserve frozen speedrun arguments and data-order semantics. Intentional behavior
  changes require an explicit experiment contract and regression coverage.
- Do not claim Megatron support for a speedrun mechanism until construction, optimizer
  grouping, checkpointing, and distributed numerical behavior have dedicated tests.
- Put reusable qualitative prompts in versioned YAML under `src/archlab/prompts`; do
  not duplicate prompt text in train or evaluation code.

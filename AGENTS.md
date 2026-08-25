# Repository working agreement

- Keep project-owned Python code under `src/next_gen_arch`; `arch` contains architecture definitions only, `training` owns all execution infrastructure, and `backends` contains integration boundaries.
- Treat `third_party/Megatron-LM` as read-only upstream source. Pin upgrades through the Git submodule and record the new commit in tests and documentation.
- Keep committed experiment paths portable. Use `env:NAME`, `package:relative/path`, or configuration-relative paths instead of machine-specific absolute paths.
- Preserve frozen speedrun arguments and data-order semantics. Any intentional behavior change requires an explicit experiment contract and regression coverage.
- Do not claim that a speedrun architecture is supported by Megatron until construction, optimizer grouping, checkpointing, and numerical behavior have dedicated tests.
- Put reusable qualitative prompts in versioned YAML under `src/next_gen_arch/prompts`; do not duplicate prompt text in training or evaluation modules.

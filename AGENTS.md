# Working agreement

- Hardware is NVIDIA B300, even when NVIDIA-SMI/NVML labels it L20D.
- Keep project Python in `src/archlab`: `architectures` owns models,
  `optimizers` local optimizer extensions, `megatron` the sole Megatron boundary,
  and `speedrun` the frozen reference backend.
- Execution adapters depend on architecture definitions, never the reverse.
  Import concrete modules; keep `__init__.py` free of public registries.
- Share genuine primitives. Give mechanisms separate modules when they have an
  independent interface, numerical oracle or distributed implementation.
- Treat Megatron Core, PyTorch, Transformer Engine, CUDA and NCCL as container
  runtime dependencies. Never vendor or patch them here; record container identity
  and resolved package versions with each run.
- Put portable contracts in `recipes`; use `env:NAME`, `package:relative/path`
  or launch overrides for machine paths. Keep staging, logs and weights ignored.
- Preserve frozen speedrun arguments and data order. Intentional changes need
  an explicit experiment contract and regression coverage.
- Claim Megatron support only after dedicated construction, optimizer grouping,
  checkpoint and distributed numerical tests pass.
- Store reusable qualitative prompts as YAML in `src/archlab/prompts`;
  do not duplicate them in training or evaluation code.

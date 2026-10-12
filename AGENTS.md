# Development agreement

- Optimize for a small, maintainable research codebase. Deliver small complete
  changes: state the contract, implement, verify, retire the superseded path,
  and update its documentation. Avoid speculative abstractions and permanent
  compatibility layers.
- Prefer pinned upstream packages over local frameworks. Let verl own RL
  execution, rollout scheduling, checkpointing and monitoring. Extend packages
  only where the experiment requires it.
- Keep project Python in `src/archlab`. Architecture definitions stay independent
  of execution adapters; Megatron integration belongs at the `megatron` boundary.
  Share utilities when there are real callers; give distinct mechanisms clear
  interfaces and numerical references.
- Use `third_party/verl` as the only verl checkout. Record minimal upstream
  patches under `scripts/patches` with pinned revisions and reproducible checks.
  Treat PyTorch, Megatron Core, Transformer Engine, CUDA and NCCL as container
  dependencies; record their versions rather than vendoring or patching them.
- Keep `resolve`, `qualify` and `train` thin. Compose typed experiment contracts
  with upstream OmegaConf settings. Use explicit execution modes and upstream
  rollout contracts instead of interacting flags or another scheduler.
- Put portable experiment contracts in `recipes`, reusable prompts in
  `src/archlab/prompts`, and machine paths in environment references or launch
  overrides. Record checkpoint lineage, data/tokenizer/template versions, seeds,
  token budgets, policy versions and truncation behavior.
- Verify the changed behavior with focused tests, then the relevant regression
  suite. Diagnose failures with the smallest reproducer before repeating costly
  admission runs. GPU support requires construction, numerical, optimizer, checkpoint
  and distributed evidence for the actual model/backend combination. Skipped
  checks do not qualify a deployment; preserve failures and acceptance bounds.
- Run production from immutable qualified snapshots. Preserve active jobs and
  resume state during refactors. Changes to scientific settings or frozen
  reference recipes require an explicit experiment contract.
- Aggressively retire obsolete launchers, probes, recipes and dedicated tests
  after checking callers and replacements. Keep checksum-verified originals in
  ignored `.legacy/`; maintained code must never import that archive.
- Use ignored `results/` for all run outputs and receipts, OSS for datasets and
  model artifacts, and private runtime configuration for credentials. Avoid root
  aliases, duplicate data and tracked staging files. Name owned dataset folders
  descriptively in lowercase kebab-case; verify ownership before reorganizing
  shared storage, and update manifests and launch references together.
- Measure correctness and end-to-end throughput before claiming improvement.
  Report completed checks, observed progress and remaining limitations plainly.
  Hardware is NVIDIA B300 even when NVIDIA-SMI/NVML labels it L20D.

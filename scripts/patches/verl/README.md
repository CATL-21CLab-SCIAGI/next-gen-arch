# Local MiMo-verl patches

Apply numbered patches in order to the pinned `third_party/verl` checkout:

```sh
for patch in scripts/patches/verl/*.patch; do
  git -C third_party/verl apply "$PWD/$patch"
done
```

- `0001-reject-incomplete-swe-rewards.patch`: a software-engineering reward requires an explicit completed verifier result. Infrastructure failures cannot become zero-reward training examples.
- `0003-allow-single-harness.patch`: allow one harness profile, matching the upstream documented contract. Empty specifications still fail.
- `0004-partition-native-optimizer-dtypes.patch`: keep native FP32 and BF16 parameters in distinct stock optimizer groups and restore aliased FP32 optimizer shards without recording autograd edges.
- `0005-native-chunked-logits-processor.patch`: let an opt-in model apply the existing verl logits processor in chunks. Limite retains its publisher sigmoid head while avoiding a full sequence-by-vocabulary activation. Other models retain the upstream path.
- `0006-optional-recomputation-patch.patch`: let native providers disable the upstream Megatron backward monkeypatch when they own activation checkpointing.

- `0007-observability-extensions.patch`: add optional trajectory-validation and metric-callback configuration, dispatch finite non-overwriting metrics through the existing trainer logger, and accept an explicit B300 MFU reference. Execution remains upstream-owned.

At deployment set `ARCHLAB_HARNESS_SPEC` to the absolute path of `recipes/rl/harness/single_bash.yaml`; the recipe forwards this to every Ray actor as `MIXED_HARNESS_SPEC`. The preset uses the official E2B SDK and an authorized private PAI sandbox. Supply a private key-file path and a verified image-to-template map through the documented environment variables. Credentials stay in the parent SDK and are never copied into task environments.

The code pilot selects `recipes/rl/harness/single_bash.yaml`. Its native Limite Base tool grammar requires the **Hermes** JSON parser, not the inherited Qwen3-Coder XML parser. The harness reuses upstream `BashOnlyAgentConfig` prompts and tool definitions. It does not support Violetto's fixed mathematical chat template.

Regression checks for the first and third patches are in `tests/test_mimo_reward_contract.py`. Patch files carry only local changes; third-party source remains pinned rather than copied into the project package.

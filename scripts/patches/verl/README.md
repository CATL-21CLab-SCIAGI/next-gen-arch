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
- `0008-preserve-truncated-agent-status.patch`: set `finished` from agent completion before stock Uni-Agent masks unfinished episodes. A completed verifier is recorded separately as `verifier_finished`; capped runs keep their real truncation reason and original trajectory evidence. Reproduce the runner patch series against the pinned revision in `docs/upstreams.json`; `tests/test_mimo_reward_contract.py` executes both the actual runner and package mask writer.
- `0009-padding-without-fa2-extension.patch`: when FA2 is absent, reuse verl's existing pure PyTorch padding helpers for FA4 installations. Padding values, masks and gradients remain unchanged; missing transitive dependencies still fail. Regression checks are in `tests/test_training_attention_padding.py`.

- `0010-upstream-9c764364.patch`, `0011-upstream-e2ac8f62.patch`: backports of XiaomiMiMo/verl `main` #7511 and #7846 (absent from `mimo-oss`), exported with `git format-patch -1 <commit> -- verl/`. The vLLM server closes a request-submission gate while generation is paused for a weight sync. It reopens the gate only after the engines resume. `colocate_async` partial rollout otherwise admitted a request onto a sleeping engine, which crashed with an illegal memory access and then hung the FSDP weight gather. #7434 is not needed: this revision already resumes vLLM weights unconditionally.
- `0012-fsdp2-untied-lm-head-in-root-unit.patch`: keep an untied `lm_head` in the FSDP2 root unit. The fused-kernel log-prob path reads `lm_head.weight` without calling `lm_head.forward`, so a separately wrapped head is never all-gathered and post-backward fails (Qwen3.5-9B, 2026-10-20). Tied-embedding models are unchanged. Verified by the 9B arm's training steps 1-114 on 8 GPUs.
- `0013-reference-offload-only-when-requested.patch`: the frozen reference policy offloads its parameters to host memory only when `ref.fsdp_config.param_offload` is set. Upstream always forced it there, which holds the full-precision shards in host RAM on every rank: 108 GB on a node for Qwen3.8-27B. The forced copy, plus rank 0's temporary full copy during construction, exceeded the 256 GB host limit twice in the 27B admission (2026-10-25). Placement changes; the reference computation does not. It costs 1/8 of the reference weights per GPU. Verified by the 27B admission run's construction and training steps.
At deployment set `ARCHLAB_HARNESS_SPEC` to the absolute path of `recipes/rl/harness/single_bash.yaml`; the recipe forwards this to every Ray actor as `MIXED_HARNESS_SPEC`. The preset uses the official E2B SDK and an authorized private PAI sandbox. Supply a private key-file path and a verified image-to-template map through the documented environment variables. Credentials stay in the parent SDK and are never copied into task environments.

The code pilot selects `recipes/rl/harness/single_bash.yaml`. Its native Limite Base tool grammar requires the **Hermes** JSON parser, not the inherited Qwen3-Coder XML parser. The harness reuses upstream `BashOnlyAgentConfig` prompts and tool definitions. It does not support Violetto's fixed mathematical chat template.

Regression checks for the first and third patches are in `tests/test_mimo_reward_contract.py`. Patch files carry only local changes; third-party source remains pinned rather than copied into the project package.

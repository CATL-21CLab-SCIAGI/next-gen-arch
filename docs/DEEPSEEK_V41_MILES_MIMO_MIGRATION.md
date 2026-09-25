# Miles / MiMo RL migration — 2026-09-24

> Historical experiment record. For the supported command and current qualification scope, use [Miles baseline](MILES_BASELINE.md). Statements below refer to their dated attempts, not live health.

The user retired the slow AutoModel RLOO backend and selected the
[MiMo-V2.6 technical report](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL/resolve/main/MiMo_V2_6_technical_report.pdf)
as the replacement algorithm contract. Restart **both normal and simplicial arms
from their matched fine-tuned step-4537 parents**, with fresh RL state.
The old RL checkpoints are recovery artifacts, not the new run's parents.

**Historical migration record:** the two 16-GPU drivers were retired. Current work uses one 32-GPU arm at a time; see [resident qualification](DEEPSEEK_V41_RESIDENT_QUALIFICATION.md). Production training is not yet verified. The portable migration
contract is `recipes/proposals/deepseek_v41_miles_mimo.yaml`. It is not an executable
recipe or a claim of Miles support for the custom models.

## Selected recipe

Use Eq. (1) and §5.1: group-centered GRPO, prompt-group token normalization,
generation-time behavior log probabilities, detached per-token importance ratios,
and separate positive/negative acceptance bounds initially `[0.2, 5.0]`.
Out-of-bound tokens are masked; this is not the stock PPO clipped surrogate.
Train all response tokens rather than the old eight-prefix gradient estimate.
Freeze routers and their correction biases. Use Muown with LR `3e-6`, no warmup
or weight decay, gradient clipping 1, momentum 0.95, Nesterov, ten Newton–Schulz
iterations, update scale 0.5, and Adam betas `(0.95, 0.95)`, epsilon `1e-8`.

The report uses 16 samples per prompt and asynchronous partial rollouts with
staleness four. Qualify synchronous rollouts first, including behavior-probability
and weight-version provenance, before enabling this reuse. No entropy threshold
or automatic bound-adjustment schedule is published; do not invent one and label
it as the paper's recipe.

Explicit adaptations: retain the approved Nemotron math split, exact verifier,
paired prompt order/seed, 2,560 context and 2,001 response budget. Start with 16
prompts per batch per arm on the existing 16 B300 GPUs per arm, rather than the
paper's 1,568 prompts. Retain the local numeric values for Eq. (4)'s success-only
length deduction. Code-agent GAR and tool-call penalties do not transfer directly
to this math task. Fresh optimizer state is a user-selected departure from MiMo's
SFT Muown row-state continuation; no such state is established for these parents.
Do not silently replace Muown with Adam, generic Muon, or the old Adafactor.

## Runtime findings

- Checkout: `/mnt/nas/evergreen/miles`, upstream `radixark/miles`, pinned to
  `6c6858a42b61459467814edc1404b1d9bfa38471` (`zczeng/dsv41-rl-support`).
  Main revision `7d1d15b1cc072d68d544e6047173b4fb251a86d3` contains V4.1
  documentation but lacks the referenced V4.1 model and launch implementation.
- Existing four-node DLC job: `dlc1hig7iitpry5n`, x86_64, eight B300 GPUs/node.
  Refreshed through `GetJob`; all four pods were Running.
- Current image: `sci-agi-zhongwei-registry-vpc.cn-zhongwei.cr.aliyuncs.com/dev/nemo:26.06`.
  Head-node `/opt/venv` reports PyTorch `2.12.0a0+0291f960b6.nv26.4.48445190`,
  Megatron Core `0.18.2`, Transformer Engine `2.16.0+b9d690e0`, Ray `2.55.1`.
  SGLang and mbridge are absent. Megatron lacks `transformer.hyper_connection`
  and `transformer.module.mark_keep_in_fp32`, required by the upstream V4.1 plugin.
- Docker Hub manifest inspection: `radixark/miles:deepseek-v41`, digest
  `sha256:6a5916d18064b2db5c19005e1e0b096fe64528b66827d4ffb402c7f534ed5d01`,
  exposes ARM64 only. It cannot replace the x86_64 job's image.
  `radixark/miles:dev` has AMD64, but its V4.1 fork/runtime compatibility is unverified.
  Docker, Podman, Apptainer, BuildKit and a Docker socket were not available on
  the head node for building a replacement image in place.

The repository agreement makes PyTorch/Megatron/TE/CUDA/NCCL container-owned.
Do not patch or pip-replace those libraries in the running NeMo image to bypass
the missing APIs. The replacement uses the existing isolated AMD64 SGLang rootfs plus a separate, pinned NAS runtime overlay; the running NeMo installation remains intact. The rootfs is `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.6.0-deepseek-v4.1-flash-dev.1`, digest `sha256:778d7062f77592db1ada294f3511b48dd213f9fc655f432737733ec76557f5ec`. The overlay provides the Miles Megatron fork, TE 2.17, matching Apex/FlashAttention, Ray 2.56 and the disk-enabled memory saver.

## Model admission still required

The upstream V4.1 plugin does not establish support for our added normal or
simplicial attention modules. Preserve every trained adapter and the parents'
numerical behavior through checkpoint conversion, Megatron construction,
optimizer grouping, SGLang generation, weight synchronization and recovery.
The Miles extension uses explicit begin/end transactions and complete coverage checks for live weight updates.
Upstream's frozen quantized Engram path must not silently replace the fine-tuned
checkpoint's Engram weights or precision.

Qualify the actual parents, all-token backward memory, distributed loss scaling,
Muown state/checkpoint restoration, cached generation and full weight refresh.
Record a new explicit numerical contract for the Miles kernels; do not claim the
old cache admission applies. A successful launch requires reward-driven updates
and durable pilot checkpoints on both arms, not just live Ray processes.

## Old run retirement

Old output directories are `production-{normal,simplicial}-retained-v1` under
`results/deepseek-v41-math-rl-shared-20260923`. Both have complete step-2
checkpoints and passing **bounded critical-state** readback receipts on 16 ranks;
these receipts are not full-model restore tests. `STOP_REQUEST` was written for
both arms. The normal arm completed another rollout without applying an optimizer
update and entered evaluation. Both arms still had exactly two optimizer updates
when terminated. SIGTERM followed by SIGKILL for remaining exact RL processes
stopped the old workers; all four nodes reported no matching processes afterward.
Uncheckpointed rollout/evaluation progress was discarded. Completed optimizer
updates remain preserved at step 2. Final termination receipts are recorded under
`results/deepseek-v41-miles-mimo-20260924/retired-*.json`; `stop-*.json` records
the earlier incomplete SIGTERM attempt.

Preserve old weights, optimizer/RNG state, logs and metrics. Do not overwrite
`STOPPED.json` to simulate the old trainer completing a safe-boundary shutdown.
The DLC allocation remains available; a stopped old trainer is not evidence that
the Miles replacement has launched.

## Current implementation and checks

Project extensions implement the matched adapters, trainable BF16 Engram tables, checksum-verified parent import, bounded FP32-master Muown updates, and transactional rollout weight refresh. TP8/PP2/EP8 uses BF16 gradient accumulation; FP32 optimizer masters and momentum are streamed through persistent storage to fit the allocation.

Four CPU tests passed. Two-GPU admission passed for TP-sharded Muown versus the full-matrix reference, exact streamed-master updates, immutable optimizer snapshots, and Engram values/gradients. Receipts are under `results/deepseek-v41-miles-mimo-20260924/numerical-admission-v1`. These component checks do not establish full-model training or rollout parity. Live status is recorded in that run directory’s `MIGRATION_STATUS.json`.

The B300 sparse-attention kernel oracle passed forward and backward against a dense reference (`kernel-admission.log`). A metadata audit of all 16 ranks found no source-name or shape mismatch: 39,408 normal entries and 39,600 simplicial entries (`layout-audit-*.json`). Engram row padding is handled separately by the bounded row importer. These checks do not replace checksum verification of the checkpoint payloads or the first production update.

All 16 ranks in each arm completed checksum-verified parent loading. The current
startup is writing disk-backed weight copies. The isolated Miles checkout has a
small optional extension hook before weight-iterator construction, recorded in
`miles-weight-readiness.patch`: the project extension waits for every rank's
backup receipt before entering collectives and skips the redundant initial
restore of weights already resident on GPU. Distributed startup/transfer timeouts
are two hours to accommodate NAS IO.

Rollout engines allocate with SGLang's dummy loader and receive the complete
verified trainer policy before serving. The project forward guard remains closed
until every parameter and Engram row has been transferred and derived hashing
state verified. This avoids rereading each SFT parent in the inference replicas.
`test_miles_initial_policy_gate.py` checks that beginning a transfer cannot enable
generation. Full production transfer and optimizer updates remain pending.
The container's `SGLANG_DIAG_BYPASS_HEALTH_GENERATE=1` makes Miles' pre-transfer
health probe readiness-only. It does not admit generation through the project
model guard; successful rollout generation must be observed separately.

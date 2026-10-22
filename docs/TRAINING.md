# Training

The primary RL path composes XiaomiMiMo/verl's code recipe and delegates execution to its V1 trainer. Local code owns the scientific contract, verified model export and dataset index. It does not implement another optimizer loop, agent scheduler or monitoring system.

## Current Qwen3.5 code RL

The two current runs use native Qwen3.5-4B, upstream FSDP2 and vLLM. The maintained
contract is [`qwen35_code.yaml`](../recipes/rl/qwen35_code.yaml), composed over
pinned upstream defaults with one YAML overlay:

- [`qwen35_code_local.yaml`](../recipes/rl/qwen35_code_local.yaml): original v3c,
  batch 8, the 113-task pool, a local Ray head.
- [`qwen35_code_efficiency.yaml`](../recipes/rl/qwen35_code_efficiency.yaml): v4e,
  batch 16, the 80-task selected curriculum, and external CPU agent workers.

Both preserve 262,144 tokens of context, eight responses per prompt, DAPO group
filtering, prompt-mean policy loss, two versions of allowed policy lag and token
importance correction capped at two. This is a configuration/curriculum
comparison; the two runs do not isolate architecture or systems efficiency.
The resolved overlay output has been compared field-for-field to both live
configurations. Historical interpolation values are intentionally preserved,
including unused `actor/ref/critic.rollout_n=4` while generation uses `rollout.n=8`.

Supply model, data, output and sandbox paths through deployment environment
variables declared by the recipe. `ARCHLAB_MODEL_MANIFEST` identifies the model
asset provenance. For an explicit restart, set both `ARCHLAB_RESUME_CHECKPOINT`
and `ARCHLAB_RESUME_MODE=resume_path`; otherwise the recipe starts fresh. For example:

```bash
archlab resolve recipes/rl/qwen35_code.yaml \
  --overlay recipes/rl/qwen35_code_efficiency.yaml --output "$ARCHLAB_OUTPUT"
```

The maintained DLC environment client now owns task discovery, file transfer and
transport semantics. Existing CPU pool snapshots continue to own isolated server
execution. Tracking credentials are read from `ARCHLAB_MLFLOW_CONFIG` by the
shared launcher and its Ray worker setup hook; token values stay out of printed
Ray configuration. The snapshot launcher supplies the qualified `PYTHONPATH`.

**The new source is not production-qualified yet.** `qualify` records the missing
Qwen-specific GPU and task evidence and `train` stops at that gate. Existing
trainers retain their immutable source and active launch handles. Exact recipe
parity is not proof of FSDP restart, hybrid prefix-cache/logprob parity, full-context
memory, verifier correctness or throughput after moving source. In particular,
training samples top-k 20/top-p 0.95 while learner replay computes full-vocabulary
probabilities: the recorded decoupled importance correction must be evaluated with
that contract, rather than applying the older Limite full-support parity bound.

## Muse agentic pilot

[muse_code.yaml](../recipes/rl/muse_code.yaml) defines a fresh native
`meta-models/Muse-Glimmer-30B` control for executable code tasks. It starts from
the publisher checkpoint with a fresh Adam optimizer, rather than Limite SFT or
old RL state. This establishes agentic capability before a separately qualified
attention replacement experiment. The recipe's presence does not mean GPU
qualification or production training has completed.

The learner uses stock verl FSDP2 across eight B300s, FP32 master parameters,
BF16 compute and native Transformers FlashAttention-4. Native vLLM serves the
publisher's Muse architecture, reasoning and tool protocol. Its global and
sliding attention, normalization, output scaling and logit softcap remain part
of the native model. No Megatron weight conversion is used for this control.
Image and video inputs are disabled for these text-only tasks.

The 131,072-token context is preserved: 16,384 prompt tokens and 114,688 response
tokens, with 8,192 tokens per agent turn. Length-capped turns remain unfinished
episodes and are excluded from policy updates. Sampling uses temperature 1,
top-p 1 and no top-k cutoff, explicitly differing from the publisher's default
top-p/top-k settings. The native template uses high reasoning strength. The
MiMo bash agent, isolated task environments and executable verifier are reused.

The initial pilot has two task groups with four responses each, at most eight
active task environments, one policy version of asynchronous lag, and upstream
token importance correction capped at two. It requests 20 fresh updates,
validation and full checkpoints every five updates, retaining the latest two
resume checkpoints. MLflow records the experiment as **Muse — RL**.

Sharded model and optimizer state stay on the GPUs. Offloading all FP32 masters
and Adam moments would exceed the target node's 256 GiB host RAM. Rollout reserves
30% of each GPU. This budget still requires actual simultaneous learner/serving
memory checks; capacity cannot be inferred from model weight size alone.

For this recipe, set `ARCHLAB_PARENT_CHECKPOINT` to the verified native publisher
directory and `ARCHLAB_MODEL_MANIFEST` to its pinned asset manifest. Use
`recipes/rl/harness/muse_single_bash.yaml` for `ARCHLAB_HARNESS_SPEC`; the remaining
deployment variables and thin `resolve` / `train` interface below are shared.
The isolated runtime records its exact package lock separately from container
dependencies. Native model construction, sampled-probability parity, all-layer
backward, full optimizer/RNG restore, full-context capacity and a real nonzero
agentic GRPO update must pass before production admission.

## Limite pilot

`recipes/rl/limite_code.yaml` starts fresh software-engineering RL from the normal Limite **2B adapter-warmup endpoint**, selected using AIME rather than cross-entropy alone. The original matched normal/simplicial checkpoints both completed 2,000,158,720 tokens at step 7,630. Their frozen publisher backbone is verified before the adapters are restored and **all weights are unfrozen for RL**. The new normal recipe explicitly uses the publisher’s FlashAttention-2 reduction for both backbone and preludes; weights and attention geometry remain unchanged. The earlier TileLang migration is preserved as a separate failed admission candidate.

The run resets the optimizer, scheduler, RNG streams to seed 42, and training counters. Admission checks the exact declared SFT endpoint and rejects post-RL weights even if supplied through a launch override. The 10B SFT and old mathematical-RL checkpoints remain archived; they are not initialization inputs.

The code data and asynchronous MiMo recipe define a separate experiment. Historical RL started at 10B SFT, so changes in initialization, data, execution and algorithm prevent attributing a difference to architecture alone. A matched normal-versus-simplicial code comparison requires both original 2B anchors and the same new data split, seed, recipe and training budget.

The original code index contains 64 training and 8 validation tasks. The current v3 readiness pilot contains 8 training and 3 validation tasks, selected from the available sandbox templates with smaller published images. Both use XiaomiMiMo/MiMo-V2.6-RL-oss at revision `639865fd3374018d6cb29b9fb82dd531406fcf5f`; the original index is preserved. The manifest records its availability and image-size selection bias, including replacement of a task whose published verifier fails before evaluating the requested behavior. It is an infrastructure pilot, not a representative benchmark. Original test patches and verifier commands remain outside model prompts. The split is task-disjoint, not repository-disjoint.

The index pins the publisher's image digests. The private sandbox mirror currently uses task tags; runtime admission verifies clean source state, the hidden test patch and executable verifier. It records the actual Git commit and template identity, but does not establish OCI digest equality between that mirror and the publisher.

The pilot allows one policy version of lag. verl recomputes the old learner probabilities, applies its token importance ratio against recorded rollout probabilities once (capped at 2), then evaluates PPO against the recomputed old policy. This also measures and corrects serving/replay numerical differences. The historical `use_rollout_log_probs` flag alone does not enable that correction in the pinned trainer.

The pilot batch has 8 tasks and 4 rollouts per task. One agent worker admits up to eight isolated task sessions through upstream's concurrency limit. Each agent reserves 0.25 local Ray CPU tokens through `UNI_AGENT_RUNNER_TASK_NUM_CPUS`: it waits on remote tools and GPUs, while its isolated sandbox still receives 2 cores and 2 GB. Reserving a full local CPU per agent limited the observed 32-core deployment to five active sessions despite eight available rollout replicas. Deployment requires dedicated sandbox capacity for eight environments. The real-policy admission probe preserves this concurrency instead of silently serializing sessions. Full-weight synchronization preserves native FP32/BF16 dtypes; delta synchronization is excluded. The native provider owns activation checkpointing, with upstream Megatron runtime monkeypatching explicitly disabled.

The current qualification candidate uses the publisher's `flash_attention_2` for both backbone and normal preludes. The backbone choice is recorded in the export configuration; the source weights remain unchanged. The original SDPA candidate produced non-finite backward gradients at 131,072 tokens. FlashAttention-2 passed that full-context check, with small measured BF16 differences at short contexts. The fresh 2B candidate also passed eight-rank gradient and optimizer restart checks. Serving/replay agreement and a real-policy learning signal remain required before production admission.

Normal serving now selects vLLM's single-sequence `FULL_DECODE_ONLY` CUDA graphs,
with Torch compilation disabled. The model extension refreshes the publisher's
folded inference buffers in place after weight loading, preserving addresses
captured by those graphs. A matched 2026-10-12 probe measured 5.7–7.6× faster
decoding across 128-, 1,100- and 2,000-token prompts, with identical generated
tokens and log probabilities before and after reloading weights produced by an
actual optimizer update. This is a decoding measurement; tools, rewards and learner time are
excluded. Full-context serving and end-to-end agentic admission must also pass
for this configuration before production starts.

Deployment supplies these paths:

```bash
export ARCHLAB_PARENT_CHECKPOINT=/path/to/complete-parent
export ARCHLAB_POLICY=/path/to/verified-hf-export
export ARCHLAB_CODE_DATA=/path/to/code-index
export ARCHLAB_OUTPUT=/path/to/new-run
export ARCHLAB_HARNESS_SPEC="$PWD/recipes/rl/harness/single_bash.yaml"
export E2B_DOMAIN=your-private-sandbox-domain
export ARCHLAB_E2B_KEY_FILE=/private/sandbox-key
export ARCHLAB_E2B_TEMPLATE_MAP=/path/to/qualified-task-templates.json
export ARCHLAB_CONTAINER_IDENTITY=registry/image@sha256:actual-allocation-digest
export ARCHLAB_THROUGHPUT_BASELINE=/path/to/matched-reference/throughput.json
archlab resolve recipes/rl/limite_code.yaml --output "$ARCHLAB_OUTPUT"
archlab train recipes/rl/limite_code.yaml --output "$ARCHLAB_OUTPUT"
```

### CPU-DLC sandbox backend

When the E2B pools refuse capacity, `archlab.training.dlc_sandbox.DLCEnvironment` serves the same MiMo environment contract from CPU DLC pods. Select it with the harness `environment_class` and set `ARCHLAB_DLC_SANDBOX_POOL` to the directory where pool servers announce themselves.

- Each pod runs `python -m archlab.training.dlc_sandbox serve <images> <pool>`.
- Each sandbox is a rootless user/mount/pid/network namespace. Its root is an overlay of the task image, with `userxattr` and a private tmpfs upper layer.
- Sandboxes have loopback only and use E2B's login shell and offline pip/uv variables. A full pod answers 503, so clients wait for capacity exactly as with E2B.
- Images are root filesystems exported on DLC from the private mirror tags, stored under `<images>/exports/<digest>.tar.gz`. `configs` fetches the digest-verified image configs.
- Admission evidence is the per-task comparison against the E2B runtime receipts: base commit, verifier return code and baseline reward. Separately, exports are compared with the publisher layers.
- Command timeouts return code 124 with partial output, as upstream's Kubernetes environment does; transport failures abort the session. Unlike E2B, `/proc` is the pod's (bound, not remounted): other processes' command lines are visible, while their environments, roots and signals are not reachable.

`train` performs admission once and starts verl only after it passes. Use `qualify` instead when you want the checks without starting production; running both commands repeats the live checks.

Install the project and the pinned verl/agent packages in an isolated environment that reuses the container's GPU runtime. Agent workers must import the same checkout and patches. Supply MLflow configuration through its standard environment variables; never put credentials in recipes or resolved artifacts. A functioning isolated task-container service is required. Dataset indexing alone does not qualify the agent loop.

Record the actual allocation image identity; when the platform exposes only a tag, record that tag explicitly rather than invent a digest. Admission also records the resolved PyTorch, CUDA, NCCL, Megatron Core, Transformer Engine, serving and agent-package versions. Scientific command-line overrides use `engine=megatron`; upstream settings use, for example, `settings.actor_rollout_ref.actor.optim.lr=0.00001`.

## Upstream patches

A fresh submodule checkout needs the local patch sets for [verl](../scripts/patches/verl/README.md), [Uni-Agent](../scripts/patches/uni-agent/README.md) and [MiMo-Agent](../scripts/patches/mimoagent/README.md). Apply each set to its documented pinned checkout in numbered order. Admission verifies both the patch-file hashes and the complete working diff against `docs/upstreams.json`; an unpatched or additionally modified checkout cannot reuse qualification. Container runtime libraries remain unmodified.

## Admission

One command records the effective configuration and runs checks for package identity, data integrity, model lineage, native numerical behavior, serving/replay agreement, optimizer/checkpoint restart, stock Ray startup, matched learner throughput and a real-policy agent/tool/verifier update. Missing, failed or skipped checks prevent launch. Each test preserves its resolved configuration, numerical receipts and checkpoint evidence alongside the report.

The upstream configuration validator runs before GPU admission. Distributed gates require an executed test on every rank, and both learner backward and serving are tested at the full native context. A deterministic reference repair proves the verifier can distinguish failure from success. The separate real-policy probe exercises two groups of four responses, assistant-only loss masks and a complete saved update. Admission also requires at least one group with mixed rewards and a finite, nonzero gradient; equal rewards leave learning admission pending even when execution succeeds. The probe has its own output directory and does not update the production source checkpoint.

The learner numerical gate uses a fixed engineering equivalence contract, declared before distributed validation. Native BF16 reductions and atomic gradient accumulation need not be bit-identical; see [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html). The limits below are our admission criteria, not an upstream guarantee.

| Observation | Required comparison |
| --- | --- |
| Initial weights, forward outputs and exported weights | Exact |
| Restored model, optimizer master weights, Adam moments, group settings, scheduler and RNG | Exact |
| Backward gradients | Relative L2 difference ≤1% globally and ≤2% per architecture block; cosine ≥0.999 |
| Next-update weights after restart | Relative L2 difference ≤0.00001 globally and ≤0.0001 per block |
| Next-update direction and magnitude | Global update cosine ≥0.999 and norm ratio between 0.98 and 1.02 |

Every gradient gate also runs two bounded controls: another backward on the unchanged native model and a separately reconstructed native model. The checkpoint gate replays the same saved update in the original engine. These native controls must meet the same fixed limits; native instability blocks admission. Tiny deterministic fixtures remain bit-exact. Negative tests reject corrupt checkpoint bits, altered optimizer settings, missing or scaled updates, and gradient perturbations. Earlier failed receipts remain preserved: the previous acceptance rule compared hundreds of extrema against a small native sample and could reject harmless native variation. Passing this revised contract establishes engineering numerical equivalence, not bit-identical training trajectories.

Do not force Limite through a generic serving implementation: it has mixed parameter dtypes, publisher-specific normalization and output transforms, and 48 additional attention layers. These require independent caches and exact weight reload behavior. The normal serving integration is restricted to tensor/pipeline parallel degree one until broader distributed qualification exists. Simplicial serving additionally needs short-axis cache support.

Serving admission compares the same sampled response tokens before and after an actual optimizer update. The independent publisher oracle uses the declared serving precision: BF16 FlashAttention-2 for both backbone and preludes. Its cache retains full key history and absolute positions; the publisher still applies each layer’s original attention window. The publisher’s physically bounded cache remains a separate diagnostic. This avoids confusing a change in FA2 tile alignment at a sliding-cache boundary with a serving weight or mask error. Both TileLang and FA2 return BF16 to the projection here; their reduction rounding differs. The current normal learner uses FA2 too. Serving-versus-learner mean and p95 log-probability errors must remain within `1.25 × native FA2 cached-versus-learner replay error + 0.01`; a separate serving-versus-native-FA2-cache check uses the native FA2 cache-versus-replay baseline. Both checks retain absolute maximum error ≤0.5 and estimated k3 KL ≤0.005. Original TileLang cached-versus-replay statistics and prompt errors remain diagnostic evidence. No response is resampled to pass. An earlier failed receipt exposed an oracle mismatch. Correcting it still left a TileLang-versus-FA2 discrepancy beyond the unchanged bound on one of six fixed-token cases. Independent FA2 replay satisfied those same bounds on all six recorded cases, motivating the explicit learner migration. A subsequent post-update discrepancy against the bounded native cache disappeared on the exact same recorded tokens when the independent oracle retained full history; all six fixed-token cases met the original bounds. This is an explicit reference-cache correction, not a changed error allowance. Historical receipts do not replace a new optimizer, serving and real-policy qualification.

## Backends

| Integration | Current status |
| --- | --- |
| Limite HF + publisher FA2 | Normal RL candidate; full deployment admission required |
| Limite HF + TileLang | Native model and export tests; new verl execution qualification in progress |
| Limite vLLM normal | Dedicated integration under GPU qualification |
| Limite vLLM simplicial | Short-axis paged-cache integration required |
| Limite Megatron learner | Eight-rank gradients, exact checkpoint-state restore and next-update checks passed; full deployment admission in progress |
| Limite TE attention | Numerical qualification required; Megatron orchestration does not establish TE layer parity |
| Historical scratch/Megatron/SFT | Existing model-specific paths retained until a qualified replacement exists |

Local compatibility archives are ignored and are not imported by the new training path. Running jobs use immutable source snapshots. New features belong in the primary package-backed path.

The primary normal learner uses Megatron with the publisher’s native FlashAttention-2 operations. The abandoned FSDP attempt was removed because its sharding groups cannot preserve this model's native mixture of FP32 and BF16 parameters. Megatron orchestration does not imply that all model layers use Transformer Engine; TE substitution needs its own numerical evidence.

## Rollout, data and performance contracts

`training/rollout.py` validates the upstream trajectory before Uni-Agent writes it to TransferQueue. It requires aligned tokens, assistant-only masks, finite recorded behavior probabilities, actual first/last policy versions and explicit completion or truncation. Typed metadata carries task, prompt-group, session and response identities through the existing transport. Tool observations remain outside policy loss. One session may produce several context segments; those segments are not extra independent GRPO responses. The engine seed and response identity are recorded, but asynchronous sampling is not claimed to be bit-exactly replayable from a seed alone.

`qualification` writes `data-audit.json` for the verified code manifest. It fingerprints tokenizer/template/generation files, reports prompt lengths and native EOS counts, rejects over-budget prompts without truncation, and records the static curriculum's selection provenance. Prompt statistics cover the task's user message and generation prefix; the agent adds system/tool messages later. Native EOS occurrences and truncation during generated trajectories are separate runtime measurements. The current pilot does not use a reward-adaptive curriculum. Transformers' token-ID return type is requested explicitly so a dictionary of tokenizer output fields cannot be mistaken for two tokens.

`training/metrics.py` registers the shared native Limite FLOP ledger with verl's existing counter. The ledger lives in `architectures/limite_cost.py` and is also used by SFT. It includes both attention stacks, native windows, GQA projections, gates, MUDD and the tied output head; it excludes embedding lookups, recomputation, padding, communication and optimizer work. Variable sequence lengths are counted separately. Its qualified scope is the normal prelude geometry. The explicit B300 reference is **2.25 PFLOP/s dense BF16 per GPU**, half of the sparse specification in [NVIDIA's Blackwell Ultra datasheet](https://resources.nvidia.com/en-us-gpu-resources/blackwell-ultra-datasheet?xs=443990). The mixed FP32/BF16 implementation makes this a useful-model-FLOP ratio against that reference, not a measurement of hardware instruction utilization. Upstream's learner/forward timers determine the reported MFU interval; it is not whole-job MFU including tools and rollouts.

A small optional callback feeds additional metrics into verl's existing MLflow/console logger: consumed model/tool tokens, useful and incomplete reward groups, EOS and truncation counts, trainer steps and policy versions. Per-phase allocated GPU-seconds separate rollout collection, replay, learning, checkpointing and synchronization. Generation time on colocated GPUs is **not** labeled idle. `perf/measured_idle_available=0` explicitly marks physical idle-time accounting as unavailable; phase accounting covers each training step, not external validation, startup or individual GPU utilization. A trainer step may contain several optimizer updates, so the callback does not mislabel that counter.

## Throughput regression gate

The unified qualification runs a disposable stock Megatron learner benchmark with the declared export, precision, optimizer and GPU topology. Defaults are one 2,048-token sequence per rank, two warmup updates and five measured updates. Timing synchronizes CUDA, uses the slowest rank and includes normal parameter/optimizer offloading. It excludes generation and sandbox tools. It never modifies the parent or production checkpoints.

Supply a measured reference receipt through `ARCHLAB_THROUGHPUT_BASELINE`. Candidate and reference must match weights, actor configuration, workload, container, runtime versions, hardware, topology and benchmark source. The median slowdown limit defaults to 1.20 in the typed `throughput` contract. Missing references remain pending; failed, skipped or mismatched measurements cannot admit training. Qualification reports preserve both receipts and their hashes. End-to-end agentic throughput still depends on response length, tool behavior and sandbox concurrency; the fixed learner gate does not certify those costs.

On 2026-10-12, the frozen pre-restructure learner and current learner both passed this eight-B300 benchmark from the same 2B SFT export: **3.127 s versus 3.050 s median** across five measured updates. This establishes no observed regression at this workload; the 2.5% difference is not evidence of a meaningful speedup. CPU contract checks and this benchmark do not replace serving/replay parity or the real-policy learning-signal gate.

The superseded Miles RL, native DeepSeek RL and SGLang prototype stacks are retired from the maintained package, including their dedicated tests and executable recipes. Their verified local backups live under the ignored `.legacy/` directory. Historical reports remain labeled as such; running immutable snapshots are independent.

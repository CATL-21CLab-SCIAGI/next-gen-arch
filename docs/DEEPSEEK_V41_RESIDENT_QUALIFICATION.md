# GPU-resident RL qualification

The selected contract is `recipes/experiments/deepseek_v41_resident_frozen_engram.yaml`:
normal then simplicial, four nodes / 32 NVIDIA B300 per arm, BF16 model and gradients,
FP32 master weights, scaled FP16 momentum and FP32 orthogonalization, frozen Engram tables, trainable Engram projections
and gates, frozen routers, full activation recomputation. Each production arm gets
7200 seconds of rollout, update, and policy-transfer time after initialization.

The user first selected FP32 momentum after BF16 failed the numerical gate
(update relative error <= 0.01 and cosine >= 0.999). BF16 receipts remain in
`results/deepseek-v41-resident-20260924/numerical-admission`; they cannot admit
production. FP32 receipts in `fp32-streaming-restore-admission` measured zero relative
update error against the same-topology FP32 reference and minimum cosine
0.9999997616. Distributed Sinkhorn and exact component optimizer resume passed.
The subsequent FP32 full-model handoff had insufficient headroom. The user then
selected a 16-bit momentum path. Scaled FP16 with FP32 orthogonalization passed
both B300 probe ranks at weight scale 0.02: maximum direct direction error
0.00245615, minimum direction cosine 0.99999708, maximum applied-update error
0.00809067, and minimum applied-update cosine 0.99996728. Thresholds remain
0.01 and 0.999 for both measurements. Power-of-two FP32 scalar scales protect
FP16 gradient history from under/overflow; recurrence and master arithmetic stay
FP32. CPU tests cover gradient scales 1e-12, 1 and 1e4. This saves about 33 GiB
per heavily loaded trainer rank relative to FP32 matrix momentum.

A 100-step two-rank probe at actual expert dimensions (2304 x 5120) also
passed: maximum direction error 0.00079368 and applied-update error 0.00615041,
with exact checkpoint restore. Admission requires both small and expert-size probes.

The corresponding BF16 + FP32 orthogonalization GPU diagnostic still failed
(1.47% direction error, 1.55% applied-update error on rank 0). Original order-one
weight diagnostics are retained, rather than treated as direct momentum-error
measurements. The selected probe records both pre-application directions and
applied weight changes; it does not discard the latter. These are component
checks, not full-model qualification.

## Why the earlier experiment design was wrong

Splitting the allocation into two 16-GPU jobs made live training state compete
with rollout state and backups, forcing CPU/disk transfers on the critical path.
Process startup and checkpoint import were incorrectly easy to mistake for RL
progress. A model fitting in steady state does not establish that gradients,
optimizer workspace, weight refreshes, and rollout KV can coexist at their peaks.
The new comparison requires completed updates and transition memory evidence.

The MiMo-V2.6 technical report and DeepSeek-V4.1 Flash report were read before
implementation. MiMo's FP32 master/state choices do not justify blindly lowering
all optimizer precision. DeepSeek's Sinkhorn specifies full logical-matrix
normalization, so sharding must preserve global column statistics; local shard
normalization is not a numerical oracle. BF16 preference remains subordinate to
the numerical gate. Frozen Engram removes table gradients, masters, and momentum
rather than merely relocating their cost to the host.

Report artifacts: MiMo PDF SHA256
`fb81e6e083801b3358f084ed6be953dc23b0d2e434690f4541d5eae03e01e7af`;
DeepSeek PDF SHA256
`0cfdda19c8e1691b5226dd0d4949e2397051268e8a823c2fb61db94ea20c4ac0`.
Sources: https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf
and https://arxiv.org/html/2609.19969v1 .

## Memory lifecycle

The trainer retains weights and optimizer state on GPU. Its gradient storage is
discarded during rollout and recreated before training. Serving weights and KV
are discarded before training; serving weights are repopulated by GPU transfer
from the trainer. Deterministic serving buffers remain on GPU. There is no saved
old/reference policy. Trainer and inference layout copies do coexist during
rollout/refresh; this is not a claim of one shared physical weight allocation.

Miles names the SGLang discard/reallocation lifecycle `--offload-rollout`.
The resident launcher needs this API flag but explicitly disables host backups;
`--no-offload-train` remains set. The allocator probe reclaimed 536870912 bytes
with both CPU and disk backup disabled and successfully recreated the buffer.
No runtime library source was modified for this implementation.

Training is TP8/PP4/EP8, while rollout is TP32/EP32 with attention DP4 (attention
TP8). Frozen training Engram tables retain the existing per-stage TP8 sharding;
rollout tables are sharded over 32. This topology is not supported for production
until both full-model pilots pass. Physical HBM sampling supplements, but does
not replace, peak/transition checks. Initial checkpoint import and explicit
checkpoint save/restore may use the host; live training state may not offload.

## Current evidence and remaining gates

FP32 numerical and allocator checks passed, including streaming distributed
optimizer checkpoint save/load and exact resumed updates. The CPU suite passed 963 tests across the initial run and targeted retries;
149 tests were skipped. Socket tests required execution outside the sandbox and
tracking tests required the optional mlflow-skinny dependency in a temporary test
environment. Repository Ruff checks pass. Focused optimizer, admission, serving
and checkpoint regressions are included. The full-model normal attempt 5 initialized all 32 ranks,
but failed before its first RL update: Miles requested an absent SGLang
`begin_weight_update` endpoint. Free HBM during the attempted policy handoff was
about 5.5%, below the 10% gate. The failed workers were retired.

Attempt 6 also failed before training and its workers were retired. It used the existing model
in-stream transaction markers instead of absent quantized-base/LoRA session RPCs,
releases unused cached allocations after replacing native Engram storage, and
started serving imports before trainer checkpoint reads. The added startup discard
repeated Miles' already-completed KV discard and aborted the memory-saver runtime.
The launcher now discards only weights at that point; attempt 7 is testing the correction. Cache release returned zero bytes, so it did not solve the headroom deficit.
All 32 GPUs were verified empty after attempt 6 cleanup. The DLC allocation is retained.
Serving session compatibility and full checkpoint verification remain unqualified. The bounded streaming checkpoint writer passed a two-rank
save/load probe; the default MCore writer staged all tensors in host RAM and was
unsuitable for this node memory budget.

Full-model update, memory-headroom, policy-parity, and checkpoint-restore gates
remain pending. Production launch fails closed until the selected variant has valid
full-model receipts; startup alone must not create those receipts.

Latest machine-readable status: `results/deepseek-v41-resident-20260924/QUALIFICATION_STATUS.json`.
Attempt 8 runs the normal arm first with scaled FP16 momentum and frozen Engram.
It checks initial response-token log-probability parity before any optimizer
update (mean absolute difference <=0.05, maximum <=0.5). After two pilot rollouts,
all ranks must demonstrate at least two optimizer updates, >=10% physical free
HBM, and exact full checkpoint restore. Passing these checks admits an in-place
7200-second continuation; it does not require reloading the parent or waiting for
the simplicial pilot. The user explicitly prioritized a healthy baseline.
Production has not yet been admitted.

Attempt 7 initialized all 32 scaled-FP16 optimizers and reached policy handoff
with approximately 48 GiB free on the constrained ranks. It failed before any
rollout because native SGLang finalized Engram after the first weight bucket.
The project loader now defers Engram validation, APE conversion and norm-cache
refresh until the full transaction ends; it joins split compressor projection
pairs within a bounded 128 MiB workspace and verifies every owned expert slice.
Three new streaming regressions pass, in addition to five parity-gate tests.
Attempt 8 passed the complete policy-transfer checks on all 32 receivers.
No production admission is claimed yet.


Attempt 8 first-rollout blocker (2026-09-24): all 32 resident optimizers
initialized, and all 32 serving receivers acknowledged policy version 1 with
1,003 parameter tensors each. Observed physical free HBM stayed at or above
13.45895% through transfer and the attempted first rollout. No optimizer update
completed; full training-peak memory, policy parity and full checkpoint restore
remain unqualified.

The first eager prefill crashed in native SGLang V4.1: attention-DP preparation
rounds 100 real tokens up to 104 slots (attention TP size 8), while
`dsv41_sparse.token_req_indices` passes the padded position count as
`repeat_interleave(output_size=104)` with request lengths summing to 100.
PyTorch reports that exact 104-versus-100 mismatch in the CUDA assertion.
The native eager `_forward_prepare` forwards unsliced tensors to
`forward_low_ratio_sources`; its separate breakable-graph helper slices to
`num_token_non_padded_cpu`. This is a serving padding defect encountered before
any optimizer update, independent of momentum storage. Removing the assertion
or fabricating requests for padding would not establish correct KV writes.
A padding-safe serving path must be qualified before another full RL launch.

The failed driver was stopped, and all 32 GPUs were verified at 0 MiB used.
The four-node allocation and persistent artifacts are retained. Evidence is in
`pilot-normal-v8.log`, `pilot-normal-v8/BLOCKER.json`, the per-rank resident
receipts and per-node HBM observations under the run root. No runtime library
source was modified to bypass this failure.

The project serving extension now bounds eager extend compressor/indexer inputs
to the real-token prefix while retaining padded attention and MoE communication
shapes. Adapter caches likewise consume only real prefill/decode tokens; padded
decode request slots write the Engram hasher's dedicated spare history row.
This addresses both the observed 104/100 assertion and subsequent stateful
padding hazards without changing container source. Twelve focused CPU tests
pass, including idle ranks and both adapter variants. Full-run qualification
of the correction is pending in attempt 9.

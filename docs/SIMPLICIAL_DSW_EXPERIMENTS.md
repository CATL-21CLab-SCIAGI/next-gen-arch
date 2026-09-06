# Controlled simplicial attention: experimental DSW pilots

The user approved the existing sampling environment and ordinary partial RoPE
on Q and both keys. The native adapter and sequential A/B/C driver are
implemented. Each campaign refuses to launch unless all three full-model probes
have passed against the exact training/model source hashes. Operational status
lives in the campaign's `CAMPAIGN.json`, not this versioned document.
This is an experimental path, not a generally supported Megatron architecture.
The original DSW campaign uses DP1. The separately authorized DLC migration
uses the DP32 contract below; containers and allocations are never restarted.

## Reproduce the bounded preflight

Use the existing container Python with project `src` on `PYTHONPATH`:

```sh
python -m archlab.benchmarks.simplicial_attention \
  --output <new-evidence-json> --batch 4 --sequence 2048 \
  --repetitions 20 --include-flash
```

The output must not exist. Source hashes, runtime/container metadata, GPU process
snapshots, all numerical errors and timing samples are saved. No packages are
installed. Both no-grad forward and forward/backward paths are warmed up before
timing. GPU memory is capped at 25%. Other services are not stopped; timings are
preliminary if a co-resident service is active.

CPU tests need only existing PyTorch and the standard library:

```sh
PYTHONPATH=src python -m unittest discover -s tests -p test_simplicial_attention.py -v
PYTHONPATH=src python -m unittest discover -s tests -p test_simplicial_campaign.py -v
```

Native adapter tests use the approved existing environment and the same native
backend switch as the pilot initializer:

```sh
python -m torch.distributed.run --standalone --nproc-per-node=1 \
  tests/test_simplicial_native.py -v
```

These check local/global equivalence inside the window, an explicit local-mask
boundary oracle, the full simplicial block's forward/backward against FP64,
unchanged parameter bytes, isolated initialization RNG, and all gate/norm/new
projection gradients. The separate full-model probes cover optimizer grouping
and native model/master-weight/momentum/scheduler checkpoint round trips.

## Code boundaries

- `src/archlab/architectures/simplicial_attention.py`: input contract, independent
  FP64-capable oracle and lazy CUDA dispatch. No trainer/optimizer dependencies.
- `src/archlab/architectures/simplicial_kernels.py`: project-owned forward and
  backward kernels using stock Triton, not TLX. Both KV groups are local. A query
  owns 12 valid heads in a masked 16-head tile. There is no TP or EP requirement.
- `src/archlab/benchmarks/simplicial_attention.py`: bounded execution adapter.
- `src/archlab/megatron/simplicial_attention.py`: retains native Q/K/V and output
  projections, Q/K norms, FP32-sigmoid output gate, and partial RoPE. Only C adds
  K2/V2 projections and a K2 norm. Installation happens after full baseline
  construction; unchanged parameter objects and names are preserved.
- `src/archlab/megatron/simplicial_pilot.py`: native Muon/Adam, forward/backward
  schedule, fixed-window held-out evaluation, checkpointing and bounded loop.
- `src/archlab/megatron/simplicial_campaign.py`: single-GPU lock, sequential fresh
  A/B/C launches, evidence/source gates, fail-stop supervision, no service control.
- `recipes/proposals/qwen38_w320_simplicial_dsw.yaml`: descriptive proposed
  A/B/C contract, not an executable training recipe.

The kernel is the coordinate-wise trilinear score with joint softmax over two
causal sliding-window axes and elementwise products of the two value branches.
It adds no KV biases and does not itself apply projections, Q/K normalization,
gating, residual mixing, or positional embeddings. Those are retained by the
enclosing adapter/backbone. It is not determinant-based attention.

Backward recomputes scores and uses FP32 atomics for shared K/V gradients.
Consequently it is not bitwise deterministic. Higher-order differentiation is
not validated. Kernel timings do not establish whole-model throughput.
Convergence is not established by numerical correctness tests.

## DLC migration: DP-only, fresh matched pilots

`recipes/proposals/qwen38_w320_simplicial_dlc.yaml` specifies the separate
four-node/32-GPU campaign. The long-running baseline is checkpointed and paused;
its checkpoint is not used to initialize any pilot. The old DSW pilot checkpoint
is also retained. All three DLC arms start fresh in the same frozen NeMo runtime.

The model, seed, 3B-token budget, global batch 4096, microbatch 4 and full-baseline
LR horizon remain unchanged. DP32 reduces accumulation from 1024 to 32 local
microbatches per update; TP/PP/EP/CP/expert-TP remain one. Global microbatch `i`
belongs to rank `i % 32`, preserving the DP1 stream's exact global token set and
order rather than giving each rank a duplicate stream. A CPU regression checks
the union and resume cursor at DP1/DP2/DP32. Fixed held-out evaluation is replicated
on each rank and averaged, counting 131,072 **unique** tokens, not 32 copies.

DLC uses native attention backend `auto` and gradient-accumulation fusion, matching
its existing runtime capabilities. These differ explicitly from DSW's compatibility
fallbacks; A/B/C use the same settings. Do not interpret a cross-host timing ratio
as an isolated architecture speedup.

`archlab.megatron.simplicial_dlc_campaign --phase probe` runs three full-shape,
three-step DP32 probes (global batch 256). `--phase train --probe-root ...` refuses
to start unless all probes match the exact model/training sources, runtime,
common initialization, data order and held-out window, and all ranks have passed
native model/master-weight/momentum/scheduler checkpoint reload. Run the native
adapter tests with `torchrun --nproc-per-node=2` first: they compare DP-mean
parameter gradients against a combined-batch reference for all three arms.

The supervisor connects only to the four supplied existing hosts using strict
SSH host-key checking. Each worker records its torchrun PID and has a per-host
advisory lock. No controller/service/allocation control is implemented. SIGTERM
to the supervisor requests checkpointed stop after the current complete update;
failed/incomplete arms prevent the next launch. Baseline resumption remains
explicit, not an automatic side effect of finishing the pilots.

## Matched pilot contract

All arms retain width 320, depth 48, 36 GDN layers, four residual streams,
32 routed experts/top-10 plus one shared expert, expert width 80, PLE, Q/K norm
and attention output gates. MTP remains disabled. Attention has 24 query heads,
two KV heads and head dimension 32. Only layers 8/16/24/32/40/48 change:

| Arm | Six selected attention slots | Total parameters |
| --- | --- | ---: |
| A | Original global gated dot-product | 387,680,960 |
| B | 128-token causal local gated dot-product | 387,680,960 |
| C | 16×128 causal gated simplicial, ordinary RoPE on Q/K1/K2 | 387,926,912 |

C is an explicitly named ordinary-RoPE variant, not a claim of trilinear
relative-position invariance or exact paper reproduction. The original
0.25 rotary fraction and theta 10,000,000 remain. K2/V2 are each `[64, 320]`,
with one additional `[32]` zero-centered RMSNorm per changed layer. New
projection initialization uses isolated RNG; every unchanged weight is hashed.

All arms are fresh seed-42 runs, DP1, TP/PP/EP/CP/expert-TP=1, BF16, sequence
2048, microbatch 4, global batch 4096 (1024 accumulation microbatches). Each
stops at 358 optimizer steps / 3,003,121,664 tokens. The LR follows the unchanged
11,921-step baseline schedule prefix, not a shortened cosine cycle. Native
Muon and Adam grouping, clipping and router objectives are inherited unchanged.

Evaluation uses the same 64 held-out sequences (131,072 tokens), at step zero,
every 32 steps and the end. This fixed small window is a paired pilot metric,
not a broad benchmark or sufficient evidence for close statistical rankings.
Checkpoints include model, FP32 optimizer master weights, momentum/Adam state,
scheduler and RNG, every 64 steps and the end. Resume reconstructs the token
cursor from the completed step. Campaigns are fresh-only; resumption is explicit.

## Existing runtime and compatibility settings

The user-approved sampling environment provides PyTorch 2.13.0+cu130,
Triton 3.7.1, Transformer Engine 2.17.1 and FLA 0.4.2. Native MCore 0.18.2
comes from the existing container checkout. Muon uses the container's already
cached Emerging Optimizers source at revision
`1effa026ff096b7fa1063ca2fba19d98be6e6cdf`, exposed on `PYTHONPATH` read-only.
No runtime packages are installed, copied into the project, or patched.

This environment lacks Apex, so gradient-accumulation fusion is disabled for
all arms. Its cuDNN fused-attention backward fails with `CUDNN_STATUS_BAD_PARAM`;
all native dot-product attention therefore uses TE's tested **unfused** backend.
Both settings are explicit in the launch contract. These runs compare
architectures in this DSW environment; their throughput is not a comparison of
fully optimized flash versus simplicial kernels, nor a matched DLC speed test.

## Launch and inspect

Use an immutable project source snapshot and inject existing machine paths:

```sh
export PYTHONPATH="$NGA_PILOT_SOURCE/src:$NGA_MEGATRON_ROOT:$NGA_EMERGING_OPTIMIZERS_ROOT"
"$NGA_PILOT_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=1 \
  --module archlab.megatron.simplicial_pilot --arm A --mode probe \
  --steps 3 --global-batch 8 --micro-batch 4 --eval-sequences 4 \
  --eval-interval 3 --save-interval 3 --run-dir "$NGA_PROBE_A" \
  --data-root "$NGA_DATA_ROOT" --tokenizer "$NGA_TOKENIZER_PATH"
```

Repeat for B/C with fresh probe directories and
`--initialization-reference "$NGA_PROBE_A/INITIALIZATION.json"`. Each probe
must complete optimizer updates, finite/nonzero gate/norm/extra-branch gradients,
and native save/reload after deliberate perturbation of model and optimizer
state. The campaign checks exact source hashes, paired initialization and token
order before accepting their `PROBE_COMPLETE.json` evidence.

```sh
"$NGA_PILOT_PYTHON" -m archlab.megatron.simplicial_campaign --detach \
  --campaign-dir "$NGA_PILOT_RUN_ROOT" --source-commit "$NGA_PILOT_COMMIT" \
  --probe-a "$NGA_PROBE_A" --probe-b "$NGA_PROBE_B" --probe-c "$NGA_PROBE_C" \
  --data-root "$NGA_DATA_ROOT" --tokenizer "$NGA_TOKENIZER_PATH"
```

The supervisor launches A, then B, then C, and never advances after a failed or
incomplete arm. `CAMPAIGN.json`, arm logs, `metrics.jsonl`, `HEARTBEAT.json`,
`RUN_CONTRACT.json`, `MODEL_SHAPES.json` and checkpoint markers are the evidence.
SIGTERM to the supervisor requests a checkpointed stop after the current
optimizer step via a run-local marker; this can take an entire accumulation
step. No DLC or Ollama processes are signaled. Operational artifacts, host
metadata and data paths are not intended for repository publication.

## References

- https://arxiv.org/html/2507.02754v1
- https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/

This implementation was written against the mathematical operation; it does
not vendor or patch Triton, Megatron, PyTorch, Transformer Engine or CUDA.

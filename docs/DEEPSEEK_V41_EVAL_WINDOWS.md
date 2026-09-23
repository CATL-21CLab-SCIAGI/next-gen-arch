# Fixed-baseline evaluation and persistent checkpoint chat

The normal baseline is permanently held at global optimizer **step 4537**, with
**756,364,650 supervised tokens**. Its full checkpoint is complete and pinned.
The previous step-5000 target is superseded. The simplicial trainer continues on
its own 16 B300 GPUs and retains an exact step-4537 checkpoint for a matched
comparison, while also saving and requesting evaluation every 500 steps.

The baseline's 16 GPUs run `archlab.automodel.deepseek_v41_checkpoint_service`.
This entry point creates no optimizer and contains no training-update loop.
It restores the pinned baseline once, validates its 64K held-out forward against
the recorded resident-training result, and retains those weights on GPUs.
It loads a complete simplicial checkpoint alongside it for paired evaluation
and persistent chat. No model weights are offloaded to CPU.

The portable policy is `recipes/experiments/deepseek_v41_eval_windows.yaml`.
Deployment resolves environment paths into JSON. The existing trainer protocol
still requires `chat_window_seconds`, but persistent serving ignores that timer.
Chat has no 30-minute expiration and cannot restart baseline training.

Individual HTTP requests have separate time limits. The shared deployment uses
EAS `metadata.rpc.keepalive=1860000` milliseconds and
`MLFLOW_GATEWAY_ROUTE_TIMEOUT_SECONDS=1800` seconds. The earlier 60-second EAS
limit truncated ordinary replies; a 155-second response now passes end to end.
These request deadlines do not stop serving or restart training.

The benchmark uses the sealed 1M held-out math targets, 1,140 MMLU questions,
128 ARC-Challenge questions, and 256 PIQA questions. Case, prompt, and tokenization
digests must match the original benchmark. Results carry each checkpoint's actual
step and token count; only the 4537/4537 pair is the requested matched comparison.
The independent 64K lightweight validation loop continues in the simplicial run.
A watcher publishes an additional evaluation request at step 4537, which lies
outside the regular 500-step grid.

When another simplicial evaluation request arrives, the service pauses chat,
releases the prior secondary model, restores the requested checkpoint, and runs
paired evaluation against the fixed baseline. Chat then remains available until
the next checkpoint refresh. Large checkpoint reads create a temporary availability
gap during refresh. `STOP_SERVING` in the baseline output ends the service.

MLflow Gateway exposes `deepseek-v41-normal-latest` and
`deepseek-v41-simplicial-latest` to the shared workspace using the existing team
authorization. The private backend implements `/v1/models` and
`/v1/chat/completions`, including streaming. Its response fingerprint identifies
the checkpoint actually being served. Normal remains step 4537; simplicial moves
to newly evaluated complete checkpoints.

Chat uses the pinned native non-thinking conversation encoder, at most 2,048
prompt-plus-output tokens, and at most 256 generated tokens. The native backbone
has no inference KV cache, so this full-prefix decoder favors numerical fidelity
over serving throughput. Reusable smoke prompts are versioned under
`src/archlab/prompts/deepseek_v41_chat_smoke.yaml`.

Checkpoint pins, pending evaluations, and reader leases protect active weights
from retention. Active mmap training datasets remain protected from offloading.
MLflow retains the existing two training run identities and checkpoint references;
no full model weights are uploaded to its artifact store.

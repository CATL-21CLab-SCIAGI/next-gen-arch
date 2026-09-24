# Stock Miles baseline

This restart consumes the upstream V4.1 recipe and unmodified `train.py` at
Miles `6c6858a42`. The V4 Flash recipe motivates FP8 rollout, but the finetuned
checkpoint has V4.1 geometry and requires the V4.1 model plugin. This experiment
uses upstream GRPO, distributed Adam, routing replay, and disk streaming of
BF16 serialized Adam moments. It does not use the previous Muown optimizer,
custom loss, custom driver, or resident admission machinery.

The compatibility boundary retains our full-checkpoint importer, trained
attention adapters, BF16 frozen Engram tables, and weight-name/transaction
mapping. FP8 mode delegates expert execution and quantization to SGLang and
Miles. It leaves the adapters in FP32 and Engram tables in BF16. Deterministic
Engram hash buffers survive serving weight discard/reload.

Launch with `archlab.megatron.miles_v41_stock_launch`, passing explicit runtime,
checkpoint, and Ray paths through its run directory. The launcher captures the
upstream recipe's arguments, supplies checkpoint compatibility hooks, and runs
the upstream driver. It preserves the dataset's already formatted prompts.
The versioned contract is `recipes/experiments/deepseek_v41_stock_fp8.yaml`.

The September 24 attempt uses four nodes / 32 B300 GPUs. All four nodes were
verified idle before launch. Each initially had about 1.53 TB of local disk free.
The initial attempt used NAS for offload and checkpoint storage; the retry below
uses local optimizer storage. Argument validation and 24 focused existing
serving regressions passed. FP8 full-model weight refresh, useful optimizer
updates, and checkpointing remain unqualified until runtime evidence exists.

The NAS attempt was retired before any update. Under allocation load a bounded
four-node probe measured only 20–26 MiB/s writes and 11–18 MiB/s reads per node.
Inspection of the upstream offload configuration showed that both gradient and
parameter buffers are discarded, not backed up. Estimated optimizer files plus
remaining Engram/FP32 offload state fit the local disks (roughly 1.3 TB on the
busiest node). The retry mounts named node-local temporary storage at the run's
`offload` path, inside the private execution namespace. Checkpoints and the
upstream weight backups remain on NAS. The mount checks filesystem type, initial
free capacity, and an empty project-local destination; seven regressions cover
these guards. No runtime library or optimizer implementation is modified.

The local-storage attempt completed all 32 parent imports and initialized all
four FP8 engines, leaving roughly 120 GiB per GPU after serving allocation.
It failed on the first weight packet because the transaction tracker assigned
to SGLang's read-only `ModelWeightParameter.weight_loader` property. The
compatibility tracker now wraps its existing backing loader and restores it
even on failure. Seven focused regressions and a GPU probe against the pinned
runtime passed. No rollout or optimizer update completed in that attempt.

The run is bounded to 64 rollouts with checkpoint interval 20. The stock save
sentinel requests an initial checkpoint after useful updates. Initialization or
completed generation alone is not healthy RL. Before leaving it unattended,
observe multiple finite nonzero updates, reward variation, stable policy-gap
metrics, a completed checkpoint, and practical measured step times. Status and
logs live in the explicit run directory; do not infer success from this document.

The next configuration removes the NAS policy-backup path entirely. Measured
trainer allocation peaked at about 92 GiB per GPU, while FP8 serving used about
148 GiB including the bounded KV cache. The actor and gradients therefore stay
on GPU (`--no-offload-train`); only stock Adam state streams to local disk.
Serving still discards its allocations during training. A narrowly scoped
adapter disables Miles's unconditional colocation backup and reads the live
actor for weight synchronization when there is no reference, teacher, or old
actor. This also makes the existing startup `_switch_model("actor")` a no-op,
instead of restoring the identical policy from NAS. Other configurations keep
upstream backup behavior. Full coexistence and training peaks remain subject
to runtime qualification.

A bounded 256 MiB probe isolated the native mapped-copy path: NAS copy plus
flush took 1.63 s versus 0.15 s on local storage; reopening after dropping the
mapping/cache and restoring took 1.50 s versus 0.13 s. The plain GPU-to-CPU
transfer was 0.074–0.075 s on both paths. A buffered NAS write plus flush took
0.64 s, so the mapped path adds overhead beyond sequential NAS bandwidth.
These are small single-process measurements, not a sustained 32-rank bandwidth
claim. The resident-actor change eliminates these policy-backup transfers
instead of moving another full copy onto nearly full local optimizer disks.
Twelve focused regressions, upstream argument validation, and a GPU probe of
the actual Miles actor's no-backup/no-restore/live-sync path passed.

The first resident run completed all 32 parent imports and skipped the NAS
backup/restore, but its first sync exposed Ray's pre-created method wrappers:
the generated subclass still called the original backup reader. The adapter
now covers those subclasses too. Thirteen focused tests and a GPU probe using
Ray's actual generated actor class pass. The retry uses `train-attempt2.log`
under `deepseek-v41-stock-fp8-resident-policy-20260924`; the original log and
receipts are retained. This fix still requires full refresh and RL qualification.

That retry reached full-policy coverage validation in about three minutes of
weight transfer, with 21–24 GiB of measured GPU headroom on an Engram training
stage. Validation rejected missing FP8 scales for the two Engram WKV
projections. The upstream V4.1 FP8 converter does not include these projections;
the compatibility iterator now sends them through the existing Miles FP8
quantizer. A GPU probe of both full-size 25600-by-6144 projections verified the
weight/scale names and native packed UE8M0 scales, with 2.65% reconstruction
error. No completeness checks were relaxed. The subsequent retry uses
`train-attempt3.log`; no successful RL update is implied by these probes.

Attempt 3 completed all four FP8 engine refreshes (1333 tensors per rank), then
generated 128 samples in 2049 seconds. Fifteen samples received reward 1;
response lengths ranged from 85 to 2001 tokens. Training rejected the batch
because all routing-replay records were missing. A controlled HTTP echo probe
proved that the installed SGLang router strips `return_routed_experts` before
forwarding requests. The same probe through the unmodified Miles router
preserved the request flag and response metadata. The launcher now selects
`--use-miles-router`; this also uses native active-request balancing. Replay
remains mandatory. Attempt 4 uses `train-attempt4.log`; the rejected rollout is
archived and no optimizer update from attempt 3 is claimed.

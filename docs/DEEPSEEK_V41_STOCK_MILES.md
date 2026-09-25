# Stock Miles baseline

The baseline passed operational qualification on September 25 in attempt 6:
three finite, nonzero updates, varied rewards, stable train/rollout policy gap,
and a complete native checkpoint with readback. The same process continues on
four nodes / 32 B300 GPUs, with FP8 rollout, frozen BF16 Engram tables, and
node-local disk optimizer state. The run retains its 64-rollout cap and native
20-rollout checkpoint interval. Evidence is in `QUALIFICATION.json` under
`results/deepseek-v41-stock-fp8-resident-policy-20260924`.

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
serving regressions passed. At that initial launch, FP8 full-model weight refresh, useful optimizer
updates, and checkpointing were still awaiting runtime qualification.

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

Attempt 4 validated routing records for all 128 samples and completed two
finite updates: gradient norms 0.193/0.309, train-rollout KL 0.00160/0.00187,
and log-probability absolute differences 0.0231/0.0278. Rollouts took 968/1006
seconds; the second training phase took about 5.5 minutes. The requested save
failed before writing: Miles's broad `"adapter"` name heuristic classified
`archlab_adapter` as LoRA and selected an adapter-only exporter requiring
Megatron Bridge. These two updates were not checkpointed.

The compatibility hook excludes only `.archlab_adapter.` from that heuristic
and preserves detection of real LoRA/PEFT modules. Both normal save dispatch
and explicit save-with-LoRA dispatch now reach the native full-model saver.
Seventeen regressions and a probe of the actual Miles dispatch passed; the
probe intercepts the final saver and is not a full checkpoint round trip.
Attempt 5 uses `train-attempt5.log` and requests a full checkpoint after its
first update. A completed full checkpoint is still required for qualification.

Attempt 5 completed one finite update (gradient norm 0.23525, train/rollout KL
0.001566, reward 18/128). All 64 optimizer manifests and 2,696 bucket files
(4.450 TB) reached NAS in about 37 minutes; sampled saved moments were finite
and nonzero. The model save then exhausted host NUMA memory, confirmed by the
kernel OOM log. One failed trainer had about 80 GB of shared memory resident.
This is an incomplete checkpoint and cannot be used as a training resume point.
It is retained under `checkpoints/attempt5-incomplete-iter_0000000`.

The synchronous MCore strategy still stages the entire rank's model through
its asynchronous writer before writing. `miles_v41_checkpoint_writer` supplies
a strategy through the existing checkpointing context that uses native
PyTorch `FileSystemWriter` with one thread, zero copy-ahead, and one tensor
per file. This pinned PyTorch retains `tensor_dict` until a file closes even
for torch serialization, so single-file-per-rank must also be disabled. MCore still
converts and validates shards and writes the completion metadata; its native
loader reads the same `torch_dist` format. CPU copies are made as individual
tensors are written, rather than staging the whole policy. The largest tensor
still determines the per-rank copy peak. Optimizer save behavior is unchanged.
No container library is modified.
An eight-B300 round trip using TMS-backed BF16 tensors passed through the
native distributed save and load APIs on every rank, including sharded objects
and common state (`serial-checkpoint-probe2.log`). This checks the writer and
format compatibility; the full RL checkpoint remains the launch gate.
The larger eight-B300 probe saved and reloaded 4 GiB per rank exactly. One-file-
per-rank writing grew peak RSS by 4.1–4.2 GiB; one-tensor-per-file writing reduced
that to 2.1–2.2 GiB with four 1 GiB tensors (`serial-checkpoint-probe-bounded.log`).
Four dispatch/contract tests and the 17 existing focused regressions passed.
Attempt 6 uses `train-attempt6.log`, with the stock initial-save sentinel retained.

Attempt 6 ran code `757dcf4` without another restart after qualification:

| RL step | Rewarded samples / 128 | Gradient norm | Train/rollout KL | Training seconds |
| --- | --- | --- | --- | --- |
| 0 | 20 | 0.191608 | 0.00162050 | 641.2 |
| 1 | 10 | 0.225659 | 0.00188834 | 334.6 |
| 2 | 10 | 0.148494 | 0.00201817 | 320.6 |

Rollouts took 972.2, 991.2, and 1,022.1 seconds; the third complete warm cycle
was 1,539.3 seconds (25.7 minutes), including weight refresh. All samples in
each rollout used a single policy version. Native KV request retractions
occurred under the 32K cache cap in rollout 2; all 128 routing-replay tensors
still had the expected token length, 40 layers, six experts, and valid IDs.
The 2,001-token response cap produced 45–78% truncation across these batches.
These observations qualify operation under the recorded limits.

The checkpoint after step 0 completed in 3,220.5 seconds (53.7 minutes).
Its 80,366 DCP files total 1.499 TB, with 80 architectural-adapter entries,
both Engram tables, backbone experts, embeddings, and the output head present.
All referenced file extents were checked; native PyTorch DCP read back four
adapter tensors with finite values. Optimizer verification checked all 64
manifests and 2,696 buckets (4.450 TB), plus 32 resident-FP32 optimizer files.
Sampled moments were finite and nonzero. This checkpoint contains the first
update; subsequent updates continue under the native checkpoint cadence.

During the save, ten-second host samples showed at least 236.9 GiB available
on every node. The highest sampled trainer RSS was 29.8 GiB, resolving the
previous whole-policy staging OOM. A snapshot during the next rollout showed
at least 9.1 GiB GPU memory free across the 32 B300s. Optimizer streaming still
uses local disk. Detailed receipts are `full-checkpoint-verification.json`,
`saved-optimizer-verification.json`, and
`attempt6-checkpoint-hostmem-summary.json` in the run directory.

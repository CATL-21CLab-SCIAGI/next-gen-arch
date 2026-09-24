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

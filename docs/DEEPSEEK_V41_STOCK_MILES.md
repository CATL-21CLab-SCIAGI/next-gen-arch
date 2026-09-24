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
verified idle before launch. Each has about 1.4 TB local disk free; offload
and checkpoint storage for this attempt is NAS, not an assumed NVMe volume.
Its performance must be measured. Argument validation and 24 focused existing
serving regressions passed. FP8 full-model weight refresh, useful optimizer
updates, and checkpointing remain unqualified until runtime evidence exists.

The run is bounded to 64 rollouts with checkpoint interval 2. Initialization or
completed generation alone is not healthy RL. Before leaving it unattended,
observe multiple finite nonzero updates, reward variation, stable policy-gap
metrics, a completed checkpoint, and practical measured step times. Status and
logs live in the explicit run directory; do not infer success from this document.

# Runtime and Backends

The current RL interface is the [package-backed training workflow](../TRAINING.md). verl owns training, rollout scheduling, checkpoints and monitoring.

| Component | Role | Required evidence |
| --- | --- | --- |
| verl | RL execution and serving coordination | Data, reward, rollout identity, policy lag and optimizer/restart checks |
| Megatron Core / Transformer Engine | Distributed training and retained pretraining | Model construction, optimizer grouping, topology, checkpoint and numerical checks |
| TileLang attention | Architecture-specific attention kernels | Forward/backward parity and end-to-end throughput for the model geometry |
| Speedrun | Frozen small-model reference | Historical arguments, packing, optimizer and data order |
| AutoModel references | Historical model loading and numerical implementations | Model-specific reference tests; no current custom RL executor |

Megatron and TileLang are not competing scheduler backends: the former distributes training, while the latter implements kernels. A configuration name or successful import does not qualify a model/backend combination.

Use the validated container's PyTorch, CUDA, NCCL, Transformer Engine and Megatron packages. Record the container identity, imported paths, resolved versions and upstream revision. Production runs use immutable qualified source snapshots.

## Parallelism

DP replicates data lanes; TP splits tensors; PP splits layers; CP splits context; EP partitions experts. FSDP shards model state across its owner group. Record the actual groups and measure throughput: more parallelism may save memory while slowing execution.

## Historical evidence

The matched 10M study found architecture-delta correlation 0.971361 and matching directions for 13/15 variants. The backends were not numerically interchangeable. Speedrun retained a cold-start advantage; Megatron improved steady-state throughput in its qualified configuration.

Legacy AutoModel execution, Miles RL and SGLang prototype launch paths are retired. See [Code Map](Code-Map.md) for the archive boundary and [Results](Results.md) for dated evidence.

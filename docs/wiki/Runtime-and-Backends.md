# Runtime and Backends

Current DeepSeek RL launch and qualification entry: [Miles baseline](../MILES_BASELINE.md).

## Supported roles

| Path | Role | Qualification boundary |
| --- | --- | --- |
| Speedrun | Cold small-model screens and historical regression oracle | Frozen arguments, packing, optimizer and data order |
| Megatron Core | Distributed pretraining and scaling | Model, optimizer grouping, topology, checkpoint and numerical tests |
| NeMo AutoModel | Pretrained Qwen/DeepSeek integration and post-training | Pinned upstream source, complete weight loading, FSDP/EP behavior |
| SGLang integration | Serving/export experiments | Checkpoint coverage, model plugin, cache behavior and inference parity |

The CLI backend registry covers speedrun and Megatron. AutoModel and serving use dedicated entries.

## Runtime contract

Use the validated container's interpreter and installed PyTorch, CUDA, NCCL, Transformer Engine, and Megatron packages. Record the container identity, imported paths, resolved versions, and upstream revision.

A support table or successful import is insufficient evidence for a new model/topology.

## Parallelism

DP replicates data lanes; TP splits tensors; PP splits layers; CP splits context; EP partitions experts. FSDP shards model state across its owner group.

Record actual DP, TP, PP, CP, EP, expert-FSDP and table-owner groups. Expert count is an architecture choice; EP size is an execution choice.

More model parallelism can save memory while reducing speed. Choose topology from measured capacity and throughput.

## Historical backend evidence

The matched 10M study found architecture-delta correlation 0.971361 and matching directions for 13/15 variants. The backends were not numerically interchangeable. Speedrun retained a cold-start advantage; Megatron improved steady-state throughput in its qualified configuration.

See [[Results]] and the versioned `docs/BACKEND_COMPARISON.md` report.

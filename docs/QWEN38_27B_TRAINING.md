# Dense Qwen3.8 pretraining

**Type:** native Megatron training reference.

| Boundary | Contract |
| --- | --- |
| Optimizer | Container-registered Muon; native Adam fallback for embeddings, head and non-matrices |
| Compute/state | BF16 model/activations; FP32 optimizer masters |
| Muon matrix precision | `medium`; FP32 state retained |
| Orthogonalization | Native Polar Express, eight Newton–Schulz steps |
| LR | Peak 5e-5; minimum 5e-6 |
| Distribution | Native distributed optimizer and communication overlap |
| Optimizer CUDA graph | Disabled for this qualified path |

The model tags optimizer routing; it does not replace Megatron's optimizer implementation.

## Launch gate

The recorded quarter-scale launcher runs a 400-step production-shaped preflight at context 2048, microbatch 4 and global batch 512. This crosses the earlier BF16-Muon failure point.

Use explicit source/data/output variables and the named launch recipe. The allocation/controller lifecycle is separate from trainer success.

[PIQA and scale evidence](QWEN38_PIQA_AND_EARLY_CURVES_20260904.md) · [Runtime guide](RUNTIMES.md)

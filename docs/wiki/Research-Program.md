# Research Program

**Question:** Which architectural changes improve useful capability for their training and inference cost?

## How the scope expanded

| Stage | Work |
| --- | --- |
| Component screens | Frozen architecture grid, paired seeds, small-model controls |
| Distributed scaling | Megatron integration, backend reproduction, parallelism and precision studies |
| Model development | Qwen pretraining and scaled hybrid backbones |
| Frontier-model adaptation | Qwen and DeepSeek with additional architectural branches |
| RL | Outcome-driven training from matched DeepSeek checkpoints |

The earlier stages remain references for later work.

## What we vary

| Axis | Examples |
| --- | --- |
| Model mechanism | Attention, GDN/KDA mixers, Engram/PLE memory, residual routing |
| Architecture composition | Width, depth, expert count, placement, component combinations |
| Training | Trainable parameters, optimizer, supervised objective, RL objective |
| Execution | Backend, kernels, precision, sharding, compilation |
| Evaluation | Held-out prediction, task accuracy, generation cost, general capability |

Declare each changed axis. Kernel speed, lower prediction loss, and task improvement are separate findings.

## Architecture adapters

An adapter adds a mechanism at a defined pretrained-model boundary. It may be trained with a frozen backbone or followed by full-weight fine-tuning.

Record the insertion equation, initialization, parameter overhead, trainable set, and ordinary control. After full-weight fine-tuning, disabling an adapter does not restore the released model.

**Next:** [[Experiment Design|Experiment-Design]] · [[Qwen Experiments|Qwen-Experiments]] · [[DeepSeek Experiments|DeepSeek-Experiments]]

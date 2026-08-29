# Next-Gen Architecture Lab

[English](README.md) · [结果](docs/RESULTS.md) · [实验合同](docs/EXPERIMENT_CONTRACTS.md) · [数据与产物溯源](docs/PROVENANCE.md) · [运行手册](docs/OPERATIONS.md) · [后端](docs/RUNTIMES.md) · [架构](docs/ARCHITECTURES.md)

这是一个受控语言模型架构实验仓库：小规模实验使用快速 PyTorch backend，扩缩实验使用容器提供的 Megatron Core。架构机制、优化器、数据顺序、评测和执行后端均作为独立变量处理。

## 三类实验不能混用

| 类型 | 固定项 | 回答的问题 |
| --- | --- | --- |
| `controlled` | 训练 token、seed、数据顺序、优化器、评测 | 单一组件是否改善质量？ |
| `fixed_compute` | 总 algorithmic model FLOPs | 单位计算量的质量是否改善？ |
| `scaling` | tokens/parameter | 整套架构如何随规模扩展？ |

不同类型的 BPB 不合并成因果 leaderboard。参数开销、executed FLOPs、吞吐、显存、失败与 seed 波动必须同时报告。

## 两个后端

- `speedrun`：冻结的 nanochat/modded-nanogpt 对照后端，保留已发布实验的 Muon、编译、packing 和数据顺序。
- `megatron`：使用 `nemo-26.06` 容器提供的 Megatron Core，负责分布式执行与 checkpoint 生命周期；本仓库不复制或修改 Megatron。

speedrun 中可运行的机制不自动等于 Megatron 原生支持。构造、优化器分组、checkpoint 和分布式数值行为均通过测试后才会提升能力等级。

## 当前证据

- Qwen GDN 在已完成的成熟规模上质量较强，但吞吐代价较大。
- Engram 是 100M/300M 上较好的质量—速度折中。
- KDA 系列经常改善 BPB，但当前实现明显更慢。
- 小幅收益可能随 backend 或规模改变；组合只由已通过独立对照的组件晋级。
- mHC、relative attention 等数值失败保留为失败，不用更早的有利 checkpoint 替代。

完整表格、日期、限制和产物位置见 [docs/RESULTS.md](docs/RESULTS.md)，机器可读证据保存在 [`results/`](results/)。

## 新实验最低产物标准

可报告的新 run 必须保存：

- clean commit 与 worktree hash；
- 内容级 dataset manifest 和 tokenizer vocabulary hash；
- comparison regime、离散 token/FLOP 预算、配对 seed 与 data-order identity；
- 共享参数初始化 hash；
- 原始 JSONL metrics、稳定 run ID 与唯一 attempt ID；
- 最终模型、optimizer、RNG，以及可重建的 dataloader cursor；
- 完整 wall time 与预先声明的稳态吞吐窗口；
- 明确的失败类型；只有临时基础设施失败可以原配置重试。

FineWeb 每次评测都会重放同一个 validation token 窗口；恢复训练时根据 checkpoint iteration 精确重建 distributed-microbatch cursor。

## 快速开始

```bash
git clone https://github.com/CATL-21CLab-SCIAGI/next-gen-arch.git
cd next-gen-arch
uv sync --extra cpu --group dev
uv run next-gen-arch verify
uv run pytest -m "not slow" -q
```

创建并完整校验数据 manifest：

```bash
uv run next-gen-arch data-manifest create \
  --root /path/to/data \
  --dataset owner/dataset \
  --revision <immutable-revision> \
  --pattern '*.bin' \
  --output /path/to/dataset.manifest.json

uv run next-gen-arch data-manifest verify \
  --root /path/to/data \
  --manifest /path/to/dataset.manifest.json \
  --mode full
```

查看冻结实验或渲染可移植 recipe：

```bash
uv run next-gen-arch list
uv run next-gen-arch show --size 300m --variant engram --seed 42
uv run next-gen-arch render \
  --config recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml
```

GPU 训练使用目标容器中的 PyTorch、CUDA、Transformer Engine、NCCL 和 Megatron。启动前在容器内执行 `next-gen-arch doctor --backend megatron`。

## 目录

```text
src/archlab/architectures/  架构机制
src/archlab/optimizers/     本地优化器扩展
src/archlab/megatron/       唯一 Megatron 集成边界
src/archlab/speedrun/       冻结的小规模参考后端
recipes/                    可移植实验配置
results/                    不可变机器可读证据
docs/                       方法、运行、结果与历史说明
tests/                      单元、数值、恢复与分布式测试
```

项目采用 MIT License；第三方论文与机制归属见 [NOTICE.md](NOTICE.md)。

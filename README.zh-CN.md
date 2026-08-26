# Next-Gen Architecture Lab

[English](README.md) · [完整结果](docs/RESULTS.md) · [10M 后端对比](docs/BACKEND_COMPARISON.md) · [优化审计](docs/OPTIMIZATION_AUDIT.md) · [复现指南](docs/REPRODUCIBILITY.md) · [运行后端](docs/RUNTIMES.md) · [架构说明](docs/ARCHITECTURES.md)

这是一个兼顾快速受控实验与 Megatron 扩缩的语言模型架构实验仓库。它把冻结 campaign 的 16 种架构机制与一个探索性的 CoLU arm 放进统一实验合同，在相同数据、tokenizer、seed、数据顺序、训练预算和评测方法下，回答一个具体问题：

> 当其他变量都被固定后，某个架构改动是否真的降低了验证集 BPB？

仓库包含统一后的训练代码、100M/300M/1B 三档参数规模与三个 seed 的冻结 144-run 合同、早期固定 token 对照、机器可读结果、测试和 CI/CD；不包含模型权重、数据集副本或私有基础设施配置。

## 背景与目标

很多架构论文比较的是整套系统，因此质量提升难以归因到单一组件。本项目将 token mixer、残差拓扑、记忆模块、注意力改造和激活函数等机制拆开，移植到共享 backbone 中进行成对对照。

项目目标是：

- 提供简洁、可读、可训练的统一实现；
- 冻结实验合同，让每个结果能追溯到命令、参数量、数据与 tokenizer 指纹；
- 同时报告质量、吞吐和参数开销，不用单一 BPB 掩盖工程代价；
- 公开失败与负结果，特别是 NaN、harness bug 和不适用的实现；
- 为后续组合实验建立可做因果归因的 baseline 与单组件对照。

## 双后端边界

- `speedrun` 保留继承自 [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) 和 campaign nanochat fork 的 Muon、编译、数据顺序及 kernel 优化，是已发布 100M–1B 结果的对照后端。
- `megatron` 使用已验证 `nemo-26.06` 容器提供的系统 MCore 训练生命周期。16 种机制均已通过约 10M、三 seed 的匹配对照；它负责后续扩缩，项目不复制、不修改、也不随包分发 Megatron 源码树。

配置按 `base + backend + scale + experiment` 分层合并。路径使用 `env:NAME` 或 CLI override，sampling prompt 是带版本的包内资产。未完成 Megatron adapter 的变体会明确拒绝运行，不会把同名架构当作数值等价实现。10M 对比采用 `TP=PP=CP=1`，验证的是 MCore wrapper 与架构信号，并不代表每种机制已经原生支持 tensor/pipeline/context parallelism。

## 核心结果

主指标是验证集 **BPB（bits per byte，越低越好）**。`Δ BPB` 是相对同规模、同 seed baseline 的成对差值，负数代表改善。

### 约 10M 参数的 Megatron—speedrun 对比

B300 safe-autotune 对比包含 `16 个变体 × 3 个 seed × 2 个后端 = 96` 个采纳
run。15 个非 baseline 变体的成对 ΔBPB 跨后端 Pearson 相关系数为 `0.971361`，
改善/退化方向一致 `13/15`，平均绝对 delta 差为 `0.009241 BPB`。

| 后端 | 最佳变体 | 平均 BPB | 成对 Δ BPB | baseline tok/s |
| --- | --- | ---: | ---: | ---: |
| Megatron safe-autotune | KDA | **1.465789** | -0.081969 | 1,576,266* |
| speedrun | Kimi K3 KDA | **1.461594** | -0.093387 | 1,360,228 |

`*` Megatron 使用稳态 step 中位数；其 warmup 后聚合吞吐为 `1,529,304 tok/s`
（speedrun 的 `1.124×`）。但新建 max-autotune cache 时，完整进程耗时为
`567.0 s`，speedrun 仅 `79.8 s`，因此一次性小模型筛选仍应使用 speedrun；
只有编译成本被复用或摊薄后，Megatron 的稳态优势才成立。max-autotune 对
KDA、Kimi K3 KDA、Qwen GDN 不安全，这 9 个采纳 run 使用默认
`torch.compile`，其余 39 个使用 max-autotune。完整表格和数值安全审计见
[后端对比报告](docs/BACKEND_COMPARISON.md)。

### FineWeb 1M 单卡 B300 筛选

seed-42 Megatron 筛选在 FineWeb/GPT-2 上运行全部 17 个 arm；全局 batch 为 192，
在单张 B300 上拆成 `16 条序列 × 12 个梯度累积 microbatch`。所有 run 均来自
clean commit `36a6830`，在 1,231 秒内 `17/17` 完成，loss 与 gradient 全部有限。

| 变体 | 最终 BPB | 相对 baseline | 稳态吞吐 |
| --- | ---: | ---: | ---: |
| Engram | **2.167429** | **-0.334663** | 0.961× |
| Kimi K3 KDA update | 2.178213 | -0.323880 | 0.301× |
| GLM MLA | 2.179100 | -0.322992 | 0.968× |
| Baseline | 2.502092 | — | 1.000× |
| CoLU | 3.010535 | +0.508443 | 0.968× |

这是单 seed、仅约 1M 参数的 37-step 筛选，baseline 对 microbatch 累积顺序高度
敏感；它适合验证 runtime 兼容性和提前淘汰，不应覆盖 100M/300M 的多 seed 证据。
完整 17-arm 表格与冻结合同见
[FineWeb 1M 结果目录](results/fineweb10b-1m-b300-dsw/)。

### 单个三节点进程组上的 100M baseline

seed 42 baseline 使用一个 `3 节点 × 每节点 5 张 B300 = 15 ranks` 作业；节点内
使用 NVLink，节点间已从 NCCL 日志确认 RoCE/GDRDMA。即使 192 不能整除 15，
exact replay 仍保持每步恰好 192 条有效序列。

| 后端 | 最终 BPB | 曲线一致性 | 稳态中位吞吐 |
| --- | ---: | ---: | ---: |
| speedrun | **0.91615027** | 相对历史 `r=0.99999787` | 1,906,064 tok/s |
| Megatron | 0.91756084 | 相对 speedrun `r=0.99983861` | **1,940,137 tok/s** |

speedrun 最终值与历史 seed-42 结果仅差 `-0.00021621 BPB`。Megatron 与当前
speedrun 曲线的平均绝对差为 `0.00122598 BPB`，最终高 `0.00141057 BPB`。
Megatron 稳态 step 中位数为 speedrun 的 `1.018×`，但含生命周期过渡的聚合
吞吐只有 `0.892×`；两者稳态 kernel 基本打平，运行开销仍不相同。完整结果见
[100M 三节点结果目录](results/100m-multinode-b300/)。

### 单节点 B300 上的 100M 级原生并行

8 个 seed-42 Megatron run 依次训练 7,372.8 万 token。DP5/PP5 使用 5 卡；
TP/CP/EP 的 factor-two 对照使用 4 卡，从而不改变 6 个 attention heads 与
6 个 experts。稳态吞吐中位数统一去掉前 10 步。

| 对照 | 稳态吞吐比 | 峰值显存变化 | 结论 |
| --- | ---: | ---: | --- |
| PP5 / DP5 | 0.617× | -44.6% | 是容量路径，不是 100M 默认配置 |
| TP2+DP2 / DP4 | 0.389× | -48.6% | 是容量路径，不是 100M 默认配置 |
| fused DP4 / unfused DP4 | **1.274×** | -26.9% | B300 上有效的性能优化 |
| CP2+DP2 / fused DP4 | 0.366× | -22.7% | 仅在长上下文需要时启用 |
| MoE EP2+DP2 / EP1+DP4 | 0.980× | -7.1% | 6 experts 下没有吞吐收益 |

8 个 run 均完成 200/200 steps，skip 与非有限迭代均为 0。本 harness 使用
indexed FineWeb-Edu 对照数据与匹配的 DeepSeek-V3 tokenizer，因此其 validation
LM loss 不能与 ClimbMix/nanochat BPB campaign 合并比较。完整报告与机器可读曲线见
[单节点原生并行结果目录](results/100m-native-parallelism-b300-1n/)。

### 参数缩放实验：约 12 个训练 token / 参数

| 规模 | 方案 | 平均 BPB | Δ BPB | 相对吞吐 | 有效 seed |
| --- | --- | ---: | ---: | ---: | ---: |
| 约 100M | Qwen GDN | **0.902994** | **-0.013177** | 0.42× | 3 |
| 约 100M | Engram | 0.908589 | -0.007582 | **0.95×** | 3 |
| 约 100M | Kimi K3 KDA | 0.909599 | -0.006573 | 0.40× | 3 |
| 约 300M | Qwen GDN | **0.799714** | **-0.008303** | 0.53× | 3 |
| 约 300M | Kimi K3 KDA | 0.802384 | -0.005633 | 0.51× | 3 |
| 约 300M | Engram | 0.803945 | -0.004072 | **0.97×** | 3 |

目前成熟规模中，Qwen GDN 的 BPB 最低；Engram 的质量—速度折中最好：100M/300M 参数分别增加约 12.2%/7.6%，同时保留约 95%/97% baseline 吞吐。

### 早期固定约 10 亿训练 token 的对照

| 深度 | baseline BPB | 最佳方案 | 方案 BPB | Δ BPB |
| --- | ---: | --- | ---: | ---: |
| d14 | 0.843770 | aligned baseline | 0.843770 | — |
| d16 | 0.830908 | mHC | **0.826490** | **-0.004419** |
| d18 | 0.820979 | mHC | **0.817117** | **-0.003862** |

这组结果使用固定训练 token 预算；上面的参数缩放实验使用约 12 token/参数。**两类实验的绝对 BPB 不能合并排序。**

100M 实验经过两次独立重复：variant 相对 baseline 差值的相关系数为 `0.9990`；去掉已知不稳定的 relative-attention 后，平均绝对差仅 `0.000105 BPB`。

## 1B 当前状态

截至 **2026-08-24（UTC+8）**，1B 参数实验 48 个 run 中完成 18、失败 3、运行中 21、等待 6。已完成的六个三-seed方案中，AttnRes 暂时最好：`0.706122 BPB`，相对 baseline 为 `-0.003726`。

这不是最终 1B 榜单：Qwen GDN 当时尚未开始，Kimi K3 KDA 仍在运行。已知问题包括：

- 三个 1B mHC run 全部出现非有限值；
- Inkling relative attention 的两个 run 出现非有限值；
- 三个 Engram run 因 d12 meta-reference 错用了 d32 的 `7,15,23` 注入层而触发断言。这是 harness bug，不是模型质量失败。仓库已按深度同比例映射 reference 层。

完整数据见 [docs/RESULTS.md](docs/RESULTS.md)、
[results/key-metrics.csv](results/key-metrics.csv) 与
[B300 safe-autotune 结果目录](results/megatron-10m-safe-autotune-b300/)；
[100M 三节点结果目录](results/100m-multinode-b300/) 保存了本次学习曲线与来源审计。

## 实验合同

| 维度 | 固定值 |
| --- | --- |
| 数据 | ClimbMix，campaign 使用 171 个 shard |
| Tokenizer | 固定 32,768-token nanochat BPE |
| 序列长度 | 2,048 |
| 精度 | BF16 |
| Seed | 42、43、44 |
| 数据顺序 | 同 seed 成对一致 |
| 总 batch | 393,216 token |
| 预算 | 约 12 个训练 token / 参数 |
| 优化器 | per-head Muon + Adam 参数组 |
| 主指标 | 验证集 BPB |

公开 manifest 保存了全部 144 个 run 的命令参数、探测参数量、源码树 hash、数据指纹、tokenizer hash 以及软硬件信息。公开副本已将内部路径和主机名替换为可移植标签。

## 快速开始

需要 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/CATL-21CLab-SCIAGI/next-gen-arch.git
cd next-gen-arch
uv sync --extra cpu --group dev
uv run next-gen-arch verify
```

CPU extra 仅用于结果检查和单元测试。GPU 训练使用已验证的 NeMo 容器，以
`--system-site-packages` 创建环境，并在启动前运行
`next-gen-arch doctor --backend megatron`。项目不会解析或替换容器内的 PyTorch、
Transformer Engine 与 Megatron 包。查看实验轴与重建冻结命令：

```bash
uv run next-gen-arch list
uv run next-gen-arch show --size 300m --variant engram --seed 42
uv run next-gen-arch command --size 300m --variant engram --seed 42 --run-name my-run
```

准备数据与 32K tokenizer：

```bash
export NANOCHAT_BASE_DIR=/path/to/next-gen-arch-data
uv run python -m archlab.speedrun.dataset -n 170
uv run python -m archlab.speedrun.tok_train --vocab-size 32768
```

渲染可移植 speedrun 合同：

```bash
export NGA_DATA_ROOT=/path/to/next-gen-arch-data
uv run next-gen-arch render \
  --config recipes/experiments/speedrun_qwen_gdn_100m_seed42.yaml
```

Megatron 示例使用 `NGA_TRAIN_DATA`、`NGA_VALID_DATA`、`NGA_DATA_CACHE`、`NGA_TOKENIZER` 和 `NGA_OUTPUT_DIR` 注入机器路径，再渲染 `recipes/experiments/megatron_baseline_1b_seed42.yaml`。

汇总 Megatron 10M campaign 并与冻结 speedrun 参考比较：

```bash
uv run python -m archlab.speedrun.campaign_compare \
  --megatron-root /path/to/megatron-results \
  --reference results/speedrun-10m-reference.csv \
  --output-dir /path/to/comparison
```

新训练的 tokenizer 适合新实验，但不保证与冻结 campaign 二进制完全一致。严格对照前请核对 [复现指南](docs/REPRODUCIBILITY.md) 中的数据与 tokenizer 指纹。

本地质量门禁：

```bash
uv run ruff check src/archlab tests
uv run python -m compileall -q src/archlab
uv run pytest -m "not slow" -q
uv build
```

## 长任务保护

训练循环会对 validation loss、training loss 和 gradient 做 NaN/Inf fail-fast，并跨 distributed rank 同步。默认每步扫描梯度；可以用 `--finite-check-every=N` 降低频率，`0` 仅关闭梯度扫描，loss 与 validation 检查仍会执行。

## 当前判断

1. Qwen GDN 是已完成 100M/300M 的质量领先方案，但吞吐代价大。
2. Engram 是当前最佳质量—速度折中。
3. Kimi K3 KDA 在约 10M–300M 持续有效，但速度较慢，1B 尚未完成。
4. sconv-KV 和 gated attention 是稳健的低额外参数改造。
5. mHC 的固定-token结果很好，但缩放设置数值不稳定，暂不应视为可靠方案。
6. 当前 DSA 与 relative attention 不适合 2K-context 实现；DSA 仍是 masked dense SDPA，不能证明真实稀疏 kernel 的速度收益。
7. 10M 架构信号总体可跨后端迁移，但后端并非无关变量：两个小效应改变方向；safe-autotune Megatron 稳态更快，speedrun 的冷启动一次性实验快得多。

## 路线图

- 完成 1B Qwen GDN、Kimi K3 KDA 和 sconv-KV 三-seed对照；
- 在不改变其余合同的前提下重跑已修复的 Engram；
- 先完成 mHC 稳定性消融，再扩大规模；
- 实现真实稀疏 DSA kernel，并补充长上下文评测；
- 将优先机制改造成 Megatron tensor/context parallel 原生实现，并在多 rank 上验证；
- 所有组合实验保留 baseline 与各单组件对照；
- 发布更完整的训练曲线和硬件归一化效率数据。

所有 Python 代码集中在 `src/archlab`：`architectures/` 只保留按架构族合并的模型定义，`megatron/` 是唯一的 MCore 适配边界，`optimizers/` 放置新增优化器，`speedrun/` 冻结继承自 modded-nanogpt 的轻量训练路径，`prompts/` 保持 prompt 可移植。实验参数统一放在 `recipes/`，Megatron 由容器作为系统依赖提供。当前 16 种机制已验证单 rank MCore wrapper，100M baseline 也已完成三节点 15-way data parallel；但尚不能据此声称全部机制支持 Megatron tensor、pipeline、expert 或 context parallelism。项目以 [MIT License](LICENSE) 发布；各架构名称和论文归原作者所有，详见 [NOTICE.md](NOTICE.md)。

---
difficulty: 中等
topic: MoE基础/结构 容量与部署权衡
summary: MoE如何用稀疏专家扩容量及为何部署不一定更快
tags: [真题, MoE, Dense, 路由, 专家并行, 待校对]
company: 蚂蚁集团、字节、小红书、美团、腾讯、蚂蚁
mastered: false
highfreq: false
---

## 题目

请解释 Dense 与 Mixture of Experts 模型的结构差异：MoE 的专家、路由器和 Top-k 怎样工作？比较两者的总参数、激活参数、FLOPs、显存、通信、训练稳定性和推理延迟，并说明专家数与 Top-k 应怎样选择。

## 要点

- Dense 对每个 token 使用同一套 FFN，MoE 只激活少数专家 FFN
- 稀疏激活把总容量与每 token 专家计算部分解耦
- 权重仍要存放，路由和 All-to-All 也有成本，理论 FLOPs 不等于实际延迟
- Top-k、专家数和容量系数没有通用最佳值，必须结合负载与拓扑验证

## 答案

**MoE 通常把 Transformer 的一个 Dense FFN 换成多个专家 FFN，再让路由器为每个 token 只选择少数专家。** 若路由器给专家 $e$ 的分数为 $p_e(x)$，选中集合为 $S_k(x)$，输出可写成

$$
y=\sum_{e\in S_k(x)}g_e(x)E_e(x).
$$

Top-k 的索引选择是离散的，但被选专家的网络和门值仍能从主任务损失反向传播；路由器也常配负载均衡目标。未选专家从该 token 得不到同样的主路径梯度，所以可能出现热门专家拥塞、冷专家训练不足和 token 丢弃等问题。

| 维度 | Dense | 稀疏 MoE |
|---|---|---|
| 参数使用 | 每 token 经过同一套权重 | 总专家很多，每 token 只经过 Top-k |
| 容量与计算 | 总参数和活跃计算绑定较紧 | 专家总容量可增长而活跃计算增长较慢 |
| 权重显存 | 只存一套 FFN | 仍需容纳或分片全部专家权重 |
| 并行通信 | 常规张量/流水并行 | 专家并行常需 token 的 All-to-All |
| 稳定性 | 路径固定、较易调试 | 还要处理负载、容量与路由抖动 |

MoE 在相近专家 FLOPs 下可提供更大的总参数容量，但不会自动降低单请求延迟：小 batch 时专家矩阵乘可能太碎，跨卡 All-to-All、负载不均、路由与权重读取都可能吞掉收益。吞吐还取决于 token 是否能按专家聚合成足够大的批次。推理可从专家放置、容量规划、通信与计算重叠、热点复制和量化入手，但每项都应实测。

设计 64 个专家时，我不会直接背 Top-1 或 Top-2。先根据质量目标、每专家宽度、设备数、网络带宽和容许溢出设基线，再观察每专家 token 数、丢弃率、All-to-All 时间和任务质量。Top-k 增大通常提高每 token 的专家计算并可能增加组合能力，也加重通信。Switch Transformer 研究了 Top-1 路由；GShard 的代表性配置使用 Top-2；MegaBlocks 重点是用块稀疏计算高效执行不均匀专家负载，而不是发明一个通用新 Top-k。共享专家可持续处理通用模式，细粒度专家提供更多可组合的专门分支，但都要重新核算活跃宽度和路由负载。GPT-4 官方未公开专家数量、Top-k 等架构细节，不能把传闻写成事实。

## 知识点

MoE 的核心是“总参数多、每 token 只激活少数专家”。激活参数、FLOPs、端到端显存和延迟是四种不同口径，任何固定倍数都必须绑定具体模型和系统。

- 一手依据：[Switch Transformer](https://arxiv.org/abs/2101.03961)、[GShard](https://arxiv.org/abs/2006.16668)、[MegaBlocks](https://arxiv.org/abs/2211.15841)、[DeepSeek-V2 / DeepSeekMoE](https://arxiv.org/html/2405.04434)、[GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)。

## 追问

- Top-k 是离散选择，路由器为什么仍能训练？
- MoE 在什么情况下会比 Dense 更慢？
- 专家坍塌和负载不均怎样发现、怎样缓解？
- 设计 64 个专家时，Top-k 与容量应依据哪些指标选择？
- 专家并行为什么常出现 All-to-All，推理显存怎样优化？

## Note

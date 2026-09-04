---
difficulty: 中等
topic: verl/批次参数
summary: verl中prompt、response、mini-batch与微批怎样换算
tags: [面经, 待校对, verl, PPO, 分布式训练]
company: 蚂蚁金服、蚂蚁
mastered: false
highfreq: false
---

## 题目

在 verl 的同步 PPO/GRPO 训练中，prompt batch、`rollout.n`、actor 更新 mini-batch、每卡 micro-batch 与 PPO epoch 之间怎样换算？平台题所写的 `rollout batch size`、`global batch size`、`micro_batch_size_per_device_for_update` 是否都是当前 verl 的正式参数，异构并行时又该怎样计算？

## 要点

- 先纠正三个并非当前公开配置字段的泛化名称
- 分清 prompt 组数和生成 trajectory 数
- 数据并行度只除一次，TP/PP 不扩大统计 batch
- 梯度累积次数不等于物理通信间隔
- 动态微批按 token 打包，不存在固定累积公式

## 答案

**当前快照里，题目的三个长名称都不能直接当作 verl 的 PPO 配置字段。** 常规同步路径使用 `data.train_batch_size` 表示每轮全局 prompt 数，`actor_rollout_ref.rollout.n` 表示每个 prompt 生成的 response 数，`actor_rollout_ref.actor.ppo_mini_batch_size` 表示一次 actor 更新包含的全局 prompt 组数，推荐用 `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` 表示每个 actor 数据并行 rank 一次前反向处理的本地 trajectory 数。

设 prompt batch 为 $B$，每题采样数为 $n$，actor mini-batch 的 prompt 组数为 $M$，actor 数据并行度为 $D$，每卡微批 trajectory 数为 $m$，PPO epoch 为 $E$。普通一问一答同步路径中：

$$
N_{trajectory}=Bn,\qquad M_{trajectory}=Mn
$$

所以每个 epoch 的 optimizer step 数是 $B/M$，总 step 数是 $E B/M$。每个数据并行 rank 在一次 optimizer step 中处理 $Mn/D$ 条 trajectory；固定微批下，梯度累积次数为：

$$
A=\frac{Mn}{Dm}
$$

这里 $D$ 已是 actor 的数据并行度，不能再乘一次 GPU 总数。TP、PP、CP 等是在多张卡上分片处理同一份数据，不增加统计 batch。梯度累积描述的是一次 optimizer step 前有几次前反向；实际每次 backward 是否通信由 FSDP、Megatron 等后端决定，不能把 $A$ 直接叫“梯度同步间隔”。

平台题里的 `rollout batch size` 可能指每轮 prompt 数、一次生成分块或推理引擎瞬时容量，三者不同；`global batch size` 也必须说明是在数 prompt 还是 trajectory。`micro_batch_size_per_device_for_update` 是跨框架说法，不是当前字段。旧的全局微批配置仍可能在历史示例中出现，但当前推荐每卡字段，二者不能同时设置。启用动态微批后，框架按每卡 token 上限和序列长度打包，每个微批的 trajectory 数会变化，此时没有固定的 $A$。

OOM 时先定位阶段。actor 训练 OOM，优先减每卡 micro-batch，或用动态 token 上限、激活重计算、分片与卸载；这些手段可通过累积保留算法级 mini-batch。rollout OOM 则调生成侧并发、token/KV cache 上限、最大回答长度及推理并行。盲目缩小 prompt batch 会改变采样统计，也未必击中峰值来源。

异构并行时分别计算生成侧 TP/DP 容量与 actor 训练侧的 $D$，通过 trajectory 交换和权重同步衔接；不能把两侧卡数一起放进同一个 batch 公式。DPO 使用固定偏好对，batch 更像监督训练的全局 batch、每卡 micro-batch 与累积关系，没有在线 `rollout.n`；PPO/GRPO 则先由 prompt 扩成 $Bn$ 条轨迹，再按 mini-batch 多轮更新，还要控制旧策略数据的陈旧程度。

## 知识点

verl、PPO/GRPO 批次、trajectory、梯度累积、动态微批、异构并行。

- 来源：[老师平台](https://course.terminiai.com/interview)，P003-Q011。
- 本题按本地 verl 快照 `3c5f6e0496d3d4fa80f2210859d5dda40a16c877`（提交时间 2026-04-29 22:24:48 +08:00，`v0.7.0-551-g3c5f6e04`）核实；其他版本需重新检查字段与计数口径。

- 真实面经来源：[B002-G01-Q009](../../docs/references/面经原题.md#b002-g01-q009)、[B002-G01-Q044](../../docs/references/面经原题.md#b002-g01-q044)、[B002-G01-Q045](../../docs/references/面经原题.md#b002-g01-q045)。
- 老师答案参考：[P005-Q009](../../docs/references/平台题/P005-Infra-001-030.md#p005-q009)、[P005-Q044](../../docs/references/平台题/P005-Infra-031-060.md#p005-q044)、[P005-Q045](../../docs/references/平台题/P005-Infra-031-060.md#p005-q045)。

## 追问

- 如果 global batch size 设置过大导致 OOM，应该优先调整哪个参数，为什么？
- verl 中 rollout 和 training 阶段通常采用不同并行策略（如 TP/DP），这种异构并行下参数如何对齐？
- 对比 DPO 和 PPO 在 batch size 设置上的核心差异是什么？

## Note

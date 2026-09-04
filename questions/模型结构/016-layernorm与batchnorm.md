---
difficulty: 简单
topic: Norm位置/计算轴与适用场景
summary: BN与LN统计哪些维度，为什么Transformer更常用LN
tags: [真题, 归一化, LayerNorm, BatchNorm, Transformer, 待校对]
company: 美团、快手、字节、小红书、滴滴、腾讯
mastered: false
highfreq: false
---

## 题目

请从计算轴、训练与推理差异、batch 大小、变长序列和部署一致性出发，比较 BatchNorm、LayerNorm、InstanceNorm、GroupNorm 与 RMSNorm，并解释 Transformer 为什么通常选择 LayerNorm 而不是 BatchNorm。

## 要点

- 先说明张量布局，再说均值和方差沿哪些轴统计
- BN 依赖一批样本的统计量，LN 按每个 token 的隐藏维独立计算
- BN 训练与推理使用的统计量不同，LN 两阶段算法一致
- Transformer 选 LN 的关键是样本独立和适配变长序列，不是背固定 batch 数字

## 答案

**Transformer 通常用 LayerNorm，因为它对每个 token 的隐藏向量独立归一化，不依赖同批样本、序列长度和运行均值；BatchNorm 会把样本与位置耦合起来。** 设序列张量为 $X\in\mathbb{R}^{B\times T\times D}$，LN 对固定的 $(b,t)$ 沿 $D$ 维计算

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu_D}{\sqrt{\sigma_D^2+\epsilon}}+\beta.
$$

BN 若用于该布局的特征维，通常对每个 $D$ 通道汇总 $B,T$ 轴；padding 是否污染统计取决于布局和是否做了有效位置掩码，不能一概而论。

| 方法 | 以 NCHW 图像为例的统计范围 | 主要特点 |
|---|---|---|
| BN | 每个通道跨 N、H、W | 训练用当前批统计，推理常用运行统计 |
| LN | 每个样本跨指定特征维 | 不依赖其他样本，训练推理计算一致 |
| IN | 每个样本、每个通道跨 H、W | 常用于强调单样本风格统计的视觉任务 |
| GN | 每个样本在一组通道及 H、W 内 | 小 batch 下仍不依赖跨样本统计 |
| RMSNorm | 常按每个 token 的 D 维算均方根 | 省去均值中心化，通常只保留缩放参数 |

BN 在 CNN 中能利用稳定的批统计，卷积的通道语义也较固定；但小批、跨设备拆批、变长序列或训练与线上 batch 组成变化时，统计噪声与运行均值偏差会增大。LN 每个位置自给自足，更适合自回归生成和可变长度输入。它通过控制表示尺度改善优化，没必要把作用只归结为有争议的“内部协变量偏移”。

$\gamma$、$\beta$ 让归一化后的各维重新学习尺度和偏移；是否省略要看具体 Norm 和实现。BN 的动量决定运行统计跟随新 batch 的快慢，数据变化快时跟随太慢会滞后，batch 噪声大时跟随太快又不稳，应以推理集验证。若一定在序列模型中用 BN，可对有效 token 做 masked statistics、同步多设备统计或冻结可靠的运行统计，但应验证训推分布变化，不能仅因总 batch 大就断定 BN 更优。Pre-LN/Post-LN 属于 Norm 的放置问题，见 [Pre-LN 与 Post-LN](017-pre-ln与post-ln.md)；RMSNorm 的中心化取舍见 [RMSNorm 与 LayerNorm](018-rmsnorm与layernorm.md)。

## 知识点

归一化方法的本质差别是“哪些元素共享统计量”。先写清 B/T/D 或 N/C/H/W，再谈优缺点；LN 与 RMSNorm 不需要运行均值，BN 需要区分训练统计和推理统计。

- 一手依据：[Batch Normalization](https://arxiv.org/abs/1502.03167)、[Layer Normalization](https://arxiv.org/abs/1607.06450)、[Group Normalization](https://arxiv.org/abs/1803.08494)、[RMSNorm](https://arxiv.org/abs/1910.07467)。

## 追问

- 为什么 LN 更适合变长序列与小 batch，padding 一定会污染 BN 吗？
- BN 的运行均值和动量怎样影响推理，训练与推理为何要区分？
- $\gamma$、$\beta$ 和 $\epsilon$ 分别做什么，可以省略吗？
- RMSNorm 相比 LN 省去了什么，为什么不保证在所有任务上更好？
- 如果必须在 Transformer 中使用 BN，你会怎样处理有效位置和跨设备统计？

## Note

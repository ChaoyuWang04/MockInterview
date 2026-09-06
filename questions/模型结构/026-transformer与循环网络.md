---
difficulty: 简单
topic: 序列模型/与RNN、GRU、CNN的建模比较
summary: Transformer与循环卷积结构如何权衡依赖路径和并行成本
tags: [真题, Transformer, RNN, GRU, CNN, 序列建模, 待校对]
company: 小红书、哔哩哔哩、美团、华为、腾讯
mastered: false
highfreq: false
---

## 题目

从依赖路径、训练并行、计算与内存、顺序建模、流式推理和归纳偏置比较 Transformer、RNN/GRU 与 CNN。为什么 Transformer 成为大模型主流，哪些场景下循环或卷积结构仍有价值？

## 要点

- RNN 沿时间递推，Transformer 同层位置并行，CNN 用局部共享卷积并行提取模式
- 自注意力任意位置可在一层交互，但稠密计算随序列长度二次增长
- “层内并行”不等于训练 $O(1)$，自回归推理也仍逐 token
- 流式、低内存、强局部先验等场景仍可能选择循环或卷积结构

## 答案

**Transformer 的主要优势是训练时同一层的所有位置可并行，并让远距离 token 在一层内直接交互；代价是稠密注意力的二次长度成本和不断增长的 KV Cache。** RNN 递推为

$$
h_t=f(x_t,h_{t-1}),
$$

第 $t$ 步必须等 $t-1$ 步，训练计算深度随序列增长，长距离梯度要跨很多递推步。LSTM/GRU 用门控缓解遗忘和梯度问题，但没有去掉时间依赖。

| 维度 | RNN/GRU | CNN 序列模型 | Transformer |
|---|---|---|---|
| 同层训练并行 | 时间步依赖强 | 各位置可并行 | 各位置可并行 |
| 远距离路径 | 随距离增长 | 需堆层、扩核或空洞卷积 | 全局注意力中一层可直接交互 |
| 顺序信息 | 递推天然带顺序 | 来自卷积位置与边界 | 需要位置编码或位置偏置 |
| 长序列成本 | 计算约随长度线性，但串行 | 通常随核宽线性 | 稠密注意力为 $O(n^2d)$ 序列项 |
| 在线状态 | 可压入固定隐藏状态 | 可缓存有限感受野 | KV Cache 通常随历史增长 |

Transformer 的矩阵乘法更适合 GPU/TPU，能扩大 batch、模型和数据规模；多头注意力也让模型按内容动态选择信息。CNN 的局部感受野与参数共享提供强局部先验，硬件执行规则；RNN 把历史压入状态，适合严格流式、持续到达且设备内存很小的信号。选择要看序列长度、延迟、可用算力与任务是否需要精确回看历史，不能只按榜单排名。

“注意力路径是一层”只描述依赖路径，不表示长序列没有优化困难：每层仍按顺序堆叠，注意力要计算许多位置对。训练时 causal mask 允许整段 token 并行算 loss；生成时第一个未知 token 产生后才能生成下一个，因此自回归推理仍串行。短序列或小设备上，RNN/CNN 的较小状态和低常数可能更合适。

Mamba 是选择性状态空间模型，不是 GRU/RNN 的同义词；RWKV 可按递归形式推理，也有线性注意力视角。QRNN、SRU 等方案则把部分门控计算并行化，减少传统递推中的串行工作，但仍有各自的状态更新。它们是否优于 Transformer 要按精确检索、质量、训练吞吐和部署内核共同验证。长序列方法见 [长序列 Attention 比较](025-长序列attention比较.md)，位置编码见 [位置编码方法](013-位置编码方法.md)，多头的独立作用见 [多头注意力原理](009-多头注意力原理.md)；业务能支持多长必须给出自己的模型、硬件、延迟和质量证据，不能编一个长度。

## 知识点

比较序列架构要同时看总计算、关键路径和硬件并行度。Transformer 的同层位置并行不等于全训练图或自回归生成并行，任意位置一步交互也不等于超长序列免费。

- 一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)、[Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122)、[Mamba](https://arxiv.org/abs/2312.00752)、[RWKV](https://arxiv.org/abs/2305.13048)。

## 追问

- Transformer 的训练并行具体发生在哪里，为什么推理不能同样并行？
- 注意力的依赖路径短，为什么超长序列仍然困难？
- RNN/GRU 在哪些流式或端侧场景仍有优势？
- Transformer 为什么需要位置编码而 RNN 通常不需要？
- Mamba、RWKV 与传统 RNN 分别是什么关系？

## Note

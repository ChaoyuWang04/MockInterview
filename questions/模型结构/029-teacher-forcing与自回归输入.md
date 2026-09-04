---
difficulty: 简单
topic: Transformer整体架构/Teacher Forcing与自回归推理的输入是什么
summary: 训练真值前缀与推理离散token输入为何不同
tags: [真题, Teacher Forcing, 自回归生成, Decoder-only, 暴露偏差, 待校对]
company: 字节、抖音
mastered: false
highfreq: false
---

## 题目

Decoder-only 模型训练和推理时，输入究竟是 token ID、one-hot、Embedding 还是完整概率分布？请说明 Teacher Forcing、因果掩码、自回归采样和暴露偏差，并解释 Beam Search 与“软概率输入”的区别。

## 要点

- token ID 查 Embedding 与 one-hot 乘 Embedding 矩阵数学等价，实际通常直接查表
- 训练用右移后的真实 token 前缀，借助因果掩码并行计算各位置损失
- 推理从 logits 得到概率并选择离散 token，再把该 token 的 Embedding 输入下一步
- 完整概率加权 Embedding 可以另行设计，但会改变标准生成语义与成本

## 答案

**标准 Decoder-only 推理把上一步选出的离散 token ID 查成 Embedding，再作为下一步输入；不会把整个词表概率分布直接喂回模型。** 设词表大小为 $V$、Embedding 表为 $E\in\mathbb{R}^{V\times d}$。token ID 为 $i$ 时，查表 $E_i$ 与 one-hot 向量 $e_i^\top E$ 数学等价，但前者无需显式构造长度为 $V$ 的稀疏向量。

自回归目标为

$$
p(x_{1:T})=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

训练时已知完整样本，把真实序列右移一位作为输入，用 causal mask 保证位置 $t$ 只能看真实前缀 $x_{<t}$，却能在一次前向中并行计算所有位置的 next-token loss。这就是 Teacher Forcing。模型输出 logits，经 softmax 得到词表概率；标签通常用 token ID 参与交叉熵，也无需 one-hot 展开。

推理时真实的 $x_t$ 不存在。模型从概率分布按贪心、温度采样、top-k/top-p 等策略选出一个 token ID，把它追加到前缀并查 Embedding。下一 token 依赖刚选出的结果，因此自回归步骤仍串行。Beam Search 确实利用完整分布给多个候选续写计分，但每条 beam 保存的仍是离散 token 前缀，并分别输入模型。

理论上可计算“软 Embedding”$\bar e=p^\top E$，但它把许多互斥 token 混成一个向量，训练与离散文本语义不一致，还要处理稠密词表概率及不同解码步的误差传播；这是另一种模型设计，不是标准推理的理由。

Teacher Forcing 训练看到真实前缀，推理看到自己的生成前缀，分布差异会使早期错误影响后续，称为暴露偏差。训练时混入模型采样 token 的 Scheduled Sampling 试图让模型接触自己的前缀，但会削弱整段并行、引入采样噪声和目标一致性问题；不能把完整概率软输入当成它的等价替代。暴露偏差是误差累积的一种解释，但重复、幻觉等现象还与数据、目标、解码和上下文有关，不能全部归为单一原因。

## 知识点

输入链路是“token ID → Embedding → Transformer → logits → 概率 → 选 token ID”。one-hot 只是查表的数学写法，概率用于选择和计分，不是标准下一步的直接输入表示。

- 一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[Scheduled Sampling](https://arxiv.org/abs/1506.03099)。

## 追问

- token ID、one-hot、Embedding 与 logits/概率分别位于哪一步？
- Teacher Forcing 为什么能并行训练，推理为什么仍需串行？
- 若用完整概率加权 Embedding，语义与计算会发生什么变化？
- Beam Search 如何使用概率分布，又为何仍输入离散前缀？
- 暴露偏差会导致什么风险，是否能解释所有生成错误？

## Note

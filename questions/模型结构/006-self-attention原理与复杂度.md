---
difficulty: 简单
topic: 注意力配件/Self-Attention原理、公式与复杂度
summary: 从QKV推导Self-Attention并计算时间空间复杂度
tags: [真题, SelfAttention, QKV, 复杂度, 待校对]
company: 字节、美团、小红书、海天、哔哩哔哩、阿里云、阿里、淘天、京东、滴滴
mastered: false
highfreq: false
---

## 题目

请从 Q、K、V 的生成开始，写出 Self-Attention 的完整公式并解释每一步的作用。长度为 $n$、隐藏维度为 $d$ 时，时间和空间复杂度分别是什么，长序列瓶颈在哪里？

## 要点

- $Q=XW_Q,K=XW_K,V=XW_V$ 的形状和语义
- 点积匹配、缩放、softmax 与 Value 加权的因果链
- 区分投影项 $O(nd^2)$ 与序列项 $O(n^2d)$
- 长序列的计算、显存限制及常见优化方向

## 答案

**Self-Attention 先计算“每个 token 应读谁”，再按这些权重汇总信息。** 设输入 $X\in\mathbb{R}^{n\times d}$：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
O=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

第 $i$ 个 Query 表示当前位置要找什么，第 $j$ 个 Key 用来与它匹配，所得权重决定读取多少第 $j$ 个 Value。缩放控制点积量级，$M$ 表示可选 mask，softmax 把每行变成和为 1 的读取权重。因为权重由当前内容动态生成，同一 token 在不同上下文中可读取不同位置。

若各投影宽度与 $d$ 同阶，Q/K/V 与输出投影耗时为 $O(nd^2)$；构造 $QK^\top$ 和乘 $V$ 为 $O(n^2d)$。训练中显式保存注意力矩阵需要 $O(n^2)$ 元素，另有 Q/K/V 的 $O(nd)$。因此不能直接写成 $O(n^3)$，除非额外假设 $d$ 与 $n$ 同阶。

### 常见追问简答

- **Q/K 维度能不同吗？** 做点积的最后一维必须相同；V 的输出维可不同。
- **与 Cross-Attention 的区别？** Self-Attention 的 Q/K/V 来自同一序列，Cross-Attention 的 Q 与 K/V 来自不同序列。
- **如何处理长序列？** 可用 FlashAttention 降低中间矩阵读写，或用局部、稀疏、近似注意力改变计算范围；前者不把核心配对计算改成线性。
- **为何能建模长距离？** 一层内任意两个允许位置可直接交互，但长距离信息是否学好仍取决于训练和模型容量。

## 知识点

Self-Attention 是内容相关的加权读取；复杂度账本要同时保留 $n$ 与 $d$，并分开投影成本和两次注意力矩阵乘。

- 依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)。

## 追问

- Q、K、V 为什么使用不同投影，各自代表什么？
- Self-Attention 与 Cross-Attention 的 Q/K/V 来源有何不同？
- 如何分别计算投影项和注意力序列项的复杂度？
- FlashAttention、稀疏注意力和线性注意力分别改变了什么？

## Note

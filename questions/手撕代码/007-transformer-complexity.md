---
difficulty: 简单
topic: Transformer/计算复杂度
summary: 分解 Transformer 各组件在训练与自回归推理中的计算和显存开销
tags: [真题, 待校对, Transformer, 复杂度, KVCache]
company: 滴滴、快手、字节、哔哩哔哩
mastered: false
highfreq: false
---

## 题目

请系统分析标准 Transformer 模型中词嵌入层、多头自注意力机制和前馈网络层的时间复杂度与空间复杂度，详细说明各组件在训练和自回归生成过程中的计算开销，并讨论其随输入序列长度增长的变化趋势。

## 要点

- 设批量为 $B$、序列长 $n$、隐藏维 $d$、FFN 宽度 $d_{ff}$，先明确统计口径。
- 注意力投影为 $O(Bnd^2)$，注意力矩阵为 $O(Bn^2d)$；FFN 为 $O(Bndd_{ff})$。
- 训练激活显存含 $O(Bn^2)$ 注意力矩阵；FlashAttention 可降中间显存但不改变标准注意力的算术量级。
- 自回归无缓存会反复计算前缀；KV Cache 后单步仍需读取全部历史 K/V。
- FFN 与注意力谁占主导取决于 $n,d,d_{ff}$、kernel 和硬件，不能写死长度阈值。

## 答案

令头数为 $h$、每头宽度 $d_h=d/h$。Embedding 查表需要 $O(Bn)$ 次索引访问，但读取并写出这些向量仍有 $O(Bnd)$ 的数据量，输出激活占 $O(Bnd)$；参数量为 $Vd$。若把输出词表投影算入语言模型头，则每个位置还要 $O(Vd)$ 计算。

一层标准 Transformer 的主要计算是：

- Q/K/V 与输出投影：$O(Bnd^2)$；
- 分数 $QK^T$ 与加权和 $PV$：合计 $O(Bn^2d)$；
- 两层 FFN：$O(Bndd_{ff})$，常见 $d_{ff}=4d$ 时约为 $O(Bnd^2)$。

训练时所有 token 并行计算，但反向传播和保存激活会放大常数。普通实现显式保存每头的 $n\times n$ 分数/概率，相关激活空间为 $O(Bhn^2)$；FlashAttention 分块重算，把这部分中间显存降下来，但标准密集注意力的 FLOPs 仍是二次量级。

自回归推理若每一步都重算整个前缀，生成 $T$ 个 token 会重复投影历史和计算历史 token 之间的注意力。KV Cache 保存各层历史 K/V 后，新 token 只做一次 Q/K/V 投影，但当前 query 仍要与 $t$ 个历史 key 做注意力；每层单步计算为 $O(d^2+td)$，生成阶段累计为 $O(Td^2+T^2d)$。缓存占 $O(BLTd)$（$L$ 为层数，忽略 GQA/MQA 的 KV 头数修正）。

不能仅按“超过某个固定长度”判断瓶颈。短序列、大 $d_{ff}$ 时 FFN 可能占主导；长序列时二次注意力通常变得显著，实际还受带宽、融合 kernel 和并行策略影响。

## 知识点

- 参数量与 FLOPs、训练激活、密集注意力的二次项、KV Cache 的时间—空间权衡、FlashAttention、GQA/MQA。


## 追问

相关真题追问：

- 线性注意力、稀疏注意力和滑动窗口分别改变了哪一项复杂度？
- batch size 与序列长度如何共同影响吞吐和显存？
- 为什么典型 FFN 参数量约为标准注意力投影的两倍，而不是固定四倍？

## Note

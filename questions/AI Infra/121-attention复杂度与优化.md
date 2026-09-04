---
difficulty: 中等
topic: 稀疏注意力/Attention复杂度与优化
summary: Attention平方复杂度可用哪些精确或近似路线优化
tags: [面经, 待校对, Attention, 稀疏注意力, 线性注意力, FlashAttention]
company: 阿里、小红书、字节、阿里云
mastered: false
highfreq: false
---

## 题目

标准多头Attention的计算和内存开销来自哪里？比较FlashAttention、局部/块稀疏、Longformer类模式、低秩、Reformer/Performer及线性Attention的机制、复杂度、精度和适用场景。

## 要点

- 标准attention的QKᵀ和PV计算为O(n²d)，注意力矩阵为O(n²)
- FlashAttention是精确IO优化，不改变平方级FLOPs
- 稀疏方法减少连接，线性/核化方法改变计算形式，都会引入结构假设
- Mamba是状态空间模型，不应归类为线性Attention
- 长上下文选型要同时看信息路径、kernel支持、KV布局与任务精度

## 答案

**标准自注意力的瓶颈来自两张 $n	imes n$ 的交互：$QK^T$ 和随后对 $V$ 的加权。** 对单头，主要算术量约为 $O(n^2d)$；若保存注意力矩阵，中间内存为 $O(n^2)$。

| 路线 | 怎样省 | 是否保持标准Attention | 主要边界 |
|---|---|---|---|
| FlashAttention | 分块、online softmax，减少HBM往返 | 是，允许浮点舍入差异 | FLOPs仍为平方级 |
| 滑窗/局部+全局 | 每个token只连附近窗口和少量全局token | 否 | 长距信息必须经过全局点或多层传播 |
| block/strided/fixed sparse | 只算指定块或固定跨距连接 | 否 | 模式需匹配任务且有高效稀疏kernel |
| 低秩近似 | 用较低秩表示交互矩阵 | 否 | 秩不足会丢复杂依赖 |
| Performer等核化线性注意力 | 用特征映射重排结合顺序并处理归一化 | 否 | 只对特定形式成立，数值与精度需验证 |
| Reformer | 局部敏感哈希把相似token分桶 | 否 | 哈希与分桶本身有开销和召回误差 |

Longformer属于“局部窗口+全局token”；滑窗是稀疏注意力的一种，不是与稀疏并列的完全不同类别。Mamba是状态空间模型，不是Linear Attention。

为什么主流LLM仍常用标准Attention加FlashAttention？因为它保持全连接表达能力，且GPU上有成熟稠密kernel；在中等长度下，近似方法的理论复杂度优势未必转成端到端收益。

100K到1M上下文可分层选：必须精确全局交互时用FlashAttention加Context/Ring Parallel；局部结构明确时用滑窗或块稀疏并保留全局token；若任务允许近似，再评估线性或低秩方法。稀疏Attention可与KV分页/量化组合，前者减少被访问的位置，后者减少缓存字节；二者都必须测长距检索和端到端kernel效率。

## 知识点

标准Attention复杂度、FlashAttention、局部/全局与块稀疏、低秩近似、Reformer、Performer、线性Attention。

真实面经：[B002-G01-Q022](../../docs/references/面经原题.md#b002-g01-q022)、[B002-G01-Q032](../../docs/references/面经原题.md#b002-g01-q032)、[B002-G01-Q120](../../docs/references/面经原题.md#b002-g01-q120)、[B002-G01-Q122](../../docs/references/面经原题.md#b002-g01-q122)、[B002-G01-Q155](../../docs/references/面经原题.md#b002-g01-q155)、[B002-G01-Q156](../../docs/references/面经原题.md#b002-g01-q156)、[B002-G01-Q158](../../docs/references/面经原题.md#b002-g01-q158)、[B002-G01-Q159](../../docs/references/面经原题.md#b002-g01-q159)。

老师参考：[P005-Q022](../../docs/references/平台题/P005-Infra-001-030.md#p005-q022)、[P005-Q032](../../docs/references/平台题/P005-Infra-031-060.md#p005-q032)、[P005-Q120](../../docs/references/平台题/P005-Infra-091-120.md#p005-q120)、[P005-Q122](../../docs/references/平台题/P005-Infra-121-150.md#p005-q122)、[P005-Q155](../../docs/references/平台题/P005-Infra-151-180.md#p005-q155)、[P005-Q156](../../docs/references/平台题/P005-Infra-151-180.md#p005-q156)、[P005-Q158](../../docs/references/平台题/P005-Infra-151-180.md#p005-q158)、[P005-Q159](../../docs/references/平台题/P005-Infra-151-180.md#p005-q159)。

## 追问

以下均为平台页面参考追问，不作为面经原话：

- 参考追问：稀疏模式怎样兼顾效率和跨段信息？
- 参考追问：线性Attention牺牲什么能力，什么场景不适合？
- 参考追问：FlashAttention与稀疏Attention能否结合？
- 参考追问：strided与fixed sparse attention的连接模式有何不同？
- 参考追问：为什么主流LLM仍常用标准Attention加FlashAttention？
- 参考追问：100K到1M上下文怎样选择精确、稀疏或线性方案？
- 参考追问：稀疏Attention怎样与KV分页和量化配合？

## Note

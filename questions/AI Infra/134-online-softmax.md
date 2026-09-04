---
difficulty: 中等
topic: FlashAttention/Online Softmax
summary: 分块注意力怎样递推最大值、分母和输出而保持等价
tags: [面经, 待校对, FlashAttention, Online Softmax, 数值稳定性]
company: 字节、阿里淘天
mastered: false
highfreq: false
---

## 题目

FlashAttention 不保存完整注意力矩阵时，怎样通过 online softmax 在分块处理中更新最大值、归一化分母和输出，并保持数值稳定与跨块数学等价？

## 要点

- 每行维护 running max、指数和及未归一化输出
- 最大值变化时，旧分母和旧输出都要重缩放
- 分块超过片上容量时继续逐块扫描，边界块用 mask
- 减少的是 HBM 中间量读写，具体收益依形状和硬件

## 答案

**关键不是只累计分母，而是每次最大值变化时，同时把旧分母和旧输出换到新的指数基准。**

对某一行，旧块统计量为最大值 $m$、指数和 $l$、加权值和 $o$；新块对应 $m_b,l_b,o_b$。合并为：

$$
m'=\max(m,m_b)
$$

$$
l'=e^{m-m'}l+e^{m_b-m'}l_b,\qquad
o'=e^{m-m'}o+e^{m_b-m'}o_b
$$

扫完所有 K/V 块后输出为 $o'/l'$。所有指数都以当前最大值为基准，避免上溢；很小的项仍可能下溢为零，但通常对应可忽略的概率质量，不能声称绝不下溢。

序列长于片上存储时，算法本来就会流式加载更多块，并不要求整行进入 SRAM。最后一个不满的块用边界判断和 mask 排除无效位置。反向通常保存少量行统计量和输出，在需要时重算分数块；这减少中间矩阵的 HBM 流量，但仍需要重算所需输入。

FlashAttention-2延续同一等价公式，主要改进工作划分、减少非矩阵乘操作和提高并行性；warp 细节依具体 kernel。短序列或小形状下，调度和重算开销可能抵消收益。FP32累计降低误差，但归约顺序不同仍可能与普通 softmax 有末位差异。IO节省应按片上容量、head维度、序列和实现分析，不能笼统写成固定倍数或简单的 $O(N^2)\to O(N)$。

## 知识点

running max、归一化分母、输出重缩放、边界 mask、前向分块与反向重算。

- 真实面经：[B002-G01-Q051](../../docs/references/面经原题.md#b002-g01-q051)、[B002-G01-Q069](../../docs/references/面经原题.md#b002-g01-q069)、[B002-G01-Q070](../../docs/references/面经原题.md#b002-g01-q070)、[B002-G01-Q125](../../docs/references/面经原题.md#b002-g01-q125)、[B002-G01-Q126](../../docs/references/面经原题.md#b002-g01-q126)、[B002-G01-Q150](../../docs/references/面经原题.md#b002-g01-q150)
- 老师答案参考：[P005-Q051](../../docs/references/平台题/P005-Infra-031-060.md#p005-q051)、[P005-Q069](../../docs/references/平台题/P005-Infra-061-090.md#p005-q069)、[P005-Q070](../../docs/references/平台题/P005-Infra-061-090.md#p005-q070)、[P005-Q125](../../docs/references/平台题/P005-Infra-121-150.md#p005-q125)、[P005-Q126](../../docs/references/平台题/P005-Infra-121-150.md#p005-q126)、[P005-Q150](../../docs/references/平台题/P005-Infra-121-150.md#p005-q150)

## 追问

- 页面参考追问：FlashAttention-2 相比第一版在 online softmax、算法和 warp 并行上改进了什么？
- 页面参考追问：online softmax 与标准 softmax 的反向传播有什么差异？
- 页面参考追问：为什么增量更新不会不断积累显著误差？
- 页面参考追问：序列超过 SRAM 容量或不能整除块大小时怎样处理？
- 页面参考追问：FlashAttention 的 IO 复杂度怎样分析？
- 页面参考追问：为什么短序列可能没有明显收益甚至变慢？
- 页面参考追问：FlashAttention 是否会带来精度差异，原因是什么？

## Note

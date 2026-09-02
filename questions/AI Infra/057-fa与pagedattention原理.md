---
difficulty: 中等
topic: FlashAttention/与PagedAttention关系
summary: FlashAttention 与 PagedAttention 各自的原理,以及两者的关系
tags: [面经, 待校对, FlashAttention, PagedAttention]
company:
mastered: false
highfreq: false
---

## 题目

讲下 FlashAttention 和 PagedAttention 的原理?两者是什么关系,能不能一起用?

## 要点

- FA 是**精确**注意力,不是近似;省的是访存不是计算量
- FA 两个机制:tiling(中间结果只活在片上)+ online softmax(边扫边重定标)
- PagedAttention 是 KV cache 版的虚拟内存分页,块表 = 页表
- 一个省时间、一个省空间,**两者正交**,现代引擎叠加使用
- 说清叠加为什么代价小:FA 本来就是分块取 K/V 的

## 答案

### FlashAttention:让那张大表不落显存

标准注意力被拆成三个 kernel,$L \times L$ 的分数矩阵要**写两遍读两遍**,一共四趟 HBM;而它只是中间产物。$L=8192$、$d=128$、fp16 时这张表 128 MiB、四趟就是 512 MiB,而 Q/K/V/O 加起来才 8 MiB——**中间量流量是输入输出的 64 倍**,算术强度约 64 FLOP/Byte,远低于 A100 的拐点,所以注意力本身是访存受限的。

FA 的两个机制:

- **tiling**:Q 与 K/V 都沿行切块,每步只把小块载入 shared memory,片上算出分数分块、直接吃掉、累加到输出,**这个分块从头到尾没离开过 SM**。
- **online softmax**:softmax 要除以整行指数和,而整行分散在所有 K 块里。解法是边扫边维护行最大值 $m$ 与指数和 $\ell$,每来一块就把旧结果重定标一次:

$$
m_j = \max\bigl(m_{j-1},\ \mathrm{rowmax}(S_j)\bigr), \qquad
\ell_j = e^{\,m_{j-1}-m_j}\,\ell_{j-1} + \mathrm{rowsum}\bigl(e^{\,S_j-m_j}\bigr)
$$

靠恒等式 $e^{x-m_{j-1}} \cdot e^{m_{j-1}-m_j} = e^{x-m_j}$,重定标是**精确的代数改写而非近似**,所以 FA 叫 exact attention。反向不存分数矩阵、只存输出和每行一个 logsumexp 标量,用重算换访存,多约 17% FLOPs 但省几倍 HBM 流量。

### PagedAttention:把 OS 分页搬到 KV cache 上

传统按 max_len 预留连续显存,实测只有 20%–40% 的 KV 显存真装着 token。PagedAttention 把 KV 切成固定大小的块(典型 16 token 一块)散落在显存池里,用**块表**把逻辑块号映射到物理块号——**块表就是页表,attention kernel 就是软件版 MMU**。按需分配、不预留,浪费封顶在「不到一个块」;块大小固定还顺手消灭了外部碎片;前缀相同的块让不同序列的块表指向同一物理块即可共享,一个字节都不用拷。收益传导链:利用率上去 → batch 变大 → 权重读取被摊薄 → 吞吐涨 2–4 倍。

### 两者的关系:正交,叠加使用

| | FlashAttention | PagedAttention |
|---|---|---|
| 解决什么 | attention **算得快**(访存 / 算力) | KV cache **存得省**(碎片 / 共享) |
| 手段 | tiling + online softmax + 融合 | 固定块 + 块表 + 按需分配 |
| KV 布局要求 | 连续(或紧凑拼接) | 任意分散 |
| 省的是 | 时间 | 空间 |

**最容易答错的一点是把它们当二选一。** 分页只改变「K/V 块从哪里读」,不改变 tiling 与 online softmax 的任何一步——因为 FA 本来就是分块取 K/V 的,只要把「基址 + 偏移」换成「查块表拿物理块号」就行。而且块表粒度和 tile 粒度对得上,查表是每块一次而不是每元素一次,开销可忽略。所以主流做法是**用 FlashAttention 的内核吃分页布局的 KV**。

## 知识点

exact attention、tiling、online softmax 与重定标、块表 / 页表类比、按需分配、正交叠加。

## 追问

- online softmax 为什么和标准 softmax 数学等价?减最大值是为了什么?
- 反向为什么要重算而不是把分数矩阵存下来?
- block size 开大开小分别有什么影响?一般用多少?

## Note

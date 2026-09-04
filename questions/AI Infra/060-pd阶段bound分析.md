---
difficulty: 中等
topic: PD分离/阶段bound分析
summary: prefill 与 decode 分别是什么 bound,从算术强度上解释
tags: [真题, 待校对, PD分离, Roofline]
company:
mastered: false
highfreq: false
---

## 题目

p 阶段和 d 阶段一般情况下分别是什么 bound?为什么?

## 要点

- 结论:prefill 计算受限,decode 访存受限
- 判据是**算术强度**与硬件脊点的比较,不能只背结论
- prefill 强度 ∝ prompt 长度(上百),decode 强度 ∝ batch(几十)
- 给一个数量级地板:权重读一遍的时间就是 decode 单步的下限
- 补充:decode attention 部分的强度恒等于 GQA 组数,与 batch、序列长度都无关

## 答案

**结论:prefill 是计算受限(compute-bound),decode 是访存受限(memory-bound)。**

### 判据:算术强度 vs 脊点

算术强度 = 每从显存搬 1 字节能干多少次浮点运算。对权重矩阵而言,一次前向必须把整份权重读一遍,而**这份权重被 batch 里的多少行复用,就是强度的量级**:

- **prefill**:一个 2048 token 的 prompt 相当于 2048 行同时用这份权重 → 强度上百,远在 A100 FP16 脊点(约 312 TFLOPS ÷ 2.0 TB/s ≈ **150 FLOP/Byte**)之上 → **算力才是墙**。
- **decode**:一步只有 $B$ 行($B=32$ 就是 32)→ 强度只有几十,卡在脊点左侧 → **带宽是墙**。

对应到矩阵形状:prefill 是 $[S, d] \times [d, d']$ 的**大矩阵乘**,Tensor Core 能吃满;decode 是 $[B, d] \times [d, d']$ 的**瘦长矩阵**,接近 GEMV,大部分时间在等数据。

### 一个必须记住的地板

7B 模型 FP16 权重 14 GB,A100 显存带宽约 2.0 TB/s,**光是把权重读一遍就要约 7 ms**。这是 decode 单步耗时的物理地板,**和 batch 多大几乎无关**——所以 decode 阶段的优化几乎全在「少读字节」上:量化、GQA、KV 压缩。

### 一个容易被追问的细节

上面算的是权重那部分。attention 读 KV cache 那部分的强度更极端:

$$
I = \frac{4 S H_q d}{2 S H_{kv} d\, b} = \frac{2}{b} \cdot \frac{H_q}{H_{kv}}
$$

分子是 $QK^\top$ 与 $\text{attn}\cdot V$ 两次乘加,分母是把这份 cache 读进来的字节数。**序列长度 $S$ 上下约掉了**——fp16 下这个强度恒等于 GQA 的组数 $g$(常见 4 或 8),与上下文长度、与 batch 都无关。

推论很重要:**增大 batch 能摊薄权重读取,却摊薄不了 KV 读取**(每个请求各读自己的一份,随 batch 线性增长)。所以 batch 加到一定程度后,KV 带宽会顶上来成为新的主导项。

### 这对部署意味着什么

两阶段资源画像相反,所以 prefill 想要算力强的新卡、decode 想要带宽大显存大的卡(老卡也能干);混在一组 GPU 上跑必然两边都不最优,这正是 PD 分离的动机。

## 知识点

算术强度、Roofline 脊点、GEMM vs GEMV 形状、权重可摊薄 / KV 不可摊薄、H20 这类砍算力保带宽的卡为何专供 decode。

## 追问

- 增大 batch 能不能缓解 KV cache 的带宽压力?为什么?
- 长上下文下 decode 为什么会「越生成越慢」?
- 开了 chunked prefill,prefill 块的 bound 属性会变吗?

## Note

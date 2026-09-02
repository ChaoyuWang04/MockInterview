---
difficulty: 困难
topic: KVCache量化/能否走int8算力
summary: KV 量化后能不能吃到 int8 算力,QK 与 PV 怎么走整数矩阵乘
tags: [面经, 待校对, KVCache量化, TensorCore]
company:
mastered: false
highfreq: false
---

## 题目

KV cache 量化之后能不能用上 int8 的算力?Q 和 K、注意力分数 P 和 V 分别怎么做 int8 矩阵乘?

## 要点

- 分 prefill / decode 两个阶段答,答案相反
- decode 的 $QK^\top$ 根本不是 GEMM,MMA 最小 16 行起步,换整数是在空闲部件上省时间
- decode 侧 KV 量化省的是访存,一分算力都没省
- prefill 侧能且真在用,但那是"量化 attention 计算",不是"量化 KV cache"
- 真走整数 MMA:Q 也要量化、$S$ 要乘 $s_q s_k$ 还原;softmax 必须 fp32,P 想走整数得再量化一次

## 答案

分两个阶段答,答案相反。

### decode 侧:能,但没意义

decode 的 $QK^\top$ **根本不是 GEMM**——query 只有一行(GQA 下把同组几个 q head 塞进 M 维也才四到八行),而 Tensor Core 的 MMA 最小 16 行起步。这个形状下算力压根没用满,换整数 MMA 是在一个**空闲部件**上省时间。

所以结论是:**decode 侧 KV 量化省的是访存,一分算力都没省。** 主流引擎当前的做法就是"存时量化、读时反量化回 bf16 再算",显存与带宽收益是实的,算力收益不一定有。

### prefill 侧:能,而且真在用

prefill 的 $QK^\top$ 是胖大 GEMM,低位宽 MMA 的算力翻倍是实打实的。FlashAttention-3 在 Hopper 上有完整 fp8 路径,论文报 fp8 下接近 1.2 PFLOPS,数值误差比朴素 fp8 attention 低 2.6 倍。

但要分清一件事:**那是"把 attention 的计算量化",不是"把 KV cache 量化"**——prefill 的 K/V 是刚算出来的、还没进缓存。两件事可以叠加,收益来源不同。

### 真要走整数 MMA,两处必须处理

1. **Q 也得量化**。$S$ 出来是整数累加器里的值,要乘上 $s_q s_k$ 才还原成真正的注意力分数(scale 与求和维无关,所以能整个提到求和号外面)
2. **$P$ 要不要再量化一次**。softmax 必须在 fp32 里做,所以 $P$ 出来是浮点;想让 $PV$ 也走整数,就得**把 $P$ 再量化一次**——多一层误差,还赔掉 V 按 token 量化那顿免费午餐($\sum_t (P_t s_t)\hat V_{t,d'}$ 里 $P_t$ 本就是寄存器里的标量,吸收一个逐 token 因子几乎不要钱;$P$ 变成整数之后就吃不下这个浮点因子了)

## 知识点

decode attention 的形状与 MMA 最小行数、访存收益与算力收益的区分、FlashAttention-3 的 fp8 路径、Q 侧量化与 $s_q s_k$ 还原、softmax 的 fp32 约束与 P 的二次量化代价。

## 追问

- GQA 下把同组 q head 塞进 M 维能凑到多少行?到多少才值得换整数 MMA?
- prefill 用 fp8 attention、decode 用 bf16 反量化,同一份 KV 两阶段怎么保持一致?
- $P$ 再量化一次带来的误差,和 KV 本身量化的误差哪个更伤?

## Note

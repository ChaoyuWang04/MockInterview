---
difficulty: 中等
topic: KVCache量化/prefill与decode的处理差异
summary: prefill 与 decode 对 KV cache 的处理和量化差异在哪
tags: [真题, 待校对, KVCache量化, Prefill]
company:
mastered: false
highfreq: false
---

## 题目

prefill 阶段和 decode 阶段对 KV cache 的处理有什么不同?量化上的差异体现在哪?

## 要点

- prefill 一次产生整段 prompt 的 KV,decode 每步只产生一个 token 的
- prefill 能看到整段分布、可段内按通道统计;decode 只看得到当前这一个 token
- 规约成本在 prefill 上被几千 token 摊掉,在 decode 上每步一次、占比显眼
- 两阶段 bound 相反,所以能不能吃到低位宽算力的答案也相反
- 但两阶段**必须用同一套 scale**,这不是选择题

## 答案

| | prefill | decode |
|---|---|---|
| 一次产生多少 KV | 整段 prompt | **1 个 token** |
| 能看到什么 | 整段分布,可段内统计 | 只有这一个 token 自己 |
| 能不能做 per-channel | **能**,段内按通道统计一次 | 不能,新 token 会改通道范围而历史已写死 |
| 规约成本 | 摊到几千 token 上,可忽略 | 每步一次,占比更显眼 |
| 本身是什么 bound | 计算受限 | 访存受限 |
| 低位宽算力吃不吃得到 | **吃得到**:$QK^\top$ 是胖大 GEMM,fp8 MMA 算力翻倍是实的 | 吃不到:query 只有一行,MMA 最小 16 行起步,量化只省访存 |

### 两条推论

**一是两个阶段必须用同一套 scale**,这不是选择题:同一条序列的 KV 会被两阶段先后写进同一份缓存,读的时候是一起读的;分别用两套 scale 就等于同一份缓存里混了两种编码。静态 scale 天然满足;动态方案必须把粒度定在"不跨阶段边界"的单位上——per-token 正好满足,per-channel 正好不满足。

**二是"prefill 的 attention 用 fp8 算"和"KV cache 量化"是两件事**。prefill 的 K/V 是刚算出来的、还没进缓存,把那次矩阵乘量化属于**量化 attention 的计算**;KV cache 量化说的是写进缓存的那份数据用什么位宽存。两件事可以叠加,收益来源不同(一个省算力、一个省访存与显存)。

### 落到实现上

写 KV 的是**同一个 kernel**,两阶段只是喂进去的 token 数不同——所以量化逻辑不该按阶段分叉。真正的差异在读侧:prefill 走的是稠密的 attention 路径,decode 走分页 gather 的路径,后者才是"多读一份散点 scale"的开销所在。

## 知识点

两阶段的 KV 产生量与可见分布、per-channel 在 decode 上不可行、bound 差异、低位宽算力的可用性、共用 scale 的强制性、量化 attention 计算与量化 KV cache 的区分。

## 追问

- 开了 chunked prefill 之后,两阶段混在一次前向里,这些差异还成立吗?
- prefill 阶段顺手统计出的分布,能不能拿来当 decode 的 scale?
- PD 分离部署下,两个实例的量化配置有什么约束?

## Note

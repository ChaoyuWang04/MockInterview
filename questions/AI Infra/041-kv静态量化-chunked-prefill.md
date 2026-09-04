---
difficulty: 困难
topic: KVCache量化/与chunked-prefill的兼容
summary: KV cache 静态量化怎么和 chunked prefill 兼容,为什么
tags: [真题, 待校对, KVCache量化, ChunkedPrefill]
company:
mastered: false
highfreq: false
---

## 题目

KV cache 静态量化怎么和 chunk_prefill 这个特性兼容?

## 要点

- 结论:静态天然兼容,chunked prefill 恰恰是"必须共用一套 scale"这条约束的最强版本
- 同一条序列的 KV 被两阶段先后写进同一份缓存、读时一起读,两套 scale = 一份缓存混了两种编码
- 开了分块预填后,一次前向里 prefill token 与 decode token 混在同一个 kernel 里写 KV
- 此时"按阶段分 scale"连表达都表达不出来
- 动态方案要兼容,粒度必须定在不跨阶段边界的单位上:per-token 行,per-channel 不行

## 答案

**结论先给:静态量化天然兼容;chunked prefill 恰恰是"两个阶段必须用同一套 scale"这条约束的最强版本。**

### 第一层:两个阶段本来就必须共用一套 scale

这不是选择题。同一条序列的 KV 会被 prefill 和 decode **先后写进同一份缓存**,而 attention 读的时候是**一起读**的。分别用两套 scale,就等于同一份缓存里混了两种编码——读侧 kernel 没有任何依据去区分某个块里的 token 是哪个阶段写的。

### 第二层:chunked prefill 把这条推到极致

开了分块预填之后,**一次前向里既有某请求的几百个 prefill token,又有别的请求的一个 decode token**,写 KV 的是同一个 kernel、同一批 slot。此时"按阶段分 scale"**连表达都表达不出来**——kernel 拿到的就是一批混合 token,没有"阶段"这个维度可供分派。

| 方案 | 兼不兼容 | 为什么 |
|---|---|---|
| 静态 scale | **天然兼容** | scale 是模型级常量,和是谁写的、什么阶段写的完全无关 |
| 动态 per-token | 兼容 | 每个 token 自己一套,谁写的都一样,不跨阶段边界 |
| 动态 per-channel(跨 token 分组) | **不兼容** | 组跨 token,而 chunk 边界不等于分组边界;一个组可能横跨两次前向 |
| 动态 per-tensor(整层 absmax) | 不兼容 | scale 每步都变,而缓存里躺着的是历史步写进去的值,没有一个统一的 scale 能解释它们 |

一句话答法:**动态方案要兼容 chunked prefill,量化粒度必须定在"不跨阶段边界"的单位上**;静态方案根本没有这个问题。

### 同一条约束的另一个出口

PD 分离下的约束也是这么来的:prefill 实例算出的 KV 要传给 decode 实例,**两端的量化格式与 scale 必须完全一致**,否则传过去的字节流没法解释。静态 scale 在这里又赢一次——它是模型的一部分,天然两端相同。

## 知识点

两阶段共用缓存 ⇒ 共用 scale、chunked prefill 的混合 batch、量化粒度不能跨阶段边界、PD 分离下的字节流可解释性。

## 追问

- 混合 batch 里 prefill token 与 decode token 的 K/V 分布真的一样吗?对静态 scale 有什么影响?
- 分块大小变化会不会影响量化参数的选取?
- PD 分离下如果两端想用不同位宽,有没有办法?

## Note

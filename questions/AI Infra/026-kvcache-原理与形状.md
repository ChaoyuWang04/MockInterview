---
difficulty: 中等
topic: KVCache/原理与形状
summary: KV cache 为什么能缓存 K/V 不缓存 Q,形状与显存怎么算
tags: [面经, 待校对, KVCache, 推理优化]
company:
mastered: false
highfreq: false
---

## 题目

KV cache 的原理是什么?它的形状是什么样的,显存占用怎么算?

## 要点

- 因果掩码 ⇒ K/V 一旦算出后续不再变,所以可缓存;Q 每步只用一次,存了没人读
- 不缓存的计算量是 $Pn^2$ 量级,缓存后是 $2Pn$,差 $n/2$ 倍
- 六维逻辑形状里只有 batch 与序列长度随负载变,其余四维是模型常量
- 每 token 字节公式里只有 KV 头数,没有 Q 头数——这正是 GQA 的省法
- 真实实现分页之后,batch 与序列长度两维被块表取代

## 答案

### 为什么能缓存

生成第 $t+1$ 个 token 时,attention 需要**当前这一个 token 的 Q**,去和**前面所有 token 的 K、V** 交互。关键前提是**因果掩码**:第 $i$ 个 token 只能看到自己和之前的内容,所以它的 K/V 一旦算出,后续任何一步都不会再变。不变 = 可缓存。Q 每步只用一次,缓存没有意义。

不缓存的代价:为了拿到第 2 层的 K/V,得先有第 1 层对所有历史 token 的输出,于是每生成一个 token 就要把前面所有 token 完整 prefill 一遍。设参数量 $P$、生成 $n$ 个 token:

$$
\text{不缓存} \approx \sum_{t=1}^{n} 2Pt = Pn^2 \qquad\text{vs}\qquad \text{缓存后} \approx 2Pn
$$

比值是 $n/2$——生成 1000 个 token,不缓存要多做约 500 倍计算。

### 形状与显存

逻辑形状六个维度:

$$
[\ \text{层数} L,\ 2\ (K/V),\ \text{batch } B,\ \text{KV 头数 } H_{kv},\ \text{序列长度 } S,\ \text{head 维度 } d\ ]
$$

每 token 占用($b$ 为每元素字节数,fp16 是 2):$m_{\text{token}} = 2 \times L \times H_{kv} \times d \times b$,总量再乘 $B \times S$。

| 模型 | $L$ | Q 头数 | KV 头数 | $d$ | 每 token(fp16) | 128K 上下文/请求 |
|---|---|---|---|---|---|---|
| Llama-3-8B | 32 | 32 | 8(GQA 4:1) | 128 | 128 KiB | 16 GiB |
| Llama-3-70B | 80 | 64 | 8(GQA 8:1) | 128 | 320 KiB | 40 GiB |

值得记住的对照:8B 的 fp16 权重是 16 GiB,而**一个 128K 上下文请求的 KV cache 也正好 16 GiB**。

两点补充:公式里没有 Q 头数,省 KV 的唯一入口就是 $H_{kv}$,这是 GQA 存在的意义;真实实现分页之后每层单独存 `[2, 块数, 块内 token 数, KV 头数, head_dim]`,$B$ 与 $S$ 被块表取代。

## 知识点

因果掩码与可缓存性、KV cache 六维形状、每 token 字节公式、GQA 对 $H_{kv}$ 的削减、分页后的物理布局。

## 追问

- 为什么 decode 是访存受限?KV cache 在带宽里占多少?
- 增大 batch 能不能缓解 KV cache 的带宽压力,为什么?
- 显存不够触发抢占时,是重算还是换出,为什么?

## Note

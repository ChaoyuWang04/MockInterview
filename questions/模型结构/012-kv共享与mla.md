---
difficulty: 中等
topic: KV共享注意力/MHA MQA GQA MLA
summary: MHA、MQA、GQA与MLA如何压缩KV Cache
tags: [MHA, MQA, GQA, MLA, KVCache, 待校对]
company: 快手、字节、阿里云、蚂蚁集团
mastered: false
highfreq: false
---

## 题目

请比较 MHA、MQA、GQA 和 MLA 的 Q/K/V 组织方式，并解释它们怎样在 KV Cache、内存带宽、计算与模型质量之间取舍。GQA 分组和 MLA 低秩压缩分别解决什么问题？

## 要点

- 按 Query 头数与 KV 头数区分 MHA、MQA、GQA
- 用完整变量推导 KV Cache 大小
- MLA 缓存联合压缩的 KV latent，并保留解耦 RoPE 分量
- 选择受质量、带宽、并行和上下文长度共同影响

## 答案

**四种机制都保留多头 Query，主要差别是历史 K/V 以多少份、什么表示被缓存。**

| 机制 | KV 组织 | 主要取舍 |
|---|---|---|
| MHA | 每个 Query 头各有一组 K/V | 容量完整，缓存与带宽最大 |
| MQA | 所有 Query 头共享一组 K/V | 缓存最小，可能损失质量 |
| GQA | $h$ 个 Query 头分成 $g$ 组，每组共享 K/V | 在 MHA 与 MQA 间折中 |
| MLA | 把 K/V 联合压成低维 latent，另处理位置分量 | 用低秩表示进一步压缩缓存 |

若每层缓存 K 和 V，batch 为 $B$、层数 $L$、长度 $T$、KV 头数 $n_{kv}$、每头维 $d_h$、每元素 $s$ 字节，则

$$
\text{KV bytes}=2BLTn_{kv}d_hs
$$

这只是 KV 张量，不含权重、临时缓冲和碎片。GQA 中每组 Query 头数为 $h/g$；$g$ 越少越省缓存，但共享约束越强。MLA 不能直接套该头数公式，它缓存的是联合压缩 latent。典型 DeepSeek-V2 MLA 仍含解耦 RoPE，也不要求额外搭配 LoRA。

### 常见追问简答

- **GQA 的组数怎样选？** 先满足 $h$ 可被 $g$ 整除，再用目标硬件上的质量、吞吐和显存实验选，没有通用最优值。
- **Llama 2 是否全系列用 GQA？** 原报告中 7B/13B 使用 MHA，70B 使用 GQA，不能把一个尺寸推广到全系列。
- **为何训练时就使用分组？** 共享 K/V 改变了模型函数，不能只在推理时无损删除独立 KV 头；可用转换加继续训练适配。
- **长上下文一定选 MLA 吗？** 不一定，还要看实现、训练稳定性、位置方案和服务内核。

## 知识点

MQA/GQA 压缩 KV 头数量，MLA 压缩 KV 表示本身；端到端显存与速度不能只用一个理论比例代替实测。

- 来源：[老师平台](https://course.terminiai.com/interview)，采集编号 P004-Q002、P004-Q022、P004-Q039、P004-Q167、P004-Q291、P004-Q306、P004-Q327。
- 依据：[MQA](https://arxiv.org/abs/1911.02150)、[GQA](https://arxiv.org/abs/2305.13245)、[DeepSeek-V2 / MLA](https://arxiv.org/abs/2405.04434)、[Llama 2](https://arxiv.org/abs/2307.09288)。

## 追问

- 怎样从 batch、层数、长度和 KV 头数估算缓存？
- GQA 的 KV 头数和每组 Query 头数怎样换算？
- 为什么不能把 MHA 在推理时直接无损改成 GQA？
- MLA 与 LoRA 的“低秩”分别作用在哪里？
- 怎样为一个长上下文服务选择 MHA、GQA 或 MLA？

## Note

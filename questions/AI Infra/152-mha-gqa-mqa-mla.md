---
difficulty: 中等
topic: KV共享注意力/MHA、GQA、MQA与MLA
summary: 比较四种注意力的KV头共享、潜变量压缩及推理质量取舍
tags: [面经, 待校对, MHA, GQA, MQA, MLA, KV Cache]
company: 小米
mastered: false
highfreq: false
---

## 题目

1. MHA、GQA（Grouped Query Attention）、MQA（Multi-Query Attention）和MLA（Multi-Loop Attention）是注意力机制的不同变体。请分别说明它们的设计动机与优化目标，特别是在推理效率、内存占用和模型性能之间的权衡。
2. MHA、GQA、MQA和MLA这些注意力变体各自的优化目标是什么？它们分别通过何种机制改进原始多头注意力在效率或效果上的局限？

## 要点

- 原题中的 Multi-Loop Attention 是误写；MLA 的正确全称是 Multi-head Latent Attention
- MHA 每个查询头有独立 KV 头；GQA 按组共享；MQA 所有查询头共享一组 KV
- MLA 不直接缓存完整 K/V，而是缓存低维潜变量及不能被吸收的位置信息
- Decode 常受 KV Cache 容量和带宽影响；Prefill 更偏矩阵计算
- 共享越强通常越省缓存，但质量取决于模型、训练方式和任务，不能写固定损失比例

## 答案

**四种方法主要在同一件事上做取舍：给每个查询头保留多少独立的 K/V 表达，以及为此付出多少 KV Cache 和带宽。**

先纠正题干中的术语：这里的 MLA 应是 **Multi-head Latent Attention**，不是 Multi-Loop Attention。

### 四种结构的核心区别

设查询头数为 $H_q$，KV 头数为 $H_{kv}$，每个 KV 头维度为 $d_h$。

| 结构 | KV组织方式 | 主要目标 | 主要取舍 |
|---|---|---|---|
| MHA | $H_{kv}=H_q$，每个查询头有独立K/V | 保留最完整的多头表达能力 | KV Cache最大，Decode读取量大 |
| GQA | 多个查询头共享一个KV头，$1<H_{kv}<H_q$ | 在质量和推理成本之间折中 | 组数越少通常越省缓存，但共享约束更强 |
| MQA | $H_{kv}=1$，所有查询头共享同一组K/V | 最大程度降低KV缓存与带宽 | KV多样性最受限制，质量需靠训练验证 |
| MLA | 把K/V相关信息压到低维latent中，需要时再投影到各头 | 用低维缓存保留较丰富的头间表达 | 结构和kernel更复杂，RoPE部分要单独处理 |

对普通 MHA/GQA/MQA，每层每个 token 的 KV Cache 元素数可粗略写为：

$$
2H_{kv}d_h
$$

乘以层数、token 数、批量和每元素字节数，才得到总缓存。GQA 和 MQA 直接通过减少 $H_{kv}$ 降低缓存。

### MLA 为什么不是简单的“更多共享”

MLA 先把输入压缩成较低维潜变量，再从潜变量恢复各个注意力头需要的 K/V 表示。推理时缓存的是潜变量，而不是每个头的完整 K/V。这样既能降低缓存，又能让不同查询头通过各自的投影得到不同表示。

在 Decode 中，一些投影矩阵可以通过矩阵结合提前吸收到查询侧或输出侧，避免每个历史 token 都显式恢复完整 K/V。它与 MQA 的差别在于：MQA 让所有查询头直接使用同一组 K/V；MLA 缓存共同的低维信息，但各头仍可通过投影形成不同的交互。

### MLA 怎样处理 RoPE

RoPE 会把位置相关旋转作用到查询和键上。若把所有键都压缩进一个潜变量，位置旋转与低秩投影通常不能随意交换，也就难以直接做矩阵吸收。

常见做法是把查询和键拆成两部分：

- 一部分不承载 RoPE，可走低秩压缩和矩阵吸收；
- 一小部分专门承载 RoPE，需要按 token 缓存相应的位置键。

因此 MLA 的缓存不是“只有一个 latent，其他什么都没有”，还要计入不能被吸收的 RoPE 键部分。

### Prefill 与 Decode 的收益不同

- **Prefill**一次处理很多 token，矩阵乘较大，计算效率相对高。KV压缩能减少写缓存和显存，但额外投影也可能带来计算成本。
- **Decode**每步只有少量新 token，却要读取全部历史缓存，常受内存带宽限制。减少每个历史 token 的缓存尺寸，通常更容易转化为吞吐或并发收益。

因此不能只看参数量判断速度。还要看序列长度、批量、kernel实现、显存带宽和是否真正避免显式恢复完整 K/V。

### GQA 分组数怎样选

组数增加时，KV头更多，模型更接近MHA，缓存和带宽成本也更高；组数减少时，更接近MQA，推理更省资源。实际选型要在目标硬件和上下文长度上，联合比较质量、首 token 延迟、单 token 延迟、吞吐和最大并发，不能只按一个固定比例选。

MLA 的低秩压缩也不要与 LoRA 混淆：MLA 的低秩表示是注意力前向结构的一部分，目标是压缩激活和缓存；LoRA 是参数高效微调方法，用低秩增量更新已有权重。

## 知识点

MHA、GQA、MQA、Multi-head Latent Attention、KV头共享、低秩潜变量、矩阵吸收、解耦RoPE、Prefill与Decode。

- 真实面经：[B002-G01-Q160](../../docs/references/面经原题.md#b002-g01-q160)、[B002-G01-Q161](../../docs/references/面经原题.md#b002-g01-q161)
- 老师参考：[P005-Q160](../../docs/references/平台题/P005-Infra-151-180.md#p005-q160)、[P005-Q161](../../docs/references/平台题/P005-Infra-151-180.md#p005-q161)

## 追问

- 参考追问：MLA相比MQA为什么能兼顾表达能力和低KV Cache？
- 参考追问：GQA的分组数如何影响精度和速度？
- 参考追问：DeepSeek MLA如何处理RoPE与低秩压缩的冲突？
- 参考追问：MLA中的低秩压缩与LoRA有什么本质区别？
- 参考追问：这些机制在Prefill和Decode阶段分别有什么性能特点？

## Note

---
difficulty: 中等
topic: KVCache/优化方法综述
summary: KV cache 的容量、布局、量化、复用和调度怎样优化
tags: [面经, 待校对, KVCache, PagedAttention, 推理优化]
company: 联通、美团、快手
mastered: false
highfreq: false
---

## 题目

KV cache 在 Transformer 解码中解决什么问题，又会带来哪些容量与管理瓶颈？请比较减少 KV、分页布局、量化、前缀复用、淘汰或卸载、连续批处理等优化方法及其适用场景。

## 要点

- 缓存历史 K/V 避免重复投影，但当前 token 仍要读取并关注历史 K/V
- 显存由层数、KV 头数、head 维度、dtype、batch 和长度共同决定
- GQA/MQA、量化减少逻辑字节；PagedAttention主要减少分配碎片
- 滑窗、选择性保留和卸载会改变可见历史或增加传输，必须评估质量与延迟
- 连续批处理提高复用率和吞吐，也需要 token 预算、抢占和公平性

## 答案

**KV cache 用显存换计算：历史 token 的 K/V 不再重复投影，但每个新 token 仍要读取历史 K/V 做注意力。** 因此它解决重复计算，却会让显存随并发和上下文长度线性增长。

设层数为 $L$、KV 头数为 $H_{kv}$、每头维度为 $d$、每元素字节数为 $b$，batch 为 $B$、缓存长度为 $S$：

$$
M_{KV}=2L H_{kv} d b B S
$$

前面的 2 表示 K 和 V。这个式子也说明该从哪里省：

| 方法 | 改变什么 | 适用边界 |
|---|---|---|
| GQA / MQA | 减少 $H_{kv}$ | 需重新训练或使用原生结构；头数越少不代表质量一定可接受 |
| KV 量化 | 减少 $b$ | K/V 的量化粒度可不同，必须用长文本和生成任务校准 |
| PagedAttention | 把逻辑序列映射到固定大小物理块 | 主要减少碎片并支持共享，不减少逻辑 KV 字节数 |
| 前缀缓存 | 相同前缀共享物理块，写入时复制 | 适合系统提示、模板重复高的负载；要按租户隔离 |
| 滑窗、StreamingLLM、H2O | 只保留部分历史 token | 真正减少 $S$，但可能丢失远距信息，需任务验证 |
| CPU/远端卸载 | 用容量换传输 | 适合低频块；热块搬运会放大 TPOT |

PagedAttention像页表：请求只维护逻辑块表，物理块不要求连续。块太大，最后一块浪费多；块太小，块表、调度和 kernel 元数据开销高，所以不能把 16 或 32 当成固定答案，应对目标模型和长度分布压测。

连续批处理在每轮 decode 后补入新请求，可提高 GPU 利用率；调度器还要限制总 token、预留显存，并按 deadline 或优先级抢占。长上下文 OOM 时通常组合 **GQA/MQA + KV 量化 + 分页分配 + 合理的上下文裁剪/卸载**。FlashAttention减少注意力中间量的 I/O，不会直接压缩持久 KV cache。

## 知识点

KV cache 显存公式、GQA/MQA、KV 量化、PagedAttention、前缀缓存、滑窗与淘汰、连续批处理。

真实面经：[B002-G01-Q005](../../docs/references/面经原题.md#b002-g01-q005)、[B002-G01-Q015](../../docs/references/面经原题.md#b002-g01-q015)、[B002-G01-Q105](../../docs/references/面经原题.md#b002-g01-q105)、[B002-G01-Q148](../../docs/references/面经原题.md#b002-g01-q148)、[B002-G01-Q166](../../docs/references/面经原题.md#b002-g01-q166)、[B002-G01-Q167](../../docs/references/面经原题.md#b002-g01-q167)、[B002-G01-Q178](../../docs/references/面经原题.md#b002-g01-q178)。

老师参考：[P005-Q005](../../docs/references/平台题/P005-Infra-001-030.md#p005-q005)、[P005-Q015](../../docs/references/平台题/P005-Infra-001-030.md#p005-q015)、[P005-Q105](../../docs/references/平台题/P005-Infra-091-120.md#p005-q105)、[P005-Q148](../../docs/references/平台题/P005-Infra-121-150.md#p005-q148)、[P005-Q166](../../docs/references/平台题/P005-Infra-151-180.md#p005-q166)、[P005-Q167](../../docs/references/平台题/P005-Infra-151-180.md#p005-q167)、[P005-Q178](../../docs/references/平台题/P005-Infra-151-180.md#p005-q178)。

## 追问

以下均为平台页面参考追问，不作为面经原话：

- 参考追问：PagedAttention怎样减少碎片，block size怎样调？
- 参考追问：KV量化怎样校准并评估精度？
- 参考追问：100K以上上下文发生KV OOM时怎样组合方案？
- 参考追问：GQA与MQA怎样选择KV头数并控制质量损失？
- 参考追问：动态批处理怎样权衡延迟、吞吐和抢占？
- 参考追问：显存不足时KV逐出、卸载和重算怎样选？
- 参考追问：StreamingLLM与H2O分别怎样减少保留的历史？

## Note

---
difficulty: 中等
topic: 线性注意力/长序列Attention方案比较
summary: 稀疏线性滑窗与FlashAttention怎样权衡长序列成本
tags: [真题, 长序列, 稀疏注意力, 线性注意力, FlashAttention, 待校对]
company: 字节、快手、阿里淘天、淘天、哔哩哔哩
mastered: false
highfreq: false
---

## 题目

标准 Attention 的时间和空间成本来自哪里？请比较全局感受野、滑动窗口、稀疏 Attention、线性 Attention、分块/序列并行与 FlashAttention，说明它们对训练、prefill 和 decode 的不同影响，以及怎样为超长文本选型。

## 要点

- 稠密注意力的序列项为 $O(n^2d)$，朴素保存分数/概率需 $O(n^2)$ 元素
- 局部与稀疏方案少算位置对，线性方案改变核形式，FlashAttention 保持精确结果
- 训练注意力中间量、prefill 激活和 decode KV Cache 要分开分析
- 全局感受野不一定更好，结构先验、噪声、硬件与任务共同决定

## 答案

**标准稠密 Attention 的瓶颈是长度为 $n$ 的序列要计算所有 $n^2$ 个位置对；不同方案分别从“少看位置、改计算形式、少搬数据或分摊序列”入手。** 隐藏维为 $d$ 时，Q/K/V 与输出投影约为 $O(nd^2)$，分数和加权求和约为 $O(n^2d)$；朴素实现显式保存注意力矩阵需要 $O(n^2)$ 个元素。

| 方法 | 改了什么 | 主要收益与代价 |
|---|---|---|
| 滑动窗口 | 每个位置只看附近窗口 | 成本随窗口线性增长，但窗口外不能当层直接访问 |
| 稀疏 Attention | 局部、全局 token、块或内容选择 | 保留部分远程访问；稀疏模式与不规则访存影响效果和速度 |
| 线性 Attention | 用核特征等改变/近似 softmax 形式并重排乘法 | 序列复杂度可降为线性；精确检索与归一化性质可能变化 |
| FlashAttention | 分块和在线 softmax，减少 HBM 读写 | 精确且省中间显存；稠密配对计算仍是 $O(n^2)$ |
| 分块/序列并行 | 把序列与计算分到设备或分段处理 | 扩展可处理长度；通信和边界信息成为成本 |

Ring Attention 属于分布式序列并行：让 K/V 块在设备间传递以完成注意力，不应叫稀疏注意力。分块若只是独立截断会丢跨块联系，若携带记忆、重叠或全局 token，则要分析新增的信息路径。

三种阶段的口径不同。训练和 prefill 同时处理整段序列，关心二次计算与中间激活；FlashAttention 可避免物化完整矩阵。自回归 decode 每次只有新 query，但需读取历史 K/V，单层每步工作和缓存都随已生成长度增长；仅优化训练矩阵不等于消除 KV Cache。

感受野也不是越大越好。全局注意力能直接捕捉远依赖，却会计算大量无关位置并缺少局部先验。原始 ViT 使用全局注意力，窗口化是 Swin 等变体的设计。BERT 的双向可见性与 GPT 的因果可见性属于 mask 问题，见 [因果与双向注意力](004-因果与双向注意力.md)；RNN/CNN 是否替代注意力见 [序列建模比较](026-transformer与循环网络.md)。为超长文档应先用“局部窗口 + 少量全局通道”建立基线，再按大海捞针、长文问答、困惑度、显存和端到端吞吐比较稀疏、线性或分布式方案；最大长度和加速倍数都必须绑定具体模型、内核和硬件。

## 知识点

长序列优化要明确它降低的是 FLOPs、中间激活、HBM IO、KV Cache 还是单卡长度。FlashAttention 是精确 IO 优化；Mamba 是选择性 SSM，不属于线性 Attention。

- 一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[Longformer](https://arxiv.org/abs/2004.05150)、[BigBird](https://arxiv.org/abs/2007.14062)、[Performer](https://arxiv.org/abs/2009.14794)、[FlashAttention](https://arxiv.org/abs/2205.14135)、[Ring Attention](https://arxiv.org/abs/2310.01889)。

## 追问

- FlashAttention 属于稀疏或线性注意力吗，为什么？
- 线性 Attention 为何不能只用“矩阵结合律”解释？
- 训练、prefill 与 decode 的内存瓶颈分别是什么？
- 局部加全局 token 的稀疏模式怎样平衡感受野与成本？
- 面向百万 token 文档，你会怎样组合并验证这些方案？

## Note

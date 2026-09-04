---
difficulty: 中等
topic: VLM结构/MLP与Q-Former连接器
summary: 比较逐token投影与查询压缩连接视觉编码器和语言模型的取舍
tags: [面经, 待校对, 多模态, VLM, MLP, Q-Former, LLaVA, BLIP-2]
company: 淘天、美团
mastered: false
highfreq: false
---

## 题目

连接视觉编码器和大语言模型时，LLaVA 式 MLP 与 BLIP-2 的 Q-Former 有什么区别？请比较它们的信息流、参数和训练成本、视觉细节、推理开销及适用条件。

## 要点

- MLP把每个视觉token投影到LLM嵌入空间，不等于先做全局平均池化
- Q-Former用少量可学习Query通过Cross-Attention读取冻结视觉特征
- 查询压缩降低送入LLM的token数，但固定容量可能漏掉小字和密集区域
- MLP结构简单，成本仍会随保留的视觉token数量和分辨率增长
- 比较结构时必须固定视觉塔、LLM、数据和视觉token预算

## 答案

**MLP 是“逐 token 翻译”，Q-Former 是“先用固定数量的问题采访图像，再交摘要”。** 两者都在解决视觉特征空间与语言模型词嵌入空间不一致的问题，但信息压缩发生的位置不同。

| 维度 | MLP连接器 | Q-Former |
|---|---|---|
| 输入输出 | 每个视觉token独立投影，通常数量不变 | 少量可学习Query读取全部图像特征，输出固定数量token |
| 模态交互 | 投影本身不在视觉token之间做注意力 | Query通过Cross-Attention选择和汇总视觉信息 |
| LLM侧成本 | 取决于保留多少视觉token | 查询数固定时，送入LLM的序列更短 |
| 训练 | 新模块简单，易先做特征对齐再指令微调 | Q-Former需要专门的视觉语言表征与生成连接训练 |
| 信息风险 | 高分辨率时token多、上下文和推理成本上升 | 压缩瓶颈可能漏掉OCR、局部属性和密集对象 |

LLaVA 的连接器对视觉编码器输出的网格 token 做线性层或两层 MLP 投影，再与文本 token 一起送入 LLM。它没有因为使用 MLP 就天然丢失全部空间结构；真正的限制来自视觉塔分辨率、位置表示、投影表达力和传入 LLM 的 token 预算。连接器预训练时可以冻结两端，但后续视觉指令微调的可训练范围要按具体版本说明。

BLIP-2 冻结图像编码器和 LLM，训练一个 Q-Former。可学习 Query 通过 Cross-Attention 从视觉特征中提取固定长度表示；第一阶段学习视觉语言表征，第二阶段把输出连接到冻结 LLM 做生成学习。若图像有 2048 个 token、最后只输出 32 个 Query，只能说交给 LLM 的 token 数缩小 64 倍，不能推出端到端计算必然降低 64 倍，因为 Q-Former 本身也要读取视觉特征。

选型没有结构级赢家。短视觉序列、快速迭代和易部署可先用 MLP；视觉序列很长且语言侧成本突出时，查询压缩有价值；小字、图表和精确定位任务则要重点检查压缩丢失。最终应在同一编码器、数据和LLM下比较任务准确率、视觉token数、首token延迟、吞吐及错误类型。

## 知识点

视觉语言连接器、逐token投影、可学习Query、Cross-Attention、视觉token压缩、信息瓶颈与分阶段训练。

- 真实面经：[B004-Q007](../../docs/references/面经原题.md#b004-g01-q007)、[Q008](../../docs/references/面经原题.md#b004-g01-q008)、[Q011](../../docs/references/面经原题.md#b004-g01-q011)
- 老师参考：[P007-Q007](../../docs/references/平台题/P007-MultiModal-001-020.md#p007-q007)、[Q008](../../docs/references/平台题/P007-MultiModal-001-020.md#p007-q008)、[Q011](../../docs/references/平台题/P007-MultiModal-001-020.md#p007-q011)
- 一手依据：[BLIP-2](https://arxiv.org/abs/2301.12597)、[LLaVA](https://arxiv.org/abs/2304.08485)

## 追问

- Query数量减少时，省掉的是哪一段计算，为什么不是等比例端到端加速？
- MLP不做token间交互，为什么仍能保留一部分空间信息？
- LLaVA从线性投影改为两层MLP说明了什么设计取舍？
- 文档理解和视频理解分别更担心哪种连接器瓶颈？
- 怎样设计MLP与查询压缩的混合连接器，并判断复杂度是否值得？

## Note

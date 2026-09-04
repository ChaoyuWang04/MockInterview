---
difficulty: 简单
topic: 检索优化/Reranker原理与训练数据
summary: 重排为何比初始召回更准，以及训练样本怎样贴近线上候选
tags: [面经, 待校对, RAG, Reranker, Cross-Encoder, 难负样本]
company: 字节、美团
mastered: false
highfreq: false
---

## 题目

RAG为什么需要Reranker，它怎样提高候选排序质量？训练Reranker时，正负样本、标签和数据来源应怎样设计？

## 要点

- 初始召回追求快和广，通常压缩了query与文档之间的细粒度交互
- Cross-Encoder联合编码query与候选，精度通常更高但不能预计算文档表示
- Reranker不只等于Cross-Encoder，late interaction等结构也可用于重排
- 训练数据应复现线上候选分布，并同时包含易负样本和难负样本
- 点击日志有曝光偏差，高分未点击文档也可能是假负例

## 答案

**Reranker解决的是“已经召回，但顺序不够准”。** 双塔检索可提前计算文档向量，速度快，却把query和文档各压成一个表示，容易漏掉数字、否定和局部对应关系。Cross-Encoder把两者一起输入，在token层交互后打分，通常更细致，但每个候选都要重新计算，所以只处理召回后的短候选集。ColBERT一类late interaction是另一种精度与成本折中，Reranker不等于Cross-Encoder。

训练时，正例应是真能支持问题答案的文档；负例既要有随机易负例，也要有当前检索器高分却不相关的难负例。数据可来自人工标注、经过偏差校正的日志、公开集和教师模型伪标签。未点击不等于不相关，要抽查假负例并保持训练候选与线上分布接近。pointwise、pairwise或listwise目标按线上排序任务选择。

验收同时看MRR、nDCG、目标答案进入上下文的比例、端到端答案质量和延迟。候选数、模型大小与阈值都应由这组曲线决定，没有通用固定值。

## 知识点

两阶段检索、Bi-Encoder、Cross-Encoder、late interaction、难负样本、假负例、pointwise、pairwise、listwise。

- 真实面经：[B003-Q037](../../docs/references/面经原题.md#b003-g01-q037)、[B003-Q042](../../docs/references/面经原题.md#b003-g01-q042)、[B003-Q075](../../docs/references/面经原题.md#b003-g01-q075)、[B003-Q084](../../docs/references/面经原题.md#b003-g01-q084)、[B003-Q094](../../docs/references/面经原题.md#b003-g01-q094)
- 老师参考：[P006-Q037](../../docs/references/平台题/P006-RAG-034-066.md#p006-q037)、[P006-Q042](../../docs/references/平台题/P006-RAG-034-066.md#p006-q042)、[P006-Q075](../../docs/references/平台题/P006-RAG-067-097.md#p006-q075)、[P006-Q084](../../docs/references/平台题/P006-RAG-067-097.md#p006-q084)、[P006-Q094](../../docs/references/平台题/P006-RAG-067-097.md#p006-q094)

## 追问

- 参考追问：重排模型与精排模型有什么区别，能否直接用LLM重排？
- 参考追问：候选集大小怎样确定，线上延迟怎样优化？
- 参考追问：重排后效果仍不好，应从哪些环节继续定位？
- 参考追问：难负样本怎样挖掘，如何避免把假负例学进去？
- 参考追问：怎样评估训练数据质量？
- 参考追问：Reranker与Embedding模型联合训练有什么利弊？

## Note

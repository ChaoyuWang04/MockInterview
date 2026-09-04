---
difficulty: 简单
topic: 推荐模型/推荐系统中的行为序列建模
summary: 比较BST排序建模与SDM长短期兴趣融合的结构和目标
tags: [待校对, SFT, 推荐模型]
company: 小红书
mastered: false
highfreq: false
---

## 题目

BST 与 SDM 如何建模推荐系统中的用户行为序列?请比较两者的任务、结构、时间信息和长短期兴趣,并说明怎样加入多模态内容特征。

## 要点

- BST 面向候选相关的排序/CTR,SDM 面向匹配召回
- BST 用 Transformer 编码行为序列
- SDM 用短期多头自注意力和长短期门控融合,不是 LSTM+Transformer
- 时间衰减可作为时间差特征、偏置或采样策略
- 多模态特征可离线编码,但需控制时延和漂移

## 答案

**BST 与 SDM 都利用行为序列,但服务阶段不同:BST 主要增强候选物品的 CTR 排序,SDM 主要学习用于大规模匹配的用户表示。** BST 把用户行为物品的 Embedding 加入位置等信息,用 Transformer self-attention 建模行为间关系,再与候选物品和其他特征拼接完成点击预测。它能并行捕捉不同历史行为的依赖,但序列变长会增加二次注意力成本。

SDM 先从当前 session 建模短期兴趣:原论文使用 multi-head self-attention 聚合短期行为;长期部分从历史行为和用户画像提取稳定偏好,再通过 long-short term gated fusion 形成最终用户向量,用于与物品向量匹配。它不能概括为“底层 LSTM、上层 Transformer”。BST 的 CTR 损失与 SDM 的匹配损失对应不同任务;若在一个系统联合训练,应明确共享哪些 Embedding、各损失权重和负采样,而不是假定论文原生就是同一分层网络。

位置编码只表达顺序,不会自动知道“昨天”和“半年前”的差异。可把时间戳、时间间隔、周期特征显式加入 Embedding,在注意力分数中加时间偏置,或在采样时提高近期行为权重。具体衰减应由数据学习或消融选择,避免固定规则抹掉长期周期兴趣。

内容社区可先用图文编码器离线提取笔记表示,与 ID、作者和行为类型特征融合;新内容可用内容向量缓解冷启动。在线模型仍要监控特征版本、缓存新鲜度、序列长度和延迟,并分别评估召回、排序和多模态切片收益。

## 知识点

BST 是排序侧行为 Transformer,SDM 是匹配侧短期注意力与长期兴趣门控融合;两者结构与损失不可混写。

来源:[深维 LLM 平台](https://course.terminiai.com/interview),P004-Q176。

一手依据:[Behavior Sequence Transformer](https://arxiv.org/abs/1905.06874)、[SDM](https://arxiv.org/abs/1909.00385)。

## 追问

- BST 如何把真实时间差与序列位置同时编码?
- SDM 的短期兴趣和长期兴趣如何融合?
- 若联合训练召回与排序,损失和共享参数怎样设计?
- 内容社区中怎样加入图像、文本和作者特征?

## Note

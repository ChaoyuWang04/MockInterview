---
difficulty: 简单
topic: 推荐模型/长行为序列兴趣建模
summary: 比较DIN候选感知激活与SIM两阶段长序列检索
tags: [待校对, SFT, 推荐模型]
company: 阿里
mastered: false
highfreq: false
---

## 题目

DIN 与 SIM 如何建模用户行为兴趣?在很长行为序列下,SIM 的两阶段检索怎样降低成本,hard search、soft search 和 Top-k 应如何理解?

## 要点

- DIN 针对当前候选物品激活相关历史行为
- DIN 的局部激活单元不等同于 Transformer self-attention
- SIM 先粗选候选行为,再做精细兴趣聚合
- hard search 与 soft search 是 GSU 的替代方案
- Top-k、序列长度与线上收益都依赖数据和系统配置

## 答案

**DIN 的核心是“同一个用户面对不同候选物品,应激活不同历史兴趣”;SIM 则先从超长历史中找相关片段,再做精细建模。** DIN 将候选物品与每个历史行为一起输入局部激活单元,得到候选相关权重,加权汇总为兴趣向量后用于 CTR 预测。它是候选到历史的逐项相关性计算,不是历史 token 两两交互的标准 self-attention。

当历史很长时,DIN 对全部行为计算精细相关性会增加延迟。SIM 把过程拆成 General Search Unit(GSU) 和 Exact Search Unit(ESU):GSU 以较低成本从长历史选出与候选相关的子序列,ESU 再对这部分行为做更细的候选感知聚合。论文中的 hard search 按物品类别等规则筛选,soft search 用可学习的相似表示检索;它们是 GSU 的两种实现,不是必须串行的两个阶段。

百万级或更长历史的关键是把行为存储与在线模型拆开:离线或增量维护行为索引,按时间窗口、类别或向量近邻粗召回,缓存物品表示,再限制 ESU 输入长度。Top-k 过小会漏掉兴趣,过大则增加计算和噪声;可按延迟预算离线网格搜索,也可让阈值或候选预算随用户活跃度变化,但必须检查召回覆盖和最终排序指标。

SIM 相比 DIN 的关键改进是两阶段搜索带来的长序列可扩展性;与 DIEN 的兴趣演化建模也不是同一维度。原论文报告的序列长度和线上收益只适用于其数据、索引与实验设置,不能当作所有业务的固定能力。

## 知识点

DIN 解决候选感知兴趣聚合,SIM 在此之前加入长序列粗检索;GSU 的 hard/soft search 需按论文定义区分。

来源:[深维 LLM 平台](https://course.terminiai.com/interview),P004-Q080、P004-Q081。

一手依据:[Deep Interest Network](https://arxiv.org/abs/1706.06978)、[Search-based User Interest Modeling](https://arxiv.org/abs/2006.05639)。

## 追问

- 百万级行为历史下怎样设计存储、检索和在线聚合?
- SIM 相对 DIN 与 DIEN 的核心改进分别是什么?
- DIN/SIM 的候选感知权重与 Transformer self-attention 有何区别?
- hard search 和 soft search 如何选择?
- soft search 的 Top-k 能否按用户或请求动态调整?

## Note

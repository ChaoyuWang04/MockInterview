---
difficulty: 简单
topic: 分类损失/类别不平衡
summary: Focal Loss如何压低易样本贡献,参数和适用边界是什么
tags: [真题, Focal Loss, 类别不平衡, 目标检测, 难样本挖掘, 待校对]
company: 小红书
mastered: false
highfreq: false
---

## 题目

请在白板上写出 Focal Loss 的完整数学公式,详细解释其设计动机、参数作用及如何有效缓解类别不平衡问题,并结合目标检测或分类任务说明其应用场景和优势。

## 要点

- 写清交叉熵、类别权重与聚焦因子,定义 $p_t$、$\alpha_t$、$\gamma$
- 解释大量易背景样本为什么会主导损失及如何降低其贡献
- 区分类别不平衡与难易样本不平衡,不声称自动筛出固定比例样本
- 回答 OHEM 对比、Transformer 检测器适用性与调参风险

## 答案

**Focal Loss 在交叉熵上乘一个随预测置信度变化的因子,压低易样本的贡献,让难样本获得更多相对权重。** 它主要针对密集检测里大量易背景样本累积后淹没前景信号的问题。

二分类中,$y\in\{0,1\}$,$p$ 为正类概率;正确标签的概率 $p_t$ 在 $y=1$ 时取 $p$,否则取 $1-p$。类别权重 $\alpha_t$ 对正类取 $\alpha$,负类取 $1-\alpha$:

$$
\mathrm{CE}=-\log p_t,\qquad
\mathrm{FL}=-\alpha_t(1-p_t)^\gamma\log p_t,\quad \gamma\ge0
$$

$\alpha_t$ 平衡类别贡献,$\gamma$ 控制压低易样本的强度。举例:$p_t=0.9,\gamma=2$ 时,聚焦因子是 $(1-0.9)^2=0.01$;若 $p_t=0.1$,因子为 0.81。它连续加权所有样本,没有固定“保留 2%”的步骤。$\gamma=0$ 时退化为**带类别权重的交叉熵**。

RetinaNet 用它减轻易背景主导的问题;收益取决于正负样本构成和噪声,并非所有分类任务都优于交叉熵。

### 相关真题追问

- **与 OHEM 有何不同?** OHEM 按损失等规则挑选难样本子集,涉及筛选数量与排序;Focal Loss 用可微权重连续调节,仍计算易样本。两者都可能过度关注错标样本。
- **Transformer 检测器还有效吗?** 关键是分类目标和样本不平衡,与是否用 Transformer 没有直接等价关系。原始 DETR 用类别负对数似然并降低空目标权重;Deformable DETR 使用 Focal Loss,是可用实例,不是任意替换都提升的保证。
- **参数怎样调?** $\gamma$ 太大可能压掉大部分信号并放大噪声样本的相对影响;$\alpha$ 不当会偏向某类。把加权 CE 当基线,联合搜索两者,按分类型召回、精确率或检测 AP 选择。

## 知识点

类别权重、聚焦因子、易负例、OHEM、标签噪声。

- 依据:[Focal Loss 原论文 §3](https://arxiv.org/abs/1708.02002)、[DETR 原论文 §3](https://arxiv.org/abs/2005.12872)、[Deformable DETR 原论文 §5](https://arxiv.org/abs/2010.04159)。

## 追问

- Focal Loss 和 OHEM 有什么区别,各有什么优缺点?
- Focal Loss 在 Transformer 检测器(如 DETR)中是否仍然有效,为什么?
- 如果 $\gamma$ 过大或 $\alpha$ 设置不当会出现什么问题,如何调参?

## Note

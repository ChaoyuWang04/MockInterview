---
difficulty: 中等
topic: 对比学习/InfoNCE与BCE
summary: InfoNCE与BCE如何归一化和分配梯度,为何都能用于图文对齐
tags: [对比学习, 待校对]
company: 阿里
mastered: false
highfreq: false
---

## 题目

请详细比较InfoNCE损失函数与二元交叉熵（BCE）损失函数在数学形式、理论推导、优化目标及典型应用场景上的异同，结合表示学习与对比学习的背景，从原理和实践角度分析两者适用条件与设计动机。

## 要点

- InfoNCE是候选集合的分类似然,BCE是逐对Bernoulli似然
- 写清分母和候选数量,互信息解释有采样条件
- BCE不要求人工标签,也能学对比表示
- 解释温度、难例及跨批次负样本的取舍

## 答案

**InfoNCE 在候选之间做 softmax 竞争,BCE 对每个样本或配对独立做 sigmoid 判别;两者都能用于表示学习,不由是否有人工标签来区分。** 令候选分数为 $z_j=s_j/\tau$,正例编号为 $+$:

$$
L_{\mathrm{NCE}}=-\log\frac{e^{z_+}}{\sum_{j=1}^{N}e^{z_j}},\qquad
L_{\mathrm{BCE}}=-\sum_j[y_j\log\sigma(z_j)+(1-y_j)\log(1-\sigma(z_j))]
$$

| 比较 | InfoNCE | BCE |
|---|---|---|
| 似然解释 | 在 $N$ 个候选中选出正例 | 每对是否匹配的伯努利似然 |
| 对分数的梯度 | $p_j-\mathbf1_{j=+}$,其他候选改变时梯度也变 | $\sigma(z_j)-y_j$,各对不靠分母直接竞争 |
| 适用 | 单正例检索、实例判别、图文对齐 | 二分类、多标签、独立配对匹配 |
| 取舍 | 依赖候选集合,要处理假负例 | 要控制正负比例、分数尺度和偏置,概率校准仍需检验 |

当一个正例来自联合分布、其余候选独立来自边缘分布等条件成立时,有 $I(X;Y)\ge\log N-L_{\mathrm{NCE}}$ 的互信息下界解释;若有 $K$ 个负例则 $N=K+1$。任意难例挖掘后不能直接沿用同一保证。BCE 也可用自动配对产生标签,不是必须人工标注。

### 图文对齐、温度与候选规模

CLIP 采用双向候选分类,让每幅图像找对应文本、每段文本找对应图像。BCE 并非不能做这件事:SigLIP 用带可学习尺度和偏置的逐对 sigmoid 损失,避免全局 softmax 归一化。优劣需在同数据预算下比较。

小温度使分布更尖锐,对高分负例更敏感,且对相似度求导多一个 $1/\tau$;过小可能放大假负例和噪声。大温度更平滑,不能固定推荐某个数值。联合调节温度与负例质量,用稳定 log-sum-exp 或带 logits 的 BCE 实现。

批内对比可通过跨设备收集表示增加负例;动量队列可在小 batch 下增加候选,代价是表示滞后。梯度缓存可节省激活显存,但仅普通梯度累积不会自动增加每次对比的候选集合。正负样本质量比盲目放大 batch 更需要验证。

## 知识点

对比学习、InfoNCE与BCE。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q067。

- 补充依据:[InfoNCE 与互信息下界](https://arxiv.org/abs/1807.03748)、[SigLIP](https://arxiv.org/abs/2303.15343)、[CLIP](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)。

## 追问


- 为什么CLIP用InfoNCE而不是BCE做图文对齐？
- InfoNCE的temperature参数如何影响优化动态？
- 在实际训练中如何缓解InfoNCE对batch size的依赖？

## Note

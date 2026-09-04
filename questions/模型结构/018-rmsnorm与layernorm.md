---
difficulty: 简单
topic: Norm位置/为何省去均值中心化
summary: RMSNorm省略均值中心化后怎样控制尺度及如何选型
tags: [真题, RMSNorm, LayerNorm, 归一化, 混合精度, 待校对]
company: 快手、小红书
mastered: false
highfreq: false
---

## 题目

RMSNorm 为什么可以省去 LayerNorm 的均值中心化？请比较两者的数学定义、参数、计算与数值稳定性，并说明在现代大模型中选择 RMSNorm 的理由、限制以及混合精度实现要点。

## 要点

- LN 同时中心化和缩放，RMSNorm 只按均方根缩放
- RMSNorm 更简单且容易高效实现，但没有固定加速比例
- 去掉中心化是否影响效果取决于模型、任务和训练配方
- 混合精度下应关注归约精度与 $\epsilon$，不能凭名称判断谁必然更稳

## 答案

**RMSNorm 保留了对向量整体尺度的控制，省去了均值中心化；很多 Transformer 能在这种更简单的约束下稳定训练，但这不是数学上的等价替换保证。** 对一个 token 的隐藏向量 $x\in\mathbb{R}^d$：

$$
\operatorname{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\frac1d\sum_i(x_i-\mu)^2+\epsilon}}+\beta,
$$

$$
\operatorname{RMSNorm}(x)=\gamma\odot\frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}.
$$

LN 同时提供重中心化与重缩放；RMSNorm 只做重缩放，标准形式通常没有 $\beta$。后一种少一次均值归约和减均值，数据流更简单，也更容易融合成高效内核。实际加速取决于隐藏维、内核、硬件、精度和整个模型中 Norm 的占比，不能背固定的 5%、10% 或 30%。

去掉均值后仍可能稳定，是因为残差网络更关键的需求常是限制输入幅度和各维尺度，让子层看到可控的数值范围；RMSNorm 的论文用实验说明这在多种设置中可行。若模型确实依赖平移不变性或均值变化携带了不希望保留的偏移，LN 仍可能更合适，需做同预算消融。

FP16/BF16 下，两者都涉及平方和归约，常在内部升到更高精度计算统计量，再转换回模型 dtype；$\epsilon$ 太小可能放大接近零向量的数值误差，太大会削弱归一化。LN 还多了均值与中心化步骤，但不能据此断言 RMSNorm 在所有实现中必然更稳定。设计变体时应先明确要解决的是均值偏移、尺度、残差增长还是注意力 logit，再决定加门控、缩放或不同位置，而不是盲目叠加 Norm。LN/BN 的统计轴见 [计算轴与适用场景](016-layernorm与batchnorm.md)，DeepNorm 与 Pre/Post-LN 的位置和残差问题见 [梯度、稳定性与收敛](017-pre-ln与post-ln.md)。

## 知识点

RMSNorm 省去的是“减均值”这项重中心化，不只是少一次除法；它保留按均方根控制尺度的能力。效果和速度都需要在具体模型、任务与内核上验证。

- 一手依据：[RMSNorm](https://arxiv.org/abs/1910.07467)、[Layer Normalization](https://arxiv.org/abs/1607.06450)、[LLaMA](https://arxiv.org/abs/2302.13971)。

## 追问

- RMSNorm 去掉均值后为什么仍可能稳定，什么场景更需要保留中心化？
- 两者的 $\gamma$、$\beta$ 和 $\epsilon$ 有什么差别？
- FP16/BF16 实现为什么常用更高精度做统计归约？
- RMSNorm 的速度收益为什么不能写成固定百分比？
- DeepNorm、Pre-LN/Post-LN 与 RMSNorm 是否在解决同一个问题？

## Note

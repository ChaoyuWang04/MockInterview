---
difficulty: 简单
topic: Norm位置/梯度 稳定性与收敛
summary: Pre-LN与Post-LN的计算图如何影响梯度和训练稳定性
tags: [Transformer, LayerNorm, Pre-LN, Post-LN, 待校对]
company: 小米、滴滴
mastered: false
highfreq: false
---

## 题目

Transformer 中 Pre-LN 与 Post-LN 分别把归一化放在哪里？请画清残差计算关系，比较它们的梯度传播、初始化与 warm-up 敏感性、深层训练和最终收敛表现；设计百层以上模型时还要考虑什么？

## 要点

- 两个公式不能颠倒，原始 Transformer 使用 Post-LN
- Pre-LN 的残差主路含恒等路径，初始化梯度通常更温和
- “更稳定”不等于一定不用 warm-up，也不等于最终效果必然高或低
- 极深模型还依赖初始化、残差缩放和学习率等整体配方

## 答案

**Pre-LN 把 Norm 放在子层之前，Post-LN 把 Norm 放在残差相加之后；前者保留了更直接的恒等梯度路径，因此深层训练通常更稳定。** 设注意力或 FFN 子层为 $F$：

$$
\text{Post-LN: }y=\operatorname{LN}(x+F(x)),
\qquad
\text{Pre-LN: }y=x+F(\operatorname{LN}(x)).
$$

原始 Transformer 使用 Post-LN。反向传播时，它从输出回到底层要反复经过 LN 的雅可比，初始化阶段靠近输出层的梯度可能偏大，所以学习率 warm-up 往往更敏感。Pre-LN 的残差主路是 $x\rightarrow y$ 的恒等加法，梯度中始终有直接传递的一项，因而对深度和初始化通常更宽容。Xiong 等人的结论来自其理论假设与实验设置，应表述为“可降低 warm-up 依赖”，不是保证所有模型都能删掉 warm-up。

两种位置使用的 LN 本身都可有逐维缩放 $\gamma$ 和偏移 $\beta$；位置改变的是计算图，不是 LN 参数的定义。Pre-LN 的表示会沿残差流累加，较深层新增分支相对主路可能变小；Post-LN 每层都重新规范残差和，但更难优化。哪个最终精度更高没有无条件结论，要在相同深度、参数量、优化器和训练预算下比较。

设计百层以上模型时，我会先用 Pre-LN 建立稳定基线，同时监控各层激活、梯度和残差分支占比；再比较残差缩放、DeepNorm 一类按深度设计的缩放与初始化、或前后都放 Norm 的混合方案。RMSNorm 主要改变归一化计算，不能单独解决 Norm 位置造成的全部梯度问题。LN 与 BN 的统计轴和训推差别属于另一维度，见 [LayerNorm 与 BatchNorm](016-layernorm与batchnorm.md)。

## 知识点

判断 Pre/Post-LN 的诀窍是看“加号之后还有没有 Norm”。Pre-LN 给梯度留下恒等主路，但极深训练仍是归一化、初始化、残差尺度、优化器与学习率共同作用的结果。

- 来源：[老师平台](https://course.terminiai.com/interview)，P004-Q231、P004-Q232、P004-Q258。
- 一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)、[DeepNet / DeepNorm](https://arxiv.org/abs/2203.00555)。

## 追问

- 为什么 Post-LN 通常对 warm-up 更敏感，Pre-LN 是否一定能省略 warm-up？
- Pre-LN 的恒等梯度路径怎样从公式中看出来？
- Pre-LN 是否一定牺牲最终精度，应怎样公平验证？
- 百层以上 Transformer 除了改 Norm 位置还要调整什么？
- DeepNorm、Sandwich-LN 与 RMSNorm 分别改变了哪个维度的问题？

## Note

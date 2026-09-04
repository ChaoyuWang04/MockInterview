---
difficulty: 简单
topic: 残差流/为何能训练深层网络
summary: 残差连接如何提供信息与梯度通路及其能力边界
tags: [残差连接, Transformer, ResNet, 梯度传播, 待校对]
company: 快手、腾讯
mastered: false
highfreq: false
---

## 题目

残差连接为什么能帮助训练深层网络？请从 ResNet 的网络退化现象讲到 Transformer 的残差流，说明恒等捷径如何影响信息与梯度传播、怎样配合 Pre-LN，以及为什么它仍不能单独保证极深网络可训练。

## 要点

- 残差块学习 $F(x)$，输出为 $x+F(x)$，最差可接近恒等映射
- 梯度包含恒等项，给深层网络提供更短、更直接的反向路径
- 网络退化是训练误差随无效加深反而升高，不等于过拟合
- 极深训练还依赖 Norm、初始化、残差尺度、学习率和数值精度

## 答案

**残差连接把“直接保留原表示”和“学习一个修正量”相加，让前向信息和反向梯度都有一条恒等通路。** 一个残差块写成

$$
y=x+F(x),\qquad \frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}.
$$

即使当前分支 $F$ 没学好，网络也可以把它推向零，使该块近似恒等映射；反向时梯度中含 $I$，无需所有信号都连乘子层的雅可比。这使优化器更容易训练加深后的模型。

ResNet 提出的“网络退化”是：在训练集上，更深的普通网络反而可能比浅网络误差更高，原因是优化困难，不能直接叫过拟合；它也不只等同于梯度消失。残差连接改善了可优化性，但若 $I+J_F$ 的连乘长期偏大或偏小，仍会爆炸或衰减。

Transformer 每层把注意力和 FFN 的输出写回同一条残差流。常见 Pre-LN 形式为

$$
x_{l+1}=x_l+F(\operatorname{Norm}(x_l)),
$$

Norm 在分支内，$x_l$ 到加号的主路保持恒等，梯度更直接；Post-LN 则在相加后再归一化，梯度路径不同。Dropout 通常作用在残差分支输出再相加，训练和推理的缩放规则必须一致。

残差连接不是“无限加深许可证”。层数很大时，残差流幅度、各层更新占比、初始化、Norm 位置、学习率与低精度误差仍可能失控。ResNet 的 bottleneck 用 $1\times1$ 卷积压缩和恢复通道，pre-activation 则把归一化与激活移到卷积分支之前；二者仍保留恒等捷径。还可比较按深度缩放残差或初始化、DeepNorm、ReZero 等方案：ReZero 用初始接近零的可学习系数控制分支注入；Fixup 则用特定初始化和缩放训练无归一化残差网络。它们解决的条件不同，需用层级激活、梯度和任务指标验证。

## 知识点

残差连接的两个核心作用是保留表示和提供恒等梯度项。它缓解优化困难，但网络退化、过拟合、梯度消失是不同概念，不能混为一个因果解释。

- 来源：[老师平台](https://course.terminiai.com/interview)，P004-Q055、P004-Q192。
- 一手依据：[Deep Residual Learning](https://arxiv.org/abs/1512.03385)、[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)、[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)、[ReZero](https://arxiv.org/abs/2003.04887)。

## 追问

- 网络退化与过拟合、梯度消失有什么区别？
- 从 $I+J_F$ 怎样理解残差连接的梯度通路？
- Transformer 为什么常把残差连接与 Pre-LN 配合？
- 残差分支与 Dropout 同用时，训练和推理要注意什么？
- ReZero、Fixup 或残差缩放分别试图解决什么问题？

## Note

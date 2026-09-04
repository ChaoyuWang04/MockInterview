---
difficulty: 简单
topic: 优化器/自适应更新与衰减
summary: Adam 的矩估计与偏差修正如何工作,AdamW 在哪里解耦衰减
tags: [待校对, 优化器]
company: 阿里云、小红书、抖音
mastered: false
highfreq: false
---

## 题目

请介绍常见深度学习优化算法,重点比较 SGD、Adam 与 AdamW 的更新原理、自适应机制、权重衰减、收敛与泛化表现,并解释 AdamW 与 Adam 加 L2 正则为何不同。

## 要点

- SGD 可配动量,Adam 用梯度一阶矩与非中心二阶矩
- 偏差修正补偿零初始化,不是保证收敛
- L2 项进入 Adam 的矩统计,AdamW 直接衰减权重
- 速度、泛化与稳定性均需同预算调参比较

## 答案

**Adam 用历史梯度的方向与尺度调整更新,AdamW 进一步把权重衰减移出矩统计。** SGD 可只用当前梯度,也可带动量;Adagrad 累积梯度平方,RMSProp 改用指数平均,Adam 结合一阶矩与平方尺度。

### Adam 的更新

设 $g_t=\nabla L(\theta_{t-1})$,各运算逐元素进行:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_t=\theta_{t-1}-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

$m$ 平滑方向,$v$ 是非中心二阶矩,不是方差或 Hessian。零初始化造成早期统计偏小,分母修正此偏差;自适应尺度并不取代全局学习率调度,也不保证每次更新或泛化优于 SGD。

### AdamW 为什么不同

给损失加 $\lambda\|\theta\|^2/2$ 后,梯度变为 $\tilde g_t=g_t+\lambda\theta_{t-1}$,它进入 Adam 的 $m,v$ 两套历史统计。AdamW 的矩只由任务梯度计算,另做:

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

因此 L2 的影响既被逐参数缩放又改变历史,不能化为统一乘法衰减。无动量普通 SGD 可在相应系数下等价,Adam 则不成立。解耦让衰减更易解释,并非“获得真正的 L2”;同组零任务更新时总收缩仍为 $\prod_t(1-\eta_t\lambda)$,调度和总步数变化后要重新验证衰减,不必机械反比调整。

### 选型与震荡

AdamW 对不同梯度尺度适应方便,可作 Transformer 基线;调好的 SGD+Momentum 在部分视觉任务可能更好,没有按领域固定胜负。震荡先查数据与实际更新尺度,再试较小学习率或 warmup;较大 $\beta_1$ 使方向更平滑但滞后,$\beta_2$ 太大也会跟不上梯度尺度变化。$\epsilon$ 太小且状态精度不足会产生数值风险,太大会削弱自适应,应检查计算精度与更新范数,不盲调大。weight decay 控制收缩,不是修复 NaN 的开关。调度见[学习率](087-学习率调度.md),Sophia/Adafactor 见[选型](069-优化器选型.md)。

## 知识点

优化器/自适应更新与衰减。

来源:[深维 LLM 平台](https://course.terminiai.com/interview),P002-Q068、P002-Q112、P002-Q137、P002-Q173。参考:[Adam](https://arxiv.org/abs/1412.6980)、[AdamW](https://arxiv.org/abs/1711.05101)。

## 追问

- 为什么AdamW在Transformer训练中几乎成为标配,而CV领域仍常用SGD+Momentum?
- 如果训练中出现loss震荡,你会如何调整AdamW的超参数(β1,β2,weight decay,lr)?
- AdamW的weight decay和L2正则化在数学上不等价,请推导说明。
- 为什么AdamW在大模型训练中比Adam更常用,能结合具体场景说明吗?
- AdamW中的weight decay和L2正则化在数学上到底有什么区别?
- 如果学习率调度采用warmup+decay,weight decay应该如何配合调整?
- AdamW与Adam的区别是什么,为什么大模型微调更常用AdamW?
- Adam的二阶矩估计在实际实现中可能遇到什么问题(如epsilon设置、数值稳定性)?
- 在什么场景下SGD+Momentum可能比Adam效果更好?
- 为什么AdamW的权重衰减实现比Adam的L2正则化更优?
- 大模型训练中AdamW的学习率调度策略有什么特殊考虑?
- 什么场景下你会考虑用Sophia或Adafactor替代AdamW?

## Note

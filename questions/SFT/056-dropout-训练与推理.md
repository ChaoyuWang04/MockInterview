---
difficulty: 简单
topic: 泛化与正则化/Dropout
summary: Dropout如何缩放激活,训练推理及归一化顺序有何区别
tags: [泛化与正则化, 待校对]
company: 小红书、美团
mastered: false
highfreq: false
---

## 题目

请说明Dropout机制的实现方式（训练与推理阶段的区别），并解释其在神经网络训练中的正则化作用及防止过拟合的原理。

## 要点

- 写出Bernoulli掩码和inverted缩放,推理恒等
- 解释随机扰动的正则化作用与计算代价
- 区分DropPath、MC Dropout和常规推理
- 说明BN/LN顺序、位置和概率选择

## 答案

**Dropout 在训练时随机置零部分激活,促使网络减少对少数特征组合的依赖;常规推理关闭随机掩码。** 设丢弃概率 $p<1$,每个元素独立采样 $m\sim\mathrm{Bernoulli}(1-p)$,inverted Dropout 的训练输出为

$$
\tilde h=\frac{m\odot h}{1-p},\qquad\mathbb E[\tilde h\mid h]=h
$$

推理直接输出 $h$。保持的是本层激活期望,不代表整个非线性网络等于所有子网络的精确平均。随机扰动可能改善泛化,也可能增加梯度噪声、减慢收敛;稠密计算通常仍执行矩阵乘,不能按置零比例宣称提速。

### 实现与缩放约定

```python
import torch

def dropout(x, p, training=True):
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0, 1]")
    if not training or p == 0:
        return x
    if p == 1:
        return torch.zeros_like(x)
    return x * (torch.rand_like(x) >= p).to(x.dtype) / (1 - p)
```

这里假定浮点输入。若训练没有除以 $1-p$,采用的是传统约定:推理须在对应层乘 $1-p$,不能只在整网最终输出乘一次,也不能两阶段重复缩放。已有 BN 统计时还要检查统计是否匹配。

### 位置、概率与推理例外

- **BN/LN 顺序:**Dropout 放在 BN 前会改变 BN 看到的方差,使训练与推理统计不匹配;可优先比较 BN 后放置。LN 不维护运行均值,但顺序仍改变函数。Pre-LN 残差块可写成 $x+\mathrm{Dropout}(F(\mathrm{LN}(x)))$,不应随意交换。
- **Transformer 位置:**注意力概率上的掩码扰动 token 间连接,FFN 输出上的掩码扰动通道表示,不是同一作用。DropPath 通常按样本丢弃整条残差分支,适合某些深残差网络,不能声称所有 Transformer 都更常用它。
- **怎样选概率:**从关闭与较小非零值作对照,按 attention、FFN、adapter 分别验证。大规模预训练已有丰富数据时,额外噪声未必有益;小数据微调可能获益。数据去重、早停、权重衰减可比较,但 LN 不是等价替代,没有通用最优 $p$。
- **MC Dropout:**推理时保持掩码,多次预测取均值与方差,在一定假设下近似贝叶斯预测不确定性。代价是多次前向,不保证自动校准;同时保留 BN 的评估统计,不要把整网都切回训练模式。

## 知识点

泛化与正则化、Dropout。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q096、P002-Q120、P002-Q145、P002-Q196、P002-Q208、P002-Q227、P002-Q258。

- 补充依据:[PyTorch Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html)、[MC Dropout 原论文](https://proceedings.mlr.press/v48/gal16.html)、[Stochastic Depth 原论文](https://arxiv.org/abs/1603.09382)。

## 追问

- 怎样手写 Dropout，怎样处理 p=0、p=1，以及训练时未做 inverted 缩放的情况?
- Dropout 与 BatchNorm、LayerNorm 的顺序为什么会影响训练和推理?
- 注意力概率与 FFN 输出上的 Dropout 各扰动什么?
- Dropout 与 DropPath 有什么区别，分别适合哪些结构?
- 为什么有的大模型配方关闭 Dropout，哪些微调场景仍值得使用?
- 怎样选择 Dropout 概率，不同层是否应使用不同数值?
- 从贝叶斯角度怎样理解 MC Dropout，推理开启时需注意什么?

## Note

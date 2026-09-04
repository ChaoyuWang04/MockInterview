---
difficulty: 中等
topic: 反向传播/常见损失梯度
tags: [真题, 待校对, 手撕代码, MSE, 交叉熵, 反向传播]
summary: 推导 MSE 与交叉熵对输出和 logits 的梯度并解释反传
company: 网易
mastered: false
highfreq: false
---

## 题目

推导 MSE、二分类交叉熵和多分类交叉熵相对于模型输出或 logits 的梯度，并说明这些梯度怎样通过链式法则更新前层参数。解释求导变量、损失归约、激活函数和分类时 MSE/交叉熵的差异。

## 要点

- MSE 的系数取决于 `mean/sum` 以及公式是否含 $1/2$，必须先声明定义。
- BCE 对概率 $p$ 的梯度不是 $p-y$；BCE 与 sigmoid 合并后，对 logit $z$ 才是 $p-y$。
- Softmax 与交叉熵合并后，对每个 logit 的梯度是 $p_j-y_j$。
- 前层梯度由权重矩阵、激活导数和后层误差信号按链式法则相乘。
- 深层梯度消失/爆炸来自 Jacobian 连乘；损失选择只是其中一环。

## 答案

**先说清对什么求导。** 对 $N$ 个标量预测，若

$$
L_{MSE}=\frac1N\sum_i(\hat y_i-y_i)^2,
$$

则

$$
\frac{\partial L}{\partial \hat y_i}=\frac2N(\hat y_i-y_i).
$$

若损失含 $1/2$，系数 2 消失；若按 `sum` 归约，也没有 $1/N$。

二分类中 $p=\sigma(z)$：

$$
L=-y\log p-(1-y)\log(1-p),\quad
\frac{\partial L}{\partial p}=\frac{p-y}{p(1-p)}.
$$

再乘 $\partial p/\partial z=p(1-p)$，才得到 $\partial L/\partial z=p-y$。多分类令 $p_j=\operatorname{softmax}(z)_j$，交叉熵 $L=-\sum_j y_j\log p_j$，联合求导同样得到

$$
\frac{\partial L}{\partial z_j}=p_j-y_j.
$$

它不是任意交叉熵对“模型输出”的通用公式，而是对 logits 的组合结果。

对层 $z^l=W^la^{l-1}+b^l$、$a^l=\phi(z^l)$，误差信号向前传播为

$$
\delta^l=((W^{l+1})^T\delta^{l+1})\odot\phi'(z^l),
$$

参数梯度是 $\partial L/\partial W^l=\delta^l(a^{l-1})^T$。优化器再用这些梯度更新参数。

分类用 sigmoid/softmax 后接 MSE 时，还会多乘容易饱和的激活导数；交叉熵组合消去了这部分，使自信答错时仍有较直接的纠错信号。但深网的梯度仍会经过许多 Jacobian，初始化、激活、残差、归一化和梯度裁剪同样重要。

## 知识点

MSE 归约、BCE、sigmoid、softmax、logits、链式法则、误差信号、Jacobian 连乘。


## 追问

相关真题追问：

- 为什么 $p-y$ 是对 logit 的梯度，而不是对概率的梯度？
- 分类使用 MSE 时，sigmoid 饱和怎样削弱梯度？
- Softmax 与交叉熵组合的 $p-y$ 怎样推导？
- 损失梯度之外，深层网络还会在哪些 Jacobian 上消失或爆炸？

## Note

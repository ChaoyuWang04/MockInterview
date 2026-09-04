---
difficulty: 简单
topic: 逻辑回归/从零训练
summary: 推导逻辑回归的概率、损失、梯度与决策边界并手写训练步骤
tags: [面经, 待校对, 手撕代码, 逻辑回归, 梯度下降]
company: 字节、得物、百度
mastered: false
highfreq: false
---

## 题目

同一题簇包含理论与代码两部分：

1. 解释逻辑回归怎样用线性打分、sigmoid、对数损失和梯度下降完成二分类，并说明概率含义和决策边界。
2. 不使用机器学习库和自动求导，给定样本 $X$ 与 $Y\in\{0,1\}$，手写前向、BCE、反向梯度以及权重和偏置的一次更新。

## 要点

- $z=w^Tx+b$，$p=\sigma(z)$；在模型假设下，$p$ 表示条件正类概率。
- BCE 是 Bernoulli 负对数似然；从 logits 计算更稳定。
- sigmoid+BCE 对 logit 的梯度为 $p-y$，继而 $\nabla_w=X^T(p-y)/B$。
- 阈值 0.5 对应 $w^Tx+b=0$；换阈值会改变决策，不会改变训练出的概率。
- 完全线性可分且无正则时，有限最大似然解可能不存在，权重范数会持续增大，不能简单说“收敛到有限参数”。

## 答案

逻辑回归先计算线性分数 $z=w^Tx+b$，再用 $p=\sigma(z)$ 得到模型假设下的正类概率。二元交叉熵就是 Bernoulli 分布的负对数似然；对单个样本有

$$
\frac{\partial L}{\partial z}=p-y,\qquad
\nabla_w L=x(p-y),\qquad
\frac{\partial L}{\partial b}=p-y.
$$

批量训练时对样本梯度求平均，再按 $w\leftarrow w-\eta\nabla_wL$、$b\leftarrow b-\eta\nabla_bL$ 更新。以 0.5 为分类阈值时，$p\ge0.5$ 等价于 $w^Tx+b\ge0$，所以决策边界是超平面 $w^Tx+b=0$。

稳定 BCE-with-logits 为

$$
L(z,y)=\max(z,0)-yz+\log(1+e^{-|z|}).
$$

它避免先算接近 0/1 的概率再取对数。下面支持单样本或批量：

```python
import numpy as np


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logistic_train_step(X, y, w, b, lr=1e-2):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    y = np.asarray(y, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float)
    if len(y) == 0:
        raise ValueError("the batch must not be empty")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if X.shape != (len(y), len(w)):
        raise ValueError("incompatible shapes")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("labels must be 0 or 1")

    z = X @ w + b
    p = sigmoid(z)
    loss = np.mean(np.maximum(z, 0) - y * z + np.log1p(np.exp(-np.abs(z))))

    error = (p - y) / len(y)
    dw = X.T @ error
    db = float(error.sum())
    return w - lr * dw, b - lr * db, float(loss)
```

这里没有用 `np.where` 同时计算正负两个指数分支，极端 logit 不会在未选分支先溢出。一次批量更新的时间 $O(Bd)$、工作空间 $O(B+d)$。

多分类可用 softmax 回归；非线性边界可先构造非线性特征、使用核方法，或换成树/神经网络。逻辑回归输出可解释为概率仍依赖模型拟合与校准，不能仅凭 sigmoid 形式保证概率校准良好。

## 知识点

- Bernoulli 最大似然、sigmoid、BCE-with-logits、解析梯度、决策边界、线性可分。
- 面经原题：[B006-G01-Q144](../../docs/references/面经原题.md#b006-g01-q144)、[B006-G01-Q148](../../docs/references/面经原题.md#b006-g01-q148)、[B006-G01-Q190](../../docs/references/面经原题.md#b006-g01-q190)、[B006-G01-Q207](../../docs/references/面经原题.md#b006-g01-q207)。
- 老师答案参考：[P009-Q144](../../docs/references/平台题/P009-LC-081-160.md#p009-q144)、[P009-Q148](../../docs/references/平台题/P009-LC-081-160.md#p009-q148)、[P009-Q190](../../docs/references/平台题/P009-LC-161-241.md#p009-q190)、[P009-Q207](../../docs/references/平台题/P009-LC-161-241.md#p009-q207)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 逻辑回归与线性回归有什么联系和区别？
- 为什么分类通常不用 MSE？
- 怎样扩展多分类或非线性边界？
- 怎样加入 L1/L2 正则，并和自动求导结果核对？

## Note

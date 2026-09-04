---
difficulty: 简单
topic: 反向传播/多层感知机
summary: 从链式法则推导多层感知机的权重梯度并手写反传
tags: [面经, 待校对, 手撕代码, 反向传播, 链式法则]
company: 腾讯
mastered: false
highfreq: false
---

## 题目

以一个简单的多层前馈神经网络为例，从损失函数出发，完整推导损失对各层权重和偏置的梯度。要求写清链式法则如何逐层传播，以及梯度怎样用于参数更新。

## 要点

- 先定义每层前向关系：$z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}$，$a^{(l)}=\phi^{(l)}(z^{(l)})$。
- 用误差项 $\delta^{(l)}=\partial L/\partial z^{(l)}$ 避免重复展开长链式乘积。
- 隐藏层递推为 $\delta^{(l)}=(W^{(l+1)})^T\delta^{(l+1)}\odot\phi'(z^{(l)})$。
- 权重梯度是外积或批量矩阵乘：$\partial L/\partial W^{(l)}=\delta^{(l)}(a^{(l-1)})^T$。
- 输出层误差取决于损失和激活；sigmoid 与二元交叉熵组合时，对 logit 的梯度是 $a-y$。

## 答案

以两层网络为例：隐藏层用 sigmoid，输出层也用 sigmoid，损失是二元交叉熵。单个样本的前向为

$$
z_1=W_1x+b_1,\quad a_1=\sigma(z_1),\quad
z_2=W_2a_1+b_2,\quad \hat y=\sigma(z_2).
$$

sigmoid 与 BCE 合并后，输出层误差直接是 $\delta_2=\hat y-y$。于是

$$
\nabla_{W_2}L=\delta_2a_1^T,\quad \nabla_{b_2}L=\delta_2,
$$

$$
\delta_1=W_2^T\delta_2\odot a_1(1-a_1),\quad
\nabla_{W_1}L=\delta_1x^T,\quad \nabla_{b_1}L=\delta_1.
$$

下面的 NumPy 代码按批次求平均梯度并更新一次：

```python
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def train_step(X, y, W1, b1, W2, b2, lr=1e-2):
    # X: (B, D), y: (B, 1)
    z1 = X @ W1.T + b1                 # (B, H)
    a1 = sigmoid(z1)
    z2 = a1 @ W2.T + b2                # (B, 1)
    pred = sigmoid(z2)

    # 稳定的 BCE-with-logits。
    loss = np.mean(np.maximum(z2, 0) - y * z2 + np.log1p(np.exp(-np.abs(z2))))

    B = X.shape[0]
    delta2 = (pred - y) / B
    gW2 = delta2.T @ a1
    gb2 = delta2.sum(axis=0)

    delta1 = (delta2 @ W2) * a1 * (1.0 - a1)
    gW1 = delta1.T @ X
    gb1 = delta1.sum(axis=0)

    W1 -= lr * gW1
    b1 -= lr * gb1
    W2 -= lr * gW2
    b2 -= lr * gb2
    return float(loss)
```

一次前向和反向的时间主要由矩阵乘法决定；对宽度为 $d_{l-1}\to d_l$ 的第 $l$ 层，时间是 $O(Bd_{l-1}d_l)$。反传需要前向激活，训练时缓存空间约为各层激活总量。

## 知识点

- 计算图、链式法则、误差项、矩阵微积分、BCE-with-logits、梯度消失。
- 面经原题：[B006-G01-Q087](../../docs/references/面经原题.md#b006-g01-q087)、[B006-G01-Q088](../../docs/references/面经原题.md#b006-g01-q088)。
- 老师答案参考：[P009-Q087](../../docs/references/平台题/P009-LC-081-160.md#p009-q087)、[P009-Q088](../../docs/references/平台题/P009-LC-081-160.md#p009-q088)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 隐藏层改成 ReLU 时反传公式怎样变化？
- 怎样缓解梯度消失、爆炸和数值不稳定？
- BatchNorm 或 Self-Attention 的反向传播怎样推导？

## Note

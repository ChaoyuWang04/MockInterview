---
difficulty: 简单
topic: 正则化/L1与L2
summary: 比较 L1 与 L2 的形式、优化影响和适用场景
tags: [真题, 待校对, 手撕代码, L1, L2, 正则化]
company: 哔哩哔哩、美团、支付宝
mastered: false
highfreq: false
---

## 题目

说明 L1 与 L2 正则化在数学形式、优化过程、参数影响和模型效果上的区别，并结合高维稀疏、共线性和稳定预测等场景分析各自的适用条件与优势。

## 要点

- L1 罚项是 $\lambda\|w\|_1$，在零点不可导；L2 常写成 $\lambda\|w\|_2^2/2$，梯度为 $\lambda w$。
- L1 的近端算子是软阈值，能把一部分参数精确压到 0；L2 连续缩小所有参数，通常不会精确为 0。
- L1 产生的多为非结构化稀疏，不使用稀疏存储/内核时不自动节省推理时间或内存。
- “L2 解唯一”只在目标严格凸等条件下成立，不能推广到一般深度网络。
- 正则强度 $\lambda$ 需要结合验证集选择；过强的 L1 或 L2 都可能造成欠拟合。

## 答案

以均方误差线性回归为例：

$$
J_{L1}(w,b)=\frac1{2n}\|Xw+b-y\|_2^2+\lambda\|w\|_1,
$$

$$
J_{L2}(w,b)=\frac1{2n}\|Xw+b-y\|_2^2+\frac\lambda2\|w\|_2^2.
$$

L2 可直接把 $\lambda w$ 加进梯度。L1 在零点用普通梯度下降容易抖动，近端梯度更清楚：先走数据损失梯度，再做软阈值。

从几何上看，二维 L1 约束集是带坐标轴尖角的菱形，高维时是 cross-polytope；损失等高面与约束边界相切时，更容易落在某些坐标为 0 的棱角或低维面上。L2 球边界光滑，没有同样的坐标轴尖角，因此通常把受罚参数连续缩小而不精确置零。这是稀疏倾向，不保证在所有数据上选出正确特征；强相关特征之间的选择尤其可能不稳定。

从最优性条件看，$w_j\ne0$ 时 $|w_j|$ 的导数是 $\operatorname{sign}(w_j)$，而在 $w_j=0$ 时次梯度是区间 $[-1,1]$。只要数据损失在该坐标的负梯度落入 $\lambda[-1,1]$，$w_j=0$ 就能满足最优性条件。L2 的导数 $\lambda w_j$ 在接近 0 时也趋近 0，不会形成这个能“吸附”到零点的区间。

```python
import numpy as np


def soft_threshold(x, threshold):
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def regularized_linear_step(X, y, w, b, lr, lam, penalty):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a non-empty 1-D array")
    if X.ndim != 2 or X.shape != (len(y), len(w)):
        raise ValueError("incompatible shapes")
    if lr <= 0 or lam < 0:
        raise ValueError("invalid lr or lambda")

    residual = X @ w + b - y
    grad_w = X.T @ residual / len(y)
    grad_b = float(residual.mean())  # 通常不正则化偏置

    if penalty == "l2":
        new_w = w - lr * (grad_w + lam * w)
    elif penalty == "l1":
        new_w = soft_threshold(w - lr * grad_w, lr * lam)
    else:
        raise ValueError("penalty must be l1 or l2")

    return new_w, b - lr * grad_b
```

近端更新中的软阈值会把绝对值不超过 $\eta\lambda$ 的中间结果直接变成 0，这对应优化过程中的精确稀疏化。L1 适合希望稀疏与变量筛选的高维线性模型，但相关特征中可能不稳定地挑一个；Elastic Net 可同时用 L1 与 L2。L2 在共线特征间平滑分摊权重，通常提高稳定性。深度网络常用 weight decay，但应区分“把 L2 项加进损失”和 AdamW 的解耦衰减。

一次更新的主要时间是矩阵乘，$O(nd)$，工作空间 $O(n+d)$。模型是否更好必须用验证集选择 $\lambda$；正则过强会欠拟合。

## 知识点

- Lasso、Ridge、软阈值、近端梯度、稀疏性、共线性、Elastic Net、AdamW。

## 追问

相关真题追问：

- L1/L2 怎样用于 LoRA 等参数高效微调？
- 为什么深度学习中 L1 通常少于 L2/weight decay？
- 怎样推导 L1 的次梯度与近端更新？

## Note

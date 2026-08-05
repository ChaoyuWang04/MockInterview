---
difficulty: 困难
tags: [Transformer, Attention, 手撕代码]
company: 阿里
mastered: false
---

## 题目

用 NumPy 手写单头 Scaled Dot-Product Attention,写出公式,并解释为什么要除以 $\sqrt{d_k}$。

## 要点

- 公式与代码一一对应
- softmax 的数值稳定性处理(减最大值)
- 缩放因子的方差论证

## 答案

公式:

$$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**为什么除以 $\sqrt{d_k}$**:设 $q, k$ 各分量独立、均值 0、方差 1,则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的方差为 $d_k$。维度越大,点积的量级越大,softmax 进入饱和区、梯度趋近于 0。除以 $\sqrt{d_k}$ 把方差拉回 1 量级,保持梯度可训练。

```python
import numpy as np

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)  # 数值稳定
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights
```

要点:`swapaxes(-2, -1)` 支持带 batch 维的输入;mask 在 softmax 之前加,把不可见位置打到 $-10^9$。

## 知识点

Scaled Dot-Product Attention、softmax 数值稳定性、注意力 mask。

## 追问

- 多头注意力相比单头多了什么?参数量怎么算?
- 为什么 mask 用加 $-10^9$ 而不是乘 0?

## Note

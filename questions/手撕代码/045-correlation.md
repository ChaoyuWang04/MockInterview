---
difficulty: 简单
topic: 统计指标/相关系数
summary: 从零实现 Pearson 相关系数并处理常数序列
tags: [面经, 待校对, 手撕代码, Pearson, 统计]
company: 哔哩哔哩
mastered: false
highfreq: false
---

## 题目

不使用第三方统计库或现成相关系数函数，手写两个等长数值列表的 Pearson 线性相关系数（PLCC）。处理空输入、长度不一致、常数序列和除零，并解释均值、协方差与标准差怎样对应公式。

## 要点

- Pearson 衡量中心化后的线性同向变化：
  $r=\sum(x_i-\bar x)(y_i-\bar y)/\sqrt{\sum(x_i-\bar x)^2\sum(y_i-\bar y)^2}$。
- 任意一个序列方差为零时 Pearson 都未定义，不能把“两条都常数”返回 0、只一条常数返回 NaN。
- 浮点计算应用 `math.fsum` 降低大量求和误差，并对舍入导致的轻微越界夹到 `[-1,1]`。
- 计算顺序要与公式对应：先求均值，再中心化，最后计算协方差项与两个平方和。

## 答案

```python
from math import fsum, sqrt


def pearson(x, y):
    x, y = list(x), list(y)
    if len(x) != len(y):
        raise ValueError("lengths differ")
    if len(x) < 2:
        raise ValueError("at least two pairs are required")

    mx = fsum(x) / len(x)
    my = fsum(y) / len(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]

    sxx = fsum(value * value for value in dx)
    syy = fsum(value * value for value in dy)
    if sxx == 0.0 or syy == 0.0:
        raise ValueError("correlation is undefined for a constant sequence")

    sxy = fsum(a * b for a, b in zip(dx, dy))
    r = sxy / sqrt(sxx * syy)
    return max(-1.0, min(1.0, r))
```

Pearson 单次遍历存中心化列表后是 $O(n)$ 时间、$O(n)$ 空间；可用 Welford/在线协方差降到 $O(1)$ 工作空间。

Pearson 对离群点敏感且只描述线性关系，也不能证明因果关系。若要比较秩相关，可在平台追问中继续讨论 Spearman。

## 知识点

- 均值、协方差、方差、零方差、数值稳定求和。
- 面经原题：[B006-G01-Q135](../../docs/references/面经原题.md#b006-g01-q135)。
- 老师答案参考：[P009-Q135](../../docs/references/平台题/P009-LC-081-160.md#p009-q135)。

## 追问

以下均为平台页面追问，不计入面经原题：

- Pearson 对异常值敏感时可用哪些稳健指标？
- 实际项目中何时用 PLCC，何时用 SRCC？

## Note

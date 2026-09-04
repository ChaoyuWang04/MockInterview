---
difficulty: 简单
topic: 数值计算/平方根
summary: 不调用 sqrt，用牛顿法计算浮点平方根并按小数位舍入
tags: [面经, 待校对, 手撕代码, 牛顿法, 二分查找, 浮点数]
company: 小红书
mastered: false
highfreq: false
---

## 题目

输入非负浮点数 x 和非负整数 k，不调用 `sqrt` 等开方函数，计算 $\sqrt{x}$ 并四舍五入保留 k 位小数。可用牛顿迭代或二分查找；说明精度、收敛、边界和极大极小值处理。

## 要点

- 牛顿迭代解 $f(y)=y^2-x=0$：$y_{t+1}=(y_t+x/y_t)/2$。
- 初值应结合数量级选择；从 $10^{\lfloor e/2\rfloor}$ 开始可避免对极大或极小数反复折半。
- 迭代要有精度控制和最大次数兜底；达到工作精度后再做十进制舍入。
- 二进制 float 无法保留任意十进制位；需要严格 half-up 时应在 `Decimal` 或整数定点域完成迭代和量化。
- 牛顿法根附近二次收敛；二分法单调稳健，但只线性收敛。

## 答案

```python
from decimal import Decimal, ROUND_HALF_UP, localcontext


def sqrt_round(x, k: int) -> Decimal:
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError("k must be a non-negative integer")
    value = Decimal(str(x))
    if not value.is_finite() or value < 0:
        raise ValueError("x must be finite and non-negative")
    quantum = Decimal(1).scaleb(-k)
    if value == 0:
        return Decimal(0).quantize(quantum)

    # sqrt(value) 的整数位数约为 value 十进制指数的一半。
    root_digits = max(1, (value.adjusted() + 2) // 2)
    with localcontext() as ctx:
        ctx.prec = max(34, root_digits + k + 12)
        # 用数量级给初值；对 1e308 或 1e-308 不必从 x/1 反复折半。
        y = Decimal(10) ** (value.adjusted() // 2)
        for _ in range(2048):
            next_y = (y + value / y) / 2
            if next_y == y:
                return next_y.quantize(quantum, rounding=ROUND_HALF_UP)
            y = next_y
    raise ArithmeticError("Newton iteration did not converge")
```

函数返回 `Decimal`，从而真实保留 k 位小数；需要普通浮点数时可由调用方再转成 `float`，但转换后可能丢掉十进制位。牛顿法进入根附近后二次收敛，也就是每轮大致把正确位数翻倍。实际轮数还受初值和 x 的数量级影响；任意精度除法的成本也会随位数增长，不能把整段计算写成与 k 无关的固定 $O(1)$。二分法可在 `[0,max(1,x)]` 上按中点平方收缩，单调稳健但线性收敛，达到误差 $\varepsilon$ 需 $O(\log((hi-lo)/\varepsilon))$ 轮。

推广到 $m$ 次方根时，可对 $f(y)=y^m-x$ 使用

$$
y_{t+1}=\frac{(m-1)y_t+x/y_t^{m-1}}{m}.
$$

工程系统通常优先调用硬件或标准库 `sqrt`，因为它们针对 IEEE-754 的舍入、次正规数和性能做了充分优化；手写迭代主要用于考察数值方法或任意精度场景。

## 知识点

- 牛顿迭代、二分查找、二次收敛、绝对/相对误差、IEEE-754、舍入模式。
- 面经原题：[B006-G01-Q151](../../docs/references/面经原题.md#b006-g01-q151)、[B006-G01-Q174](../../docs/references/面经原题.md#b006-g01-q174)。
- 老师答案参考：[P009-Q151](../../docs/references/平台题/P009-LC-081-160.md#p009-q151)、[P009-Q174](../../docs/references/平台题/P009-LC-161-241.md#p009-q174)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 牛顿法和二分法的收敛速度、复杂度怎样比较？
- 如果不能使用除法，怎样实现平方根？
- double 精度不够的大数开方怎样处理？
- 怎样把牛顿迭代推广到任意 $m$ 次方根？
- 初值怎样结合指数范围选择，为什么标准库通常仍优先使用硬件 `sqrt`？

## Note

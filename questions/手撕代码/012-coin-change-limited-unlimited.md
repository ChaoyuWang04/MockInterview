---
difficulty: 中等
topic: 动态规划/硬币组合计数
summary: 分别计算无限和有限硬币凑额组合数并避免排列重复
tags: [面经, 待校对, 动态规划, 完全背包, 多重背包, 组合计数]
company: 未标
mastered: false
highfreq: false
---

## 题目

给定不同面额的硬币和一个总金额，硬币数量可以是有限或无限，要求计算凑成该金额的所有组合总数。请使用动态规划方法实现解决方案，明确写出状态转移方程，并对比分析有限数量与无限数量场景下的设计差异。

## 要点

- 组合计数必须以硬币种类为外层；交换循环会把不同顺序重复计数。
- 无限数量时金额正序；每枚仅一次时金额倒序。
- 有界计数不能直接用二进制拆分成 0-1 物品，否则相同枚数可能有多种 bundle 表示而过计数。
- 有限数量可用枚举数量 $O(CAq)$，或按同余类滑动窗口降到 $O(CA)$。

## 答案

无限硬币的标准组合计数：

```python
def count_unbounded_combinations(coins, amount):
    if amount < 0 or any(c <= 0 for c in coins):
        raise ValueError("invalid amount or coin")
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:                 # 种类在外，避免把顺序计入答案
        for value in range(coin, amount + 1):
            dp[value] += dp[value - coin]
    return dp[amount]
```

有限数量 `limits[i]` 可按每个面额的同余类维护长度 `limit+1` 的窗口：

```python
def count_bounded_combinations(coins, limits, amount):
    if len(coins) != len(limits) or amount < 0:
        raise ValueError("invalid input")
    if any(c <= 0 for c in coins) or any(q < 0 for q in limits):
        raise ValueError("invalid coin or limit")
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin, limit in zip(coins, limits):
        new = [0] * (amount + 1)
        for rem in range(min(coin, amount + 1)):
            window = 0
            for value in range(rem, amount + 1, coin):
                window += dp[value]
                drop = value - (limit + 1) * coin
                if drop >= 0:
                    window -= dp[drop]
                new[value] = window
        dp = new
    return dp[amount]
```

两者空间都是 $O(A)$；无限版时间 $O(CA)$，有界滑窗版也是 $O(CA)$。

## 知识点

- 组合与排列、完全背包、多重背包、金额正序/倒序、同余类滑动窗口、有界计数。

- 面经原题：[B006-G01-Q022](../../docs/references/面经原题.md#b006-g01-q022)；老师答案参考：[P009-Q022](../../docs/references/平台题/P009-LC-001-080.md#p009-q022)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 怎样输出实际硬币组合而不只计数？
- 为什么有界组合计数不能直接用二进制拆分？
- 面额很大但 amount 很小时怎样减少无效循环？

## Note

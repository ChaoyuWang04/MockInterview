---
difficulty: 中等
topic: 动态规划/网格路径固定长度加成
summary: 用段前段中段后三阶段动态规划最大化路径和与长度k连续段和
tags: [真题, 待校对, 动态规划, 网格路径, 状态机]
company: 拼多多
mastered: false
highfreq: false
---

## 题目

给定一个n×m的网格，每个格子具有价值a(i,j)。机器人从(1,1)出发，每次只能向右或向下移动，最终到达(n,m)，经过的格子可获得其价值。定义基础卸货得分为路径上所有格子价值之和，批量卸货得分为路径中连续k个格子价值之和的最大值。请设计算法，求出一条路径使得基础卸货得分与批量卸货得分的总和最大，并返回该最大值。

## 要点

- `最大路径 (基础和 + 最大段和)` 可改写为同时选择路径和其中一个长度 k 段。
- 被选择段内的格子贡献两次，段外贡献一次。
- 状态分为段前、段中长度 `1..k`、段后，保留必要的路径历史。
- 普通二维 `dp[i][j]` 或跨路径单调队列不能表达这段历史。

## 答案

对任一路径 $P$，有

$$\text{base}(P)+\max_S\text{sum}(S)=\max_S\left(\text{base}(P)+\text{sum}(S)\right),$$

其中 $S$ 是路径上长度恰为 $k$ 的连续段。因此可以在走路径的同时选择一段：段内格子权重为 2，其他格子权重为 1。

```python
def max_delivery_score(grid: list[list[int]], k: int) -> int:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    rows, cols = len(grid), len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("grid must be rectangular")
    path_length = rows + cols - 1
    if not 1 <= k <= path_length:
        raise ValueError("k must be between 1 and every path length")

    neg_inf = float("-inf")
    # 状态 0：段尚未开始；1..k：正在段中且长度为该下标；k+1：段已结束。
    prev = [[neg_inf] * (k + 2) for _ in range(cols)]

    for i in range(rows):
        cur = [[neg_inf] * (k + 2) for _ in range(cols)]
        for j in range(cols):
            value = grid[i][j]
            if i == 0 and j == 0:
                cur[j][0] = value
                cur[j][1] = 2 * value
                continue

            def parent(state: int) -> float:
                from_up = prev[j][state] if i > 0 else neg_inf
                from_left = cur[j - 1][state] if j > 0 else neg_inf
                return max(from_up, from_left)

            cur[j][0] = value + parent(0)
            cur[j][1] = 2 * value + parent(0)
            for length in range(2, k + 1):
                cur[j][length] = 2 * value + parent(length - 1)
            cur[j][k + 1] = value + max(parent(k), parent(k + 1))
        prev = cur

    return int(max(prev[-1][k], prev[-1][k + 1]))
```

状态数是 $O(nmk)$，每个状态只看上方和左方，时间复杂度 $O(nmk)$。滚动行后空间是 $O(mk)$。已有答案中的 `pass` 和“用单调队列降到 $O(nm)$”没有给出合法的跨路径合并证明；不同前驱路径的窗口不能直接放入同一个普通滑动窗口。

所有单调路径都恰有 $n+m-1$ 个格子。若 $k$ 超过这个长度，题面没有定义批量得分；代码选择报错，面试时应先确认是视为不存在、0，还是允许不足 k 的段。

## 知识点

- 多阶段 DP、路径状态、固定长度连续段、max 交换、滚动数组、题意边界。


## 追问

相关真题追问：

- 如果 k 大于路径长度，批量得分应如何定义？
- 怎样用滚动数组降低空间？
- 如果要求输出具体路径，怎样保存或重建决策？

## Note

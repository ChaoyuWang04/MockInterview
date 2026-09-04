---
difficulty: 简单
topic: 动态规划/网格最大路径和
summary: 用一维动态规划求只向右或向下的网格最大得分
tags: [真题, 待校对, 动态规划, 网格, 路径规划]
company: 拼多多
mastered: false
highfreq: false
---

## 题目

在一个n×m的网格中，机器人从左上角（1,1）出发，每次只能向右或向下移动，最终到达右下角（n,m）。每个格子包含一个非负得分值，机器人经过的路径总得分为所经过格子得分之和。请设计一个动态规划算法，求出机器人所能获得的最大总得分，并说明状态转移方程、边界条件及算法的时间与空间复杂度。

## 要点

- 状态只依赖上方与左方。
- 第一行/第一列只有一种来路，需显式初始化或用负无穷哨兵。
- 一维滚动数组长度应选较短维度，才是 $O(\min(n,m))$ 空间。

## 答案

```python
def max_path_score(grid):
    if not grid or not grid[0]:
        return 0
    if any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("grid must be rectangular")

    rows, cols = len(grid), len(grid[0])
    # 逻辑上沿较长维逐层扫描，沿较短维维护滚动状态；不复制转置矩阵。
    transposed = cols > rows
    outer, inner = (cols, rows) if transposed else (rows, cols)

    dp = [float("-inf")] * inner
    dp[0] = 0
    for i in range(outer):
        for j in range(inner):
            from_left = dp[j - 1] if j > 0 else float("-inf")
            value = grid[j][i] if transposed else grid[i][j]
            dp[j] = value + max(dp[j], from_left)
    return dp[-1]
```

`dp[j]` 更新前代表逻辑上方，`dp[j-1]` 已更新为逻辑左方。矩阵较宽时只交换索引含义，转置会把“向右/向下”互换，但不改变路径集合。每格访问一次，时间 $O(nm)$；滚动数组长度为较短维，辅助空间严格为 $O(\min(n,m))$。

## 知识点

- 网格 DP、滚动数组、最大路径、边界初始化、空间复杂度统计口径。


## 追问

相关真题追问：

- 允许上下左右且不能重复访问时为何不再是同一 DP？
- 网格边长很大但稀疏时怎样存储？
- 加入障碍格后转移怎样修改？

## Note

---
difficulty: 中等
topic: 网格搜索/最大岛屿边界
tags: [面经, 待校对, 手撕代码, DFS, BFS, 连通分量]
summary: 求最大岛屿面积及最左最右列并处理并列规则
company: 字节
mastered: false
highfreq: false
---

## 题目

给定由 0 和 1 组成的二维网格，岛屿按上下左右连接。返回最大岛屿的面积，以及该岛屿覆盖的最小列索引和最大列索引；先约定多个岛屿面积相同时的选择规则，并分析大网格、原地标记和并行处理。

## 要点

- 每次 DFS/BFS 遍历一个连通分量，同时累计面积、最小列和最大列。
- 全局答案必须整体更新 `(area,left,right)`，不能只更新面积后丢失对应边界。
- 原题没有定义并列最大岛屿的选择规则；代码需明确“保留扫描到的第一个”或其他规则。
- 原地把 1 改成 0 可去掉 visited 数组，但递归栈或显式队列仍可能是 $O(mn)$。
- 超大网格需按分块边界合并连通分量；并行的难点是跨分区组件的合并。

## 答案

**在普通最大岛屿 DFS 上多维护两个量：访问单元格 `(r,c)` 时，令 `left=min(left,c)`、`right=max(right,c)`。** 下面在面积并列时保留按行扫描遇到的第一个岛屿，并原地修改网格：

```python
def largest_island_bounds(grid):
    if not grid or not grid[0]:
        return (0, -1, -1)
    rows, cols = len(grid), len(grid[0])
    best = (0, -1, -1)

    for sr in range(rows):
        for sc in range(cols):
            if grid[sr][sc] != 1:
                continue
            stack = [(sr, sc)]
            grid[sr][sc] = 0
            area, left, right = 0, sc, sc
            while stack:
                r, c = stack.pop()
                area += 1
                left, right = min(left, c), max(right, c)
                for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)):
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 0       # 入栈时标记，避免重复入栈
                        stack.append((nr, nc))
            if area > best[0]:
                best = (area, left, right)
    return best
```

每个格子至多入栈一次，时间 $O(mn)$。修改输入省掉了 `visited` 矩阵，但显式栈最坏仍为 $O(mn)$；因此不能把“原地标记”直接说成严格 $O(1)$ 辅助空间。若不允许修改输入，就使用同规模 visited。

网格无法一次装入内存时，可按行带状或块读取，块内标记连通分量，再用并查集合并相邻块边界上的同一分量，并汇总面积和左右边界。并行也采用相同的“局部组件 + 边界合并”思路。

## 知识点

连通分量、DFS/BFS、入队标记、边界聚合、分块连通性、并查集。

- 面经原题：[B006-G01-Q189](../../docs/references/面经原题.md#b006-g01-q189)、[B006-G01-Q218](../../docs/references/面经原题.md#b006-g01-q218)。
- 老师答案参考：[P009-Q189](../../docs/references/平台题/P009-LC-161-241.md#p009-q189)、[P009-Q218](../../docs/references/平台题/P009-LC-161-241.md#p009-q218)。

## 追问

- 平台页面追问（不计入面经原题）：修改原数组后能否把空间复杂度降为 $O(1)$？
- 平台页面追问（不计入面经原题）：超大网格怎样做分块流式处理并合并跨块组件？
- 平台页面追问（不计入面经原题）：怎样并行处理多个岛屿的搜索？
- 自拟：多个岛屿面积相同时，可以怎样定义稳定的 tie-break？

## Note

---
difficulty: 简单
topic: 二分查找/二维严格峰值变体
summary: 二分列并扫描列最大值寻找二维峰值同时澄清缺失的存在性约束
tags: [真题, 待校对, 二分查找, 矩阵, 题意澄清]
company: 小米
mastered: false
highfreq: false
---

## 题目

在一个二维整数数组中，峰值元素定义为严格大于其上下左右所有相邻元素的元素。请设计一个高效算法找出任意一个峰值元素的位置，并详细分析算法的时间复杂度。要求说明如何将一维二分查找思想扩展到二维场景，并比较不同方法（如逐行扫描、列优先二分等）的优劣。

## 要点

- 原题未保证严格峰存在，也未保证相邻元素不相等。
- 在标准前提下，二分一列并找列最大值，再向更大的左右邻居方向舍半。
- 越界邻居应视为负无穷，不能写死为 `-1`，因为矩阵允许负整数。
- 二分列为 $O(r\log c)$；也可按较短维度选择方向。

## 答案

若矩阵允许相邻元素相等，严格峰可能不存在，例如全相等矩阵。下面的实现对原题保持安全：相邻不等时走标准二分；遇到破坏二分证明的平台，则回退到完整扫描，找到严格峰或返回 `None`。

```python
def find_peak_2d(matrix: list[list[int]]) -> tuple[int, int] | None:
    if not matrix or not matrix[0]:
        return None
    rows, cols = len(matrix), len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("matrix must be rectangular")

    def is_strict_peak(row: int, col: int) -> bool:
        value = matrix[row][col]
        return all(
            value > matrix[r][c]
            for r, c in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            if 0 <= r < rows and 0 <= c < cols
        )

    lo, hi = 0, cols - 1
    neg_inf = float("-inf")
    while lo <= hi:
        col = (lo + hi) // 2
        row = max(range(rows), key=lambda r: matrix[r][col])
        value = matrix[row][col]
        left = matrix[row][col - 1] if col > 0 else neg_inf
        right = matrix[row][col + 1] if col + 1 < cols else neg_inf

        if is_strict_peak(row, col):
            return row, col
        if left > value:
            hi = col - 1
        elif right > value:
            lo = col + 1
        else:
            # 竖直或水平平台使“沿严格上坡方向必到峰值”的证明失效。
            break

    for row in range(rows):
        for col in range(cols):
            if is_strict_peak(row, col):
                return row, col
    return None
```

当前元素是该列最大值；在相邻不等的标准前提下，它严格大于上下邻居，只需根据左右较大项决定二分方向。每轮扫描一列花 $O(r)$，共二分 $O(\log c)$ 轮，总时间 $O(r\log c)$、辅助空间 $O(1)$。若行数更小，可交换行列思路得到 $O(c\log r)$。

原题没有补充存在性或相邻不等条件，因此最坏情况下必须允许完整扫描，本文实现的最坏时间是 $O(rc)$。如果面试官补充标准前提，回退不会触发，才有 $O(r\log c)$ 的保证。不能把这道变体无条件等同于 LC 1901。

## 知识点

- 二维二分、上坡不变量、边界哨兵、存在性前提、LC 1901 变体。


## 追问

相关真题追问：

- 如果要求全局最大峰值，算法会怎样变化？
- 非矩形或稀疏矩阵怎样定义边界？
- 能否做到 $O(r+c)$？

## Note

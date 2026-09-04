---
difficulty: 简单
topic: 区间算法/现场选题
summary: 现场任选中等算法题，示范用排序和贪心完成合并区间
tags: [面经, 待校对, 手撕代码, 合并区间, 贪心]
company: 滴滴
mastered: false
highfreq: false
---

## 题目

现场手写一道中等难度的 LeetCode 题，例如最长回文子串或合并区间，要求给出完整代码并解释思路。

本答案沿用页面老师答案的选择：**合并区间**。原始问法是开放选题，不能把它改写成面试官指定了某一个题。

## 要点

- 先说明自己选择的具体题和输入输出约定。
- 合并区间先按左端点排序，再只和结果中最后一个区间比较。
- 闭区间 `[a,b]` 与 `[b,c]` 在端点相接时重叠；半开区间需另定规则。
- 时间由排序主导，为 $O(n\log n)$；结果空间最坏 $O(n)$。

## 答案

```python
def merge_intervals(intervals):
    if not intervals:
        return []

    ordered = sorted((start, end) for start, end in intervals)
    if any(start > end for start, end in ordered):
        raise ValueError("interval start must not exceed end")

    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:              # 闭区间发生重叠或端点相接
            last[1] = max(last[1], end)
        else:
            merged.append([start, end])
    return merged
```

排序后，后一个区间的左端点不会比前一个更小。因此它若能与已有结果重叠，只可能先碰到最后一个合并区间；扩展最后一个右端点就是当前局部最优，也不会妨碍后续合并。

时间复杂度 $O(n\log n)$，输出占 $O(n)$；`sorted` 自身也需要实现相关的辅助空间。若输入已按左端点有序，扫描时间为 $O(n)$。

## 知识点

- 开放式现场选题、排序、区间贪心、闭区间边界、复杂度表达。
- 面经原题：[B006-G01-Q106](../../docs/references/面经原题.md#b006-g01-q106)。
- 老师答案参考：[P009-Q106](../../docs/references/平台题/P009-LC-081-160.md#p009-q106)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 区间数量极大、内存放不下时怎样外部排序？
- 输入已经按起点有序时能否做到 $O(n)$？
- 线段树、差分数组与区间合并各适合什么问题？

## Note

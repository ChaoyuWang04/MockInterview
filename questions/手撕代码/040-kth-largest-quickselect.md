---
difficulty: 简单
topic: 选择算法/Quickselect
summary: 用随机 Quickselect 在线性期望时间求第 k 小或第 k 大
tags: [面经, 待校对, 手撕代码, Quickselect, TopK, 分区]
company: 脉脉
mastered: false
highfreq: false
---

## 题目

解释 Quickselect 怎样复用快速排序的分区，只处理包含目标名次的一侧，在无序数组中求第 k 小或第 k 大。给出可运行代码，分析平均、最坏时间和空间，并说明随机化怎样改善期望性能。

## 要点

- 第 k 小若 k 从 1 开始，目标下标是 `k-1`；第 k 大可转成第 `n-k` 小。
- 分区后只保留目标所在区间，而不是像快排一样处理两侧。
- 随机 pivot 下期望时间 $O(n)$；最坏仍为 $O(n^2)$。
- 三路分区把等于 pivot 的区间一次确定；目标落在其中时可以直接返回。
- 原地迭代实现的额外空间为 $O(1)$；若复制 `lows/highs`，空间是 $O(n)$。

## 答案

```python
from random import Random


def kth_smallest(nums, k, seed=None):
    """返回第 k 小，k 从 1 开始；会原地修改 nums。"""
    if not 1 <= k <= len(nums):
        raise ValueError("k out of range")

    target = k - 1
    lo, hi = 0, len(nums) - 1
    rng = Random(seed)

    while lo <= hi:
        pivot = nums[rng.randint(lo, hi)]
        lt, i, gt = lo, lo, hi
        while i <= gt:
            if nums[i] < pivot:
                nums[lt], nums[i] = nums[i], nums[lt]
                lt += 1
                i += 1
            elif nums[i] > pivot:
                nums[i], nums[gt] = nums[gt], nums[i]
                gt -= 1
            else:
                i += 1

        if target < lt:
            hi = lt - 1
        elif target > gt:
            lo = gt + 1
        else:
            return nums[target]

    raise AssertionError("unreachable")


def kth_largest(nums, k, seed=None):
    if not 1 <= k <= len(nums):
        raise ValueError("k out of range")
    return kth_smallest(nums, len(nums) - k + 1, seed)
```

每轮扫描当前候选区间，随后只保留包含目标名次的一侧。随机 pivot 让后续候选规模在期望上不断缩小，因此各轮扫描量之和为 $O(n)$；不能把它写成 $2n\ln n$，那是 $n\log n$ 量级。极端 pivot 连续出现时区间只缩 1，最坏 $O(n^2)$。代码是迭代且原地分区，额外空间 $O(1)$。

若数据是流式输入或不能修改数组，用大小为 k 的最小堆，时间 $O(n\log k)$、空间 $O(k)$。若必须保证最坏 $O(n)$，可用 median-of-medians，但常数较大。

## 知识点

- 选择问题、顺序统计量、三路分区、随机化期望、Top K、堆。
- 面经原题：[B006-G01-Q120](../../docs/references/面经原题.md#b006-g01-q120)。
- 老师答案参考：[P009-Q120](../../docs/references/平台题/P009-LC-081-160.md#p009-q120)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 固定选第一个元素时，什么输入会触发最坏情况？
- Quickselect 和大小为 k 的堆怎样选？
- 重复元素很多时，二路与三路分区有何差别？

## Note

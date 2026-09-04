---
difficulty: 简单
topic: 数组/合并两个有序去重数组
summary: 用双指针在线性时间内合并两个有序数组并消除重复元素
tags: [面经, 待校对, 数组, 双指针, 去重]
company: 未标公司
mastered: false
highfreq: false
---

## 题目

请编写一个高效的函数，将两个已排序的整数数组合并为一个无重复元素的升序数组，要求时间复杂度尽可能低，并分析算法的时间复杂度。需注意边界情况处理，如空数组、全重复元素等。

## 要点

- 两个指针各自只向右移动，每个输入元素最多处理一次。
- 两边相等时可以同时前进，减少无用比较。
- 用“结果为空或当前值不同于结果末项”统一完成输入内和跨输入去重。
- 输出最坏有 $m+n$ 个元素，不能声称总空间为 $O(1)$。

## 答案

```python
def merge_sorted_unique(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    result: list[int] = []

    def append_once(value: int) -> None:
        if not result or result[-1] != value:
            result.append(value)

    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            append_once(a[i])
            i += 1
        elif a[i] > b[j]:
            append_once(b[j])
            j += 1
        else:
            append_once(a[i])
            i += 1
            j += 1

    while i < len(a):
        append_once(a[i])
        i += 1
    while j < len(b):
        append_once(b[j])
        j += 1
    return result
```

两个指针总共至多移动 $m+n$ 次，时间复杂度为 $O(m+n)$。结果数组最坏包含 $m+n$ 个元素，所以包含输出时空间为 $O(m+n)$；除输出外只使用常数个变量，辅助空间为 $O(1)$。空数组、全重复数组和一边提前耗尽都由同一流程处理。

## 知识点

- 有序性、双指针、在线去重、输出敏感的空间复杂度、边界条件。

- 面经原题：[B006-G01-Q035](../../docs/references/面经原题.md#b006-g01-q035)；老师答案参考：[P009-Q035](../../docs/references/平台题/P009-LC-001-080.md#p009-q035)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 如果数组很大，无法一次性加载到内存，怎么处理？
- 如果要求原地修改其中一个数组，空间复杂度如何优化？
- 如果输入是 k 个有序数组，如何扩展算法？

## Note

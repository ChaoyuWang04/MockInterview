---
difficulty: 简单
topic: 二分查找/旋转数组
summary: 在含重复元素的旋转有序数组中查找目标并返回一个索引
tags: [真题, 待校对, 手撕代码, 二分查找, 旋转数组]
company: 阿里云
mastered: false
highfreq: false
---

## 题目

一个非递减数组经过旋转，数组可能含重复元素。查找目标值，存在时返回任意一个索引，否则返回 -1。说明如何判断有序半区、怎样处理三端相等，以及为什么最坏时间不能保证 $O(\log n)$。

原页面同时要求“允许重复”和“严格 $O(\log n)$”，这两个条件在比较模型下不能同时保证；应以平均/典型 $O(\log n)$、最坏 $O(n)$ 为正确目标。

## 要点

- `nums[mid]` 命中时立即返回。
- 若 `nums[left] == nums[mid] == nums[right]`，无法判断旋转点在哪边，只能收缩边界。
- 否则至少有一半可确定为有序，再判断 target 是否落在该闭开范围。
- 无重复时每轮必减半，为 $O(\log n)$；大量重复时可能每轮只去掉两个端点，为 $O(n)$。
- 需约定“旋转 0 位”是否允许；下面允许。

## 答案

```python
def search_rotated_with_duplicates(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid

        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
            continue

        if nums[left] <= nums[mid]:  # 左半段非递减
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:                        # 右半段非递减
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

例如 `[1,1,1,1,0,1,1]` 中查 0，多个位置都相等时比较无法透露 0 在哪侧，算法会线性剥掉边界。更强地说，数组几乎全相等时，任意比较式算法都可能必须检查线性多个位置才能区分“全是 1”和“只有某处是 0”。因此不能承诺最坏 $O(\log n)$。

空间复杂度 $O(1)$。若保证所有元素不同，可删除三端相等分支，每轮严格排除一半。

## 知识点

- 旋转有序数组、二分不变量、重复元素、信息论下界、平均与最坏复杂度。

## 追问

相关真题追问：

- 没有重复元素时怎样简化？
- 怎样找到旋转点，也就是最小元素索引？
- 旋转数组查找在循环日志等工程场景怎样出现？

## Note

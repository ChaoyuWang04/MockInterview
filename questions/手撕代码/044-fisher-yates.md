---
difficulty: 简单
topic: 随机算法/Fisher-Yates洗牌
summary: 手写等概率 Fisher-Yates 洗牌并证明每个排列概率相同
tags: [真题, 待校对, 手撕代码, Fisher-Yates, 随机算法]
company: 百度
mastered: false
highfreq: false
---

## 题目

实现一个高效且公平的数组 shuffle，使每个排列出现概率相同。说明算法、不变量、正确性、时间和空间复杂度，并比较“随机 key 排序”等替代方案。

## 要点

- 从后向前时，第 i 步必须在 `[0,i]` 中均匀选 j；从前向后也可以，但范围必须对应为 `[i,n-1]`。
- 每一步固定一个尚未固定的位置，不能始终从整个数组随机交换。
- 均匀性依赖底层 `randint` 本身无模偏差。
- 原地时间 $O(n)$、额外空间 $O(1)$。

## 答案

```python
from random import Random


def shuffle_in_place(a, rng=None):
    rng = rng or Random()
    for i in range(len(a) - 1, 0, -1):
        j = rng.randrange(i + 1)  # 0 <= j <= i，且每个 j 等概率
        a[i], a[j] = a[j], a[i]
```

证明可以看一个指定的最终排列。位置 `n-1` 的目标元素在第一步被选中的概率是 $1/n$；随后位置 `n-2` 的目标元素从剩余元素中被选中的概率是 $1/(n-1)$，一直到最后。因此该排列的概率为

$$
\frac1n\cdot\frac1{n-1}\cdots\frac12\cdot1=\frac1{n!}.
$$

每个排列都对应唯一的一串选择，所以全部 $n!$ 个排列同概率。循环 $n-1$ 次，时间 $O(n)$，原地额外空间 $O(1)$。

给每个元素生成随机 key 再排序需要 $O(n\log n)$，而有限 key 空间还会出现碰撞与偏差。始终令每个 i 与 `[0,n-1]` 的随机位置交换通常也不均匀，因为它产生 $n^n$ 条选择序列，而 $n!$ 一般不整除 $n^n$。

## 知识点

- 随机置换、循环不变量、概率乘法、无偏随机整数、原地算法。

## 追问

相关真题追问：

- 高并发场景怎样管理独立随机状态？
- 怎样用统计检验检查 shuffle 是否均匀？
- 数据无法放入内存时，怎样做外存随机打乱？

## Note

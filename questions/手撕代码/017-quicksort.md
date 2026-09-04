---
difficulty: 简单
topic: 排序算法/快速排序
summary: 手写快排，解释分区、复杂度、退化条件与工程优化
tags: [真题, 待校对, 手撕代码, 快速排序, 分治]
company: 小红书
mastered: false
highfreq: false
---

## 题目

同一考点的十二种真题问法可归并成五个要求：

1. 快速排序最优情况何时出现？时间复杂度是多少，为什么？
2. 快速排序最坏情况是什么？什么输入会触发，怎样避免？
3. 手写快速排序，包含完整的 `partition` 和递归调用。
4. 说明边界条件，并考虑随机 pivot、三数取中、重复元素等优化。
5. 比较平均与最坏时间复杂度的成因。

实现一个能正确处理空数组、单元素和大量重复元素的原地快速排序。

## 要点

- 分区结束后要有明确不变量，递归区间不能再次包含已经归位的区域。
- 二路 Lomuto 分区容易在大量重复元素时退化；三路分区把数组分成 `< pivot`、`== pivot`、`> pivot`。
- 随机 pivot 使期望时间为 $O(n\log n)$，但理论最坏时间仍是 $O(n^2)$。
- 普通递归平均栈深 $O(\log n)$、最坏 $O(n)$。先递归较短一侧、再循环处理较长一侧，可把调用栈严格控制在 $O(\log n)$。
- 快排不稳定；小数组切换插排的阈值需要基准测试，不能把固定数字当普遍规律。

## 答案

下面用随机 pivot 加三路分区。三路分区既简化重复值处理，也不会出现指针停住的死循环。

```python
from random import Random
from typing import MutableSequence, TypeVar

T = TypeVar("T")


def quicksort(a: MutableSequence[T], seed: int | None = None) -> None:
    rng = Random(seed)

    def sort(lo: int, hi: int) -> None:
        # 循环处理较长一侧，只递归较短一侧，限制调用栈深度。
        while lo < hi:
            pivot = a[rng.randint(lo, hi)]
            lt, i, gt = lo, lo, hi

            # 不变量：
            # a[lo:lt] < pivot
            # a[lt:i] == pivot
            # a[gt+1:hi+1] > pivot
            while i <= gt:
                if a[i] < pivot:
                    a[lt], a[i] = a[i], a[lt]
                    lt += 1
                    i += 1
                elif a[i] > pivot:
                    a[i], a[gt] = a[gt], a[i]
                    gt -= 1
                    # 新换到 i 的值还没检查，所以 i 不动。
                else:
                    i += 1

            left_size = lt - lo
            right_size = hi - gt
            if left_size < right_size:
                sort(lo, lt - 1)
                lo = gt + 1
            else:
                sort(gt + 1, hi)
                hi = lt - 1

    sort(0, len(a) - 1)
```

每次分区扫描当前区间一次。最优情况是 pivot 每次把数组分成大小相近的两半：递归树高为 $\Theta(\log n)$，每层分区总共扫描 $\Theta(n)$ 个元素，因此时间是 $\Theta(n\log n)$。若每次都分成大小为 0 与 $n-1$ 的两侧，总工作为 $n+(n-1)+\cdots+1=\Theta(n^2)$。随机 pivot 不能消除理论最坏情况，但使期望时间为 $O(n\log n)$。

三数取中能减少固定端点在有序输入上的退化，但仍不保证最坏界；Introsort 在递归过深时切换到堆排序，才能把最坏时间保证为 $O(n\log n)$。只做尾递归或短侧递归优化控制的是栈深，不会改变坏 pivot 下的 $O(n^2)$ 比较次数。

代码只递归短侧，因此最坏调用栈也是 $O(\log n)$；除栈和随机数状态外为原地排序。若要求稳定排序，应改用归并排序或另建缓冲区，原地快排本身不稳定。

## 知识点

- 分治、三路分区、随机化、递归栈、排序稳定性、Introsort。

## 追问

相关真题追问：

- 快排为什么不稳定？怎样获得稳定排序？
- Top K 怎样复用分区思想做到期望 $O(n)$？
- 大量重复元素、外存数据或并行场景怎样处理？
- 工程标准库为什么常组合插排、堆排和快排？

## Note

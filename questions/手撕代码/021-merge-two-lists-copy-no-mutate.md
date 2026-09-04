---
difficulty: 简单
topic: 链表/复制合并两个升序链表
summary: 不修改输入链表地复制并合并两个有序单链表
tags: [真题, 待校对, 链表, 双指针, 哨兵节点]
company: 美团、字节
mastered: false
highfreq: false
---

## 题目

请实现一个函数，合并两个已排序的单链表，并返回合并后新链表的头节点。要求时间复杂度为O（m+n），空间复杂度为O（1）。需处理各种边界情况，如空链表、长度不等的链表等，且不能修改原链表结构。

## 要点

- “不修改原链表”且返回独立新链表，必须复制输出节点。
- 哨兵节点统一处理空输入和新链表头。
- 双指针每次选择较小值，剩余部分也逐节点复制。
- 只有把输出空间排除后，才能称辅助空间为 $O(1)$。

## 答案

题干的空间要求存在口径冲突：若新链表必须与输入独立，就要为 $m+n$ 个值创建节点，新增总空间是 $O(m+n)$。下面实现满足不修改输入；其辅助指针空间是 $O(1)$，输出空间是 $O(m+n)$。

```python
class ListNode:
    def __init__(self, val: int = 0, next_node: "ListNode | None" = None):
        self.val = val
        self.next = next_node


def merge_sorted_copies(
    first: ListNode | None, second: ListNode | None
) -> ListNode | None:
    dummy = ListNode()
    tail = dummy
    left, right = first, second

    while left is not None and right is not None:
        if left.val <= right.val:
            tail.next = ListNode(left.val)
            left = left.next
        else:
            tail.next = ListNode(right.val)
            right = right.next
        tail = tail.next

    current = left if left is not None else right
    while current is not None:
        tail.next = ListNode(current.val)
        tail = tail.next
        current = current.next
    return dummy.next
```

每个输入节点访问一次，时间为 $O(m+n)$。若允许复用原节点，才可通过改写 `next` 指针把新增节点空间降为 $O(1)$，但那会违反本题的“不修改原链表”。

## 知识点

- 双指针、哨兵节点、所有权与可变性、辅助空间和输出空间。


## 追问

相关真题追问：

- 如果允许修改原链表，如何复用节点？
- 如果要求递归实现，时空复杂度如何变化？
- 如何合并 K 个有序链表？

## Note

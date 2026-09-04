---
difficulty: 简单
topic: 二叉树/基础操作
tags: [真题, 待校对, 手撕代码, 二叉树, DFS, BFS]
summary: 手写二叉树遍历、重建、路径和、深度、平衡与最近公共祖先
company: 小红书
mastered: false
highfreq: false
---

## 题目

面试官会从二叉树基础操作中选择一项或继续追问：

- 递归或迭代实现前序、中序、后序、层序遍历；
- 求最大深度或判断平衡二叉树；
- 根据前序和中序序列重建二叉树；
- 判断是否存在给定路径和；
- 求两个节点的最近公共祖先。

要求先说明当前实现的具体操作和输入前提，并正确处理空树等边界情况。

## 要点

- 先明确选择哪一种遍历或典型操作，以及节点值是否唯一、目标节点是否保证存在等输入前提。
- DFS 可用递归或显式栈；层序遍历使用队列，二者辅助空间分别受树高和最大层宽约束。
- 若选择平衡判断，应在同一次后序遍历中同时返回高度和失败标记，做到 $O(n)$。
- 若选择 LCA，而题目不保证 p、q 都在树中，就必须额外验证存在性。

## 答案

原题允许从多种操作中选择。下面选“迭代中序遍历、判断平衡、最近公共祖先”三个代表：它们分别展示显式栈、后序汇总和分治。面试时应先说清自己选择哪一项，而不是默认所有操作都必须一次写完。

```python
from dataclasses import dataclass


@dataclass(eq=False)
class TreeNode:
    val: int
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


def inorder(root):
    ans, stack = [], []
    cur = root
    while cur or stack:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        ans.append(cur.val)
        cur = cur.right
    return ans


def is_balanced(root):
    def height(node):
        if node is None:
            return 0
        left = height(node.left)
        if left == -1:
            return -1
        right = height(node.right)
        if right == -1 or abs(left - right) > 1:
            return -1
        return 1 + max(left, right)

    return height(root) != -1


def lowest_common_ancestor(root, p, q):
    # p、q 都存在时返回 LCA；缺少任一节点时返回 None。
    def dfs(node):
        if node is None:
            return None, 0
        left_node, left_count = dfs(node.left)
        if left_count == 2:
            return left_node, 2
        right_node, right_count = dfs(node.right)
        if right_count == 2:
            return right_node, 2

        count = left_count + right_count + int(node is p) + int(node is q)
        if count >= 2:
            return node, 2
        candidate = left_node or right_node
        if node is p or node is q:
            candidate = node
        return candidate, count

    node, count = dfs(root)
    return node if count == 2 else None
```

三个操作都只访问每个相关节点常数次，时间为 $O(n)$。中序显式栈占 $O(h)$；平衡判断和 LCA 的递归栈也是 $O(h)$，其中 $h$ 是树高。若题目进一步追问链状深树，应把相应递归改成显式栈，避免 Python 调用栈溢出。

## 知识点

- 树遍历、递归与显式栈、层序队列、后序信息汇总、分治、输入前提。

## 追问

相关真题追问：

- Morris 遍历怎样做到 $O(1)$ 额外空间？
- 序列化到文件时选前序还是层序？
- 极深二叉树怎样避免递归栈溢出？
- BFS 与 DFS 求深度各自的空间上界是什么？

## Note

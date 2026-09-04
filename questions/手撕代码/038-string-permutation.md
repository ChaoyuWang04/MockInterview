---
difficulty: 简单
topic: 回溯/字符串全排列
summary: 生成字符串全排列，并用计数回溯消除重复结果
tags: [面经, 待校对, 手撕代码, 回溯, 全排列]
company: 快手
mastered: false
highfreq: false
---

## 题目

面经中有两种递进问法：

1. 给定字符互不重复的字符串，使用回溯输出所有字符全排列。
2. 字符串可能含重复字符，结果中不能出现重复排列；说明去重、时间和空间复杂度。

## 要点

- 路径长度达到原字符串长度时保存结果，递归返回后恢复计数。
- 用字符频次而不是“结果放集合后去重”，可以从搜索树上直接剪掉重复分支。
- 若有 $n$ 个字符，字符 $c$ 出现 $f_c$ 次，唯一排列数是 $U=n!/\prod_c f_c!$。
- 生成并复制全部结果的时间至少是 $\Theta(nU)$；工作栈与路径是 $O(n)$，结果存储是 $O(nU)$。

## 答案

同一个计数版实现同时覆盖“无重复”和“有重复”两种输入：

```python
from collections import Counter


def string_permutations(s: str) -> list[str]:
    counts = Counter(s)
    chars = sorted(counts)  # 只为让输出顺序确定
    path: list[str] = []
    result: list[str] = []

    def backtrack() -> None:
        if len(path) == len(s):
            result.append("".join(path))
            return

        for ch in chars:
            if counts[ch] == 0:
                continue
            counts[ch] -= 1
            path.append(ch)
            backtrack()
            path.pop()
            counts[ch] += 1

    backtrack()
    return result
```

输入 `"abc"` 时每个字符频次都是 1，得到 $3!=6$ 个结果。输入 `"aab"` 时，递归每层只按“还剩多少个 a/b”选择，不区分两个 a 的身份，因此只得到 `aab、aba、baa`。

空字符串的数学排列通常定义为一个空排列，所以代码返回 `[""]`。若产品约定空输入返回空列表，可在入口单独改成 `if not s: return []`。

## 知识点

- 回溯、状态恢复、多重集排列、频次剪枝、输出敏感复杂度。
- 面经原题：[B006-G01-Q110](../../docs/references/面经原题.md#b006-g01-q110)、[B006-G01-Q124](../../docs/references/面经原题.md#b006-g01-q124)。
- 老师答案参考：[P009-Q110](../../docs/references/平台题/P009-LC-081-160.md#p009-q110)、[P009-Q124](../../docs/references/平台题/P009-LC-081-160.md#p009-q124)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 怎样用迭代的 `next_permutation` 生成字典序排列？
- 不生成前 $k-1$ 个结果，怎样直接求第 k 个排列？
- 字符串很长、结果放不下内存时怎样改成生成器？

## Note

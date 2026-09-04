---
difficulty: 简单
topic: 字符串/字符频率排序
summary: 统计字符频率并按频率降序和字典序稳定输出
tags: [真题, 待校对, 字符串, 哈希表, 排序]
company: 通义实验室
mastered: false
highfreq: false
---

## 题目

给定一个字符串，请统计每个字符出现的频率，并按频率从高到低排序；当频率相同时，按字符字典序排序。要求实现算法并分析时间、空间复杂度。

## 要点

- 用哈希表计数 $O(n)$。
- 复合键写作 `(-frequency, character)`。
- 不同字符数为 k，总时间 $O(n+k\log k)$，不能在未限定 ASCII 时简化成 O(n)。
- 先明确输出是字符—频率对还是展开后的字符串。

## 答案

下面返回 `(字符, 频率)` 列表：

```python
from collections import Counter


def frequency_order(s: str):
    counts = Counter(s)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))
```

计数时间 $O(n)$、排序 $O(k\log k)$、结果和哈希表空间 $O(k)$。若字符集固定且很小，可扫描定长计数表；若需要按频率展开字符串，再额外输出 $n$ 个字符，输出成本为 $O(n)$。Unicode 的“字典序”是代码点顺序还是 locale collation 也应由题目约定。

## 知识点

- 频率哈希、复合排序键、不同字符数 k、Unicode 排序口径、输出空间。


## 追问

相关真题追问：

- 十亿字符无法一次装内存时怎样分块归并计数？
- 频率范围不大时怎样用桶排序？
- 在线流式统计怎样提供近似 Top-K？

## Note

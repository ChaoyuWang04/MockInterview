---
difficulty: 简单
topic: 字符串/最高频字符
summary: 在线性时间返回最高频小写字母并按字典序打破平局
tags: [面经, 待校对, 字符串, 计数, 哈希表]
company: 网易
mastered: false
highfreq: false
---

## 题目

给定一个仅包含小写字母的字符串，编写程序找出出现频率最高的字母；如果存在多个相同最高频率的字母，则返回字典序最小的一个。要求实现高效算法，并说明时间与空间复杂度。

## 要点

- 小写字母范围固定，可用长度 26 的数组。
- 按字典序从小到大扫描计数表，只在频率严格更大时更新，天然保留最小字符。
- 必须定义空字符串的返回或异常契约。

## 答案

```python
def most_frequent_lowercase(s: str) -> str:
    if not s:
        raise ValueError("s must be non-empty")
    counts = [0] * 26
    for ch in s:
        idx = ord(ch) - ord("a")
        if not 0 <= idx < 26:
            raise ValueError("s must contain only lowercase a-z")
        counts[idx] += 1

    best = 0
    for i in range(1, 26):
        if counts[i] > counts[best]:
            best = i
    return chr(ord("a") + best)
```

遍历字符串为 $O(n)$，再扫描固定 26 项；辅助空间 $O(1)$。若字符集改为 Unicode，应使用字典，空间变为 $O(k)$（k 为不同字符数）。

## 知识点

- 固定字符集计数、频率平局、字典序、输入契约、流式统计。

- 面经原题：[B006-G01-Q018](../../docs/references/面经原题.md#b006-g01-q018)；老师答案参考：[P009-Q018](../../docs/references/平台题/P009-LC-001-080.md#p009-q018)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 输入是十亿字符流时怎样处理？
- 怎样返回频率最高的前 k 个字符？
- 字符集改为 Unicode 后复杂度如何变化？

## Note

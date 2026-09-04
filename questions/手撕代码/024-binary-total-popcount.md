---
difficulty: 中等
topic: 位运算/前缀比特计数
tags: [面经, 待校对, 手撕代码, 位运算, 动态规划, popcount]
summary: 统计零到 n 所有整数二进制中 1 的总数并比较复杂度
company: 得物
mastered: false
highfreq: false
---

## 题目

给定非负整数 $n$，求 $0$ 到 $n$ 的所有整数二进制表示中 1 的总个数。先分析逐数暴力统计的复杂度，再给出 $O(n)$ 动态规划和 $O(\log n)$、$O(1)$ 额外空间的按位周期算法。

## 要点

- 逐数右移统计是 $O(n\log n)$；Kernighan 对单数是 $O(\operatorname{popcount})$，总量仍可能达到同阶。
- DP 可用 `bits[i] = bits[i >> 1] + (i & 1)`，时间、空间均为 $O(n)$。
- 第 $k$ 位以 $2^{k+1}$ 为周期：前半为 0，后半为 1。
- 按位算法必须统一计算区间 $[0,n]$，处理完整周期与尾段，时间 $O(\log n)$。
- Python 大整数下复杂度还受整数位宽影响；这里采用常见 word/bit 操作模型。

## 答案

**最快的思路是不逐个数检查，而是按二进制位统计这一位在 $0..n$ 出现了多少次 1。** 对权值 `bit = 1,2,4,...`，一个周期长 `2*bit`，后半段的 `bit` 个数该位为 1：

```python
def total_set_bits(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    total = 0
    size = n + 1                 # [0, n] 一共有 n+1 个数
    bit = 1
    while bit <= n:
        cycle = bit << 1
        full, remain = divmod(size, cycle)
        total += full * bit
        total += max(0, remain - bit)
        bit <<= 1
    return total
```

例如 $0..7$ 在最低位上有 4 个 1，在第二位和第三位上也各有 4 个，总数 12。循环次数等于二进制位数，因此时间 $O(\log n)$、额外空间 $O(1)$。

暴力法对每个 `i` 反复右移，最坏执行 $\lfloor\log_2 i\rfloor+1$ 次，总时间 $\Theta(n\log n)$。用 `x &= x-1` 每次删除最低的 1，对单个数是 $O(\operatorname{popcount}(x))$，但对密集 1 的许多数并不会把总体无条件降成 $O(n)$。若还要返回每个数的计数，可用 DP：`bits[i]=bits[i>>1]+(i&1)`，时间和输出空间都是 $O(n)$。

最大 popcount 则不是求总和：对 $0..n$，只需看不超过 $n$ 的最大形如 $2^k-1$ 的数，它含 $k=\lfloor\log_2(n+1)\rfloor$ 个 1。

## 知识点

位周期、完整周期与尾段、Kernighan 位计数、DP 比特计数、复杂度模型。

- 面经原题：[B006-G01-Q052](../../docs/references/面经原题.md#b006-g01-q052)、[B006-G01-Q172](../../docs/references/面经原题.md#b006-g01-q172)、[B006-G01-Q173](../../docs/references/面经原题.md#b006-g01-q173)、[B006-G01-Q206](../../docs/references/面经原题.md#b006-g01-q206)、[B006-G01-Q208](../../docs/references/面经原题.md#b006-g01-q208)。
- 老师答案参考：[P009-Q052](../../docs/references/平台题/P009-LC-001-080.md#p009-q052)、[P009-Q172](../../docs/references/平台题/P009-LC-161-241.md#p009-q172)、[P009-Q173](../../docs/references/平台题/P009-LC-161-241.md#p009-q173)、[P009-Q206](../../docs/references/平台题/P009-LC-161-241.md#p009-q206)、[P009-Q208](../../docs/references/平台题/P009-LC-161-241.md#p009-q208)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 不保存 DP 数组时，怎样做到 $O(1)$ 额外空间？
- 若要返回每个数字的 popcount，为什么输出本身就需要 $O(n)$ 空间？
- 这题与 LC191、LC338 的输入输出有什么不同？
- 位计数在 bitmap、权限掩码和集合运算中怎样使用？

## Note

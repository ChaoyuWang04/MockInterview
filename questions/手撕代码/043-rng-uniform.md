---
difficulty: 简单
topic: 随机算法/均匀整数生成
summary: 从伪随机位流生成无模偏差的区间均匀整数，并比较常见 PRNG
tags: [面经, 待校对, 手撕代码, 随机数, 拒绝采样, LCG]
company: 百度
mastered: false
highfreq: false
---

## 题目

实现一个基础随机数生成器，支持在指定整数闭区间内均匀采样。解释线性同余法、梅森旋转等伪随机算法的原理与局限，正确处理边界，并避免简单取模造成的偏差。

## 要点

- PRNG 的“看起来随机”来自确定性状态转移；同一个种子产生同一序列。
- 若底层等概率空间大小 $M$ 不是目标宽度 $w$ 的倍数，直接 `% w` 会有模偏差。
- 拒绝掉末尾不足一个完整余数周期的值，再取模即可严格均匀。
- LCG 和 MT 都可预测，不用于密码学；密码学应调用操作系统 CSPRNG。
- 区间宽度、上下界、整数位宽和溢出语义必须明确。

## 答案

下面用一个 64 位全周期 LCG 演示状态生成，再用拒绝采样映射区间。它适合讲原理和可复现实验，不适合安全场景。

```python
class LCG64:
    MOD = 1 << 64
    MASK = MOD - 1
    A = 6364136223846793005
    C = 1442695040888963407

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.A * self.state + self.C) & self.MASK
        return self.state

    def randint(self, low: int, high: int) -> int:
        """闭区间 [low, high]；要求区间宽度不超过 2^64。"""
        if low > high:
            raise ValueError("low must not exceed high")
        width = high - low + 1
        if width > self.MOD:
            raise ValueError("range is wider than the generator space")

        limit = self.MOD - (self.MOD % width)
        while True:
            x = self.next_u64()
            if x < limit:
                return low + (x % width)
```

`0..limit-1` 的大小是 width 的整数倍，每个余数恰好出现 `limit/width` 次，所以返回值均匀。若 `width` 本身整除 $2^{64}$，不会拒绝；否则拒绝概率小于 `width/2^64`。

LCG 状态小、速度快，但低位相关性明显且容易由输出反推状态。MT19937 周期很长、统计质量比简单 LCG 好，但状态大、也可预测。PCG/xoshiro 等现代非密码学 PRNG 常有更好的速度与统计折中。生产代码应使用语言的 `uniform_int_distribution`/`randrange`；密钥、令牌等必须使用 `secrets`、`getrandom` 或等价系统 CSPRNG。

单次底层生成是 $O(1)$；拒绝采样的期望轮数是 $M/\text{limit}$，空间 $O(1)$。

## 知识点

- PRNG 状态机、LCG、周期、模偏差、拒绝采样、CSPRNG、整数溢出。
- 面经原题：[B006-G01-Q128](../../docs/references/面经原题.md#b006-g01-q128)。
- 老师答案参考：[P009-Q128](../../docs/references/平台题/P009-LC-081-160.md#p009-q128)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 怎样选择 LCG 参数以获得完整周期？
- 为什么 LCG/MT 不能用于密码学？
- 怎样用卡方检验、序列相关等方法检查随机质量？

## Note

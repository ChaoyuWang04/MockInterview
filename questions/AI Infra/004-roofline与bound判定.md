---
difficulty: 中等
topic: Roofline与Bound分析/模型与bound判定
summary: Roofline 是什么,怎么判断算子是带宽 bound 还是算力 bound
tags: [面经, 待校对, Roofline, 算术强度, bound分析]
company:
mastered: false
highfreq: false
---

## 题目

什么是 roofline 模型?怎么判断一个算子是带宽 bound 还是算力 bound?

## 要点

- 算术强度的定义,以及它由算法 + 实现决定、不是硬件属性
- 屋顶线就是一个 min:斜线段是带宽供给,水平段是算力封顶
- 拐点 = 峰值算力 / 峰值带宽,是硬件自身的性质(machine balance)
- 判据:强度小于拐点 → 带宽 bound,大于 → 算力 bound
- 能心算几个常见算子的强度和几张卡的拐点
- 知道 Roofline 管不了什么(它只给上界)

## 答案

### 一、算术强度:算子的「体质」

$$
I = \frac{\text{总浮点运算次数 FLOPs}}{\text{总访存字节数 Bytes}} \quad (\text{FLOP/Byte})
$$

每搬 1 字节能榨出多少次浮点运算。强度高 = 数据搬一次能用很多遍;强度低 = 搬来一个数用一下就扔。它由**算法 + 实现**共同决定,**不是硬件属性**——同一个矩阵乘,分块做得好强度就高。

### 二、屋顶线与拐点

$$
P_{\text{可达}} = \min\bigl(P_{\text{peak}},\ BW_{\text{peak}} \times I\bigr)
$$

这个 min 就是「屋顶」的全部含义:左边是斜率为峰值带宽的**斜线**,右边是峰值算力的**水平线**,交汇处是**拐点**:

$$
I_{\text{拐点}} = \frac{P_{\text{peak}}}{BW_{\text{peak}}}
$$

拐点是**硬件自身的性质**(文献里叫 machine balance / ridge point)。判据一句话:

- $I <$ 拐点 → 斜线段 → **带宽 bound(访存受限)**
- $I >$ 拐点 → 水平段 → **算力 bound(计算受限)**

### 三、必须能当场心算的数(fp16)

| 算子 | 强度 $I$ |
|---|---|
| 向量加 `c = a + b` | 0.17 |
| GELU / softmax 之类 elementwise | ~2.5 |
| GEMV / batch=1 的线性层 | 1.0 |
| 方阵 GEMM(边长 $n$) | $n/3$,n=4096 时 **1365** |

| 卡 | 峰值算力 | HBM 带宽 | 拐点 |
|---|---|---|---|
| A100 / A800 80GB | 312 TFLOPS | 2.0 TB/s | **153** |
| H100 SXM | 989 TFLOPS | 3.35 TB/s | **295** |
| H20 96GB | 148 TFLOPS | 4.0 TB/s | **37** |

拐点高达 153,意味着**数据搬一次得用上一百多次才算「够本」**。回头看上表:除了大 GEMM,几乎所有算子都远在拐点左边——这就是「绝大多数算子访存受限」的定量版本。而且新卡算力涨得比带宽快,拐点在**逐代右移**,对算法的复用率要求越来越苛刻。

### 四、Roofline 管不了什么

它只给**上界**:occupancy 不足、warp divergence、bank conflict、同步等待会让你贴不到屋顶,但屋顶本身不动;**kernel launch 开销根本不在模型里**。所以 Roofline 只负责第一步定方向,定完之后拿 profiler 找具体原因。

## 知识点

算术强度、Roofline 的 min 结构、拐点 / machine balance、访存受限与计算受限的判据、Roofline 的适用边界。

## 追问

- 算力利用率、带宽利用率怎么理论计算?
- prefill 和 decode 分别落在屋顶线的哪一段?为什么?
- 量化之后 bound 会不会迁移?
- 判定完是访存受限,接下来具体怎么优化?
- 为什么说拐点在逐代右移是个坏消息?

## Note

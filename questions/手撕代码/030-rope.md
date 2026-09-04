---
difficulty: 中等
topic: Transformer/旋转位置编码
summary: 推导并用相邻偶奇维配对正确实现RoPE作用于查询和键
tags: [面经, 待校对, Transformer, RoPE, 位置编码]
company: 得物
mastered: false
highfreq: false
---

## 题目

请手动实现旋转位置编码（Rotary Position Embedding, RoPE），要求推导其核心数学公式，解释其工作原理，并用Python实现一个可运行的版本，支持将位置信息旋转注入到查询（Q）和键（K）向量中。

## 要点

- 头维度必须为偶数，把相邻偶奇维看成二维向量。
- 第 $i$ 对维度使用频率 $\theta_i=\text{base}^{-2i/d}$。
- 位置 $p$ 施加角度 $p\theta_i$ 的二维旋转。
- 同时平移 Q/K 位置不会改变点积，注意力中的位置项只依赖相对位移。

## 答案

对一对分量 $(x_{2i},x_{2i+1})$，位置 $p$ 的旋转为

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}=
\begin{bmatrix}\cos(p\theta_i)&-\sin(p\theta_i)\\
\sin(p\theta_i)&\cos(p\theta_i)\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.
$$

```python
import torch


def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """x: (..., seq_len, head_dim); positions: (seq_len,)."""
    dim = x.shape[-1]
    if dim % 2 != 0:
        raise ValueError("head_dim must be even")
    if positions.numel() != x.shape[-2]:
        raise ValueError("positions must match seq_len")

    frequencies = base ** (
        -torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim
    )
    angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * frequencies[None, :]
    cos = angles.cos().to(dtype=x.dtype)
    sin = angles.sin().to(dtype=x.dtype)

    pairs = x.reshape(*x.shape[:-1], dim // 2, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def rope_qk(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return apply_rope(q, positions), apply_rope(k, positions)
```

这里代码和公式都按相邻两维配对。不能一边用 `chunk(2)` 把前后半维配对，一边又用 `repeat_interleave(2)` 生成相邻维频率；那会混合两种布局。

旋转矩阵满足 $R_m^TR_n=R_{n-m}$，所以 $\langle R_mq,R_nk\rangle=q^TR_{n-m}k$，只依赖相对位置 $n-m$。该点积一般还含正弦交叉项，不能简化成单一的 $q^Tk\cos((m-n)\theta)$。正确测试可检查同时把 $m,n$ 平移相同偏移后点积不变；随机独立 Q/K 的对角线无需天然最大。

## 知识点

- 二维旋转矩阵、复数表示、相对位置、频率谱、偶数维配对、张量广播。

- 面经原题：[B006-G01-Q061](../../docs/references/面经原题.md#b006-g01-q061)；老师答案参考：[P009-Q061](../../docs/references/平台题/P009-LC-001-080.md#p009-q061)。

## 追问

以下均为平台页面追问，不计入面经原题：

- RoPE 相比绝对位置编码和其他相对位置编码有什么优势？
- RoPE 长度外推为何会退化，NTK-aware 和 YaRN 怎样调整频率？
- 为什么通常旋转 Q/K 而不旋转 V？

## Note

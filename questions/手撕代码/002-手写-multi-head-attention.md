---
difficulty: 中等
topic: Attention/多头注意力
summary: 手写 MHA,说清多头比单头多了什么、参数量怎么算
tags: [Transformer, Attention, 手撕代码, 面经, 待校对]
company: 腾讯、小米、美团、字节、快手、联通、海天、百度
mastered: false
highfreq: true
---

## 题目

手写多头注意力(Multi-Head Attention),并说明:多头相比单头多了什么?参数量怎么算,和头数有关系吗?

## 要点

- 能写对拆头/合头的维度变换(reshape + transpose,顺序不能反)
- 缩放用的是每个头的维度 $d_h$,不是总维度 $d$
- 参数量 $\approx 4d^2$,与头数**无关**
- 知道 mask 加在 softmax 之前

## 答案

**一句话理解**:单头注意力是"一个人从一个角度读句子";多头是"派 8 个人各读一个角度"——有人盯语法搭配,有人盯指代关系,有人盯远距离依赖,最后把每个人的结论拼起来再融合一次。

### 流程

输入 $x$ 形状 $(B, L, d)$:

1. 三个线性层得到 Q、K、V,形状仍是 $(B, L, d)$
2. **拆头**:$(B, L, d) \to (B, L, h, d_h) \to$ transpose 成 $(B, h, L, d_h)$,其中 $d_h = d/h$
3. 每个头独立做缩放点积注意力
4. **合头**:$(B, h, L, d_h) \to$ transpose 回 $(B, L, h, d_h) \to$ reshape 成 $(B, L, d)$
5. 再过一个输出线性层 $W^O$

$$
\mathrm{MHA}(x) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\, W^O, \quad
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q,\, KW_i^K,\, VW_i^V)
$$

```python
import torch, torch.nn as nn, torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        assert d % h == 0
        self.h, self.dh = h, d // h
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, mask=None):          # x: (B, L, d)
        B, L, d = x.shape
        def split(t):                          # (B,L,d) -> (B,h,L,dh)
            return t.view(B, L, self.h, self.dh).transpose(1, 2)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))

        scores = q @ k.transpose(-2, -1) / self.dh ** 0.5      # (B,h,L,L)
        if mask is not None:
            # 约定 True 表示允许关注；形状须能广播到 (B,h,L,L)
            allowed = mask.to(device=x.device, dtype=torch.bool).expand_as(scores)
            if not allowed.any(dim=-1).all():
                raise ValueError("每个 query 至少要保留一个可见 key")
            scores = scores.masked_fill(~allowed, float('-inf'))
        out = F.softmax(scores, dim=-1) @ v                    # (B,h,L,dh)

        out = out.transpose(1, 2).contiguous().view(B, L, d)   # 合头
        return self.wo(out)
```

### 参数量

四个 $d \times d$ 的线性层 → $4d^2$(加 bias 再 $+4d$)。**和头数无关**:头数只决定怎么把 $d$ 切开,总宽度没变。所以"多头不增加参数量,只改变信息的组织方式"。

### 常见坑

- `view(B, L, h, dh)` 之后**必须** transpose,直接 `view(B, h, L, dh)` 会把序列和头维度搅乱
- 合头前要 `.contiguous()`,否则 view 报错
- 缩放除的是 $\sqrt{d_h}$;写成 $\sqrt{d}$ 会让注意力过于平滑

自回归训练时还要传下三角 causal mask,让位置 $t$ 只能读取 $0\ldots t$。否则模型训练时会偷看目标 token 右侧的信息,loss 虽低却不能用于逐 token 生成。增量推理只查询当前新 token,历史 K/V 由 KV cache 提供;此时仍要保证当前 query 不能访问尚未生成的位置。

## 知识点

多头注意力、拆头/合头的维度变换、参数量估算、注意力 mask;GQA/MQA 是在这个基础上共享 K/V 头。

- 来源:[老师平台](https://course.terminiai.com/interview),P004-Q011/Q198；本批真实面经 [B006-Q001](../../docs/references/面经原题.md#b006-g01-q001)、[Q026](../../docs/references/面经原题.md#b006-g01-q026)、[Q079](../../docs/references/面经原题.md#b006-g01-q079)、[Q098](../../docs/references/面经原题.md#b006-g01-q098)、[Q104](../../docs/references/面经原题.md#b006-g01-q104)、[Q117](../../docs/references/面经原题.md#b006-g01-q117)、[Q196](../../docs/references/面经原题.md#b006-g01-q196)、[Q215](../../docs/references/面经原题.md#b006-g01-q215)。老师答案与逐图纠错见 [P009](../../docs/references/写作参考索引.md#p009)。
- 依据:[Transformer](https://arxiv.org/abs/1706.03762)。

## 追问

- 为什么多头有效?(不同子空间学不同关系,类似 CNN 多个卷积核)
- 头数怎么选?$d_h$ 太小会怎样?
- KV cache 存的是哪一部分?为什么 GQA 能省显存?
- 不加 causal mask 训练 Decoder-only 模型会发生什么?

## Note

---
difficulty: 困难
topic: 并行推理/张量并行Transformer层
summary: 手写支持 TP 的 transformer layer,含 MHA 与 MLP 切分
tags: [真题, 待校对, 手撕代码, 张量并行, TP, all-reduce]
company:
mastered: false
highfreq: false
---

## 题目

实现一个支持张量并行(TP)的 transformer layer,包含 MHA 和 MLP 两部分。说明每个权重怎么切、通信点放在哪里、为什么这样切通信量最小。

## 要点

- 列切后接行切,一个子层只需一次 all-reduce
- MHA 按头切:QKV 列切,输出投影行切,每卡持有完整的若干头
- MLP 同构:up/gate 列切,down 行切
- 通信点在每个子层的末尾,共两次 all-reduce
- LayerNorm 和残差不切,每卡各算一份,避免额外通信

## 答案

**核心模式只有一条:列切接行切(column-parallel → row-parallel),这样一个子层只需在末尾做一次 all-reduce。**

设 TP 组大小为 $P$。对 $Y=XA$,若按列切 $A=[A_1,\dots,A_P]$,每卡得到 $Y_i=XA_i$,是完整结果的一部分列,**无需通信**;紧接着的 $Z=YB$ 按行切 $B$,每卡算 $Z_i=Y_iB_i$ 得到部分和,再 all-reduce 相加。若反过来先行切再列切,中间就要多一次通信。

```python
class TPTransformerLayer(nn.Module):
    def __init__(self, d, n_heads, tp):
        super().__init__()
        assert n_heads % tp == 0                 # 头数必须能整除
        self.nh, self.hd = n_heads // tp, d // n_heads
        self.qkv  = ColumnParallelLinear(d, 3 * d)   # 列切,不通信
        self.proj = RowParallelLinear(d, d)          # 行切,末尾 all-reduce
        self.up   = ColumnParallelLinear(d, 4 * d)
        self.down = RowParallelLinear(4 * d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        h = self.ln1(x)                              # 每卡各算一份,不切
        q, k, v = self.qkv(h).chunk(3, dim=-1)       # 每卡 nh 个完整头
        q, k, v = (t.view(*t.shape[:2], self.nh, self.hd).transpose(1, 2)
                   for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(*x.shape[:2], -1)
        x = x + self.proj(o)                         # ← all-reduce #1
        h = self.ln2(x)
        x = x + self.down(F.gelu(self.up(h)))        # ← all-reduce #2
        return x
```

**MHA 按头切是关键**:QKV 列切之后,每张卡拿到的是 $n_h/P$ 个**完整的头**,注意力计算在卡内闭合,不需要跨卡通信;只有输出投影这一步产生部分和。若按 head_dim 切,单个头会被拆开,softmax 就得跨卡,通信量暴涨。

**LayerNorm 和残差不切**:它们参数量小、计算便宜,每卡各算一份即可。切了反而要为归一化的统计量再加一次通信。

所以整层通信是**两次 all-reduce**,各 $O(b\cdot s\cdot d)$;前向两次、反向两次(all-reduce 的反向仍是 all-reduce,列切层的反向需要对梯度做 all-reduce)。

边界:`n_heads % tp == 0` 是硬约束;GQA/MQA 下 KV 头数更少,切分要以 KV 头数为准,否则会出现某些卡没有 KV 头。原理与通信量核算见 [并行方式全景](../AI%20Infra/091-并行方式全景.md) 和 [通信算子与通信量](../AI%20Infra/092-通信算子与通信量.md)。

## 知识点

列切与行切的配对、按头切分、卡内闭合的注意力、all-reduce 位置、LayerNorm 不切的理由、GQA 下的 KV 头约束。

## 追问

- 反向传播时,列切层和行切层各需要什么通信?
- GQA 且 KV 头数小于 TP 度时怎么办?
- 序列并行(SP)接进来后,通信算子会变成什么?
- 为什么不把 LayerNorm 也切开?

## Note

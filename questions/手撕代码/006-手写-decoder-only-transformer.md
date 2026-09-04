---
difficulty: 简单
topic: Transformer整体架构/手写Decoder-only Transformer
summary: 用PyTorch实现训练前向完整的Decoder-only Transformer
tags: [待校对, 手撕代码, Transformer]
company: 字节
mastered: false
highfreq: false
---

## 题目

请用 PyTorch 手写一个完整的 Decoder-only Transformer,包含 token/位置 Embedding、因果多头自注意力、FFN、LayerNorm、残差和多层堆叠,并说明训练前向、增量推理和最大序列长度的边界。

## 要点

- 因果 mask 禁止位置 $t$ 读取未来 token
- 多头拆分与合并的张量形状正确
- 使用 Pre-Norm 残差块并说明与 Post-Norm 的区别
- 训练整段并行计算,增量推理需为每层缓存 K/V
- 可学习位置表不能无条件外推到训练长度之外

## 答案

**Decoder-only Transformer 用带因果掩码的 self-attention 预测下一个 token;训练时整段并行,下面代码实现的是清楚可运行的教学版训练前向。**

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split_heads(t):
            return t.view(batch, length, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future, torch.finfo(scores.dtype).min)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        y = weights @ v
        y = y.transpose(1, 2).contiguous().view(batch, length, width)
        return self.resid_drop(self.out(y))


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, max_len, d_model=512,
                 n_heads=8, n_layers=6, d_ff=2048, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, token_ids, targets=None):
        batch, length = token_ids.shape
        if length > self.max_len:
            raise ValueError(f"length {length} exceeds max_len {self.max_len}")
        pos = torch.arange(length, device=token_ids.device)
        x = self.drop(self.token_emb(token_ids) + self.pos_emb(pos)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
```

训练时把相邻 token 对齐成输入与下一 token 标签：输入是 $[x_0,\ldots,x_{T-1}]$，targets 是 $[x_1,\ldots,x_T]$；causal mask 让各位置并行计算却看不到未来。推理时每步只新增一个 token,生产实现应让每层接收并追加 K/V Cache,避免重算全部前缀;上面代码未实现缓存,不能宣称是生产级解码器。Decoder-only 只有 masked self-attention,Encoder-Decoder 还让 Decoder 用 Cross-Attention 读取 Encoder 输出。

代码采用 Pre-Norm:$x+F(\operatorname{LN}(x))$,通常有利于深层训练初期的梯度;原始 Transformer 的 Post-Norm 是 $\operatorname{LN}(x+F(x))$,两者没有跨任务固定胜负。LayerNorm 对每个 token 的隐藏维归一化,不依赖 batch 统计,因此更适合变长序列和单样本解码;BatchNorm 的统计量还受 batch、padding 与训练/推理模式影响。

这里使用长度为 `max_len` 的可学习位置表。超长输入不能直接索引;可扩表并继续训练,或改用可外推/可插值的位置方案并重新验证,不能假定未训练位置自然有效。

## 知识点

完整数据流是 token/位置 Embedding → N 个 Pre-Norm 注意力与 FFN 残差块 → Final Norm → 词表头;训练 mask 与推理 KV Cache 是两条不同执行路径。

来源:[深维 LLM 平台](https://course.terminiai.com/interview),P004-Q015。

一手依据:[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)、[Layer Normalization](https://arxiv.org/abs/1607.06450)。

## 追问

- 为什么 GPT 通常使用 LayerNorm 而不是 BatchNorm?
- 序列超过训练最大长度时,位置表示怎样扩展并验证?
- Pre-Norm 与 Post-Norm 在结构和训练稳定性上有何差异?
- 训练整段前向与增量推理的 causal mask、KV Cache 有何不同?

## Note

---
difficulty: 困难
topic: FlashAttention/kernel入参差异
summary: 三种形态下 attention kernel 的入参差异与 causal 对齐坑
tags: [面经, 待校对, FlashAttention, chunked-prefill]
company:
mastered: false
highfreq: false
---

## 题目

prefill 阶段的 chunked prefill 用到的 attention 有什么差异?整段 prefill、纯 decode、chunked prefill 这几种形态的函数入参主要差在哪?

## 要点

- chunked prefill 的特征形态:**query 长度 ≠ KV 长度**
- 最容易踩的坑:$L_q \ne L_k$ 时 causal mask 必须**右下对齐**,偏移 = past
- 连续布局 kernel 与分页 kernel 的入参差别:累积序列长度 vs 块表
- 混合 batch 里同时有 query 长 1 的 decode 和 query 长 c 的 prefill 块
- 「varlen 接口」为什么存在

## 答案

### chunked prefill 对 attention 提出了什么新要求

chunked prefill 把一条长 prompt 切成若干 chunk 分批送进模型。于是一次前向里,**query 只有新 chunk 的那几百个 token,要 attend 的 KV 却是「已缓存的历史 + 本 chunk」**——出现了整段 prefill 里没有的情况:**query 长度 ≠ KV 长度**。

### 最容易踩的坑:causal mask 要右下对齐

当 $L_q \ne L_k$ 时,chunk 里第 $i$ 个 query 的绝对位置是 $\text{past} + i$,它应该能看到 KV 的第 $0 \ldots \text{past}+i$ 位。也就是说 **mask 的对角线必须从矩阵右下角出发,偏移量正是 $L_k - L_q = \text{past}$**。

如果按左上对齐画对角线(query $i$ 只看 KV $0..i$),chunk 里的 token 就看不见自己的历史,结果直接错——而且是**loss 不炸、只是效果变差的隐蔽错误**。flash-attn 在 2.1 版把 $L_q \ne L_k$ 时 causal 的语义从左上对齐改成右下对齐,就是为了这个场景。

### 三种形态的入参对照

| 形态 | query 长度 | KV 长度 | causal | 关键入参 |
|---|---|---|---|---|
| 整段 prefill | $L$ | $L$ | 左右对齐等价 | 一套累积序列长度 `cu_seqlens` 够用 |
| 纯 decode | 1 | past + 1 | 退化,不需要 mask | KV 长度数组 + 块表 |
| **chunked prefill(混合 batch)** | chunk 长 $c$,各请求不同 | past$_i$ + $c$,各请求不同 | **必须右下对齐,偏移 = past$_i$** | `cu_seqlens_q` 与 `cu_seqlens_k` **两套**、各自的 max_seqlen、块表 |

**核心差别一句话**:从「一套累积序列长度描述一切」变成「query 和 KV 各有一套长度前缀和,再加每条序列的历史偏移」。同一个 batch 里既有 query 长度为 1 的 decode 请求、又有 query 长度为 $c$ 的 prefill chunk,kernel 靠这两套前缀和把它们统一处理——**这就是 varlen 接口存在的理由**。

### 另一条正交的入参差别:连续 vs 分页

上面讲的是「query/KV 长度怎么描述」,还有一条是「KV 存在哪」:

- **连续布局的 FlashAttention**:给基址 + stride 就能算出第 $j$ 块 K/V 在哪;变长 batch 只是把多条序列紧凑拼接,额外传一套 `cu_seqlens`。
- **分页布局(PagedAttention)**:kernel **多吃一张块表**,取第 $j$ 块之前先查表拿物理块号;写侧还要一份槽位映射,把新算出的 K/V 散写回块池。

所以一句话总结两者的入参差异:**`cu_seqlens` vs block table**——前者描述「序列边界在哪」,后者描述「KV 块落在物理显存的哪一格」。两者不冲突,分页 kernel 通常两个都要。

## 知识点

varlen 接口、`cu_seqlens_q` / `cu_seqlens_k` 两套前缀和、causal 右下对齐、块表与槽位映射、混合 batch。

## 追问

- 为什么左上对齐会造成「loss 不炸但效果变差」的隐蔽错误?
- 为什么把 prefill 块和 decode 混进同一步反而更划算?
- 块大小和 kernel 的 tile 粒度有什么约束关系?

## Note

---
difficulty: 简单
topic: 解码策略/重复惩罚与ngram约束
summary: no-repeat n-gram和重复惩罚怎样改logits,各自会误伤什么
tags: [面经, 待校对, 解码策略, Repetition Penalty, n-gram, 文本生成]
company: 快手、腾讯
mastered: false
highfreq: false
---

## 题目

1. No Repeat N-gram Size机制在文本生成中有何作用？它是如何防止局部重复短语出现的？
2. 请详细解释Repetition Penalty在文本生成过程中的作用机制，包括其实现原理、超参数影响以及在Hugging Face等框架中的应用方式。

## 要点

- no-repeat n-gram 是硬约束：把会形成已出现 n-gram 的候选 token 设为不可选
- $n$ 越大，匹配条件越苛刻，约束通常越弱；$n=1$ 最强
- repetition penalty 是软惩罚，Hugging Face 对正负 logit 的处理方向不同
- frequency penalty 看出现次数，presence penalty 只看是否出现
- 所有方法都会误伤必要重复，需按任务验证并与训练侧问题分开

## 答案

**no-repeat n-gram 是“禁止再次拼出某个已经出现过的短语”，repetition penalty 是“让见过的 token 更难再选”。前者是硬封禁，后者是软降权。**

### no-repeat n-gram 怎样工作

设 $n=3$，当前末尾两个 token 是“机器 学习”。解码器会在已有序列中查找所有以“机器 学习”开头的三元组；如果以前出现过“机器 学习 模型”，就把候选“模型”的 logit 设为 $-\infty$。softmax 后它的概率为 0，因此不会再次形成相同三元组。

实现上无需枚举所有短语。维护“长度为 $n-1$ 的前缀 → 曾跟随它的 token 集合”即可，每步用当前末尾 $n-1$ 个 token 查禁用集合。

有一个容易答反的边界：**$n$ 越大，约束通常越弱。** 因为要完全重复更长的片段才触发。`no_repeat_ngram_size=1` 会禁止任何 token 再出现，几乎不可用；较小的 $n$ 更容易误伤姓名、术语、代码和固定格式。

### Hugging Face repetition penalty

对已经出现过的 token，常见 Hugging Face 语义在惩罚系数 $r>1$ 时为：

$$
z'=
\begin{cases}
z/r,& z>0\\
z\times r,& z<0
\end{cases}
$$

正 logit 变小，负 logit 变得更负，两种情况都降低被选概率。把所有 logit 统一除以 $r$ 是错的：负数除以大于 1 的数会更接近 0，反而提高概率。

惩罚哪些历史 token、是否包含 prompt、tokenization 如何切分，取决于具体框架和配置。系数也没有所有任务通用的“最佳范围”。

### 三种软惩罚的区别

| 方法 | 依据 | 直觉 |
|---|---|---|
| repetition penalty | 修改见过 token 的 logit，常为乘除式 | 已经见过就降权 |
| presence penalty | 是否出现过 | 出现一次和十次惩罚相同 |
| frequency penalty | 出现次数 | 越重复惩罚越大 |

它们的参数量纲不同，不能把同一个数直接横向比较。

### 与 Beam Search 和其他方法

Beam Search 中也可以在每条 beam 自己的历史上应用 n-gram blocking 或 repetition penalty。Diverse Beam Search 主要减少**不同 beam 之间**的相似，no-repeat n-gram 主要禁止**单条序列内部**的局部重复；两者可以组合，但约束叠得太多会让候选集枯竭。

若重复来自训练数据重复、模型退化或长文本状态丢失，解码约束只能治表面。还可配合数据去重、合理的温度/top-p、对比解码和任务提示。限制 attention 窗口不是通用去重方法，它也可能损害长程一致性。

验收时同时看重复率、任务正确性和人工可读性。法律条款、代码标识符、诗歌副歌等本来就需要重复，不能只追求 n-gram 越少越好。

## 知识点

n-gram blocking、logit 硬掩码、Hugging Face repetition penalty、presence/frequency penalty、Beam Search 多样性。

- 真实面经：[B002-G01-Q104](../../docs/references/面经原题.md#b002-g01-q104)、[B002-G01-Q185](../../docs/references/面经原题.md#b002-g01-q185)
- 老师参考：[P005-Q104](../../docs/references/平台题/P005-Infra-091-120.md#p005-q104)、[P005-Q185](../../docs/references/平台题/P005-Infra-181-199.md#p005-q185)

## 追问

- 参考追问：如果 $n$ 设置得太大，会有什么负面影响？
- 参考追问：no-repeat n-gram 与 Diverse Beam Search 有什么区别，能否组合？
- 参考追问：除了 n-gram 去重，还有哪些方法能缓解文本重复？
- 参考追问：Repetition、Frequency 和 Presence Penalty 有什么区别？
- 参考追问：Repetition Penalty 怎样与 Beam Search 配合？
- 参考追问：怎样避免误伤关键词的合理重复？

## Note

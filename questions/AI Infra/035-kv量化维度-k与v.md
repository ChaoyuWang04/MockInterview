---
difficulty: 困难
topic: KVCache量化/K与V的量化维度
summary: 为什么 K 按 channel 量化、V 按 token 量化,维度怎么判断
tags: [面经, 待校对, KVCache量化, 量化粒度]
company:
mastered: false
highfreq: false
---

## 题目

对 KV cache 做量化时,量化维度是怎么判断的?为什么 K 按 channel、V 按 token?

## 要点

- 判断维度要同时看两件事:**分布事实**和**算术事实**
- 分布上 K 沿 head_dim 有固定的大通道,V 沿哪个方向切都平(KIVI 的结论)
- 算术上 attention 有两次乘法、求和维不同,scale 落在求和维上就提不出来
- 结果是 K 的分布想要 per-channel、K 的算术想要 per-token,两边打架
- 另有两重矛盾:RoPE 抹匀通道结构、decode 逐 token 追加使历史失效

## 答案

判断量化维度只看两件事:**这一维上分布长什么样**,以及**这一维是不是当前这次乘法的求和维**。

### 分布事实

KIVI 系统测过 KV cache 的元素分布,结论是**非对称**的:K 的幅度沿 head_dim 有几条固定的"大通道"(与 Massive Activations 描述的输入无关的极大激活是同一类现象),而 V 沿哪个方向切都差不多平。所以 **K 想要 per-channel,V 无所谓**。

### 算术事实

attention 要**分两次乘法**看:

$$
S_{t} = \sum_{c=1}^{d} Q_c K_{t,c}, \qquad O_{d'} = \sum_{t} P_t V_{t,d'}
$$

第一次沿 **channel** 求和,第二次沿 **token** 求和。一个 scale 免不免费,只看它落不落在**当前这次求和的那一维**上:

| 张量 | 粒度 | 在求和维上? | 代价 |
|---|---|---|---|
| K | per-token($s_t$) | 否 | **免费**:MMA 出来后每行乘一个数 |
| K | **per-channel($s_c$)** | **是** | 只能逐元素反量化;或折进 $Q$(数学等价,但要求同组 token 共用一套 $s_c$,kernel 遍历顺序被绑死) |
| V | per-token($s_t$) | 是 | **仍然便宜**:$\sum_t (P_t s_t)\hat V_{t,d'}$,$P_t$ 本就是寄存器里的标量 |
| V | per-channel | 否 | 便宜,但 V 没有离群通道,做了没收益 |

矛盾于是很清楚:**K 的分布想要 per-channel,K 的算术想要 per-token;V 两边都不为难。** 注意 V 那顿免费午餐有前提——**只有 $P$ 还是浮点时才成立**。

### 还有两重矛盾

- **RoPE 抹匀通道结构**:RoPE 把 head_dim 两两配对做旋转、旋转角随位置变,post-RoPE 的 K 里同一通道在不同位置是两个原始通道的随位置摆动的组合,per-channel 想抓的"跨 token 稳定的通道结构"正好被破坏。KVQuant 的解法是 **pre-RoPE 量化**,读出来反量化后再补 RoPE——decode 本来访存受限、算力闲着,这是少见的"拿算力换精度"真划算的地方
- **per-channel 的 scale 跨 token,而 decode 逐 token 追加**:新 token 若在某通道上更大,该通道 scale 就要变,一变则此前所有已量化的值全部作废。KIVI 的解法是按固定大小分组、每组封口时定一次、最近若干 token 保持全精度;KVQuant 干脆离线校准、运行时不更新

**结论**:per-channel K 是学术线(2/3-bit 靠它),per-token / per-head 是工程线(fp8/int8 靠它)。

## 知识点

KIVI 的 K/V 分布非对称结论、attention 两次乘法的求和维、scale 可提取性、RoPE 对通道结构的破坏、追加写与 per-channel scale 的冲突。

## 追问

- 把 per-channel 的 $s_c$ 折进 $Q$ 数学上成立,工程上为什么还是难做?
- pre-RoPE 量化多出的那一步旋转,在 decode 上真的免费吗?
- 想让 $PV$ 也走整数矩阵乘,V 的那顿免费午餐还在吗?

## Note

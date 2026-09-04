---
difficulty: 简单
topic: RNN与门控循环网络/门控结构和选型
summary: GRU与LSTM如何用不同门控保留和更新序列状态
tags: [真题, GRU, LSTM, RNN, 门控机制, 待校对]
company: 小红书
mastered: false
highfreq: false
---

## 题目

请比较 GRU 与 LSTM 的门控结构、参数量、计算、长依赖能力和适用场景，写出更新门/重置门与输入门/遗忘门/输出门的作用，并说明实际任务中怎样选基线及如何面向端侧压缩。

## 要点

- LSTM 把 cell state 与 hidden state 分开，用输入、遗忘、输出门控制
- GRU 合并状态，用更新门决定保留/写入，用重置门控制候选状态读取历史
- 同输入和隐藏维下 GRU 通常参数更少，但差值不是固定百分比
- 最终效果没有固定排名，要在相同预算、数据和隐藏维下比较

## 答案

**GRU 用两道门和一个状态简化 LSTM，通常参数与计算更少；LSTM 用独立 cell state 和三道门提供更细的记忆控制。** 一种常见 GRU 写法是

$$
z_t=\sigma(W_zx_t+U_zh_{t-1}),\quad
r_t=\sigma(W_rx_t+U_rh_{t-1}),
$$

$$
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})),\quad
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
$$

这里更新门 $z_t$ 决定保留旧状态还是写入候选，重置门 $r_t$ 决定生成候选时读取多少历史。不同资料可能把 $z_t$ 与 $1-z_t$ 的命名方向互换，解释时要与公式保持一致。

LSTM 则计算输入门 $i_t$、遗忘门 $f_t$、输出门 $o_t$ 与候选 $g_t$：

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t).
$$

$c_t$ 是较直接的记忆通路，遗忘门保留旧记忆，输入门写新内容，输出门决定对外暴露多少。它的控制更细，但每步有更多投影和中间状态。

若输入维为 $m$、隐藏维为 $h$，忽略实现差异，GRU 约有三组 $(m\times h,h\times h)$ 变换，LSTM 约有四组；这只说明同维度下 GRU 通常更小，不能宣称固定少 25%。投影层、双向结构、偏置和为了对齐效果而调整的隐藏维都会改账。也没有“超过 100 步必选 LSTM”或效果固定相差 2% 的规律。

实践中先按延迟与内存选一个尺寸可比的 GRU 和 LSTM，再在相同训练预算下比较验证损失、长依赖切片、吞吐和峰值内存。严格流式、端侧、小数据场景可优先试 GRU；需要更细记忆控制时可把 LSTM 作为强基线。端侧还能结合较小隐藏维、低比特量化、剪枝或蒸馏，但要回归长依赖与实时延迟。Transformer 出现后循环模型仍有价值的原因与整体效率比较，见 [Transformer 与循环网络](026-transformer与循环网络.md)。

## 知识点

GRU 的“更新/重置”和 LSTM 的“输入/遗忘/输出”都是数据依赖的软门。参数多少由实际矩阵形状决定，任务性能必须在相同预算下实测。

- 一手依据：[Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078)、[Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)。

## 追问

- GRU 的更新门和重置门怎样协作，为什么不同资料的更新公式看似相反？
- LSTM 的 cell state 与 hidden state 分别承担什么作用？
- GRU 的参数为什么通常更少，为什么不能背固定百分比？
- 实际 NLP 或时序任务中怎样公平选择 GRU 与 LSTM？
- 资源受限设备上还能怎样压缩循环模型？

## Note

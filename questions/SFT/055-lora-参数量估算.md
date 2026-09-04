---
difficulty: 简单
topic: LoRA/参数量估算
summary: 从矩阵形状推导 LoRA 参数量,估算 LLaMA 7B 与多卡训练显存
tags: [SFT, LoRA, 参数量, 显存估算, 待校对]
company: 淘天
mastered: false
highfreq: false
---

## 题目

请解释 LoRA(Low-Rank Adaptation)的基本原理,并推导其在 Transformer 层中引入的额外参数量计算方式,以注意力权重为例说明。请以一个具体模型估算可训练参数规模,并说明参数节省与实际训练显存之间的区别。

## 要点

- 对每个实际目标矩阵求和 r(d+k),不能按基座比例猜测
- MHA 与 GQA 的 K/V 形状不同
- 明确 LLaMA 7B 的层数、维度、目标层与秩
- 显存还含冻结基座、激活、优化器及通信缓冲,分片才可能按卡分摊

## 答案

**LoRA 减少的是需要训练的小矩阵参数,不能把这一比例直接当作总显存降幅。** 对 $W_j\in\mathbb R^{d_j\times k_j}$,冻结原矩阵,训练 $A_j\in\mathbb R^{r_j\times k_j}$、$B_j\in\mathbb R^{d_j\times r_j}$,有效权重为 $W_j+(\alpha_j/r_j)B_jA_j$。

### 参数量逐项计算

$$
N_{\rm LoRA}=\sum_{j\in\mathcal T}r_j(d_j+k_j)
$$

$\mathcal T$ 是实际适配的矩阵集合,另有解冻的 bias、词嵌入或输出头时须单独加上。一个 $4096\times4096$ 矩阵取 r=16,新增 131,072 个参数,为原矩阵的 1/128。

经典多头注意力中,Q/K/V/O 都是 $d\times d$,全部适配每层为 $8rd$,仅 Q/V 为 $4rd$。GQA 的 K/V 输出维度可能较小,必须按真实形状重算;也不能漏掉已适配的 FFN gate/up/down。

原始 LLaMA 7B 的隐藏维度为4096、32层。只在每层 Q/V 加 LoRA:
$r=8$ 时为 $32\times4\times4096\times8=4,194,304$;r=64 时为33,554,432。后者相对论文约6.7B基座约0.50%,不是先猜“Q/V占30%”再估算。

### 显存与多卡怎么记账

在 BF16 权重及梯度、FP32 主权重与 Adam 两个状态的假设下,每个可训练参数约16字节,因此可先估:

$$
M\simeq 2N_{\rm base}+16N_{\rm LoRA}
+M_{\rm activation}+M_{\rm workspace}
$$

单位为字节;不同优化器和精度要换系数。基座被冻结仍需前向及输入梯度计算,LoRA 也有额外分支,总速度不按参数缩减倍数增长。

普通数据并行每卡复制模型及训练状态,不能把上式直接除以卡数。采用参数、梯度或优化器分片时,分别按实际分片范围计算,再加本卡激活、临时聚合和通信峰值;没有配置不能报一个“多卡 LoRA 固定显存”。

初始化、rank 和方法对比复用 [原理与秩选择](002-lora-原理与秩选择.md);全参效果差距及多适配器部署见 [LoRA 与全参数微调](066-lora与全参数微调.md)。4-bit 下的65B/48GB结果来自 [QLoRA](051-lora与qlora.md) 的特定训练配置。

## 知识点

LoRA、参数量、显存估算。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q087、P002-Q088、P002-Q104。
- 依据:[LoRA](https://arxiv.org/abs/2106.09685)、[LLaMA 表2](https://arxiv.org/html/2302.13971v1)、[QLoRA](https://arxiv.org/abs/2305.14314)。

## 追问

- 全参微调的精度优势会在哪些情况下扩大,怎样缓解?
- LoRA 与 Adapter、Prefix-tuning 各适合什么场景?
- 为什么 A 随机、B 零初始化?
- 多个 LoRA 如何切换与合并?
- QLoRA 为什么能在单卡48GB上微调65B?

## Note

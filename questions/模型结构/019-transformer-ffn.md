---
difficulty: 简单
topic: FFN与激活/维度扩展 激活与门控
summary: Transformer FFN怎样逐token变换及为何先升维再降维
tags: [Transformer, FFN, MLP, SwiGLU, 待校对]
company: 快手、腾讯、网易、抖音、阿里云
mastered: false
highfreq: false
---

## 题目

请说明 Transformer FFN 的结构、维度变化、激活函数和作用：为什么常先从 $d_{model}$ 升到 $d_{ff}$ 再降回去，$4d$ 是否有理论必然性，FFN 能否删除，以及 GELU FFN、SwiGLU 与 MoE 有什么关系？

## 要点

- FFN 对每个 token 独立使用同一组参数，负责通道变换与非线性
- 升维提供更宽的中间特征，降维后才能接回残差流
- $d_{ff}=4d$ 是原始 Transformer 的配置，不是理论最优常数
- SwiGLU 多一条门控投影，MoE 把单个 FFN 扩成稀疏选择的多个专家

## 答案

**FFN 是每层里对各 token 独立执行的非线性变换：注意力先在 token 之间交换信息，FFN 再沿特征维加工每个位置。** 经典形式为

$$
\operatorname{FFN}(x)=W_{down}\,\sigma(W_{up}x+b_{up})+b_{down},
$$

其中 $W_{up}\in\mathbb{R}^{d_{ff}\times d}$，$W_{down}\in\mathbb{R}^{d\times d_{ff}}$。所有位置共享这组权重，但彼此不在 FFN 内通信。升维让模型能产生更多中间特征和激活模式，激活函数引入非线性，降维则恢复到 $d$，以便与残差主路相加。忽略偏置时，参数量约为 $2dd_{ff}$，每 token 的主要矩阵计算也是 $O(dd_{ff})$。

原始 Transformer 采用 $d_{ff}=4d$ 和 ReLU；后来常见 GELU。四倍只是效果、参数和算力之间的经验配置，不存在普适推导证明它优于 $2d$ 或 $8d$。实际宽度会随参数预算、门控结构、硬件对齐和任务调整。增加宽度与增加深度也不是可直接互换的：前者扩大单层特征容量，后者增加连续变换与通信次数。

SwiGLU 使用两条升维投影，一条经 SiLU 形成门，一条提供内容，再逐元素相乘并降维：

$$
\operatorname{SwiGLU}(x)=W_{down}\bigl(\operatorname{SiLU}(W_gx)\odot W_ux\bigr).
$$

它有三个大矩阵，所以公平比较时常调整 $d_{ff}$ 以对齐参数或 FLOPs。FFN 不宜随意删除：去掉后会失去主要的逐位置非线性与通道扩展，但整个 Transformer 仍含 softmax 等非线性，不能称为“纯线性网络”。FFN 与 Attention 分工不同，直接交换或合并会改变信息先跨 token 还是先跨通道传播。剪枝、量化时也不能预设谁更敏感，应按层记录误差和任务掉点；激活差别见 [GELU 与 SwiGLU](020-激活函数与swiglu.md)。MoE 通常把一个 FFN 换成多个专家 FFN并稀疏路由，具体权衡见 [Dense 与 MoE](022-dense与moe.md)。

## 知识点

Attention 负责 token mixing，FFN 负责 channel mixing。经典 FFN 是“升维—激活—降维”，门控 FFN 要重新核算宽度、参数与计算，不能照搬 $4d$。

- 来源：[老师平台](https://course.terminiai.com/interview)，P004-Q016、P004-Q045、P004-Q180、P004-Q190、P004-Q191。
- 一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

## 追问

- $d_{ff}=4d$ 是怎样来的，为什么不是普适最优值？
- FFN 与自注意力各自混合哪个维度，能否交换或合并？
- 删除 FFN 后网络还算不算非线性，能力会损失在哪里？
- SwiGLU 为什么有两条升维分支，怎样公平比较参数和计算？
- MoE 怎样从普通 FFN 扩展而来？

## Note

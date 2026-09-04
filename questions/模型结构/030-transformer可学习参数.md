---
difficulty: 简单
topic: Transformer整体架构/Transformer中有哪些可学习参数
summary: Transformer各组件有哪些参数及冻结策略如何影响更新
tags: [真题, Transformer, 参数量, Embedding, 反向传播, 待校对]
company: 美团
mastered: false
highfreq: false
---

## 题目

标准 Transformer 训练时会更新哪些可学习参数？请按 Embedding、Q/K/V/O、FFN、Norm、位置表示和 LM Head 列出作用与常见形状，并说明 MHA/GQA、门控 FFN、权重绑定、bias 和冻结策略怎样改变这份清单。

## 要点

- 参数包括输入表示、注意力投影、FFN、Norm 仿射参数与输出头
- 位置编码可能固定也可能可学习，不能统一回答“有参数”或“没参数”
- 实际形状依注意力头、KV 头、门控、bias 与权重绑定变化
- 反向传播计算梯度，优化器只更新被设为可训练且纳入参数组的张量

## 答案

**训练会更新所有参与损失、允许求梯度且交给优化器的可学习张量；“标准 Transformer”只能给出组件清单，不能给唯一参数模板。** 设词表 $V$、隐藏维 $d$、中间维 $d_{ff}$、Query 头数 $h_q$、KV 头数 $h_{kv}$、头维 $d_h$：

| 组件 | 常见参数形状与作用 |
|---|---|
| Token Embedding | $V\times d$，把 token ID 映射为隐藏向量 |
| Q 投影 | $d\times(h_qd_h)$，产生查询 |
| K/V 投影 | 各为 $d\times(h_{kv}d_h)$，产生键和值 |
| O 投影 | $(h_qd_h)\times d$，合并注意力头输出 |
| 经典 FFN | $d\times d_{ff}$ 与 $d_{ff}\times d$，升维、激活、降维 |
| 门控 FFN | gate、up 两个升维矩阵和 down 矩阵 |
| Norm | 每层常有长度 $d$ 的 $\gamma$；LN 常有 $\beta$，RMSNorm 标准形式通常没有 |
| LM Head | $d\times V$，把隐藏状态映射到词表 logits |

矩阵转置方向会随框架约定变化，bias 也可有可无。MHA 通常有 $h_{kv}=h_q$；GQA/MQA 减少 KV 头，因此 K/V 矩阵与缓存更小。QKV 可以打包为一个大矩阵，但这不等于三者一定共享同一映射；真正共享会限制表示，需要以质量和效率验证。

位置部分取决于方案：learned absolute embedding 有可学习的 $L_{max}\times d$ 表；原始固定正弦位置编码没有；典型 RoPE 是对 Q/K 的固定旋转，也没有位置表参数；相对位置偏置可固定或可学习。不能按“相对/绝对”四个字直接判断。

输入 Embedding 和 LM Head 可以共享同一权重，参数不能重复计数。经典 FFN 与门控 FFN 的矩阵数不同，$d_{ff}$ 也会调整，所以“FFN 固定占三分之二”不是通则。前向时这些参数产生表示和 logits，反向传播由损失链式求导；预训练通常更新全部参数，微调可全参更新，也可冻结基座只训练 LoRA、Adapter 或任务头。冻结参数仍可参与前向，是否保存其梯度和优化器状态则取决于训练设置。

## 知识点

参数清单要与具体架构配置一起读：头数决定 Q/K/V 形状，FFN 类型决定矩阵数，Norm 类型决定仿射项，位置方法与权重绑定决定是否新增参数。

- 一手依据：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[Grouped-Query Attention](https://arxiv.org/abs/2305.13245)、[RMSNorm](https://arxiv.org/abs/1910.07467)、[LLaMA](https://arxiv.org/abs/2302.13971)。

## 追问

- MHA、GQA 与 MQA 的 K/V 投影形状怎样变化？
- Q/K/V 打包成一个矩阵是否等于共享参数？
- LayerNorm 的 $\beta$ 为什么有时不存在，RMSNorm 又有哪些参数？
- 位置编码在什么情况下有可学习参数？
- 预训练、全参微调与 LoRA 微调分别更新哪些张量？

## Note

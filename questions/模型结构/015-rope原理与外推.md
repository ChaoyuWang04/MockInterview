---
difficulty: 中等
topic: RoPE/原理与长度外推
summary: RoPE怎样表示相对位置以及如何扩展上下文
tags: [RoPE, 长上下文, 位置插值, YaRN, 待校对]
company: 字节、小红书、百度
mastered: false
highfreq: false
---

## 题目

请推导 RoPE 如何把位置注入 Q/K 并使内积体现相对位移，解释不同频率的作用。超过训练长度时为什么会退化，位置插值、NTK-aware、Dynamic NTK 和 YaRN 分别怎样扩展上下文？

## 要点

- 把相邻通道组成二维向量并按位置旋转
- $R_m^\top R_n=R_{n-m}$ 使 QK 内积依赖相对位移
- 频率在超训练长度时进入未学过的相位分布
- 扩窗方法改变位置或频率，并常需继续训练和目标长度评测

## 答案

**RoPE 不把位置向量加到 token 上，而是按位置旋转每一对 Q/K 通道；两次旋转在点积中相减，因此注意力分数自然包含相对位移。** 第 $i$ 个二维通道对在位置 $m$ 旋转角为 $m\theta_i$：

$$
q_m=R(m\theta_i)q,\qquad k_n=R(n\theta_i)k
$$

由于旋转矩阵满足 $R(m\theta_i)^\top R(n\theta_i)=R((n-m)\theta_i)$：

$$
q_m^\top k_n=q^\top R((n-m)\theta_i)k
$$

不同通道使用不同频率，高频区分近邻，低频覆盖更长尺度。标准实现同时旋转 Q 和 K，V 通常不旋转。该公式能计算训练长度外的位置，但模型没学过的新相位组合仍可能导致质量下降；RoPE 也不保证注意力随距离单调衰减。

| 扩窗方法 | 核心动作 | 边界 |
|---|---|---|
| 位置插值 | 把目标位置按比例压回原训练范围 | 分辨率改变，论文配合微调 |
| NTK-aware | 按频率调整尺度，减少各频段失真 | 具体公式和参数依实现 |
| Dynamic NTK | 根据实际上下文长度动态设缩放 | 训练和推理配置必须一致验证 |
| YaRN | 分频段插值，并配合 attention 缩放等设计 | 不是只改一个 base 即无条件生效 |

### 常见追问简答

- **base 越大越好吗？** 不是；它改变频率分布，要结合原训练长度、目标长度和微调数据选择。
- **RoPE 与 ALiBi 怎么选？** RoPE 旋转 Q/K，ALiBi 在 logits 加距离偏置；选型看 checkpoint、内核和扩窗实验。
- **能否直接扩到 128K 或百万 token？** 不能只凭公式承诺，需要长序列继续训练或微调，并测困惑度、检索和生成。
- **能直接用于所有线性注意力吗？** 不一定，核映射和矩阵重排后是否保留旋转关系要逐算法确认。

## 知识点

RoPE 的可靠结论是 Q/K 内积显式依赖相对位移；“天然无限外推、自然远程衰减、显式保存绝对位置”都属于过强表述。

- 来源：[老师平台](https://course.terminiai.com/interview)，采集编号 P004-Q017、P004-Q018、P004-Q035、P004-Q036、P004-Q063、P004-Q099、P004-Q166、P004-Q240、P004-Q283、P004-Q284、P004-Q292、P004-Q293、P004-Q296、P004-Q326。
- 依据：[RoFormer / RoPE](https://arxiv.org/abs/2104.09864)、[Position Interpolation](https://arxiv.org/abs/2306.15595)、[YaRN](https://arxiv.org/abs/2309.00071)。

## 追问

- 怎样从二维旋转矩阵推导 RoPE 的相对位置性质？
- RoPE 为什么能计算表外位置，却不保证长度外推质量？
- 位置插值、NTK-aware、Dynamic NTK 与 YaRN 分别改变什么？
- RoPE 的 base 和目标长度应如何联合选择？
- RoPE 与 ALiBi 的注入位置和扩窗思路有何不同？

## Note

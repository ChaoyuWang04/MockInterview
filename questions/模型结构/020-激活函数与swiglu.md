---
difficulty: 简单
topic: FFN与激活/常见激活函数与LLM中的GELU SwiGLU
summary: 常见激活怎样影响梯度，GELU和SwiGLU为何用于LLM
tags: [激活函数, GELU, SiLU, SwiGLU, 待校对]
company: 美团、小红书、网易
mastered: false
highfreq: false
---

## 题目

为什么神经网络需要非线性？请比较 Sigmoid、Tanh、ReLU、Leaky ReLU、GELU、SiLU/Swish，并解释 LLM 中 GELU 与 SwiGLU 的设计动机、梯度和量化权衡；sin/cos 能不能作为激活函数？

## 要点

- 没有非线性，多层仿射变换仍可合并成一层
- 饱和区、零梯度、平滑性、输出范围和硬件成本共同影响选型
- SwiGLU 是门控 FFN 结构，不是单个标量激活函数
- sin/cos 并非不能训练，但周期性与初始化使其不适合作为通用默认项

## 答案

**激活函数让多层网络不再等价于一层线性变换。** 若两层都没有非线性，$W_2(W_1x+b_1)+b_2=(W_2W_1)x+b'$，再堆层也只是仿射映射，无法形成复杂的分段或弯曲决策边界。

| 函数 | 形式或范围 | 主要特点与常见用途 |
|---|---|---|
| Sigmoid | $\sigma(x)\in(0,1)$ | 大绝对值处梯度小；适合二分类概率和循环门控 |
| Tanh | $(-1,1)$ | 零中心，但两端同样饱和；常见于循环状态候选 |
| ReLU | $\max(0,x)$ | 正半轴梯度恒定、计算简单；负半轴可能长期零梯度 |
| Leaky ReLU | $\max(ax,x)$ | 给负半轴保留小梯度，斜率 $a$ 是设计选择 |
| GELU | $x\Phi(x)$ | 平滑地按输入大小调节，常见于 Transformer FFN |
| SiLU/Swish | $x\sigma(x)$ | 平滑且允许少量负输出；SiLU 与 Swish 常指同一形式 |

GELU 与 SiLU 都是平滑、轻微非单调的门；没有理论说明它们在所有模型中必胜 ReLU，最终要做控制参数量与训练预算的消融。SiLU 的导数为 $\sigma(x)+x\sigma(x)(1-\sigma(x))$。量化时要看激活范围、离群值、是否产生精确零以及目标硬件内核；“更平滑”不自动等于“更好量化”。

SwiGLU 不是一条激活曲线，而是 $\operatorname{SiLU}(W_gx)\odot W_ux$ 的门控 FFN：门分支按输入动态缩放内容分支，再由第三个矩阵降维。它比普通 GELU FFN 多一条投影，比较显存与速度时必须先对齐中间宽度、参数量和实现。LLaMA 公布的架构采用 SwiGLU，不能写成普通 GELU。

sin/cos 当然是非线性，也能用于隐藏层；问题是周期性会把相隔多个周期的输入映到相同值，梯度方向反复变化，常规初始化下不易作为通用默认激活。SIREN 通过专门的频率尺度与初始化训练正弦网络。位置编码用 sin/cos 是预先构造坐标特征，与把它作为每层可训练激活不是一回事。

选型时在同参数、数据和训练预算下比较收敛、任务质量、激活分布和目标硬件吞吐，模型变大不产生唯一答案。即使某个 FFN 去掉标量激活，门控乘法与注意力 softmax 仍可提供非线性，不能只凭“无 ReLU/GELU”断言全网退化。输出层用 softmax 还是其他归一化目标属于概率建模与损失设计，不应和隐藏层激活混为一谈。

## 知识点

激活函数的选型看表达、梯度、数值分布和内核；不要求全局单调或可逆。SwiGLU 属于门控拓扑，不能和 GELU 当成同层级的单函数直接比较。

- 来源：[老师平台](https://course.terminiai.com/interview)，P004-Q014、P004-Q064、P004-Q096、P004-Q159、P004-Q170、P004-Q177、P004-Q178、P004-Q210、P004-Q211、P004-Q217、P004-Q220、P004-Q242。
- 一手依据：[GELU](https://arxiv.org/abs/1606.08415)、[Swish](https://arxiv.org/abs/1710.05941)、[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)、[SIREN](https://arxiv.org/abs/2006.09661)、[LLaMA](https://arxiv.org/abs/2302.13971)。

## 追问

- 为什么无激活的多层网络仍等价于单层线性模型？
- ReLU 缓解了什么梯度问题，是否彻底解决深层网络的梯度问题？
- GELU、SiLU 和 SwiGLU 是什么关系？
- 激活函数怎样影响低精度量化，应如何公平验证？
- sin/cos 为什么能做位置编码，SIREN 又为何能把正弦用于隐藏层？

## Note

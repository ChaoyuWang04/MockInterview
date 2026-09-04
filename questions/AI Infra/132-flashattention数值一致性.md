---
difficulty: 中等
topic: FlashAttention/训练推理数值一致性
summary: 分块、精度和内核差异为何会造成训推数值偏差
tags: [真题, 待校对, FlashAttention, 数值稳定性, 训练推理一致性]
company: 蚂蚁金服
mastered: false
highfreq: false
---

## 题目

在使用 FlashAttention 进行大模型训练和推理时，若采用固定的分块策略，是否会导致训练与推理阶段出现数值不一致问题？请分析根因、影响、解决办法及权衡。

## 要点

- 分块 online softmax 在实数数学上与全局 softmax 等价
- 浮点加法顺序、dtype、kernel、mask 和形状都会造成舍入差异
- 固定 block size 不能单独保证逐 bit 一致
- 用分层误差测试判断偏差是否会改变任务结果

## 答案

**固定分块本身不会让 FlashAttention 在数学上变成近似算法，但训练和推理也不保证逐 bit 相同。差异通常来自浮点执行路径，而不是 softmax 公式失效。**

online softmax分块维护行最大值 $m$ 和指数和 $l$，新块到来时把旧统计量重标定到新的最大值：

$$
m'=\max(m,m_b),\qquad
l'=e^{m-m'}l+e^{m_b-m'}l_b
$$

这个恒等变换在实数中与整行 softmax 相同。但 GPU 上加法不满足结合律；训练反向和推理前向可能采用不同 tile、warp 归约顺序、累加精度、融合 kernel、padding/mask、确定性设置，所以出现末位误差。FP32统计能减小误差，不能保证与另一实现逐 bit 相同。

排查时固定输入和随机性，逐层比较：先核对 shape、mask、dtype 与 scale，再比较普通 attention 和 FlashAttention 的输出、梯度，最后看误差是否在残差层中放大并改变任务指标。若必须复现，锁定框架、内核版本、设备、算法配置和形状，并使用确定性路径；代价可能是速度下降。

FlashAttention各版本主要在并行划分、工作分配和硬件利用上演进，数值行为要按具体实现测试。除softmax外，矩阵乘的 TF32/低精度累加、dropout、归一化、量化和非确定性归约也会造成训推差异。

## 知识点

online softmax、浮点结合律、归约顺序、累加精度、确定性算法、逐层误差定位。


## 追问

- FlashAttention 1/2/3 在数值稳定性上有何改进？
- 如果必须固定分块，如何在推理阶段复现训练的数值行为？
- 除了 softmax，Attention 中还有哪些算子可能引入训推不一致？

## Note

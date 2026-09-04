---
difficulty: 中等
topic: 损失函数/分类回归与度量学习
tags: [面经, 待校对, 损失函数, Triplet Loss, Focal Loss, 对抗训练]
summary: 比较交叉熵、MSE、对比、Triplet、Focal 和对抗损失
company: 小红书
mastered: false
highfreq: false
---

## 题目

解释交叉熵、MSE、对比损失、Triplet Loss、Hard Triplet、Focal Loss 和对抗损失的定义、数学形式与典型场景，并说明 margin、难样本挖掘及 Focal 的 $\alpha,\gamma$ 怎样影响训练。

## 要点

- 损失必须匹配输出含义、噪声假设和最终任务，不能只按流行程度选择。
- 对比与 Triplet 都学习相对距离；Hard Triplet 是样本挖掘策略，不是完全不同的基本公式。
- Focal Loss 用 $(1-p_t)^\gamma$ 压低易例贡献，适合极度不平衡检测，也可能放大噪声难例。
- “对抗损失”不是单一公式，需说明 GAN 的判别器/生成器目标或其他具体对抗任务。
- margin、温度和 mining 决定梯度集中在哪些样本，必须结合假负例与稳定性调节。

## 答案

**按任务组织最容易记：分类预测概率，回归预测数值，度量学习安排距离，生成对抗匹配数据分布。**

下表中 $p$ 是预测概率，$y$ 是标签，$d$ 是嵌入距离；对比损失约定 $y=1$ 表示相似样本，$m$ 是间隔。

| 损失 | 常见形式 | 典型场景与边界 |
|---|---|---|
| 交叉熵 | $-\sum_c y_c\log p_c$ | 分类、token 预测；依赖标签质量和概率建模 |
| MSE | $\frac1N\sum_i(\hat y_i-y_i)^2$ | 回归、高斯噪声假设；大残差影响很强 |
| 对比损失 | $y d^2+(1-y)[m-d]_+^2$ | 成对相似/不相似；结果受 pair 采样支配 |
| Triplet | $[d(a,p)-d(a,n)+m]_+$ | 让正例比负例至少近 margin $m$ |
| Focal | $-\alpha_t(1-p_t)^\gamma\log p_t$ | 降低大量易例权重；可能过度关注错标样本 |

Hard Triplet 通常在 batch 或候选池中挑“最远正例、最近负例”再套 Triplet 公式。它提高有效梯度，却容易挑到假负例或异常值；可用 semi-hard、距离加权或课程式 mining 缓和。margin 太小约束弱，太大会让大量三元组长期有损失，需看类内/类间距离分布与验证指标。

Focal 的 $\gamma=0$ 退化为带 $\alpha$ 的交叉熵；增大 $\gamma$ 会更集中难例。$\alpha$ 处理类别权重，$\gamma$ 处理难易权重，两者应联合验证，并关注概率校准。

GAN 的“对抗损失”至少包含判别器区分真假的目标和生成器欺骗判别器的目标；还可使用非饱和、hinge 或 Wasserstein 形式，因此不能只写一个公式代表所有对抗训练。选型时还要看评价指标、样本构造、数值稳定性和多任务权重。

## 知识点

交叉熵、MSE、对比损失、Triplet Loss、难样本挖掘、Focal Loss、GAN 对抗目标。

- 面经原题：[B006-G01-Q217](../../docs/references/面经原题.md#b006-g01-q217)。
- 老师答案参考：[P009-Q217](../../docs/references/平台题/P009-LC-161-241.md#p009-q217)。

## 追问

以下均为平台页面追问，不计入面经原题：

- Focal Loss 的 $\alpha$ 与 $\gamma$ 分别控制什么，怎样联合调节？
- Triplet margin 太大或太小会怎样改变有效三元组比例？
- 对比损失与 Triplet Loss 的监督单位和负样本依赖有什么不同？

## Note

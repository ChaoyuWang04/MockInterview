---
difficulty: 中等
topic: KL散度/Reference Model约束
summary: Reference Model 为何固定,KL 怎样约束 PPO 与 GRPO
tags: [RL, RLHF, PPO, GRPO, Reference Model, KL散度, 待校对]
company: 字节、抖音、百度
mastered: false
highfreq: false
---

## 题目

PPO 或 GRPO 用于大模型对齐时,为什么需要固定的 Reference Model 和 KL 约束?请解释 KL 的计算与放置方式,区分 Reference Model 和 rollout 旧策略,并说明系数调节、工程开销及替代方案。

## 要点

- Reference Model 是累计行为漂移的固定锚点,rollout 旧策略服务于单批更新
- 写清当前策略到参考策略的 KL 方向、采样与 token/序列聚合口径
- 区分 PPO clip、PPO/TRPO 的近端控制和 RLHF reference KL
- 比较 PPO 折入奖励与 GRPO 独立 KL 项,不把 KL 当作 Critic 替代品
- 说明系数过大/过小、无 reference 的风险、显存优化和其他约束

## 答案

**Reference Model 是一把固定标尺,用来限制策略相对初始对齐模型的累计漂移。** 奖励模型有漏洞时,策略若不受约束会跑到参考模型几乎不会生成的区域,出现奖励过优化、语言退化、模式收缩或能力遗忘。KL 能降低风险,但不能判断内容是否正确或安全。

常见目标惩罚当前策略到固定参考策略的反向 KL：

$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\rm ref})
=\mathbb E_{a\sim\pi_\theta}\left[\log\frac{\pi_\theta(a\mid s)}{\pi_{\rm ref}(a\mid s)}\right].
$$

在自回归模型中可沿实际采出的 token 累加 log-prob 差。单个 token 的差可以为负,只有匹配采样分布后的期望才是 KL；汇报数值必须说明方向、按 token 还是序列、长度归一化和评估 prompt。

### 两个“旧模型”不能混用

| 对象 | 是否固定 | 用途 |
|---|---|---|
| Reference Model | 通常固定为 SFT/阶段起点 | 约束长期行为漂移 |
| Rollout old policy | 每批采样后冻结,下一批更新 | 计算 PPO/GRPO 新旧概率比与 clip |

PPO clip 只让本批样本上朝有利方向越界后不再继续获益,并不把全部概率比或 KL 硬锁在区间。RLHF 常把 $-\beta\log(\pi/\pi_{\rm ref})$ 折进逐 token reward,再经 GAE 影响优势。原始结果奖励版 GRPO（**Group Relative Policy Optimization**）保留 PPO 式新旧比率裁剪,常把非负的 KL 估计作为独立惩罚项；这是一种放置差异,不是“KL 代替了 Critic”。

TRPO 用二阶近似求解带平均 KL 约束的更新；PPO-penalty 用软 KL,PPO-clip 用一阶裁剪代理目标。它们主要控制相邻策略的单次步长,不能自动替代固定 reference 对累计漂移的约束。DPO 的 reference log-ratio 则来自 KL 正则奖励目标的闭式推导,标准损失不等于逐样本显式加一个 KL 项。

### 系数与工程

$\beta$ 太小,策略可能快速偏离并利用奖励漏洞；太大,策略几乎停在 reference,任务奖励学不动。可设目标 KL,按同一口径的实测偏移高于目标时增大 $\beta$、低于目标时减小,同时看真实任务、人评、熵和奖励分量。不存在跨模型通用的 KL 区间。

Reference 通常不参与梯度更新。`no_grad` 省激活,不省模型权重；可用低精度、量化、卸载或并行减少驻留成本。LoRA 训练时可共享冻结基座,只区分适配器。固定离线数据可预计算 reference log-prob,但 PPO/GRPO 的新 rollout 不能预先穷举。

不用 Reference Model 时可比较更强的在线早停、较小学习率、旧策略信赖域、SFT 混合目标、规则约束或 reference-free 偏好方法。它们改变了约束对象或训练目标,并不与 reference KL 等价,必须重新验证分布漂移和任务质量。KL 计算方便且能利用模型 log-prob、还可按自回归链分解；JS 或 Wasserstein 并非不能用,但会改变理论目标和估计成本。

## 知识点

Reference Model、反向 KL、PPO clip、old policy、TRPO、PPO-penalty、GRPO。

- 来源:[老师平台](https://course.terminiai.com/interview),P003-Q063、Q064、Q088、Q089。
- 依据:[PPO](https://arxiv.org/abs/1707.06347)、[TRPO](https://arxiv.org/abs/1502.05477)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[DPO](https://arxiv.org/abs/2305.18290)。

## 追问

- KL 为什么常用当前策略到 Reference Model 的方向?为什么不选 JS 或 Wasserstein?
- Reference Model 是否更新?怎样用共享基座、预计算、量化或卸载降低开销?
- 不使用 Reference Model 会有什么问题,有哪些替代约束?
- PPO 与 GRPO 把 KL 放在哪里,这种差异带来什么影响?
- KL 系数过大或过小会怎样?如何按目标 KL 动态调节?
- PPO 中的 reference KL 与 DPO 的 reference log-ratio 有什么区别?
- TRPO、PPO-penalty、PPO-clip 和其他稳定化手段分别控制什么?
- 实际训练的 KL 应控制在什么范围?为什么不能直接照搬固定阈值?

## Note

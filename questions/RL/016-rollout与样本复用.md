---
difficulty: 中等
topic: PPO/Rollout与样本复用
summary: 区分采集轨迹、样本使用次数与PPO多轮更新
tags: [待校对, PPO, Rollout]
company: 蚂蚁
mastered: false
highfreq: false
---

## 题目

强化学习中的“真实采样数量”是否一定等于 rollout 数量？请先明确环境交互、轨迹、transition、token、优化器看到的样本次数等口径，再解释 PPO 多个 epoch 复用同一批 rollout 对稳定性和样本效率的影响。

## 要点

- 计数前必须先定义样本单位
- 多个 epoch 增加样本使用次数，不增加环境交互
- mini-batch 数决定 optimizer step，不是每条样本的复用次数
- 区分新旧策略约束与固定 reference 的 KL
- 数据过旧时优先减少复用并刷新 rollout

## 答案

**两者不一定相等，因为“真实采样”没有统一计数单位。** 一次 rollout 可以是一条完整轨迹，也可以被拆成多个 transition；在语言模型里，一个回答通常算一条 trajectory，每个生成 token 又可算一步 action。并行环境只改变同时采集多少条，截断长度会改变一条轨迹含多少步，都不会把这些口径自动变成同一个数。

PPO 先用旧策略采一批数据，再把这批数据打乱成 mini-batch，训练多个 epoch。设采到 $N$ 条样本，训练 $E$ 个 epoch，每个 epoch 分成 $K$ 个 mini-batch，则：

- 新的环境样本或轨迹仍是 $N$；
- optimizer step 数是 $E\times K$；
- 总样本呈现次数是 $N\times E$；
- 每条样本通常被使用 $E$ 次，而不是 $E\times K$ 次。

例如采集 16,384 条数据，训练 4 个 epoch、每个 epoch 分 4 个 mini-batch，会得到 16 个 optimizer step，但每条数据只出现 4 次，总呈现量是 65,536。多 epoch 提高了昂贵交互数据的利用率，却会让后续更新看到越来越旧的数据。PPO 用新旧策略概率比和裁剪限制过大的代理收益，但裁剪不是硬 KL 约束；复用过多仍会过拟合本批奖励噪声，增加比例尖峰、clip fraction 和近似 KL。

RLHF 里还要分开两种约束：概率比比较当前更新与**生成本批数据的旧策略**，处理样本复用；reference KL 比较当前策略与固定 SFT/reference，限制长期行为漂移。reference KL 不能让陈旧 rollout 重新变成 on-policy，减少 epoch、按近似 KL 提前停止和及时重采才直接控制数据陈旧。

如果“真实采样数量远大于 rollout 数量”指训练侧的样本使用次数远大于新采数据，先核对是不是把 optimizer step 或 token 当成了新轨迹。确认复用过度后，优先减少 epoch 或每轮更新步数并刷新 rollout；再结合更小学习率、较保守裁剪、梯度裁剪和 KL 早停控制单步变化。学习率与 clip 没有脱离奖励尺度、batch 和模型的固定答案，应监控 reward、近似 KL、ratio、clip fraction、熵及独立验证集，按同一口径比较。

## 知识点

PPO、rollout、环境交互、样本复用、mini-batch、策略陈旧、reference KL。

- 来源：[老师平台](https://course.terminiai.com/interview)，P003-Q062。
- 一手依据：[PPO](https://arxiv.org/abs/1707.06347)。

## 追问

- PPO 中为什么通常设置多个 epoch 重复利用同一批 rollout 数据，这不会导致过拟合吗？
- 在 RLHF 的 PPO 阶段，如何处理 KL 散度约束与样本复用之间的关系？
- 如果真实采样数量远大于 rollout 数量，如何设计学习率或 clip 参数来保持训练稳定？

## Note

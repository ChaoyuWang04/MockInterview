---
difficulty: 中等
topic: PPO/On-policy与Off-policy
summary: 区分 On-policy、Off-policy、离线数据与有限复用
tags: [真题, RL, PPO, GRPO, Off-policy, 待校对]
company: 美团、华为、字节
mastered: false
highfreq: false
---

## 题目

请比较 On-policy 与 Off-policy 强化学习在数据来源、策略更新、样本复用和稳定性上的区别，并说明 PPO、GRPO、DQN、DDPG、TD3、SAC 与 DPO 应如何归类。为什么 PPO/GRPO 可以有限复用旧策略 rollout，LLM 对齐又为何很少直接使用 DQN 或 SAC？

## 要点

- 用目标策略与行为策略是否一致定义 On-policy 和 Off-policy
- 区分 on/off-policy 与 online/offline 两组概念
- 解释 PPO/GRPO 的旧策略快照、概率比、有限 epoch 和策略陈旧度
- 正确定位 DPO 为离线直接偏好优化，而非传统 Off-policy RL
- 比较典型算法及其在大模型长序列离散动作中的适用边界

## 答案

**On-policy 用当前或近邻目标策略产生的数据更新自己；Off-policy 可以用其他行为策略产生的数据更新目标策略。** 核心不是数据存在哪里，而是训练分布和目标策略是否匹配，以及算法怎样处理不匹配。

| 类型 | 典型方法 | 数据复用与代价 |
|---|---|---|
| On-policy | REINFORCE、A2C、PPO、标准 GRPO | 数据新鲜，偏差较小；策略一变，旧数据很快失效 |
| Off-policy | Q-learning、DQN、DDPG、TD3、SAC | 可长期用 replay，但要处理覆盖与分布偏移 |

Online/offline 是另一条轴：online 表示训练时还能收集新交互，offline 表示只能使用固定数据集。DPO 在固定偏好对上直接优化策略，通常称为 **offline preference optimization**；它没有 Q-learning、行为策略校正和环境交互循环，直接叫“Off-policy RL”会掩盖这些差异。

### PPO 与 GRPO 为什么仍是 On-policy

PPO 先冻结 $\pi_{old}$ 生成一批 rollout，再用 $\pi_\theta/\pi_{old}$ 的概率比做若干 minibatch epoch。只要策略滞后有限、更新次数受控、比率计算正确，并在下一轮刷新 rollout，它仍是近似 On-policy；它不会长期混用任意陈旧 replay。相比 TRPO 显式近似求解 KL 约束，PPO 用裁剪代理目标和一阶优化，工程更简单；相比裸策略梯度，它又限制了单批次的过大有利更新。GRPO（Group Relative Policy Optimization）也是这样，只是用同题多回答的相对奖励替代独立 critic。去掉 critic 不改变数据分布属性。

如果 GRPO 要长期复用离线轨迹，就必须面对支持集不足和重要性权重高方差，通常需要截断校正、保守目标或直接改用专门的离线方法，不能只把 epoch 调大。DeepSeekMath/R1 路线报告使用同题多回答、规则或模型奖励和组内相对优势；能力提升来自数据、奖励、训练阶段和算法共同作用，不能全部归因于 GRPO。论文中的组内采样来自冻结的旧策略快照；具体分布式实现还要用批量生成、并行打分并监控 actor 与 rollout worker 的版本差，不能仅凭算法名断言“严格 On-policy”。

### 典型 Off-policy 方法

DQN 用 replay buffer 打散相关性并复用样本，用 target network 减慢 bootstrap 目标的变化；两者可有替代设计，但直接去掉通常更不稳。DDPG 用确定性连续动作策略并额外加探索噪声；TD3 用双 critic、延迟 actor 更新等减少过估计；SAC 学随机策略并最大化奖励加熵。SAC 可通过 $J(\alpha)=\mathbb E[-\alpha(\log\pi(a\mid s)+\mathcal H_{target})]$ 调节温度，使实际熵靠近目标熵；$\alpha$ 大则更重探索，$\alpha$ 小则更重奖励。

LLM 每步动作虽是有限词表，但状态是不断增长的文本前缀，回报常在长序列末尾出现。DQN 要在大量前缀上学习稳定的 bootstrap Q 值，误差会逐步传播；DDPG/SAC主要面向连续动作，也不能直接套到 token 采样。可研究动作候选缩减、分布式 Q 表示、序列级价值或保守离线价值学习，但目前并非通用替代。PPO 能直接利用预训练语言模型的随机策略分布和序列奖励，因此历史上更易落地，但样本效率仍低。DPO 则用监督式成对损失训练静态偏好数据，省去 rollout、critic 和显式奖励模型；它较省显存且梯度流程简单，但会受偏好噪声和覆盖不足影响。

数据极贵时，可先判断能否学到足够准确的环境模型。模型可预测时，Model-based 方法可省真实交互，但要承担模型偏差和规划成本；否则更适合用 replay、离线数据、示范预训练或限制策略滞后的校正方法提高 Model-free 样本效率。不存在只按 On/Off-policy 标签就能选出的“最高效算法”。

## 知识点

行为策略、目标策略、replay buffer、策略陈旧度、online/offline、GRPO。

- 依据:[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300)、[DPO](https://arxiv.org/abs/2305.18290)、[SAC](https://arxiv.org/abs/1801.01290)。

## 追问

- PPO 为什么比 TRPO 更常用于 LLM，它和 Off-policy 方法各有什么代价？
- PPO 使用旧策略 rollout 和多个 epoch，为什么仍称为 On-policy？
- GRPO 去掉 critic 后为何仍是 On-policy，怎样控制策略滞后？
- 若 GRPO 长期复用离线数据，目标需要怎样处理分布偏移？
- DPO 是 On-policy、Off-policy，还是离线直接偏好优化？
- DQN 的 replay buffer 和 target network 分别解决什么问题？
- DDPG、TD3 与 SAC 怎样探索，SAC 温度 $\alpha$ 怎样调节？
- LLM 对齐为什么很少直接使用 DQN 或 SAC？
- 数据收集成本很高时，怎样在 Model-based 与 Model-free 方法间选择？
- 怎样提高 PPO/GRPO 的样本效率而不让 rollout 过度陈旧？

## Note

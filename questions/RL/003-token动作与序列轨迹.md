---
difficulty: 简单
topic: PPO/动作与轨迹
summary: 区分语言模型 RL 中的 token 动作、状态与完整轨迹
tags: [RL, PPO, RLHF, 序列生成, 待校对]
company: 蔚来
mastered: false
highfreq: false
---

## 题目

在语言模型的 PPO/RLHF 中，state、action、trajectory 分别指什么？token 级信号和完整回答的序列级奖励怎样结合，多轮对话又该如何定义这些概念？

## 要点

- token MDP 中状态是上下文前缀，动作是下一个 token
- 完整回答通常称 trajectory、rollout 或 episode，不是另一种 token 动作
- 说明终局奖励、token 级 KL 与优势估计怎样配合
- 解释稀疏序列奖励带来的信用分配问题
- 给出上下文 bandit 和多轮环境的其他合法建模口径

## 答案

**最常见的 token MDP 建模是：状态等于当前可见上下文，动作等于选择下一个 token，完整回答是一条 trajectory 或 rollout。** 第 $t$ 步可写成 $s_t=(x,y_{<t})$、$a_t=y_t$；策略 $\pi_\theta(a_t\mid s_t)$ 就是语言模型的下一个 token 分布。生成 EOS 后得到完整 episode。

“整段回答也是 action”只在上下文 bandit 的粗粒度建模下成立：环境一次接收整段文本并返回一个结果。答题时要先声明建模口径，不能同时把序列说成 token MDP 的第二种动作粒度。

### 奖励怎样落到 token

RLHF 奖励模型常对完整回答给一个终局分数，而参考策略 KL 可以逐 token 计算。可把终局分数放在 EOS 位置，把 token 级 KL 作为每步塑形奖励，再用 return、GAE 或其他优势估计把后面的信号分配给前面的动作。若只有序列末尾奖励，早期 token 与结果相隔很远，优势方差会变大，critic 估计也更难；过程奖励能提供更密的信号，但前提是中间评分可靠。

PPO 的 critic 通常输入当前前缀并输出标量 $V(s_t)$，估计从该前缀继续生成的期望回报。它不是奖励模型：奖励模型判断完整回答符合偏好的程度，critic 预测当前策略从某个状态出发还能拿多少回报。

### 多轮对话

若模型每轮只生成一条回复，可以把“到当前轮的完整对话”作为 state，把本轮回复看作一段由多个 token action 组成的子轨迹；工具结果、用户回复和环境状态转移进入下一轮 state。奖励可按 token、单轮或整段会话给出。关键是 state 必须包含做当前决策所需的信息，并明确哪些文本由策略生成、哪些事件由环境返回。

clip 仍比较本轮旧策略与新策略的 token 概率比；对固定 SFT reference 的 KL 则限制长期行为漂移，两者作用不同。

## 知识点

token MDP、state、action、trajectory、终局奖励、信用分配。

- 来源:[老师平台](https://course.terminiai.com/interview),P003-Q018（简单；蔚来）。
- 依据:[PPO](https://arxiv.org/abs/1707.06347)、[InstructGPT](https://arxiv.org/abs/2203.02155)。

## 追问

- token-level reward 和 sequence-level reward 怎样结合？
- 为什么 RLHF 常只在序列末尾给奖励，会带来什么问题？
- 多轮对话中的 state、action 和 episode 应怎样定义？
- PPO 的 critic 在文本生成中输入什么、输出什么？

## Note

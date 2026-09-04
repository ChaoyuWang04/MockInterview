---
difficulty: 中等
topic: RLHF与RM/Reward Model与Critic
summary: Reward Model 与 Critic 各自预测什么,为何 PPO 两者都需要
tags: [RL, RLHF, PPO, Reward Model, Critic, 待校对]
company: 华为、蔚来、蚂蚁、蚂蚁金服
mastered: false
highfreq: false
---

## 题目

在 PPO 式 RLHF 中,已有 Reward Model 为回答打分,为什么还要训练 Critic?请比较二者的目标、输入输出和训练方式,并说明它们怎样与 Actor 协同完成策略更新。

## 要点

- Reward Model 学人类偏好,通常给完整回答一个外部评分
- Critic 估计每个 token 前缀的未来回报,作为优势估计的基线
- 讲清终局 RM 分、逐 token KL、TD/GAE、PPO 裁剪的信号链
- 共享参数、初始化与更新频率都是工程选择,RM 不能直接替代 Critic
- 能分析 Critic 失准、RM 偏差,以及 REINFORCE、DPO、RLAIF 的差别

## 答案

**RM 决定“最终回答有多好”,Critic 估计“走到这个前缀后还能拿多少回报”。** 两者都输出标量,但学习目标不同,不能因为数值长得像就互相替代。

| 角色 | 学什么 | 常见输入与输出 | 何时更新 |
|---|---|---|---|
| Reward Model(RM) | 人类或 AI 的相对偏好 | prompt+完整回答 → 偏好分 | 先用偏好对训练,策略优化阶段常冻结 |
| Critic/Value Model | 当前策略下的期望未来回报 $V(s_t)$ | prompt+已生成前缀 → 每个状态一个值 | 跟随最新 rollout 做价值回归 |
| Actor | 哪个 token 应更可能被采样 | 前缀 → token 分布 | 用优势加权的 PPO 目标更新 |

### 三者怎样协作

策略先生成回答。RM 常在结尾给序列级分数,训练奖励还可包含逐 token 的 reference KL 惩罚。Critic 用这些奖励构造回报目标,例如

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t),\qquad
\hat A_t=\sum_{l\ge0}(\gamma\lambda)^l\delta_{t+l}.
$$

$\hat A_t$ 表示该 token 相对 Critic 基线“比预期好多少”,Actor 再把它放进 PPO 裁剪目标。Critic 自己用 $\big(V(s_t)-\hat R_t\big)^2$ 或 Huber loss 拟合回报。Critic 的梯度训练价值网络,Actor 的梯度来自优势加权的 log-prob；不是把 RM 或 Critic 反向穿过离散采样直接传给 Actor。

直接把终局 RM 分乘 $\nabla\log\pi$ 也能做 REINFORCE,并非数学上不能训练；问题是同一个终局分要分给整条序列,方差通常很大。Critic 给不同前缀提供随状态变化的基线,GAE 再在偏差与方差之间取舍,所以 PPO 更新通常更可控。

### 共享、初始化和失准

Actor 与 Critic 可以共享 Transformer 主干、各接一个头,也可以完全独立。共享能省参数和前向,但策略目标与价值回归可能争抢同一表征；独立更易隔离梯度和扩缩容,代价是显存与计算更高。用 SFT 或 RM 主干初始化 Critic 可能迁移语言理解能力,但 value head 仍须用当前策略回报重训；RM 只评偏好,不能替代 $V(s_t)$。

RM 与 Critic 技术上也能共用底座或由同一 checkpoint 初始化,但常分开训练和部署：RM 希望作为相对稳定的外部评委,而 Critic 必须跟随当前策略回报更新。若在 PPO 阶段联合共享可训练参数,价值回归可能改变 RM 的评分边界,RM 梯度也可能干扰价值估计；要共享就应分头、隔离梯度并分别验证,不能把“共享”理解成一个分数同时承担两个目标。

Critic 失准会让优势符号或尺度错误,表现为 value loss 上升、explained variance 下降、回报方差变大、KL/clip fraction 异常或真实性能退步。先核对 terminal 与 padding mask、奖励尺度和 bootstrap,再调 value 学习率、更新轮数、GAE 的 $\lambda$、value clipping 或损失权重；Actor/Critic 更新比没有通用答案,要以留出回报和生成质量验证。

RM 本身会受标注偏差、分布外失效和 reward hacking 影响。DPO 直接用离线偏好对训练策略,所以不需要 Critic,但仍继承偏好数据问题；RLAIF 把部分人类标注换成 AI 反馈,也没有自动消除评委偏差。

## 知识点

Reward Model、Critic、状态价值、TD 残差、GAE、策略梯度基线、价值回归。

- 来源:[老师平台](https://course.terminiai.com/interview),P003-Q004、Q044、Q058、Q070、Q087、Q120。
- 依据:[InstructGPT](https://arxiv.org/abs/2203.02155)、[PPO](https://arxiv.org/abs/1707.06347)、[GAE](https://arxiv.org/abs/1506.02438)、[DPO](https://arxiv.org/abs/2305.18290)。

## 追问

- 为什么不能直接用 Reward Model 的分数替代 Critic?
- Actor 与 Critic 应共享参数还是独立部署?Critic 能否从 RM 初始化?
- Critic 估计不准会怎样,应该看哪些指标并如何缓解?
- Critic loss 怎样设计,它与 policy loss 的更新频率如何配合?
- 为什么早期 RLHF 常用 PPO 而不是 REINFORCE?DPO 又为何不需要 Critic?
- Reward Model 的偏差、分布外失效和 reward hacking 怎样处理?RLAIF 改变了什么?

## Note

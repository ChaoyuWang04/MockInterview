---
difficulty: 简单
topic: PPO/Reward与Advantage
summary: 区分即时奖励、累计回报、价值与优势函数
tags: [RL, PPO, Reward, Advantage, 待校对]
company: 字节
mastered: false
highfreq: false
---

## 题目

Reward、return、value 和 advantage 分别是什么？请说明它们在 PPO 中的计算和作用，为什么策略梯度通常使用优势而不是直接使用即时奖励。

## 要点

- 区分环境给出的即时奖励与折扣累计回报
- 写出 $V$、$Q$ 与 $A=Q-V$ 的定义关系
- 说明 TD 残差只是优势估计，不是优势定义
- 解释状态基线为何不改变期望梯度却能降低方差
- 说明 GAE 的 $\lambda$ 与直接使用奖励的问题

## 答案

**Reward 是环境在某一步给出的信号；return 是从当前时刻开始累积的未来奖励；advantage 则回答“这个动作比当前状态下的平均选择好多少”。**

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k},\quad
V^\pi(s)=\mathbb E[G_t\mid s_t=s]
$$

$$
Q^\pi(s,a)=\mathbb E[G_t\mid s_t=s,a_t=a],\quad
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

奖励不一定是单步密集信号，也可以只在终局给出、由规则塑形或来自奖励模型；return 才是累计量。

### 为什么使用优势

若所有动作都乘同一个很大的回报，策略梯度会很抖。减去只依赖状态的基线 $b(s)$ 不改变期望梯度，因为：

$$
\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)b(s)]=b(s)\nabla\sum_a\pi(a\mid s)=0
$$

取 $b(s)=V(s)$ 后，优势把“绝对得分”变成“相对这个状态的预期好多少”，通常能降低方差。critic 学习 $V(s)$，actor 用估计优势更新。

一步 TD 残差 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ 在特定条件下是优势的估计，不是 $A(s,a)$ 的定义。GAE 再把未来 TD 残差按 $(\gamma\lambda)^l$ 加权：$\lambda$ 小更依赖 critic、方差低而偏差可能大；$\lambda$ 大更接近长回报、偏差小而方差可能大。

在 PPO 中直接用即时奖励，会忽略未来影响和状态难度差异；终局奖励场景下，大多数 token 甚至拿到零，无法做有效信用分配。即使把同一个终局分数复制给所有 token，也会产生高方差。至少要用 return-to-go 和基线，常见做法是 critic + GAE，再配合 PPO 裁剪。

## 知识点

Reward、return、$V(s)$、$Q(s,a)$、advantage、TD 残差。

- 来源:[老师平台](https://course.terminiai.com/interview),P003-Q075（简单；字节）。
- 依据:[GAE](https://arxiv.org/abs/1506.02438)、[PPO](https://arxiv.org/abs/1707.06347)。

## 追问

- 状态基线为什么能降低策略梯度方差，又不改变期望梯度？
- GAE 中 $\lambda$ 怎样控制偏差和方差？
- PPO 若直接用 reward 而不用 return 或 advantage，会发生什么？

## Note

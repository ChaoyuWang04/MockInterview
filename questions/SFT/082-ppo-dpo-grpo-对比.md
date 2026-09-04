---
difficulty: 中等
topic: RLHF与RM/对齐算法比较
summary: 用目标函数和训练信号区分 PPO、DPO、GRPO,解释裁剪与组大小
tags: [SFT, PPO, DPO, GRPO, 偏好优化, 待校对]
company: 美团
mastered: false
highfreq: false
---

## 题目

请详细解释 DPO、PPO 和 GRPO 三种偏好优化算法的核心原理,并比较它们在目标函数、训练流程和实际应用中的主要区别。

## 要点

- DPO直接学习离线偏好对;PPO/GRPO在训练中采样并使用奖励
- PPO用critic估计优势;GRPO用组内相对奖励而非独立critic
- 区分对旧策略的裁剪和对固定参考策略的KL
- 组大小影响采样成本、覆盖与优势信号,模式崩溃需独立评测

## 答案

**三者都可改善模型输出,但训练信号和优势估计不同。** 这里比较标准离线 DPO、常见 actor-critic PPO 与结果奖励版 GRPO;PPO/GRPO 需要奖励信号,不一定需要学习出来的奖励模型。

| 方法 | 输入与更新目标 | 主要成本 |
|---|---|---|
| DPO | 离线偏好对,增大相对参考策略的优劣 logp 分差 | 策略训练,参考分数可预计算 |
| PPO | 当前策略采样,用奖励和 critic 估计优势,优化裁剪目标 | rollout、打分、价值模型训练 |
| GRPO | 同题采样一组回答,用组内奖励相对化形成优势 | 组内采样与打分,省独立 critic |

记 $h=\log(\pi_\theta(y_w|x)/\pi_{\rm ref}(y_w|x))-\log(\pi_\theta(y_l|x)/\pi_{\rm ref}(y_l|x))$,DPO 损失为 $-\log\sigma(\beta h)$。它省去显式 RM 和在线 RL 循环,但不会自动获得离线偏好未覆盖的新反馈。

### PPO 裁剪具体限制什么

记新旧策略的 token 概率比为 $\rho_t$,优势为 $\hat A_t$,PPO 最大化:

$$
\mathbb E_t[\min(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)]
$$

正优势时,概率比超过上界后不再奖励继续增大;负优势时,低于下界后不再奖励继续减小。它抑制朝有利方向过度更新,不是把参数或所有概率硬锁住。RLHF 还常加对固定 reference 的 KL 惩罚;这与 rollout 旧策略的比值裁剪作用不同,都不保证训练稳定。

### GRPO 组大小与方法选择

结果奖励版可取 $\hat A_i=(r_i-\bar r)/(\operatorname{std}(r)+\varepsilon_{\rm num})$,同一回答各 token 使用该优势,再配裁剪及 KL。组大有助于采到不同质量答案,但增加生成成本;固定总回答预算时,也会减少不同问题的覆盖。组小估计更噪,全对或全错组的相对优势为零,组大小1更无比较信号。应比较非零方差组比例、任务收益与每单位计算收益,没有通用8或16的最优值。

有可靠离线偏好可先用 DPO;有持续奖励和在线采样预算可比较 PPO/GRPO,数学代码的验证器可提供信号,通用质量仍需可靠评判。DPO 训练过久若出现输出套路化,先固定解码检查多样性与分桶质量,再早停、降低优化强度、补充多样偏好或刷新当前策略数据;详见 [DPO 训练诊断](012-dpo-训练诊断.md)。

## 知识点

PPO、DPO、GRPO、偏好优化。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q214。
- 依据:[DPO](https://arxiv.org/abs/2305.18290)、[PPO](https://arxiv.org/abs/1707.06347)、[DeepSeekMath §4](https://arxiv.org/pdf/2402.03300)。

## 追问

- DPO 为什么可能模式崩溃,怎样缓解?
- PPO 的 clipping 怎样工作,为什么需要 clip ratio?
- GRPO 的 group size 如何影响训练效果?

## Note

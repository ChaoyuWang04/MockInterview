---
difficulty: 中等
topic: DPO/偏好优化方法比较
summary: RLHF、DPO、ORPO 如何利用反馈,ORPO 的赔率比与数据边界是什么
tags: [SFT, DPO, ORPO, RLHF, 偏好优化, 待校对]
company: 淘天
mastered: false
highfreq: false
---

## 题目

在监督微调(SFT)之后,各种后训练方法(如 RLHF、DPO、ORPO 等)在目标和优化方向上有何本质区别?请说明它们各自的设计目的与适用场景。

## 要点

- RLHF是利用人类反馈的方法范畴,比较时明确采用RM加PPO路线
- DPO用参考概率比,ORPO联合优选回答似然与赔率比损失
- ORPO不需要参考模型,不预设其计算必然比DPO多
- BT假设存在偏好异质性与非传递等边界;只含正例不足以训练标准成对目标

## 答案

**区别在于怎样把反馈变成训练信号,不是三种算法依次升级。** RLHF 是利用人类反馈优化策略的方法范畴;以下用典型的 RM+PPO 路线与直接偏好方法比较。SFT 本身属于后训练,ORPO 也不必放在独立 SFT 阶段之后。

| 方法 | 目标与设计目的 | 适用条件及风险 |
|---|---|---|
| RM+PPO | 拟合人类偏好奖励,再在线最大化奖励并限制策略偏移 | 能持续采样和可靠打分;需防RM过拟合、奖励投机和耦合不稳 |
| DPO | 用离线偏好对提高相对固定参考策略的优劣概率差 | 已有可靠偏好且希望简化训练;受离线覆盖及过优化限制 |
| ORPO | 将优选回答的SFT目标与偏好赔率比目标合并 | 希望联合学习示范和偏好并省去参考模型;要平衡两个目标 |

### ORPO 的目标与成本

令 $p_\theta(y|x)=\exp(\frac1{|y|}\sum_t\log\pi_\theta(y_t|x,y_{<t}))$,这是由平均 token logp 得到的分数,不能当作整段归一化概率。记 $o_\theta=p_\theta/(1-p_\theta)$,ORPO 使用:

$$
L=L_{\rm SFT}(y_w)
-\lambda\log\sigma\left(\log\frac{o_\theta(y_w|x)}{o_\theta(y_l|x)}\right)
$$

SFT 项提高好答案似然,赔率比项区分胜负。相对单独 SFT,需要处理劣选回答;相对 DPO,两者都处理优劣对,ORPO 无需参考前向,SFT项又可复用优选侧结果,所以不能断言一定更贵。DPO 若预缓存参考分数会缩小成本差异,应按相同 token、硬件和质量比较。

### 偏好假设与缺少负例

DPO 的 Bradley-Terry 模型用单一标量效用差解释成对偏好,难以完整表达群体分歧、上下文标准变化和循环偏好。先统一或显式区分评判标准,对分歧复标并保留验证切片,可比较可靠性加权或稳健变体;不能靠增大训练强度解决冲突。

只有高质量正例时可以做 SFT,但标准 DPO、ORPO 和成对RM缺少比较信号。可另采候选并可靠标注成对数据;不能随便把随机文本当负例。若有单条好/坏反馈可考虑 KTO;若另有可验证奖励,PPO/GRPO仍可训练,但反馈条件已改变。ORPO 不保证极少数据就能对齐,任何方法都须评测目标收益和原能力回退。

## 知识点

DPO、ORPO、RLHF、偏好优化。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q240。
- 依据:[ORPO §4](https://arxiv.org/html/2403.07691v2)、[DPO](https://arxiv.org/abs/2305.18290)、[InstructGPT](https://arxiv.org/abs/2203.02155)、[KTO](https://arxiv.org/abs/2402.01306)。

## 追问

- DPO 的 BT 假设有哪些局限,怎样处理?
- ORPO 相比 DPO 的计算开销一定更高吗?
- 只有正例没有负例时,这些方法还能怎样使用?

## Note

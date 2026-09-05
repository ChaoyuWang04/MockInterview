---
difficulty: 困难
topic: KL散度/On-PolicyDistillation
summary: OPD为什么用学生轨迹以及正向反向KL怎样选择
tags: [真题, 待校对, OPD, 知识蒸馏, KL散度, 后训练]
company:
mastered: false
highfreq: false
---

## 题目

进行 On-Policy Distillation(OPD)的动机是什么?请从数学上解释正向 KL 与反向 KL 的方向、采样分布和优化倾向,并说明数据充分时能否跳过预训练直接做 OPD。

## 要点

- OPD 让学生在自己会访问的轨迹上接受教师逐 token 反馈,缓解训练与生成分布错位
- 前向 $D_{KL}(p_T\Vert p_S)$ 由教师概率加权,反向 $D_{KL}(p_S\Vert p_T)$ 由学生概率加权
- 反向 KL 常更惩罚学生落入教师低概率区,前向 KL 更在意覆盖教师质量
- OPD 需要有基本语言与任务能力的学生起点,不能用数据量替代预训练能力

## 答案

**OPD 的核心不是“用了在线数据”,而是学生先按当前策略生成,教师再在学生真正走到的前缀上提供密集 token 分布。** 普通离线蒸馏只看教师或固定数据的轨迹,学生推理时一旦走偏,后续状态在训练中可能从未出现;on-policy 采样正是为缩小这种 exposure bias。

记教师分布为 $p_T$,学生为 $p_S$:

$$
D_{KL}(p_T\Vert p_S)=\mathbb E_{y\sim p_T}\left[\log\frac{p_T(y)}{p_S(y)}\right],\qquad
D_{KL}(p_S\Vert p_T)=\mathbb E_{y\sim p_S}\left[\log\frac{p_S(y)}{p_T(y)}\right]
$$

前向 KL 由教师覆盖的概率质量加权,学生漏掉教师模式会被重罚;反向 KL 在学生自己的分布上加权,学生把概率放到教师低概率区域会被重罚,在学生容量有限时常表现出更强的选峰倾向。方向和“轨迹由谁生成”是两个维度:输入前缀可以来自学生,token 级目标仍可选择不同散度。

数据很多也不意味着可以从随机模型直接 OPD。随机学生生成的前缀远离有效语言分布,教师反馈虽然密集,却要同时教词法、知识、指令和任务,支持集差距巨大且生成成本高。通常先用预训练或一个有基本能力的较小模型建立起点,再用 SFT/蒸馏缩小差距,最后 OPD 专门纠正学生自己会犯的生成错误。能否减少某个阶段应以起始能力和消融验证决定,不是由原始数据条数决定。

## 知识点

On-Policy Distillation、exposure bias、教师/学生分布、正向 KL、反向 KL、知识蒸馏。

- 依据:[GKD/On-Policy Distillation](https://arxiv.org/abs/2306.13649)、[MiniLLM](https://arxiv.org/abs/2306.08543)。
- 训练流程与三方对比见 [OPD 与离线蒸馏和 SFT 对比](035-opd与蒸馏和sft对比.md);教师拿不到 logits 时的退路见 [OPD 的 logits 依赖](026-opd-无logits替代.md)。

## 追问

- on-policy 只表示学生采样吗,与使用正向或反向 KL 有什么关系?
- 教师只能返回文本、拿不到 logits 时,还能怎样做 OPD?(见 [026](026-opd-无logits替代.md))
- 反向 KL 为什么可能降低输出多样性,怎样监控?

## Note

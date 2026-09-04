---
difficulty: 中等
topic: RLHF与RM/自定义奖励函数工程接口
summary: 在训练框架中实现和接入可审计的自定义奖励函数
tags: [真题, 待校对, RLHF, GRPO, Reward Function, ms-swift]
company:
mastered: false
highfreq: false
---

## 题目

在 ms-swift 一类强化学习训练框架中,自定义 reward function 应怎样实现和接入?为什么接口要按批返回标量奖励,并保留样本字段、训练状态、异步调用和多奖励聚合这些能力?

## 要点

- 区分奖励函数的业务含义与框架适配层,先定义可验证契约
- 输入包含 completions、必要数据列和训练状态,输出与样本一一对应的有限标量
- 注册、配置、权重顺序和同步/异步执行都属于训练语义,不能只写一个打分公式
- 先用固定样本做单测,再监控各分量分布、超时、异常值和 reward hacking

## 答案

核心契约可以抽象为 `score(completions, context) -> rewards`:一批生成结果进入,返回等长的有限标量列表。批接口便于框架把奖励与 rollout 样本、组内优势和分布式 batch 对齐；额外数据列承载标准答案、工具结果或任务元数据；训练状态允许做有明确依据的课程或阈值调度,但不应让同一答案仅因进程或重试不同而随机变分。

在 ms-swift 中,自定义逻辑可实现为奖励类,通过外部 plugin 注册,再由 `reward_funcs` 选择；`completions`、保留下来的数据列与 `trainer_state` 会进入调用参数。若依赖 API、数据库或外部验证器,异步实现可以并发等待 I/O；若使用多个 reward function 或 reward model,必须核对返回顺序与 `reward_weights` 的对应关系。具体类名和参数可能随版本变化,面试中应先说清版本与接口契约,再落到当前官方实现。

实现时至少处理五类边界:

1. 校验返回长度、`NaN/Inf`、异常范围与缺失字段,不要静默错位。
2. 对解析失败、超时和外部服务错误制定可区分的降级策略,不能把系统失败当成模型得零分。
3. 分开记录每个奖励分量、总奖励、通过率和耗时,避免只看聚合均值。
4. 用固定 completions 覆盖正确、错误、格式异常、超长和投机样本,检查确定性与单调性。
5. 在小规模 rollout 中检查奖励上涨是否伴随真实任务、人评或验证器指标改善,防止利用规则漏洞。

为什么不把所有逻辑硬编码进 trainer?因为奖励变化频繁,还可能使用规则、程序验证器或外部模型。插件边界让训练循环只依赖统一输入输出,而评分逻辑可独立测试、版本化和审计；异步接口则把 I/O 等待与策略计算解耦。灵活性不等于可任意访问上下文,生产上还要固定数据版本、超时、重试和外部服务版本,保证实验可复现。

## 知识点

自定义 Reward Function、ms-swift plugin、批量对齐、异步奖励、多奖励聚合、可观测性、可复现性。

参考:[ms-swift 自定义奖励函数](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)、[ms-swift 自定义奖励模型](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_model.md)。

## 追问

- 奖励函数超时或验证器宕机时,为什么不应直接返回 0?
- 多奖励的量纲差异和权重顺序怎样验证?
- 生成式奖励模型放在 trainer 内部与独立部署各有什么代价?
- 怎样为 reward function 设计反投机测试集?

## Note

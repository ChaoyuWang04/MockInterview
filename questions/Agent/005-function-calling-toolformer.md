---
difficulty: 中等
topic: 工具调用/工具学习范式
summary: 比较 Function Calling 与 Toolformer 的机制和选型
tags: [面经, 待校对, Agent, Function Calling, Toolformer, 工具调用]
company: 高德
mastered: false
highfreq: false
---

## 题目

请比较 Function Calling 与 Toolformer 在大模型调用外部工具时的本质差异。从理念、训练与推理机制、系统集成、优缺点和适用场景说明如何选型。

## 要点

- Function Calling 是运行时显式声明工具，Toolformer 是从数据中学习调用模式
- 两者都只产生调用意图，外部系统仍负责验证和执行
- 比较新增工具、可控性、数据与训练成本、实时性和审计能力
- 能结合实时路况等场景给出选型与混合方案

## 答案

**两者的差别在于“工具契约何时交给模型”：Function Calling 在请求时给，Toolformer 在训练数据中学。它们都不会替应用直接执行 API。**

| 维度 | Function Calling | Toolformer |
| --- | --- | --- |
| 机制 | 请求携带名称、说明和 schema；模型返回结构化调用意图 | 先生成候选 API 调用，执行并按语言模型损失筛选，再用保留样本继续训练 |
| 推理 | 宿主校验参数、鉴权、执行，并把结果回填模型 | 模型生成调用标记，运行时拦截、执行并插回结果 |
| 变化成本 | 可通过更新注册表接入工具，但仍要回归评测 | 工具分布或调用语法变化时通常要更新数据并再训练或适配 |
| 取舍 | 易审计、路由和降级；工具描述会占上下文 | 调用模式可融入模型；数据构造、更新和安全治理更重 |

实时路况、订单等数据频繁变化且动作有副作用，应优先 Function Calling：服务端可做权限、幂等、超时和确认。稳定、高频、接口长期不变的工具可研究 Toolformer 式训练；混合方案可训练通用“何时用工具”能力，运行时仍用显式 schema 和执行器控制具体调用。

## 知识点

Function Calling、Toolformer、自监督工具调用数据、JSON Schema、外部执行器、幂等与权限。
- 论文：[Toolformer](https://arxiv.org/abs/2302.04761)。
- 本批真实面经：[B005-Q001](../../docs/references/面经原题.md#b005-g01-q001)。
- 本批老师参考：[P008-Q001](../../docs/references/平台题/P008-Agent-001-030.md#p008-q001)。

## 追问

以下均为平台页面参考追问，不作为面经原话：

- 高德导航场景下，实时路况 API 用哪种方案更合适？为什么？
- Toolformer 如果工具 API 升级或下线，模型如何适应？
- 能否设计一个混合架构结合两者优势？

## Note

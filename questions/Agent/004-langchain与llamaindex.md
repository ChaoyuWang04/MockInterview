---
difficulty: 简单
topic: Agent框架/开发框架选型
summary: 比较主流 Agent 框架的定位和选型依据
tags: [真题, 待校对, Agent, LangGraph, LlamaIndex, AutoGPT, AutoGen, CrewAI]
company: 淘天、字节、阿里云
mastered: false
highfreq: false
---

## 题目

请比较 LangChain/LangGraph、LlamaIndex、AutoGPT 等 Agent 开发框架的技术定位、核心能力、RAG 与工具集成方式、适用场景和局限，并说明项目中如何选型；同时简要说明 AutoGen、CrewAI 一类多 Agent 框架处在什么位置。

## 要点

- 区分有状态编排、数据与检索、自主任务循环和多 Agent 协作
- 说明框架能力有重叠，不能把“编排”和“检索”画成互斥边界
- 从持久化、调试、评测、部署和团队维护成本选型
- 对版本敏感的 API 与成熟度结论标明时间

## 答案

**截至 2026-09，框架适合按“最难维护的部分”选择，而不是按品牌列功能；具体 API 必须以项目锁定版本的官方文档为准。**

| 框架/生态 | 更强的起点 | 常见场景 | 主要风险 |
| --- | --- | --- | --- |
| LangChain/LangGraph | 工具封装、有状态图编排、检查点与人工介入 | 多步骤业务流程、需要恢复和审计的 Agent | 抽象层多，版本迁移与调试成本高 |
| LlamaIndex | 数据接入、索引、检索、重排与 RAG 评测 | 企业知识问答、复杂检索链 | Agent 能力也在扩展，不能只把它当向量库外壳 |
| AutoGPT | 目标驱动的自主任务循环；当前项目也提供更完整的平台与可复用组件 | 探索自主执行、快速验证长任务闭环 | 自主步骤会放大错误和成本，而且项目定位随版本变化明显 |
| AutoGen、CrewAI | 多参与者会话或角色任务协作 | 研究型多 Agent、角色边界清晰的原型 | 多角色不等于可靠分工，仍需终止条件、状态治理和生产护栏 |

知识密集客服可用 LlamaIndex 管数据与检索，用 LangGraph 控制检索、订单工具、转人工和失败恢复；目标开放、需要模型自己拆解许多步骤的探索性任务，才更接近 AutoGPT 代表的自主循环。若一个框架已覆盖核心需求，避免为“组合”增加两套状态。选型应做最小原型，比较任务成功率、P95 延迟、可恢复性、可观测性、部署依赖和团队熟悉度。

## 知识点

Agent 框架、有状态工作流、RAG 数据链、自主任务循环、多 Agent 运行时、检查点、可观测性。
- 官方资料（截至 2026-09）：[LangChain Agents](https://docs.langchain.com/oss/python/langchain/agents)、[LlamaIndex 文档](https://docs.llamaindex.ai/)、[AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)。

## 追问

相关真题追问：

- 如果从零设计电商客服 Agent，会选择哪个框架，如何改造？
- 这些框架在工具调用失败时的容错机制有何不同？
- 在项目中应选择 LangChain/LangGraph、LlamaIndex，还是组合使用？
- AutoGen、CrewAI 一类多 Agent 框架适合解决什么问题？
- LangChain Expression Language 与普通顺序调用有什么区别？
- LlamaIndex 的不同索引和检索结构应如何按数据特点选择？

## Note

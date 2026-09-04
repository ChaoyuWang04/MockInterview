---
difficulty: 简单
topic: Agent框架/选型
summary: LangChain与LlamaIndex当前侧重点、重叠能力和选型方法
tags: [面经, 待校对, Agent, LangChain, LlamaIndex, 框架选型]
company: 阿里云
mastered: false
highfreq: false
---

## 题目

LangChain与LlamaIndex在架构设计和核心功能上有什么差异？它们分别适合哪些场景，设计企业级RAG时又该怎样选择或组合？

## 要点

- LangChain当前强调由模型、工具、Prompt和中间件组成的可配置Agent框架
- LangChain Agent建立在LangGraph之上，可使用持久执行和人工介入等编排能力
- LlamaIndex围绕数据摄取、索引、检索、查询、响应合成和Agent工作流组织能力
- 两者能力重叠，不能简单断言谁的Agent或RAG一定更强
- 选型依据是主复杂度、现有生态、可观测性、评测和运维成本

## 答案

**两者都能做RAG和Agent，差别主要在默认抽象与使用重心，不是一条绝对分界线。**

LangChain官方当前把核心Agent描述为可配置的harness：把模型、工具、Prompt和中间件装进模型调用循环；其Agent构建在LangGraph上，适合需要工具调用、路由、状态、人工介入和复杂编排的应用。

LlamaIndex官方概念体系从数据连接与摄取开始，继续覆盖Document/Node、索引、Retriever、Query Engine、响应合成和Agent/Workflow。团队若主要难点是把私有数据可靠地解析、索引、检索并评测，通常更容易从这些数据抽象切入。

企业级RAG可以只选一个，也可以让LlamaIndex负责数据与检索，把检索能力暴露给LangChain Agent编排。选择前用同一批真实查询做小型验证，比较检索质量、流程可控性、追踪调试、部署依赖和团队熟悉度，并锁定版本。具体API和功能边界变化很快，不应用连接器数量或静态功能清单做结论。

## 知识点

Agent harness、工具调用、LangGraph、数据摄取、Document/Node、Retriever、Query Engine、Workflow、框架选型。

- 真实面经：[B003-G01-Q066](../../docs/references/面经原题.md#b003-g01-q066)
- 老师参考：[P006-Q066](../../docs/references/平台题/P006-RAG-034-066.md#p006-q066)
- 官方资料（核对于2026-09-04）：[LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)、[LlamaIndex High-Level Concepts](https://developers.llamaindex.ai/python/framework/getting_started/concepts/)

## 追问

- 参考追问：基于两者设计企业级RAG系统时，怎样取舍或组合？
- 参考追问：LlamaIndex的索引抽象怎样对应不同数据和查询场景？
- 参考追问：LangChain的LCEL解决什么问题，与传统Chain相比有什么差异？

## Note

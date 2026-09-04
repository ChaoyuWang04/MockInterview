---
difficulty: 中等
topic: Agent架构/Skill-Memory-Harness
summary: Agent的Skill、Memory与Harness分别解决什么问题
tags: [真题, 待校对, Agent, Skill, Memory, Harness]
company:
mastered: false
highfreq: false
---

## 题目

Agent 系统中的 Skill、Memory 和 Harness 分别是什么,各自解决什么问题?请说明三者怎样与模型、工具、上下文和评测闭环协作,以及常见的职责混淆。

## 要点

- Skill 是可复用的任务能力与执行规程,Memory 是跨步骤或会话保存并检索的状态
- Harness 是包围模型的运行时与控制环境,负责上下文、工具、权限、循环和反馈
- Skill 可以被 harness 按需加载,Memory 由 harness 读写但不能代替工具能力
- 可靠性来自可观测、可验证和受限执行,不只是更长 prompt

## 答案

**Skill 决定“这类任务应该怎么做”,Memory 保存“过去发生过什么”,Harness 负责“让模型在什么环境里持续做事并接受反馈”。** 它们处在不同层级。

| 组件 | 典型内容 | 主要边界 |
|---|---|---|
| Skill | 任务说明、步骤、领域规则、模板、脚本或工具用法 | 是可调用能力,不是某次运行的个人状态 |
| Memory | 用户偏好、事实、历史结果、阶段摘要及其时间/来源 | 要检索、更新、过期和隔离,不能默认全部塞进上下文 |
| Harness | agent loop、上下文装配、工具协议、沙箱、权限、重试、持久化、追踪和评测 | 是运行系统,不等于模型本身或某一个框架名 |

一次执行中,harness 根据任务和权限选择 Skill,把相关说明与检索到的 Memory 放入上下文;模型据此选择工具并行动;harness 验证工具参数、记录观察、控制预算与失败恢复,再把必要结果写回 Memory。Skill 可以引用工具,但不应偷偷扩大权限;Memory 可以提示“上次失败在哪里”,却不能替代真实环境查询。

常见混淆有三种:把一长段 prompt 叫 harness;把工具 API 叫 skill;把完整聊天历史都叫 memory。更可靠的设计会给每层独立生命周期和测试:Skill 测任务成功率,Memory 测召回正确性与新鲜度,Harness 测权限、恢复、可观测性和端到端完成率。

## 知识点

Agent Skill、长期与工作记忆、agent harness、工具调用、沙箱、评测闭环。

- 参考:[OpenAI Harness Engineering](https://openai.com/index/harness-engineering/)、[OpenAI Agents SDK 演进](https://openai.com/index/the-next-evolution-of-the-agents-sdk/)。

## 追问

- Skill 和 Tool 有什么区别,一个 Skill 能否编排多个 Tool?
- Memory 应在什么时候写入、召回和淘汰,怎样避免错误记忆长期污染?
- Harness 中哪些约束应该写成可执行检查而不只是提示词?

## Note

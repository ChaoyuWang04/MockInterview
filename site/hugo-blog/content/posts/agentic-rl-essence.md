---
title: "Agentic RL 的本质：从\"给模型接工具\"到\"给模型放进环境\""
date: "2026-04-25"
summary: "从信息闭环、状态性、不可逆动作和决策质量度量重新定义 agentic RL，区分它与 RLM、single-turn tool use 和 verifiable multi-turn tool use 的本质边界。"
tags: ["Agentic RL", "Reinforcement Learning", "Tool Use", "Agents"]
---

# Agentic RL 的本质：从"给模型接工具"到"给模型放进环境"

## 把多轮工具调用回归到第一性原理

---

## 问题定位：方向的名字和方向要解决的问题不是同一件事

multi-turn tool call 已经是 2025–2026 年最拥挤的研究区域之一，但"多轮"和"工具"两个词的组合本身并没有定义一个问题——它只描述了一种**形式**。Search-R1 在多轮检索上做 RL，ReTool 在代码工具上做 RL，ToolRL、ARTIST、ReCall 各自在不同工具链上刷榜，形式上都符合 multi-turn tool call 的描述，但这些工作解决的是同一个问题还是不同问题？如果是同一个，为什么 benchmark 互不兼容？如果是不同的，那 agentic RL 作为一个统一的研究方向，其真正的本质问题是？

"multi-turn tool call 到底解决什么"这个问题如果回答不了，后续所有方法论层面的选择——reward 怎么设计、credit 怎么 assign、benchmark 怎么造——都会悬空。

## 三个范式的信息闭环对比

讨论 agentic RL 的本质之前，先把它和上游的两个范式画在同一张对比表里。核心维度不是"用不用工具"或"几轮对话"，而是**信息闭环的形状**。

| 范式 | 代表工作 | 信息闭环 | 环境性质 | 从权重拿不到什么 |
|---|---|---|---|---|
| Pure CoT / RLM | DeepSeek-R1 | x → think → y | 无外部 | — |
| Single-turn tool | Toolformer | x → call → y | 只读、幂等 | 确定性事实 |
| Verifiable multi-turn | Search-R1 / ReTool | (think→call→info)×N → y | 无状态、可逆 | 外部知识 |
| Agentic RL | 尚未收敛 | 含规划、追问、不可逆 action 的 MDP | 有状态、部分可逆 | policy 本身 |

前三行之间的 delta 是递增的知识接入量，但第四行与前面的 delta **不是量变而是质变**。前三行共享一个隐含假设——环境是无状态、幂等、只读的外部数据库；agent 所要做的全部事情是"检索对的东西、组合到答案里"。

## 核心论点：RLM 解决推理，Tool-use 解决知识接入，Agentic RL 解决的是「在 stateful environment 里做 long-horizon sequential decision 且 action 有不可逆后果」的问题。

Agentic RL 与 verifiable multi-turn 的分界线，可以用一个具体例子锁死。任务"帮我订下周去东京、避开樱花季尾声、符合差旅预算的机票"在 Search-R1 框架下**不可解**，原因不是工具种类不够，而是四条结构性差异：

| 维度 | Search-R1 世界观 | Agentic RL 世界观 |
|---|---|---|
| 约束结构 | 平铺，可并列检索 | 有偏序：日期 > 预算 > 航司 |
| 信息来源 | 只读外部 API | 外部 + 需向人类追问 |
| 动作效应 | 幂等，不改变未来 | 改变未来可行动作空间 |
| 可逆性 | 全可逆 | 含不可逆动作（支付、发送） |

四条里任何一条被打破，Search-R1 的方法都不够用。把这四条翻译成经典 RL 的术语：**agentic RL 的本质是在高维、部分可观测、有约束偏序结构、存在不可逆动作的状态空间里做 sequential decision making**。verifiable multi-turn 只是这个大问题在"环境无状态、动作可逆、reward 可验证"三个假设同时成立时的退化特例。

这个重新定位的直接推论是：**multi-turn tool call 这个形式不是问题的定义，stateful decision 才是**。把工具数量从 1 个加到 10 个、把对话轮数从 3 轮加到 20 轮，都没有推进本质问题一步；改变环境的状态性、引入不可逆性、构造偏序约束，才是在推进。

## 串行 vs 并行：一个被忽视的结构学习问题

multi-turn tool call 的"轮"本身也没有被严肃分析过。定义每次 tool call 获得一个随机变量 Y_i，最终推断目标 T，两种调用模式的信息结构如下：

并行调用的信息量是边际独立贡献的累加：

$$I(T; Y_1, Y_2, Y_3) = \sum_i I(T; Y_i \mid Y_{<i})$$

但并行决策在调用前就被锁定，每个 Y_i 的分布**不依赖**其他 Y_j 的实际取值。

串行调用的本质是：第 k 步的 action a_k 本身是 Y_1,...,Y_{k-1} **取值的函数**。

判据可以精确写成一句话：**当 a_k 的最优选择依赖于 Y_<k 的实际取值而非仅仅其存在性时，必须串行**。"查北京天气 + 查上海天气"应当并行；"先查用户位置 → 再查当地天气"必须串行，因为第二次调用的**参数本身**是第一次调用的值。

| 维度 | 并行 | 串行 |
|---|---|---|
| 信息量上限 | 调用前就定 | 动态依赖前序结果 |
| 可否聚焦 | 不能 | 能 |
| 类比 | 一次性问卷 | 医生问诊 |
| 信息论本质 | 边际互信息和 | 条件互信息链 |

这里隐藏着一个 paper-worthy 的研究缺口：当前所有 GRPO-style 的 multi-turn RL 都定义在 token 序列上，loss 和 reward 都沿着一个线性时间轴传播。**token 生成的线性结构天然偏置 policy 学成纯串行**，要让模型输出并行 DAG，必须在动作空间或 loss 结构上显式建模拓扑。目前主流 benchmark（HotpotQA、Musique、ToolBench）全部在串行范式下评测，**并行 tool use 的 RL 训练几乎空白**——这不是因为问题不重要，而是因为 benchmark 不考，方法就没人做。

## 为什么这里必须是 RL 而不是 SFT

SFT 在工具使用上能走多远的问题，2024–2026 学术圈存在真实争论。一派认为 RL 只是把 pass@k 放大为 pass@1（Yue et al. 2025），一派认为 RL 能产生 SFT 无法达到的泛化。两派各有证据，区分它们的关键是**任务结构**。

在可验证 + reward 密集的窄域（math、code、EM-scored QA），证据倾向于第一派：TinyZero 在 Qwen2.5-3B + Countdown 上涌现的是推理**格式**（多方案枚举、自验证、自修正）和枚举习惯，不是算术能力本身；response length 先塌到 100 再回升到 400+ 的 U 型曲线，是模型在"找格式"而非"学算术"的结构特征。

但在部分可观测、动作空间大、reward 稀疏的 agentic 场景，SFT 的失败模式和 RL 的成功模式**定性不同**。SFT 学的是"给定 context 下一个 token 是什么"，训练集里的轨迹默认老师"知道该干嘛"，**模型学不到"不确定时该追问"这种元认知动作**，因为老师的 demonstration 里从不展示不确定状态。RL 能学到的是 SFT 永远学不到的东西：**面对分布外 state 如何通过 exploration 找到好动作**，这不是知识是 policy。

更准确的表述不是"RL 能不能泛化"，而是：**RL 是唯一能学到"OOD state 下的行为策略"的训练范式**。这个区分在单步任务上看不出来，因为没有探索空间；但在 multi-turn + stateful 任务上被放大，这正是 agentic RL 在训练层面与 tool-augmented SFT 的分界线。

## 当前 benchmark 测错了东西

回到"credit assignment 做对了又如何"这个根问题。如果在 HotpotQA / Musique / ToolBench / MATH 上通过 step-level reward 涨 3-5 个点，这个结果之所以不触发任何直觉上的"解决了问题"的感觉，根源是**这些 benchmark 本身没有测 agentic 能力**：

第一，reward 可验证的窄域 benchmark，其环境无状态、动作可逆、约束平铺，**不包含任何 agentic RL 的本质特征**。在这类 benchmark 上 credit assignment 的精细化只是优化一个错题。

第二，benchmark 的"正确性"度量是 final answer 对不对，**不度量决策质量**。一个 agent 追问了本该追问的、避开了不可逆错误、正确识别了约束偏序——这些 agentic 意义上的"做得好"在 EM 分数上不体现。一个只靠运气猜对 final answer 的 trajectory 和一个完美执行决策链的 trajectory 在当前 benchmark 下不可区分。

第三，benchmark 环境的工具拓扑固定单一，**不度量跨拓扑泛化**。一个在 Wikipedia search 上训好的 policy 迁移到订机票工具链上完全不工作，但这件事在现有 benchmark 体系下不被惩罚。

## 真正值得推进的三个方向

基于前述分析，agentic RL 里只有三个方向在直接推进本质问题：

| 方向 | 问题 | 当前状态 | 学术性 vs 工程性 |
|---|---|---|---|
| P1 环境建设 | 造有状态、偏序约束、不可逆动作的 benchmark | 空白 | 工程为主，学术价值 90% |
| P2 决策度量 | 定义"轨迹质量"而非 final answer 正确性 | 无共识 | 学术为主 |
| P3 拓扑泛化 | 训练在 A 工具集，zero-shot 泛化到 B 工具集 | 概念未收敛 | 学术为主 |

三者共享一个元判据：**是否增加了环境的复杂度、是否改变了度量的对象、是否检验了 policy 的迁移**。在 HotpotQA 上做再精细的 credit assignment 都不属于这三项中的任何一项，因此即使工程上成功，在 agentic RL 的本质意义上也不是前进。

相反，这三个方向任一切片都是真正的前进：构造一个带"向用户追问"和"不可逆支付"的 toy agent 环境，哪怕只有三个工具；定义一个同时度量"追问质量 + 偏序识别 + 并行利用率"的复合指标并在现有 baseline 上测出负面结果；用 TinyZero pipeline 在两个工具拓扑上训练-测试看泛化崩塌的定量规律。这些切片算力需求不大，学术价值远高于在大 benchmark 上刷点。

## 方法论层面的提醒

**形式相似性会误导方向选择**。multi-turn tool call 这个名字统一了 Search-R1、ReTool 和东京机票问题，但前两者和后者在解空间的几何结构上不是同一个问题。被名字绑架会导致在"用 Search-R1 的方法解东京机票"这个错误坐标系里打转。**从"信息闭环的形状"重新分类，比从"工具数量 + 轮数"分类更本质**。

**方法创新和问题定义是两个独立的 axis**。当前 agentic RL 的绝大部分工作是在"问题定义不变"的前提下优化方法（新的 reward shaping、新的 loss masking、新的 advantage estimator），这些工作即使成功也只是把一个定义错的问题解得更精。更稀缺的工作是**重新定义问题本身**——提出新的环境、新的度量、新的泛化协议。position paper 和 benchmark paper 在这个意义上被系统性低估，因为它们没有 SOTA 数字但推动的是整个领域的坐标系。

**"RL 是否泛化"这个争论的真正意义是选任务不是选方法**。如果任务本身是可验证窄域，RL 大概率只是在放大 SFT 的能力；如果任务是 stateful + OOD state + 稀疏 reward，RL 做的事 SFT 做不了。**争论的答案取决于任务选择，不取决于算法细节**，所以"我该用 SFT 还是 RL"这个问题在问法上就错了，正确的问法是"我的任务有没有 OOD state 需要被 policy 应对"。

## 留下的裂缝

**stateful agent benchmark 的缺失是方向的主要瓶颈**。当前所有被广泛使用的 agentic benchmark 要么是无状态检索（HotpotQA 家族）、要么是可逆沙盒（ToolBench、MINT），没有一个真正包含"向人类追问 + 不可逆支付 + 动态约束"的组合。WebArena、OSWorld 部分触及但评测依赖 final state 匹配，仍然回到"结果对不对"而非"决策好不好"。**构造一个小而锋利的 stateful benchmark 是当前方向里 ROI 最高的动作**，但绝大多数研究者回避这类工作因为它看起来不够"算法性"。

**决策质量指标尚无形式化定义**。"轨迹好"目前只能靠人类评审判断，无法自动化因此无法进入 RL 的 reward loop。是否可能定义一个不依赖 final answer 的 trajectory-level reward，覆盖"追问合理性、偏序识别、并行利用率、不可逆动作的风险控制"四个维度？这件事没有被严肃尝试过，是一个开放问题。

**跨工具拓扑的泛化协议未被定义**。"把 Wikipedia search 训练的 agent 迁移到订机票"这类泛化当前没有标准测试协议，什么算"不同拓扑"、什么算"泛化成功"、zero-shot 和 few-shot 的 baseline 是什么，全部悬空。这个问题的形式化可能本身就是一篇 position paper。

**credit assignment 的价值依赖于任务选择**。step-level reward 的研究在错任务上是屠龙术，在对任务上是刚需。**判断它是不是刚需的 litmus test 很简单：这个任务上 outcome-only reward 会不会让模型学到 reward hacking**。如果不会（如 Search-R1 的 EM），credit assignment 锦上添花；如果会（如有不可逆动作的环境，一次错误支付会让整条轨迹的 outcome reward 失真传播），credit assignment 是唯一解。当前 credit assignment 的论文几乎全在前者上做实验，这是方向内部的自我消解。

## 结论

agentic RL 真正推进的不是"多轮工具调用"这个形式，而是"在 stateful、部分可观测、含不可逆动作的环境里学习 sequential decision policy"这个本质问题。Search-R1 / ReTool / Toolformer 都是这个大问题在强假设下的退化特例，把它们的方法直接套到真实 agent 任务上，失败不是因为方法不好而是因为**假设不成立**。

这个重新定位的直接后果是对研究选题的重新排序。在现有可验证 benchmark 上做 reward shaping 和 credit assignment 的工作即使技术上正确，也不推进本质问题，因为 benchmark 本身没有测 agentic 能力。真正的 ROI 集中在三件事：造出有状态、偏序、不可逆的新环境；定义超越 final answer 的决策质量度量；验证跨工具拓扑的泛化协议。这三件事每一件都比"在 HotpotQA 上涨 4 个 EM"更难发表但更有长期价值，**短期投稿回报和长期研究品味在这里必须做一次显式的取舍**。

从更广的视角看，agentic RL 重复了 RLM 走过的同一种思维迁移——从"把更多东西塞进模型"转向"把模型放进更复杂的世界"。RLM 把 prompt 从 context window 搬到 REPL 环境里；agentic RL 要把 decision 从可验证的输出搬到有状态的环境里。**两者共享同一个设计哲学——让模型做它擅长的认知决策，让符号系统和外部环境承担各自擅长的部分**。眼睛看得更多是 context scaling，手伸得更远是 tool use，真正的 agent 是要走进去站着的那个身体。这条路不短，但它的第一个路标必须是把问题本身定义清楚。

# Agentic Reasoning：把推理当成与环境互动的控制回路

<!-- release-date: 2026-01-18 -->

**本文依据**：封面正式标题 **Agentic Reasoning for Large Language Models**，副标题 Foundations · Evolution · Collaboration，arXiv 2601.12538v1（2026-01-18），135 页。作者 Tianxin Wei†、Ting-Wei Li†、Zhining Liu† 等；第一单位 University of Illinois Urbana-Champaign，另有 Meta、Amazon、Google DeepMind、UCSD、Yale。首发日取本地 PDF 头已印的 v1 提交日 2026-01-18。配套清单 [weitianxin/Awesome-Agentic-Reasoning](https://github.com/weitianxin/Awesome-Agentic-Reasoning)。综述声明覆盖到 2025 年（PDF p.4）。标「外部补充」的段落不来自本文。

## 一句话

闭世界里把中间步骤写出来（思维链、分解、程序辅助）已经够用；开放、会变的环境里，模型必须计划、调用工具、记住经验、再改自己。这篇综述把这件事叫 **Agentic Reasoning（智能体式推理）**：推理不再是一次前向，而是贯穿感知、计划、决策、验证的组织原则。地图的主贡献不是「收了多少篇」，而是两把正交的刀：环境动态切成基础 / 自演化 / 集体三层；优化方式切成上下文编排与训练后内化。

## 一、这张地图切了几刀

### 和「LLM 推理综述」「Agent 架构综述」差在哪

作者把自己摆在交叉点上（PDF p.7–8）。LLM 推理综述（Huang & Chang、长思维链、强化推理、前沿推理）多半还在**静态推理**：提示、推理时缩放、训练后对齐，一次前向把痕迹写完。Agent 综述（强化学习搜、自演化终身系统）多半从**架构模块**讲：规划器、工具、记忆怎么接。本文把推理本身当成统一机制：交互、反馈、协作如何把静态推断变成可适应的行为。

表 1 把对照写死（PDF p.7）：被动对交互、单次对多步、上下文窗口对外存、离线预训练对持续改进、提示反应对显式目标与规划。一句话：**从缩放内部算力，改成缩放测试时交互**。

### 第一把刀：环境动态三层

定义框（PDF p.3）把三层写进同一句话：基础能力（规划、工具、搜索）、自演化适应（反馈与记忆驱动）、集体协调（多 Agent 协作）；实现路径可以是上下文编排，也可以是训练后优化。

```mermaid
flowchart TB
    subgraph env [环境动态]
        F[基础层<br/>稳定但复杂]
        S[自演化层<br/>反馈与记忆]
        C[集体层<br/>角色与共享目标]
    end
    subgraph opt [优化方式]
        I[上下文推理<br/>不改参数]
        P[训练后推理<br/>SFT 与 RL]
    end
    F --> S --> C
    I --- P
```

图是机制示意，对应摘要与 PDF p.3–4、p.10。三层不是并列菜单，而是环境变难时叠上去的能力：先能在稳定环境里拆目标、调工具、核结果；再能跨回合改记忆与策略；最后多人分工、辩论、共享记忆。

### 第二把刀：上下文 vs 训练后

贯穿每一层（PDF p.3–4、p.8–9）。

- **上下文推理（In-context Reasoning）**：参数冻结。用编排、搜索式规划、自适应工作流，在测试时把交互做长。ReAct 在思想 $z$ 与动作 $a$ 之间贪心交替；Tree-of-Thoughts 把部分思想当节点，用启发式 $v̂$ 搜路径（PDF p.9 式 (2)）。
- **训练后推理（Post-training Reasoning）**：把成功的推理或工具策略写进权重。代表是 DeepSeek-R1 一类推理模型，以及 Search-R1、DeepRetrieval 这类多轮工具 RL。PPO 仍常见，但推理任务大量改用 **GRPO**：用同题一组样本的相对奖励当优势，省掉价值网络（PDF p.9 式 (3)–(4)）。ARPO、DAPO 再处理稀疏奖励与工具环境的稳定性。

判据很清楚：要不要改 $\theta$。改不了就搜 $Z$；要内化就优化 $\theta$。

### 形式化：把「想」和「做」拆开

环境写成 POMDP，再加内部推理变量（PDF p.8）。策略分解为

$$
\pi_\theta(z_t,a_t\mid h_t)=\pi_{\mathrm{reason}}(z_t\mid h_t)\cdot\pi_{\mathrm{exec}}(a_t\mid h_t,z_t)
$$

先在 $Z$ 里算，再在 $A$ 里提交。历史 $h_t$ 可用记忆 $m_t$ 压缩。多 Agent 扩成 Dec-POMDP，观察里带通信信道 $C$：别人的外部动作可以当自己的提示（PDF p.9）。自演化则是跨回合元更新 $S_{k+1}\leftarrow U(S_k,\tau_k,F_k)$（PDF p.10 式 (5)），$S$ 可以是文字反思、工具库，或系统源码本身。

**轴本身就是贡献**。读者用这张图可以对号入座：某个系统是「稳定环境 + 提示编排」还是「会变的环境 + RL 改记忆策略」，而不是混在「都叫 Agent」里。

## 二、每一区里有什么

### 基础层：规划、工具、搜索

单 Agent 的工作流被写成循环：规划拆目标 → 工具改世界 → 搜索取证；推理决定何时做哪一步（PDF p.10）。

**规划。** 上下文一侧按风格切：工作流分期（感知–推理–执行–验证）、树搜索（BFS / DFS / A* / MCTS / beam）、过程形式化（PDDL、代码）、解耦分解（ReWOO 把观察与推理拆开）、外援（KG、世界模型、HuggingGPT）。表 2 再按语言 Agent 与视觉 / 具身 Agent 分栏（PDF p.12）。共识：层次树搜索能回溯、能验证再提交不可逆动作。分歧：长程会累积误差，所以要增量验证和记忆（PDF p.11）。训练后一侧把规划当成奖励设计与最优控制：Reflexion、Reflect-then-Plan、扩散轨迹优化、离线 RL（PDF p.13）。

**工具。** 三档（PDF p.14–17）。上下文：ReAct 交错推理与动作；ART 检索成功示范；文档写清楚就能零样本用新工具；GEAR 把选工具交给小模型。训练后：SFT 启动（Toolformer 自监督插 API、ToolLLM 上万真实 API、ToolAlpaca 用多 Agent 仿真造对话），但会过拟合示范、选工具发脆；RL 才谈掌握（SWE-RL、ReSearch、ReTool、ToolRL），作者认为比纯 SFT 更能迁到域外。编排：HuggingGPT 中心规划、TaskMatrix.AI 对海量 API、OctoTools 训练免费工具卡、Chain-of-Tools 在冻结模型上组合未见工具。共识：单模型单工具不够；分歧：编排该学进权重，还是训练免费地用工具卡与搜索。

**搜索。** 传统 RAG 一次检索再生成；Agentic Search 要在推理中决定何时、搜什么、怎么搜（PDF p.17–20 图 4）。上下文：ReAct、Self-Ask、IRCoT、Self-RAG 按需检索。结构增强：Agent-G、GeAR、ARG 对着知识图动态问。训练后：Toolformer / INTERS 做 SFT；WebGPT、RAG-RL、Search-R1、Deep-Researcher、ReSearch 做 RL，会出现中途发 `<Search>`、再分解、再核验。共识：检索必须是策略，不是管道前置步骤。

### 自演化层：反馈、记忆、把基础能力再长一截

反馈三条 continuum（PDF p.20–23 图 5、表 5）。

1. **反思式**：推理时改轨迹，不改参数。Reflexion、Self-Refine、Constitutional AI、ToT / GoT 比选、Least-to-Most 分解。
2. **参数适应**：把反馈写进权重。AgentTuning、ReST、蒸馏 CoT、偏好对齐。代价是训练，灵活性下降。
3. **验证器驱动**：单元测试、仿真、环境成功信号，失败就重采样。高效，但反馈**不诊断**：不知道错在哪一步，也不改推理习惯。

记忆从「加长窗口」改成推理回路的一部分（PDF p.24–27）。扁平：事实缓冲（MemGPT、MemoryBank）对经验痕迹（Workflow Memory、ACE 把上下文当演化 playbook、Reasoning Bank 复用失败痕迹、Evo-Memory 在流式任务上测自演化记忆）。结构化：GraphRAG / MEM0 / Zep 的图，MemTree 的树，多模态 Optimus-1、M3-Agent。训练后把读写当策略：MemAgent 用 DAPO 学覆盖；Memory-R1 拆 Memory Manager 与 Answer Agent；Memory-as-Action 把增删改写进策略。共识：记忆要可编辑。作者点名的缺口：多 Agent 记忆的训练后共演化几乎没人做（PDF p.41）。

基础能力的自演化（PDF p.27–29）：规划侧自造题与自奖励；工具侧合成新技能（Voyager 把代码技能永久扩进 $A$）；搜索侧动态检索与知识合成。结构演化的极端是 AlphaEvolve：把 Agent 源码当假设空间，LLM 当变异算子（PDF p.10）。

自演化按 $S$ 的形态还切成言语（Reflexion 写错误日志）、程序（Voyager 技能库）、结构（改架构 / 代码）（PDF p.10）。时间上再切 intra-test-time（回合内改）与 inter-test-time（跨任务巩固）（PDF p.38–39）。

### 集体层：角色、协作、共演化

角色税分通用（经理–工人–批评者）与领域专用（PDF p.30–31）。协作同样分上下文（AutoGen、CAMEL 固定角色扮演）与训练后（GPTSwarm 把拓扑当图、MAGRPO / MHGPO 把 GRPO 搬到对话轮、COPY 双 Agent 共训、MAPoRL 用验证器给讨论质量打分）（PDF p.9、p.36–38）。

多 Agent 记忆四维：架构（层次 vs 异构角色模板）、拓扑（中心验证 / 公私碎片 / 无控共享池）、内容（语义六类 MIRIX、任务积木 LEGOMem、认知阶段 MAPLE）、管理（压缩遗忘 vs 验证准入 vs 学模式）（PDF p.39–41 图 10）。MIRIX 在多模态问答上写了 **35%** 准确率增益（PDF p.41）——这是作者转述该工作的数字，不是本综述自己的实验。

训练集体演化：Multi-Agent Evolve 的提出者–求解者–评判者闭环；MARFT 指出经典 MARL 假设对不上 LLM 组织（角色异构、动态协调、长对话）；Stronger-MAS 把 GRPO 扩到角色 × 轮次分组；MALT / MARS 做流水线与快慢系统分工（PDF p.42）。

共识：辩论和角色分工常常强过单 Agent。分歧：拓扑该手设计还是学；群体信用分配几乎没被理解（第八节原话）。

### 应用与基准：地图怎么落地

应用按同一三层扫五块（PDF p.43–63）：数学探索与 vibe coding、科学发现、具身、医疗、网页与自主研究。数学基准（GSM8K、MATH、AIME）趋于饱和，FrontierMath 仍偏终答；作者把数学改写成探索域（AlphaEvolve、奥林匹克几何、程序搜索发现）。编码从一次生成改成多轮协作（Copilot、Cursor 被当作流行工作流，不是本文贡献）。

基准分机制（工具、搜索、记忆与规划、MAS）与应用（具身、科学、研究、医疗、Web、通用工具）（PDF p.64–72）。机制侧点名 AgentBench、MultiAgentBench；Web 侧 WebArena、Mind2Web、WebVoyager。作者的态度是：有的基准切单能力，有的测端到端，不要混用分数。

## 三、作者的判断

下面与「他们综述到的事实」分开。

1. **主矛盾是交互，不是再写长一点的思维链。** 闭世界技巧假定静态短程；开放环境要行动、适应、用反馈改进（PDF p.2）。
2. **两把刀要一起用。** 上下文灵活但受冻结模型与窗口限制；训练后能内化，但贵、且不如提示好改。设计 Agent 是在两者之间选约束，不是站队。
3. **SFT 启动工具，RL 才掌握工具。** 纯模仿会脆（PDF p.16）。验证器重试能抬正确率，但不解释失败（PDF p.22–23）。
4. **记忆必须从缓冲变成策略。** 扁平检索加 prompt 不够支撑目标分解与长程（PDF p.24–25）。多 Agent 记忆的训练后共演化被写成明确空白（PDF p.41）。
5. **集体智能的瓶颈是机制设计，不是再加一个角色提示。** 通信拓扑与激励要对齐分散的 $\pi_{\mathrm{reason}}$；CTDE 被当作稳定合作的候选（PDF p.9）。手设计拓扑仍是主流，可学协作策略刚起步（PDF p.73）。
6. **开放问题六条是议程，不是已解决问题**（PDF p.72–73）：用户为中心的个性化与非平稳目标；长程跨 token / 工具 / 技能 / 记忆的信用分配（Voyager 仍展示误差迅速复合）；世界模型的校准与因果贡献；可适应、可解释、部分可观察且可能对抗下的协作策略；潜空间推理的可审计；治理必须覆盖模型、Agent 策略、生态系统，因为失败会跨时间与组件出现，现有护栏偏短程。

作者没有声称自己跑了统一排行榜。数字几乎都来自被引工作。

## 四、值得单读的几篇

正文点名、且能独立成一张机制图的优先。标题以各文自己的常用名为准；arXiv 来自本综述参考文献（PDF p.74 起）。**不要把本条当成已改索引。**

| 标题 | arXiv | 为什么单读 |
|---|---|---|
| ReAct: Synergizing Reasoning and Acting in Language Models | [2210.03629](https://arxiv.org/abs/2210.03629) | 基础层的默认循环：思想与动作交错，后面几乎所有上下文工具 / 搜索都从这里分叉。 |
| Toolformer: Language Models Can Teach Themselves to Use Tools | [2302.04761](https://arxiv.org/abs/2302.04761) | SFT 启动工具的原型：自监督插 API 再微调；用来对照后来的 RL 掌握叙事。 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | [2303.11366](https://arxiv.org/abs/2303.11366) | 言语演化的标杆：错误日志写成下次条件，不改权重。 |
| Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning | [2503.09516](https://arxiv.org/abs/2503.09516) | 训练后搜索：推理中途发检索动作，活网环境。 |
| Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning | [2508.19828](https://arxiv.org/abs/2508.19828) | 记忆读写当 RL 策略；Manager / Answer 双体，对应「记忆不是缓冲」。 |
| OpenHands: An Open Platform for AI Software Developers as Generalist Agents | [2407.16741](https://arxiv.org/abs/2407.16741) | 仓库级把推理、规划、测试收进同一环，基础层落地。 |
| Voyager: An Open-Ended Embodied Agent with Large Language Models | [2305.16291](https://arxiv.org/abs/2305.16291) | 程序演化：技能库永久扩大动作空间；长程误差复合的反面教材也在第八节。 |
| AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversations | 见 COLM 2024；综述 [66] | 固定角色多轮对话的集体层基线，用来对照后来可训练拓扑。 |
| CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society | NeurIPS 2023；综述 [67] | 角色扮演社会的早期集体实验。 |
| AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery | [2506.13131](https://arxiv.org/abs/2506.13131) | 结构演化：源码当假设、LLM 当变异；数学 / 算法发现应用的尖端。 |
| Evo-Memory: Benchmarking LLM Agent Test-Time Learning with Self-Evolving Memory | [2511.20857](https://arxiv.org/abs/2511.20857) | 综述作者自己的流式自演化记忆基准，经验复用是否成立可以在这里验。 |
| AgentBench: Evaluating LLMs as Agents | [2308.03688](https://arxiv.org/abs/2308.03688) | 机制评测入口，避免只用 GSM8K 谈 Agent。 |
| MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents | [2503.01935](https://arxiv.org/abs/2503.01935) | 集体层要单独的合作 / 竞争基准。 |

AutoGen / CAMEL 在参考文献里以会议正式版为主，登记时不要凭印象填错号。ReAct 的 arXiv 号是外部常用编号，封面式参考文献写的是 ICLR 2023；登记前按 [09] 再核一次 API，不要直接抄第三方清单。

## 局限、没写清的，以及可搬走的

**综述自己划的边界**：范围是「推理驱动适应行为」的系统，总结到 2025（PDF p.4）；不替代纯架构或纯内部推理综述。开放问题写成未解，不是已有产品清单。

**没写成完整实验的缺口**：没有跨方法的统一表；MIRIX 的 35% 等数字是转述；Awesome 列表会继续涨，正文不会自动同步。GRPO 公式是通式，不是作者新算法。vibe coding 引用维基与科普文，不是受控实验。

**对自己项目能搬走的**：

1. 先问环境稳不稳、要不要改参数，再选论文，不要从框架名开始。
2. 工具：示范 SFT 只够启动；要迁域就准备结果奖励。
3. 有便宜验证器可以重试，但别以为模型「学会了」——它可能只是抽到了能过测试的样本。
4. 记忆读写应进动作空间；多 Agent 还要先定拓扑与公私边界，再谈共享池。
5. 评测：机制基准与应用基准分开看。数学终答饱和不代表 Agent 能力饱和。
6. 治理从第一天按长程、记忆持久、真实动作来想，短程拒答不够。

## 关键词回看

- **Agentic Reasoning**：推理作为计划–行动–学习回路的组织原则。
- **三层环境动态**：基础 / 自演化 / 集体。
- **In-context vs Post-training**：搜 $Z$ 还是改 $\theta$。
- **GRPO**：组内相对优势，省价值网络。
- **言语 / 程序 / 结构演化**：改的是反思文本、技能库，还是系统代码。
- **Dec-POMDP**：集体层把通信当成推理的延伸。

## 参考资料

- 论文：arXiv [2601.12538](https://arxiv.org/abs/2601.12538)
- 清单：[weitianxin/Awesome-Agentic-Reasoning](https://github.com/weitianxin/Awesome-Agentic-Reasoning)

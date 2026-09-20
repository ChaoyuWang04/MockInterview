# Tülu 3：把开放后训练做成可复现的多阶段配方，最后再用可验证奖励做定向 RL

<!-- release-date: 2024-11-22 -->

**本文依据**：`Tülu 3: Pushing Frontiers in Open Language Model Post-Training`，arXiv **2411.15124v5**（[cs.CL] 14 Apr 2025），**82 页**。第一作者 Nathan Lambert，封面编号机构 **1 = Allen Institute for AI**（同页 **2 = University of Washington**）；核心贡献者以 ♥ 标注。封面**未印会议**。原件首次公开日取任务给定的 arXiv **v1** 日 **2024-11-22**；解读依据本地已核的 **v5**（`pdfinfo` Pages: 82，CreationDate 2025-04-16）。v5 日期不回写 `release-date`。文中数字都标 PDF 页码。标「外部补充」的段落不来自本文。

## 一句话

闭源后训练已经是多轮、多目标、数据与超参都不公开的工程；开放侧还停在「指令微调再加一轮便宜 DPO」。Tülu 3 在 Llama 3.1 基座上给出一整条可复现配方：先按核心技能筛提示并去污染，再 SFT，再长度归一化 DPO（含 on-policy 合成偏好），最后用 **可验证奖励强化学习（Reinforcement Learning with Verifiable Rewards，RLVR）**——没有可学习奖励模型，只有程序化校验对了才给常数奖励。70B 在其开发评测上超过同尺寸开源 instruct，并压过 GPT-4o-mini、接近 Claude 3.5 Haiku；405B 与 DeepSeek-V3 / GPT-4o（11-24）同场可比（PDF p.1, p.5–6, Table 2, Table 4）。

## 一、矛盾：后训练决定行为，开放配方却落后于闭源管道

后训练（post-training）是指令微调、人类反馈强化学习及其变体的总称，用来把基座从「会续写」改成「会按意图做事」（PDF p.5）。闭源 frontier 模型把这一段做成多轮、混合人类与合成数据、多种目标；开放对照（Tülu 2、Zephyr-β）实现更简单、更便宜，但 MATH、IFEval、GSM8K 等核心能力已经落后（PDF p.5, p.7）。截至 2024-11-20，LMSYS ChatBotArena 前 50 没有一家公开后训练数据（PDF p.7 脚注 2）。

Tülu 3 要补的不是又一个聊天分数，而是三件可移交的东西（PDF p.5）：

1. **Tülu 3 Data**：宽松许可、对准核心技能的数据；
2. **Tülu 3 Eval**：开发集 / 未见集拆开的评测，以及按该套件去污染训练提示的工具；
3. **Tülu 3 Recipe**：SFT → DPO → RLVR 的多阶段管道，外加能把 PPO 拉到 405B 的异步 RL 基建。

基座是 Llama 3.1 的 8B / 70B / 405B；发布物含中间 checkpoint、数据、`open-instruct` 训练代码、`olmes` 评测（PDF p.1, p.6）。

```mermaid
flowchart LR
  prompts[公开数据加人格合成提示]
  decon[相对评测套件去污染]
  sft[SFT 得到 Tulu3-SFT]
  dpo[长度归一化 DPO]
  rlvr[RLVR 程序化校验奖励]
  prompts --> decon --> sft --> dpo --> rlvr
```

机制示意，根据 Figure 1（PDF p.5）。四段不是平行技巧，而是同一条模型上的接力：每一段吃的数据形态不同（完成、偏好对、可验证提示）。

## 二、先定技能，再定评测，再让数据为评测服务

他们先列通用模型该补的技能：知识回忆、推理、数学、代码、精确指令跟随、闲聊、安全（PDF p.7–8）。Table 3 把每项技能拆成 **开发评测** 与 **未见评测**（PDF p.7）：

| 技能 | 开发 | 未见 |
|---|---|---|
| 知识 | MMLU、PopQA、TruthfulQA | MMLU-Pro、GPQA |
| 推理 | BigBenchHard、DROP | AGIEval English |
| 数学 | MATH（flex）、GSM8K | DeepMind Mathematics |
| 代码 | HumanEval / HumanEval+（pass@10） | BigCodeBench |
| 指令跟随与闲聊 | IFEval、AlpacaEval 2 | IFEval-OOD、HREF |
| 安全 | 六项平均 | 无未见安全集 |

开发分指导混数与超参；未见分在做模型时**不看分数**，用来检查有没有把开发基准训穿（PDF p.8）。总评是各任务等权平均；生成评测输出长度 4096（PDF p.11）。安全没有未见集，作者写明了（PDF p.8）。

可迁移：后训练实验若只用一张排行榜迭代，很容易把数据混成「这张表的过拟合」。把开发 / 未见切开，比再加一个算法更先要做。

## 三、数据：提示是所有阶段的上游

Table 7 汇总提示池：合计 **23,327,961** 条，SFT 用 **939,344**，DPO 侧计数 **425,145**（脚注 γ：8B / 70B 混数不完全相同）（PDF p.12）。新造的 Persona 系列、CoCoNot、WildJailbreak、WildGuardMix、IF-augmented 等在表里单独上色。

### 公开源怎么挑

多样性：WildChat（真实用户）、Open Assistant、No Robots、FLAN v2、去污染后的 UltraFeedback（PDF p.11–12）。技能向：OpenMathInstruct 2、NuminaMath-TIR、Evol CodeAlpaca、Aya（多语）、SciRIFF、TableGPT（PDF p.12）。许可：排除 ShareGPT 及 UltraFeedback / HelpSteer2 中相关子集，因为来源协议不清（PDF p.13）。

### 人格驱动合成：用来补技能缺口，不是用来替代真实对话

LLM 合成容易模式崩塌。他们跟 Chan 等的 Persona Hub，用约 **25 万** 个人格条件生成（PDF p.13）：

- **精确指令跟随**：覆盖 IFEval 的 25 类约束；人工写 33 条种子，用 GPT-4o（默认 `GPT-4o-2024-08-06`）扩成 **29,980** 对（If-Persona-Sft）。另把 Tülu 2 SFT 指令与 Zhou 等约束拼接成 IF-augmented，**只进 DPO 与 RLVR**（PDF p.13）。
- **数学与代码**：零样本按人格出题；数学解答用 GPT-4o，Python 用 Claude 3.5 Sonnet。约 **22 万** 数学、**3.5 万** 代码（PDF p.13）。
- **安全与不服从**：沿 CoCoNot / WildGuard / WildJailbreak 的分类，覆盖不完整、不支持、不确定、拟人化请求等（PDF p.13–14）。

### 去污染：8-gram，提示侧，超过 2% 就算脏

只比提示（或多轮用户轮），因为完成常被模型重写（PDF p.14）。8-gram：测试实例超过 50% token 与同一训练实例共享含该 token 的 8-gram，则算显著重叠（PDF p.14）。某训练集若与任一开发 / 未见评测超过 **2%** 实例重叠，视为污染。与未见集污染的整表丢掉；与开发集污染的，若去掉整表不明显伤分就整表丢，否则只删匹配实例（PDF p.14）。Table 8：NuminaMath-TIR 相对 MATH 删了 **11.3%**，Evol CodeAlpaca 相对 HumanEval 删 **3.5%**，WildChat GPT-4 相对安全删 **5.4%**（PDF p.14）。附录 Table 37 列了更多公开集污染。

可迁移：去污染阈值是工程选择，不是定理；他们承认嵌入匹配理论上能抓改写，但实践分不清「分布相近」和「改写泄漏」，所以用 n-gram（PDF p.14）。

## 四、SFT：先做技能上限，再往通用混数里折回来

### 混数怎么长出来

基线是 Llama 3.1 + Tülu 2 SFT。对落后技能单独做专家混数，逼近该技能上限，再合并成 preview，再加减数据、去污染、对过大集合降采样（PDF p.16，Figure 3）。完成：人类或前沿模型的原回答尽量保留；WildChat 只用最强模型子集；Persona 等无回答的用 GPT-4o 补；硬编码提示手写（PDF p.15）。

Table 9：Tülu 3 8B SFT 平均 **60.1**，高于 Tülu 2 8B SFT 的 **48.3**；70B SFT **72.6** vs Tülu 2 70B SFT **63.6**（PDF p.16）。注意 Table 6 里 8B SFT 平均写 **60.6**，与 Table 9 的 60.1 不完全同一列口径，引用时跟表走。

Table 10 消融（8B SFT 平均 60.1）（PDF p.16）：

- 去掉 WildChat：平均 58.9，AlpacaEval 2 从 12.4 掉到 **7.5**；
- 去掉安全数据：平均 58.0，安全从 93.1 掉到 **74.7**，其余大体不动——安全与通用技能近似正交；CoCoNot 一类对比提示用来减轻过度拒答；
- 去掉 Persona：IFEval 从 72.8 掉到 **53.6**；
- 去掉数学数据：GSM8K 76.2→**64.1**，MATH 31.5→**23.5**。

Figure 4：分层子采样显示平均分随数据量升到全量最好；TruthfulQA 反而随数据增多下降（PDF p.17, p.19）。他们不再继续堆 SFT，因为剩下的提示留给偏好阶段。

### 训练配方

8B：32 GPU、约 **6 小时**，学习率 $5\times 10^{-6}$；70B：64 GPU、约 **50 小时**，学习率 $2\times 10^{-6}$。有效 batch 128，最长 4096，**2 epoch**，线性学习率（PDF p.18，Table 11）。4–16 台 8×H100（PDF p.18）。

换基座（同一 SFT 混数，Table 12）：Llama 3.1 8B 的 GSM8K / MATH 为 76.2 / 31.5；70B 为 91.1 / 53.7；Qwen 2.5 7B 为 79.2 / 49.4；Qwen 2.5 Math 7B 为 86.3 / 56.4。尺寸和「数学向预训练」都抬下游数学（PDF p.18）。

聊天模板：去掉助手消息末尾换行；「把换行换成 eos」平均略高（53.0 vs 52.8），但为避免后段生成不一致而没用（PDF p.18，Table 13）。

随机种子：8B 默认种子 42 平均 59.9，种子 123 为 **60.1**；70B 种子 456 为 **72.6**，默认 42 为 71.8。最好的 model soup 并不稳定超过最好单次，故最终用最好单次 SFT（PDF p.18–19，Table 14）。

### 损失聚合：padding 会把「按 token」变成「按样本」

Transformers 默认对 padding 做平均时，若梯度累积或分布式把样本拆开，损失从

$$
L=\frac{\ell_{n_1}+\ell_{n_2}}{n_1+n_2}
$$

变成两个样本等权平均（PDF p.19 式 1–2）。他们改成 **sum loss**（去掉分母，所有 token 等权），并重调学习率。在 Llama 3.0 + Tülu 2 混数上，$5\times 10^{-6}$ 的 sum loss 最好；加 epoch 超过 2 没有再涨（PDF p.20，Figure 5–6）。

可迁移：SFT 复现对不齐时，先查损失是按 token 还是按样本，再查聊天模板末尾空白，再查种子。混数消融显示「真实闲聊」和「技能合成」不是互相替代。

## 五、偏好微调：主路径是长度归一化 DPO，不是 PPO

### 目标

标准 RLHF 先训奖励 $r_\phi$，再最大化带 KL 的奖励（PDF p.20–21 式 3–4）。DPO 把同一目标写成对策略对数比的分类损失（式 5）。长度归一化 DPO 再除以完成长度，减轻偏好里的长度偏差（PDF p.21 式 6）：

$$
\max_{\pi_\theta}\ \mathbb{E}\bigl[\log\sigma\bigl(\tfrac{\beta}{|y_c|}\log\tfrac{\pi_\theta(y_c\mid x)}{\pi_{\mathrm{ref}}(y_c\mid x)}-\tfrac{\beta}{|y_r|}\log\tfrac{\pi_\theta(y_r\mid x)}{\pi_{\mathrm{ref}}(y_r\mid x)}\bigr)\bigr]
$$

Table 18 在早期 SFT + UltraFeedback 上扫 SimPO、DPO、PPO、DPO-norm：只有长度归一化 DPO 超过 SFT 基线平均 55.7；其中 LR $5\times 10^{-7}$、$\beta=5$、1 epoch、batch 32 得到 **57.3**（PDF p.28）。PPO 在未精细调奖励模型时平均略低于 DPO，且约 **28 小时 / 两节点** vs DPO **4 小时 / 单节点**（PDF p.28–29）。因此混数实验全程用 DPO-norm，PPO 留给 RLVR。

最终 DPO 超参（Table 20）：8B / 70B 学习率 $5\times 10^{-7}$ / $2\times 10^{-7}$，batch 128，最长 2048，$\beta=5$，warmup 0.1，1 epoch（PDF p.28）。8B 约 10 小时 / 8×H100；70B 约 19 小时 / 64×H100（PDF p.27）。70B 学习率消融（Table 19）：随混数不同，$2\times 10^{-7}$ 或 $5\times 10^{-7}$ 更好，最终取 $2\times 10^{-7}$（PDF p.28）。

70B 显存：缓存参考策略对数概率，不再常驻参考模型；chosen / rejected 分开前向，避免拼接加倍 batch（PDF p.29–30，Figure 17）。

### 偏好数据管道（Figure 7）

1. 提示：SFT 用过的、同库未用过的、清洗后的 UltraFeedback、加约束的新提示（PDF p.21–22）。
2. 每个提示从约 **22** 个模型池随机抽 4 个模型生成；池含更新后的开源 / 闭源，并强制混入 Tülu SFT 的 on-policy 完成（PDF p.22）。
3. GPT-4o-2024-08-06 按有用、指令跟随、诚实、真实四维打 1–5 分，再二值化成 chosen / rejected（PDF p.22）。

最好混数合计 **354,192** 条；8B 用 **271,409**，70B 用 **334,302**（PDF p.22–23，Table 15）。含 SFT 复用 off-policy、IF-augmented、WildChat 若干切片、UltraFeedback、Persona IF 等。

### 数据消融里真正起作用的几条

- **独特提示数量**比重复提示重要：固定偏好、增加独特提示，多项上涨（Figure 8）；UltraFeedback 把四回答两两配对扩到 383k，平均几乎不涨，DROP / GSM8K / AlpacaEval 还略降（Figure 9）（PDF p.23–24）。
- **未在 SFT 见过的提示**略优于纯复用；最好是两者混合（Figure 10）（PDF p.24）。
- **On-policy** 优于纯 off-policy（Figure 11）（PDF p.24）。
- 裁判：GPT-4o、Llama 3.1 405B、GPT-4 Turbo 接近，GPT-4o 平均略高（Table 17：57.3 / 57.2 / 57.0）（PDF p.25）。
- 最好混数明显超过只训 UltraFeedback；70B 增益大于 8B（+3.3 vs +1.8），作者猜测 UltraFeedback 的完成多来自弱于 70B 的模型（PDF p.25，Figure 12）。
- Persona 偏好里 **只有 Persona IF** 抬平均和 IFEval；Persona Math / Code 伤平均且不抬对应项，故不进最终混数（PDF p.25，Figure 13）。
- 用同一管道重生成 HelpSteer2 / UltraFeedback / MultiPref 的完成，下游好于原偏好对（Figure 15）（PDF p.27）。

可迁移：偏好阶段先扩「新提示」，再扩 on-policy，再考虑算法变体。长度归一化是他们在「简单、能扫混数」约束下的选择，不是声称 PPO 上限更低。

## 六、RLVR：把奖励模型换成校验函数

### 定义

任务要有可程序化判定的对错（数学最终答案、指令约束）。目标与 KL 约束 RLHF 同构，只是 $r_\phi$ 换成校验 $v$（PDF p.30–31 式 7–8）：

$$
\max_{\pi_\theta}\ \mathbb{E}_{y\sim\pi_\theta(\cdot\mid x)}\bigl[v(x,y)-\beta\,\mathrm{KL}[\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)]\bigr]
$$

对了给 $\alpha$，否则 0。试探后 $\alpha=10$，未再调（PDF p.31）。优化器用 PPO。作者把它看成 STaR / 执行反馈 RL 的简化：在线、二元奖励、现成 PPO，并接到通用后训练的最后一段，而不是只刷数学（PDF p.30, p.50）。

### 数据（Table 22）

约 **29,946** 条（PDF p.31, p.30 Table 22）：

| 源 | 条数 | 校验 |
|---|---:|---|
| GSM8K 训练集 | 7,473 | 抽最终数字精确匹配；训练时加评测用的 8-shot CoT |
| MATH 训练集 | 7,500 | flex 抽答案；加 3-shot CoT |
| IF 可验证 | 14,973 | 按约束模板的校验函数 |

代码执行类校验留待将来（PDF p.31）。

### 实现细节（跟 Huang 等 PPO 实践）

价值模型从通用奖励模型初始化；dropout 全关（否则 rollout 与学习两阶段 logprob 对不齐，PPO 比率全被 clip）；可多 epoch 并在 epoch 间打乱；无 EOS 罚 **-10**；advantage 去均值除标准差（PDF p.32）。消融约 $100{,}000/7{,}473\approx 13$ epoch；最终 run 每 40–100 step 看开发集挑点（PDF p.32）。

### 关键实验结论（PDF p.33–34）

1. GSM8K / MATH / 约束提示上，训练可验证奖励上升，对应测试也上升；**全套平均不保证升**（Figure 19）。
2. 价值函数用通用 RM 初始化，好于从锚定 DPO 初始化（Figure 21）。
3. **不要**把 RM 分数叠在可验证奖励上，噪声更大（Figure 22）。
4. 从较弱 SFT 出发也能把训练可验证奖励拉到与从 DPO 出发相近，但同样 $\beta$ 下 KL 更大，测试通常仍是强起点更好（Figure 20）。
5. $\beta$ 过小、KL 过大，平均分下降，即过优化；约束跟随的过优化见附录 B.4。

### 基建

ZeRO-3；策略、参考、价值三模型；推理 GPU 专给 vLLM PagedAttention；训练与推理异步并行，策略吃「倒数第二批」推理数据以减轻 stale（PDF p.34–35）。最终 8B RM：9 小时 / 8×H100；8B RL：65 小时 / 8 GPU；70B RL：60 小时 / 48 GPU；405B RL：46 小时 / 256 GPU。各规模都取**早于训练终点**的 checkpoint（PDF p.34）。

### 最终 8B / 70B（Table 23，PDF p.36）

从最好 DPO 出发，提示用三源合并。8B 曾试更高 $\beta$（至 0.15）；70B 用学习率 $1\times 10^{-7}$、warmup 0.1、回复长 2048、400,000 episode、有效 batch 640。正文 6.4 写 $\beta=0.7$；Table 21 脚注写最终 70B 为 $\beta=0.07$、$\omega=0.07$，**两处不一致**，引用超参以表注为准并保留正文数字。8B 按 MATH+IFEval 最好点；70B 每 40 step 评估。

相对各自 DPO：8B 的 MATH 42.0→**43.7**，GSM8K 84.3→**87.6**，IFEval 81.1→**82.4**，平均 64.4→**64.8**。部分 8B run GSM8K 到 **89.4%**、IFEval 到 **84.8%**，但其他项更差、平均更低。70B 的 MATH 62.3→**63.0**，IFEval 82.6→**83.2**，GSM8K 停在 **93.5**（接近饱和）；全程 KL 远小于 1，作者归因为更低学习率（PDF p.36）。安全平均在 RLVR 后略降（8B 87.2→85.5，70B 89.0→88.3）。

可迁移：RLVR 是「有校验器的技能」的定向旋钮，不是全面对齐替代品。价值模型初始化、只用二元奖励、从已经较强的 DPO 出发，是他们复现清单里比 $\alpha$ 更敏感的项。

## 七、主结果：开发套件上的位置，以及未见套件泼的冷水

### 开发套件（Table 2 / 5 / 6）

生成长度 4096；平均等权（PDF p.11）。Table 2（PDF p.6）：

| | Tülu 3 8B | Qwen2.5 7B Inst. | Llama 3.1 8B Inst. | Tülu 3 70B | Qwen2.5 72B Inst. | Llama 3.1 70B Inst. | GPT-3.5 Turbo | GPT-4o Mini | Claude 3.5 Haiku |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 平均 | 65.1 | 66.5 | 62.9 | **76.2** | 72.8 | 74.1 | 64.7 | 69.6 | 75.3 |
| MATH flex | 43.7 | 69.9 | 42.5 | 63.0 | 75.9 | 56.4 | 41.2 | 67.9 | 68.0 |
| GSM8K | 87.6 | 83.8 | 83.4 | 93.5 | 89.5 | 93.7 | 74.3 | 83.0 | 90.1 |
| IFEval loose | 82.4 | 74.7 | 80.6 | 83.2 | 87.6 | 88.0 | 66.9 | 83.5 | 86.3 |
| AlpacaEval 2 LC | 34.5 | 29.0 | 24.2 | 49.8 | 47.7 | 33.4 | 38.7 | 49.7 | 47.3 |
| 安全六项均 | 85.5 | 75.0 | 75.2 | 88.3 | 87.0 | 76.5 | 69.1 | 84.9 | 91.8 |

8B 平均 **低于** Qwen 2.5 7B Instruct（65.1 vs 66.5），尤其 MATH / HumanEval 仍落后；70B 平均超过同场开源 instruct，并超过 GPT-4o-mini、略超 Haiku 的 75.3（PDF p.6, p.11）。表中 ♢ 为 MICE 插补，⊤ 来自 Claude 模型卡，不是本套件实测（PDF p.6）。若干最低分来自 few-shot 格式失败或重复错误（PDF p.9–10）。

阶段贡献：8B 平均 SFT 60.6 → DPO 64.7 → RLVR 65.1（Table 6）；70B 为 72.6 → 76.2 → 76.2（Table 5，DPO 与最终同平均）（PDF p.9–10）。AlpacaEval 主要在 DPO 跳升（8B 12.4→33.5，70B 26.3→49.6）。

### 405B（Table 4，PDF p.8）

不含安全的平均：Llama 3.1 405B Instruct 78.1，Hermes 3 405B 74.4，DeepSeek-V3 79.0，GPT-4o（11-24）80.5；Tülu 3 405B 的 SFT / DPO / RLVR 为 **76.3 / 79.0 / 80.0**。含安全：最终 80.7 vs GPT-4o 81.6、Llama Instruct 79.0。MATH flex：SFT 63.4 → DPO **59.9**（下降）→ RLVR **67.3**。GSM8K RLVR **95.5**。AlpacaEval 2：SFT 30.4 → DPO 49.8 → RLVR 51.4，仍低于 DeepSeek-V3 的 53.5 与 GPT-4o 的 65.0。TruthfulQA 与 MMLU 多选因 logprob 基建不兼容未报（PDF p.8）。

405B 工程（PDF p.46–47）：32 节点 256 GPU；NCCL 超时与硬件故障需重启。RLVR 推理 16 路张量并行 vLLM，其余 240 GPU 训练；每步推理约 550s、权重广播约 25s、训练约 1500s。价值模型用 **8B** 以省算力。因 GSM8K 已饱和、IF 数据初期帮助不大，405B RLVR **只训 MATH 训练集**。约 25 step MATH 提高超过 5 点；因异步 RL 不稳只训 **75 step**，MATH 测试尚未饱和（Figure 25）。SFT / DPO 超参见 Table 34：SFT LR $2\times 10^{-6}$、batch 256、2 epoch；DPO LR $2\times 10^{-7}$、$\beta=5$（PDF p.47）。

### 未见套件（Table 33，PDF p.46）

相对 Llama 3.1 Instruct，Tülu 3 **并不全面领先**：

- 8B 平均 34.2 vs Instruct **36.4**；GPQA 35.7 高于 Instruct 的 28.8，但 BigCodeBench-Hard pass@10 仅 **7.4** vs 15.5，DeepMind Math 35.4 vs 39.3。
- 70B 平均 47.2 vs Instruct **51.3**；GPQA 48.0 高于 43.8，DeepMind Math 49.8 vs 62.4，IFEval-OOD 27.8 vs 34.5。

作者认为精确约束跟随很难泛化到新约束，IFEval 高分可能过拟合该约束集；HREF 与 AlpacaEval 相对名次也不一致（70B 在 HREF 11 个子任务里 5 个超过 Llama Instruct）（PDF p.46）。**其他模型训练数据未公开，不能排除它们训过这些未见集**（PDF p.46）。

评测框架本身（第 7 节）：OLMES 工具、统一模板与推荐设置、安全六项平均、新造 IFEval-OOD 与 HREF（PDF p.36–43）。用未见集回看设计决策：开发期选择并非处处迁移（PDF p.44–45）。

## 八、没进最终配方的路，以及报告承认没覆盖的技能

**Online DPO**：用 RM 在线标当前策略的一对完成再走 DPO。通用 RM 用 Skywork 约 82k 偏好；数学再在 on-policy 数学偏好上续训。在已有 DPO checkpoint 上 200k episode，GSM8K 几乎不动、MATH 下降（PDF p.48）。

**拒绝采样**：对每个 SFT 提示采 $n$ 个回答，RM 或 LLM 裁判留最好，其余可做偏好对，再整条管道循环。他们觉得算力换来的增益太小；公开裁判选最优不稳；**把原始回答也放进候选** 好于只在新生成里选（PDF p.48–49）。

未覆盖：长上下文与多轮（混数平均 **2.4** 轮，多数样本短于 2048 token）；多语主要靠 Aya；工具与 Agent 未训未评（PDF p.49）。

相关工作把 RLVR 对照 STaR、Quiet-STaR、TRICE、VinePPO 与代码反馈 RL：强调在线二元奖励 + 标准 PPO + 多技能接入通用管道（PDF p.49–50）。

## 九、读完可以带走的判断

1. **开放后训练的主矛盾是配方与数据透明度，不只是再发明一个损失。** 技能清单 → 开发 / 未见评测 → 去污染 → 分阶段数据形态，比单点刷榜更可复用。
2. **SFT 混数是上限管理**：先做技能专家逼近上限，再折回通用；真实 WildChat 保闲聊，Persona 保 IF / 数学，安全数据几乎正交。
3. **偏好阶段，独特 on-policy 提示 + 长度归一化 DPO** 是他们在扫混数预算下的主路径；PPO 并非不能打平，而是更贵、奖励模型更难评。
4. **RLVR 适合「答案能程序化判定」的最后一公里**，8B 数学与 IF 有可见跳点，70B / 405B 更像补 MATH；叠通用 RM 分数会加噪；KL 过小会过优化。
5. **开发套件上的 70B / 405B 叙事，不能直接当成未见泛化胜利。** Table 33 里对 Llama Instruct 的平均落后，是正文自己提供的刹车。

对自己项目：有单元测试或符号答案的任务，可以在 SFT+DPO 之后加一段二元奖励 PPO，不必先上过程奖励模型。没有校验器的开放对话，这条不迁移。损失按 token 聚合、参考对数概率缓存、推理 / 训练 GPU 拆开，是把规模从 8B 推到 70B / 405B 时先要修的工程，而不是算法论文的附录装饰。

## 关键词回看

- **后训练（post-training）**：基座之后的指令、偏好与强化学习阶段。
- **Tülu 3 Eval**：开发集指导迭代，未见集事后检验过拟合。
- **人格合成（persona-driven synthesis）**：用大量人格条件降低合成数据模式崩塌。
- **8-gram 去污染**：提示侧重叠超过评测实例 50% token 则匹配。
- **Sum loss**：避免 padding 平均在梯度累积下变成按样本加权。
- **长度归一化 DPO**：对数比除以完成长度，减轻偏长。
- **On-policy 偏好**：偏好对里至少一侧来自将要继续训的 SFT 模型。
- **RLVR**：校验函数给出 $\alpha$ 或 0，PPO 最大化带 KL 的该奖励。
- **异步 PPO**：vLLM 推理与 ZeRO 训练并行，吃次新 rollout 以换吞吐。

# 报告库存

按方向分组的报告索引,是 `/reports` 页面分组与排序的唯一数据源;规则见 [docs/10-材料解读流程.md](../docs/10-材料解读流程.md) 第八节。

- `## 方向` 是页面分组;方向内的行序即页面顺序。每个方向的已发布卡片按首发日从新到旧,同日按报告名字面值升序;未解读排在已发布后面,发布后插到对应日期位置
- 「报告」等于 `reports/<公司>/<报告>.md` 去掉 `.md`;「公司」等于目录名
- 状态由文件推导,不手写:正文存在为已发布,`_<报告>.md` 为施工中,否则未解读;`npm run reports:status` 现场打印
- 网页原件在 `papers/` 下没有文件,在一句话里注明并留 URL 线索

## 语言基模

本组已发布卡片按首发日从新到旧;未解读排在后面,发布后插到对应日期位置。

| 报告 | 公司 | 一句话 |
|---|---|---|
| Model-Growth-Scaling-Exponents | NYU | 架构能改预训练的缩放指数:以循环 Transformer 为锚,训练中加循环数的模型增长改变指数最多,边界算子也有较小提升 |
| DeepSeek-V4.1-Flash | DeepSeek | 552B CED,prefill 8B/decode 16B,CSA2+FP4 把全局 KV 压到 890 B/token |
| Qwen3.8-Max | Alibaba | 2.4T/95B;真正的内容是 RL 系统——环境按 Task/Workspace/Harness 三轴解耦扩展、统一奖励系统、在线数据均衡器压批间梯度方差。**原件是网页**,见 qwen.ai/blog?id=qwen3.8 |
| Kimi-K3 | Moonshot | 序列、深度、宽度一起扩的系统级设计 |
| Hy3 | Tencent | 295B/21B,192 专家 top-8 加 3.8B MTP,256K 上下文;快慢思考融进同一个模型,不出独立 thinking 版。**原件是网页**,见 huggingface.co/tencent/Hy3 与 github.com/Tencent-Hunyuan/Hy3,只有 model card 与仓库 README,无技术报告 |
| Nemotron-3-Ultra | NVIDIA | 550B/55B,Hybrid Mamba-Attention + LatentMoE + NVFP4;官方已出 65 页技术报告 |
| ZAYA1-8B | Zyphra | CCA + top-1 专家,AMD GPU 训练 |
| Laguna-M1-XS2 | Poolside | 逐层 query 头预算,编码向开放权重 |
| DeepSeek-V4 | DeepSeek | 百万 Token 长上下文的成本重构(**写作标杆**) |
| Ling-Ring-2.6 | AntGroup | 1T,Lightning + MLA 混合线性;靠架构迁移预训练而非从零训 |
| Gemma-4 | Google | 小模型家族,原生多模态与 encoder-free |
| Mistral-Small-4 | Mistral | 119B/6B,128 专家 top-4,256K;把 Magistral(推理)、Pixtral(多模态)、Devstral(编码)三条产品线并回一个模型。**原件是网页**,见 mistral.ai/news/mistral-small-4,训练数据、算力、超参与消融均未公开 |
| GLM-5 | Z.ai | 从 vibe coding 到 agentic engineering 的全栈重设计 |
| Qwen3-Coder-Next | Alibaba | 编码 agent 方向的 Qwen 分支 |
| Step-3.5-Flash | StepFun | 196B/11B,SWA 3:1 + MTP-3 训推都用 |
| Trinity-Large | Arcee | 400B,滑窗 + 全局交错注意力,极高稀疏度的粗粒度 MoE;稳定性优先的现场记录 |
| LongCat-Flash-Thinking-2601 | Meituan | 上一篇的 agentic 续作 |
| MiMo-V2-Flash | Xiaomi | 309B/15B,128 窗口 SWA 5:1 + 注意力 sink,27T Token |
| Nemotron-3 | NVIDIA | 权重、数据、recipe 一并公开;原件是 13 页白皮书,Nano 另有技术报告 |
| Ministral-3 | Mistral | 参数高效的 dense 家族:从 24B 父模型级联剪枝蒸馏而来,14 页方法说明非完整报告 |
| Mistral-Large-3 | Mistral | 675B/41B 稀疏 MoE,3000 张 H200 训练;与 Ministral-3 同场发布,但那篇 arXiv 论文只覆盖 Ministral。**原件是网页**,见 mistral.ai/news/mistral-3 与 HF mistralai/Mistral-Large-3-675B-Instruct-2512 |
| INTELLECT-3 | PrimeIntellect | prime-rl 训练栈产出的模型 |
| Olmo-3 | Ai2 | 完全开源路线:交付整条模型流水线而非只交权重(依据 arXiv v2,118 页) |
| MiniMax-M2 | MiniMax | 掉头回全注意力 GQA 的反例,与 MiniMax-M1 的线性路线对着读 |
| LongCat-Flash-Thinking | Meituan | 领域并行训练再融合,DORA 异步 rollout |
| LongCat-Flash | Meituan | 560B,零计算专家做动态算力预算 + 快捷连接 MoE |
| GPT-5-System-Card | OpenAI | 63 页里没有一个能力分数;主轴换成评估域、Preparedness 分级与缓解充分性论证(依据 arXiv v2,官方 CDN 版反而更旧) |
| gpt-oss | OpenAI | 开源权重不是开关而是光谱:harmony 格式与风险评测写满,训练配方整段留白(依据 arXiv v1,35 页) |
| GLM-4.5 | Z.ai | ARC 三能力先分开练,再蒸馏回一套权重 |
| Kimi-K2 | Moonshot | 高质量数据见底后的 Token 与轨迹再利用 |
| Qwen3 | Alibaba | 一套权重兼顾深思、快答与预算控制 |
| Gemini-2.5 | Google | 闭源前沿少有的公开报告 |
| Command-A | Cohere | 报告自述稠密、SWA 与全注意力 3:1 交错;企业能力靠六个后训练专家做参数合并,23 种语言(依据 arXiv v2,55 页) |
| Gemma-3 | Google | SWA 5:1 与 496 KiB/token 的 KV cache 对照基准 |
| DeepSeek-V3 | DeepSeek | 671B MoE,FP8、DualPipe、无辅助损失负载均衡 |
| Phi-4 | Microsoft | 合成数据占到 55%、同一批读 13.8 轮不见过拟合;代价是纯合成伤知识(依据 arXiv v1,36 页) |
| Nova | Amazon | 家族四员零参数量与架构披露;真写出来的是评测口径、307 项红队分类与 97% goodput 的基础设施(依据 arXiv v1,48 页) |
| Qwen2.5 | Alibaba | Qwen 上一代:架构几乎不动,增量全在数据与后训练(依据 arXiv v2) |
| Llama-3 | Meta | 405B dense,复杂度从架构挪到数据与运维 |
| DeepSeek-V2 | DeepSeek | MLA 与 DeepSeekMoE 的出处 |
| Phi-3 | Microsoft | 小模型靠数据质量取胜;依据 arXiv v4,覆盖 phi-3 与 phi-3.5 共六个成员 |
| Llama-2 | Meta | 7B–70B,GQA 与 4k 上下文,公开预训练加对话对齐 |
| GPT-4 | OpenAI | 能力与安全评测,训练细节保留 |
| LLaMA | Meta | 开源基模时代的起点:7B–65B dense,只用公开数据(2023) |
| OPT | Meta | 175B 开源复现 GPT-3,附完整训练日志与故障记录(2022) |
| GPT-3 | OpenAI | few-shot 学习与 scaling 的源头 |
| BERT | Google | 双向编码器预训练,微调范式的起点(2018) |
| Seed2.0-Model-Card | ByteDance | 字节旗舰基模 2.0 的模型卡:面向真实世界复杂度的能力与评测口径(arXiv 2607.00248) |
| Seed1.5-Thinking | ByteDance | 豆包 1.5 推理线的 RL 配方与稳定化(arXiv 2504.13914) |
| Gemini-3 | Google | 本组最新停在 2.5,补 3 代的卡与安全评估。**原件待核**(deepmind 的卡 PDF 本机抓不到) |
| Llama-4 | Meta | Maverick 与 Scout 的 MoE,配 iRoPE、NoPE、MoD。**原件待核**(ai.meta.com 全域抓取失败) |
| Muse-Spark | Meta | Superintelligence Labs 首个模型,含 Safety 与 Preparedness 报告。**原件待核** |
| MAI-Thinking-1 | Microsoft | Building a Hill-Climbing Machine:MAI 自研推理基模的完整复盘。**原件待核**(PDF 未开) |
| Phi-4-Reasoning | Microsoft | Divide and Conquer 课程化 RL 的 reasoning 专线报告(arXiv 2504.21318) |
| Grok-4.6 | xAI | xAI 全线零收录,先补这一代模型卡。**原件待核**(x.ai 抓取失败) |
| Step-3 | StepFun | 321B MoE 的模型-系统协同设计,低成本解码是主线(arXiv 2507.19427) |
| Qwen3.5 | Alibaba | 主线在 3.8-Max 之前缺的一代。**原件待核**(qwen.ai 是 SPA,正文未抓到) |
| Qwen3.7 | Alibaba | 晚于 3.5 的一代,同样是代际缺口。**原件待核** |
| Intern-S1-Pro | ShanghaiAILab | 万亿规模科学多模态基模与知识图谱 RL(arXiv 2603.25040) |
| Falcon-H1 | TII | Transformer-Mamba 混合头的开放权重家族;推理续作 H1R 见 arXiv 2601.02346 |
| LFM2 | LiquidAI | 为端侧效率反推架构的开源小基模家族(arXiv 2511.23404) |
| SmolLM3 | HuggingFace | 3B 长上下文多语推理,训练方法与配方全公开。**原件待核**(HF 博客抓取失败) |

## 注意力与长上下文

| 报告 | 公司 | 一句话 |
|---|---|---|
| MiniMax-Sparse-Attention | MiniMax | MSA:GQA 之上的块稀疏,在未压缩 KV 上做选择;MiniMax-M3 的注意力底座 |
| Mixture-of-Depths-Attention | ByteDance | 让 Query 额外看前层同位置的 depth KV,缓解深层信号稀释(Seed) |
| IndexCache | Z.ai | 每四个稀疏注意力层复用同一个 indexer,1M 上下文每 token FLOPs 降 2.9×;GLM-5.2 的 IndexShare 出处,GLM-5 报告未覆盖 |
| DeepSeek-V3.2 | DeepSeek | DSA 稀疏注意力,先让长上下文变便宜 |
| Kimi-Linear | Moonshot | KDA:通道级遗忘门的线性注意力,Kimi-K3 的骨干 |
| Recursive-Language-Models | MIT | 把长上下文当外部变量,让模型递归调用自己来处理 |
| CCA | Zyphra | 直接在压缩空间里做注意力,MLA 的激进版 |
| FlashAttention-4 | Stanford | Blackwell 上的非对称扩张:软件模拟指数、TMEM、2-CTA MMA |
| MiniMax-M1 | MiniMax | 混合线性注意力,把测试时算力成本压回线性 |
| Gated-Attention | Alibaba | 注意力输出加门控,消 attention sink 并提升稀疏性;Qwen3-Next 采用 |
| MoBA | Moonshot | 块注意力混合,与 NSA 同期对打 |
| NSA | DeepSeek | 原生可训练、硬件对齐的稀疏注意力 |
| MiniMax-01 | MiniMax | lightning attention 的出处 |
| From-Attention-to-Activation | Huawei | 首 token 注意力集中与激活离群值的成因,OrthoAdam 消除(诺亚方舟) |
| FlashAttention-3 | Stanford | 异步与低精度(依据 NeurIPS 2024 正式版) |
| FlashAttention-2 | Stanford | 并行与工作划分改进 |
| FlashAttention | Stanford | IO 感知的精确注意力:多算 FLOP 换少搬字节 |
| Transformer | Google | 注意力机制的出处:Attention Is All You Need(2017) |
| Hybrid-Architectures-for-LM | Meta | 线性与全注意力混合配比的系统性消融,41 页(FAIR 与 KAIST,arXiv 2510.04800) |
| LongCat-Sparse-Attention | Meituan | 跨层索引的流式感知块稀疏注意力;同线还有 ZigZag(arXiv 2608.01662) |
| RePo | Sakana | 上下文重定位:用检索替代 KV 复用(ICML 2026,arXiv 2512.14391) |

## 训练方法与强化学习

| 报告 | 公司 | 一句话 |
|---|---|---|
| Repulsive-Self-Distillation | ETH | 特权信息同时改了教师「知道什么」与「怎么表现」:吸引式与排斥式自蒸馏引起相反的行为偏移,两者对比组合后偏移相消,只剩更贴近正确性的 token 级信号 |
| Privileged-Info-OPSD | NUS | on-policy 自蒸馏里给教师看答案到底多加了什么:5,319 题六种推理视角的 AMPLE-Math 对照无参考蒸馏,发现大部分提升来自蒸馏本身,参考信息的额外收益有限且依赖学生 |
| Score-Centering | TogetherAI | 训推不一致下 RL 失稳主要来自逐步累积的漂移:加一项可加的 score centering 修正抵消它,0.6B 到 30B 上单用即可比肩或超过重要性采样,还能与之叠加 |
| APEX-Agents-SkyRL-Recipe | Mercor | 397B 知识工作 Agent 的六步 RL recipe;**原件是官方博客,`papers/` 下无 PDF** |
| SOAP-Muon-and-Beyond | NVIDIA | 优化器 scaling 对照,附开源实现 |
| Behavior-Leverage-Imbalance | AntGroup | 多教师 OPD 的 top-K 丢掉决策坐标,导致过调用 |
| AReaL-2.0 | AntGroup | position paper:轨迹协议、数据代理、演化控制面三支柱 |
| MOPD | Xiaomi | 多教师 on-policy 蒸馏做能力整合,MiMo-V2-Flash 的后训练方法(北大合作) |
| Counteraction-Aware-OPD | Kuaishou | 多教师在线策略蒸馏:恢复通用能力同时保住领域能力 |
| Polar | NVIDIA | harness 当黑盒的 agentic RL;同模型同算法换 harness 差 22 分 |
| Continuous-Latent-Diffusion-LM | ByteDance | 连续潜空间上的扩散式语言模型(Seed) |
| Model-Spec-Midtraining | Anthropic | 对齐训练如何泛化 |
| FIPO | Alibaba | 未来 KL 影响的策略优化,引出深度推理(Qwen Pilot) |
| ProRL-Agent | NVIDIA | Polar 前作,Rollout-as-a-Service |
| Terminal-Data-Engineering | NVIDIA | 扩展 LLM 终端能力的数据工程 |
| AReaL-DTA | AntGroup | 动态树注意力,共享前缀只算一次 |
| AReaL-SEA | AntGroup | 多 agent 合成对话 + 每实例可执行 checker |
| Endless-Terminals | Stanford | 规模化生成 terminal agent 的 RL 环境 |
| Stabilizing-RL-with-LLMs | Alibaba | RL 稳定性的形式化与实践(与 GSPO 篇立场相反,已在文中摆明) |
| SkyRL-Agent | Berkeley | 多轮长程 agent 的异步流水线调度 |
| AgentEvolver | Alibaba | 通义,自进化 agent 系统:自出题、自导航、自归因 |
| FP16-Training-Inference-Mismatch | SeaAILab | 训推数值不一致导致 RL 崩溃,换 FP16 即可解决 |
| Ouro | ByteDance | 参数循环复用做潜空间推理的 scaling(Seed) |
| Laminar | ByteDance | 可扩展的异步 RL 后训练框架(依据 arXiv v1;EuroSys '26 正式版数字一致) |
| Evolution-Strategies-at-Scale | Cognizant | 进化策略替代 RL 微调十亿参数模型 |
| rStar2-Agent | Microsoft | agentic reasoning 的训练 |
| Agent-Lightning | Microsoft | agent 执行与训练完全解耦,span 流入 LightningStore |
| GSPO | Alibaba | 序列级重要性比 |
| AsyncFlow | Huawei | 服务化异步流式 RL,生产者-消费者工作流 |
| ROLL | Alibaba | 面向 RLHF / 推理 / 多轮 agentic 的框架论文 |
| AReaL | AntGroup | 大规模异步 RL 系统 |
| GPG | Alibaba | 去掉 GRPO 多余项后的最简 RL 基线(AMAP) |
| DAPO | ByteDance | 长思维链 RL 的四项工程改动 |
| Muon-is-Scalable-for-LLM-Training | Moonshot | Muon 规模化的两个前提:权重衰减与更新尺度对齐 |
| DeepSeek-R1 | DeepSeek | 结果奖励训练推理能力 |
| Kimi-k1.5 | Moonshot | 把搜索压进上下文,用 RL 扩展模型能力 |
| Qwen-Math-PRM | Alibaba | 开发数学过程奖励模型的教训:MC 估计的坑与共识过滤 |
| Coconut | Meta | 连续潜空间推理,不吐出 CoT token(FAIR) |
| HybridFlow | ByteDance | verl 的论文,RLHF 框架编程模型 |
| FineWeb | HuggingFace | 15T token 预训练语料的清洗与消融配方 |
| DeepSeekMath | DeepSeek | GRPO 的出处与数学语料流水线 |
| RFT | Alibaba | 拒绝采样微调:推理路径越多样,数学能力提升越大(DAMO) |
| Lets-Verify-Step-by-Step | OpenAI | 过程监督优于结果监督,PRM800K 的出处 |
| LIMA | Meta | 1000 条精选样本就够对齐,能力来自预训练 |
| WizardLM | Microsoft | Evol-Instruct:让 LLM 自己把指令进化得更复杂 |
| Constitutional-AI | Anthropic | 用原则和 AI 反馈替代人工有害性标注,RLAIF 的出处 |
| InstructGPT | OpenAI | SFT → 奖励模型 → PPO 的三段式 RLHF 范式出处 |
| Dr.GRPO | SeaAILab | 拆开 GRPO 的长度偏置与标准化偏置各自怎么毒害训练(arXiv 2503.20783) |
| Jet-RL | NVIDIA | BF16 训练加 FP8 rollout 会崩,精度流要统一;与 FP16-Training-Inference-Mismatch 对着读(arXiv 2601.14243) |
| RollArt | Alibaba | agentic RL 各阶段映射到最合适的硬件,省 1.35 到 2 倍(arXiv 2512.22560) |
| KTO | ContextualAI | 无配对数据的对齐:用 Prospect Theory 做拒绝式优化(arXiv 2402.01306,ICML 2024) |
| Pythia | EleutherAI | 154 个 checkpoint 的训练动力学与单元测试套件(arXiv 2304.01373,ICML 2023) |
| DCLM | Stanford | 数据侧的 ImageNet 时刻:固定模型只比数据配方(arXiv 2406.11794,NeurIPS 2024) |
| Nemotron-CLIMB | NVIDIA | 聚类迭代搜数据配方,用 512 个模型的网格反推预训练混合(arXiv 2504.13161) |
| Nested-Learning | Google | 把「深度」重述为多尺度更新,连续学习的优化视角(arXiv 2512.24695,NeurIPS 2025) |
| IER-OPD | MBZUAI | 稀疏 on-policy 蒸馏选哪些 token:用信噪分解定义信息效率比 IER 衡量梯度估计的可靠性,与已有有用性分数组合后,0.1%–1% 的 token 预算即可匹配全量 OPD |
| Value-Flattening | SJTU | PPO critic 的系统性失效:真实状态价值沿回答剧烈变化,critic 预测却几乎是平的;SP3O 每条回答只在几个相隔较远的状态上算价值损失来缓解 |

## Agent 训练与工具使用

| 报告 | 公司 | 一句话 |
|---|---|---|
| Skill2Env | NVIDIA | 把 3.4k 个公开 Agent Skills 转成 8k 个带程序化测试与行为 rubric 的终端环境,300 步 RL 让 Qwen-3.8 27B 在 Terminal-Bench 2.1 上涨 4.7 个点 |
| RetireOPD | ZJU | 多轮 agent 的自蒸馏:先用环境奖励优化带技能的教师,再让学生 RL 加 OPD 联合训练,差距不再缩小且达到目标成功率就自行退掉教师 |
| ScienceIDE | Oxford | 让 agent 按专家定义的案例与验收标准把科学代码仓改造成可执行环境,用于 SFT、RL 与评测,并训出 PhAI-IDE 72B/9B/4B |
| Cordis | DeepSeek | DeepSeek Harness 底下的插件内核;把动态组合拆成时间(可完全撤销副作用)与空间(响应式依赖)两维并给出演算 |
| DR-Venus | AntGroup | 只用 1 万条开放数据训边缘规模深研 agent |
| Agent-World | ByteDance | 规模化合成真实环境,演化通用 agent(Seed,与人大合作) |
| Beyond-Stochastic-Exploration | Alibaba | agentic 搜索的训练数据凭什么有价值(阿里云) |
| WebWorld | Alibaba | 用网页世界模型当 agent 训练环境(Qwen) |
| Agent-World-Model | Snowflake | 无限合成环境做 agentic RL |
| MT-GRPO | Huawei | 多任务 GRPO,按最差任务加权(诺亚方舟) |
| AgentRL | Tsinghua | 多轮多任务 agentic RL 框架(THUDM,Z.ai 同源) |
| MUA-RL | Meituan | 多轮用户交互的 agentic 工具使用 RL |
| MemAgent | ByteDance | 多轮 RL 训练的记忆 agent 处理长上下文(Seed) |
| WebSailor | Alibaba | 高不确定性 web agent 的数据合成与 RL(通义) |
| SimpleTIR | ByteDance | 端到端多轮工具集成推理 RL,过滤 void turn 稳定训练(TikTok) |
| BalanceSFT | AntGroup | 工具调用 SFT 的数据均衡(Inclusion AI AWorld) |
| ReTool | ByteDance | 代码解释器工具调用的 RL(Seed) |
| ToolACE | Huawei | 函数调用数据合成,小模型上 BFCL 榜首 |
| Voyager | NVIDIA | Minecraft 里的终身学习 agent:技能库 + 自动课程 |
| Toolformer | Meta | 模型自学何时调 API 的自监督方法 |
| WebGPT | OpenAI | 浏览器辅助问答,人类反馈训练的早期 web agent |
| Fara-1.5 | Microsoft | 电脑操作 agent 的可扩展学习环境与小模型数据配方;前作 Fara-7B 见 arXiv 2511.19663 |
| CodeMidas | Xiaomi | 只用源码本身造编码 RL 环境:agent 探索已实现功能、写行为规格与测试并反复验证,得到 3,185 个仓库的 5,545 个任务 |

## 推理服务与架构探索

| 报告 | 公司 | 一句话 |
|---|---|---|
| DSpark | DeepSeek | 半自回归草稿 + 置信度调度验证,投机解码 |
| Slicing-and-Dicing-MoE | Washington | MoE 配置的系统性搜索(158 页,含大量附录) |
| DFlash | UCSD | 块扩散做并行草稿的投机解码,ICML 2026 |
| Engram | DeepSeek | 可扩展查表式条件记忆,稀疏的新维度(依据 arXiv v2) |
| mHC | DeepSeek | 流形约束的超连接 |
| LiquidGEMM | ByteDance | W4A8 GEMM 内核的硬件高效实现(Seed,与上交合作) |
| DeepSeek-V3-Insights | DeepSeek | 从 V3 回看硬件与模型协同设计的取舍,ISCA 2025;与 DeepSeek-V3 报告互补不重复 |
| Mooncake | Moonshot | Kimi 的 KVCache 中心 PD 分离服务架构 |
| Sarathi-Serve | Microsoft | 分块预填充 + 无停顿调度,吞吐与延迟兼得(MSR India) |
| DistServe | Peking | 把 prefill 与 decode 拆到不同实例,按 goodput 优化 LLM 服务 |
| DeepSeekMoE | DeepSeek | 细粒度专家 + 共享专家,专家特化的出处;DeepSeek-V2 报告沿用 |
| SGLang | Berkeley | RadixAttention 前缀缓存与结构化输出的服务框架(LMSYS) |
| Splitwise | Microsoft | 把 prompt 计算与 token 生成拆到异构 GPU,按阶段配硬件 |
| PagedAttention | Berkeley | vLLM 的论文,KV cache 的分页管理 |
| Efficient-Large-Scale-MoE | Meta | MoE 与 dense 在零样本与微调上的对照研究(2022) |
| Switch-Transformers | Google | top-1 路由把 MoE 扩到万亿参数 |
| LMetric | Alibaba | 调度与 KV 命中率只用一个乘法指标就赢,OSDI 2026(SJTU IPADS 与阿里云) |
| ECHO | Huawei | 稀疏注意力模型的 KV 换页为什么仍然打不满带宽,OSDI 2026 |
| Strata | Stanford | 分层上下文缓存的碎片化把长上下文服务拖成 I/O bound(arXiv 2508.18572,OSDI 2026) |
| NanoFlow | Washington | 整卡吞吐靠算子内并行而非实例内 batching(arXiv 2408.12757,OSDI 2025) |
| EAGLE-3 | SafeAILab | 用训练期测试把草稿模型从特征拟合换成多层特征(arXiv 2503.01840) |
| XGrammar | CMU | 约束解码的上下文无关文法执行开销吃掉整条延迟(arXiv 2411.15100,MLSys 2025) |
| KTransformers | Tsinghua | CPU 与 GPU 混合推理 MoE,把内存当一层慢显存用(SOSP 2025) |

## 分布式训练与并行

| 报告 | 公司 | 一句话 |
|---|---|---|
| Ultra-Scale-Playbook | HuggingFace | 512 卡上 4000+ 次实验测出的并行训练手册;**原件是网页**(依据 2026-09-07 线上版,页面自报发布日 2025-02-19),本地打印件仅作 2025-02 的冻结快照 |
| Zero-Bubble-Pipeline-Parallelism | SeaAILab | 拆分反向传播消除流水线气泡,DualPipe 的前作 |
| ZeRO-Infinity | Microsoft | 把 NVMe 与 CPU 内存纳入训练,突破 GPU 显存墙 |
| Megatron-LM-2 | NVIDIA | 张量 + 流水 + 数据并行组合(PTD-P),万卡训练万亿参数 |
| GShard | Google | 条件计算 + 自动分片,MoE 规模化的出处 |
| ZeRO | Microsoft | 优化器状态、梯度、参数三级切分 |
| Megatron-LM | NVIDIA | 张量并行的出处,层内切分训练十亿级模型 |
| GPipe | Google | 微批流水线并行的出处 |
| MegaScale | ByteDance | 万卡训练的故障定位与通信优化实战;续作 MegaScale-MoE、MegaScale-Omni 同系列并入(arXiv 2402.15627) |
| FlashRecovery | iFlytek | 4800 卡故障恢复压到 150 秒(与 USTC、华为合作,arXiv 2509.03047) |
| Tessera | Alibaba | 异构 MoE 千卡流水线的负载均衡与 bubble(OSDI 2026) |
| Syncopate | UCSD | 多卡 kernel 里通信成为一阶瓶颈:chunk-centric 自动重叠(arXiv 2601.20595,OSDI 2026) |
| Flexible-Context-Parallelism | Huawei | 数据异构下上下文并行的严重负载不均(arXiv 2602.21788) |
| Pangu-Ultra | Huawei | 在 Ascend NPU 上训 dense 大模型的全栈工程;MoE 版见 arXiv 2505.04519 |
| NVFP4-Pretraining | NVIDIA | 4-bit 浮点做预训练时梯度溢出与缩放怎么治(arXiv 2509.25149) |
| Triton-distributed | ByteDance | 在 Triton 里写跨卡重叠 kernel,对标 DeepEP(arXiv 2504.19442;06 侧另列了收藏项目) |

## 多模态理解与 Omni

| 报告 | 公司 | 一句话 |
|---|---|---|
| MiMo-V2.6 | Xiaomi | 全模态 MiMo-V2.6 系列的 RL 扩规模报告:每步 1,568 条样本、上下文到 1M 的异步训练,环境覆盖代码、通用、视觉与网安,冻结 MoE 路由并多层防奖励黑客 |
| Thinking-with-Visual-Primitives | DeepSeek | 把坐标当思考的最小单位;**官方仓库已删除**,原件只剩本地件与社区镜像 |
| GLM-5V-Turbo | Z.ai | 把感知放进决策回路,而不是当输入接口 |
| Qwen3.5-Omni | Alibaba | Thinker/Talker 全模态,难点在流式与延迟 |
| Kimi-K2.5 | Moonshot | 视觉与文本联合优化,加上 Agent Swarm 的并行编排 |
| ERNIE-5.0 | Baidu | 2.4T 原生全模态,统一理解与生成;ERNIE 5.1 是纯文本版、无独立报告 |
| LongCat-Flash-Omni | Meituan | 560B 全模态,实时音视频交互 |
| DeepSeek-OCR | DeepSeek | 把文字渲染成图,用视觉 Token 换文本 Token |
| Qwen3-VL | Alibaba | Qwen 视觉理解线,dense 与 MoE 双形态 |
| Qwen3-Omni | Alibaba | 证明「全模态不退化可以做到」的实证工作,Qwen3.5-Omni 的前作 |
| Qwen2.5-Omni | Alibaba | Thinker-Talker 全模态架构的出处 |
| Chameleon | Meta | 早期融合的混合模态基模,图文统一 token(FAIR) |
| Flamingo | Google | 冻结视觉编码器 + 冻结 LM 的少样本视觉语言模型(DeepMind) |
| SenseNova-U1 | SenseTime | NEO-unify 架构统一多模态理解与生成;商汤此前两库零收录(arXiv 2605.12500) |
| LongCat-Next | Meituan | 把各模态词法化为离散 token 的统一基模(arXiv 2603.27538) |
| DeepSeek-OCR-2 | DeepSeek | 用视觉因果流做上下文压缩的下一代(arXiv 2601.20552) |
| GLM-4.5V | Z.ai | 用可扩展的多阶段 RL 训视觉推理模型(arXiv 2507.01006) |
| Keye-VL-2.0 | Kuaishou | 长视频与多图理解的 VLM 基模;1.0 见 arXiv 2507.01949 |
| Molmo2 | Ai2 | 开放权重 VLM:视频理解加指点接地的数据机器(CVPR 2026,arXiv 2601.10611) |
| DINOv3 | Meta | 稠密特征与 42 亿图蒸馏;日常研读侧只收了 DINOv2(arXiv 2508.10104) |
| SAM-3 | Meta | 按概念提示做分割与追踪;日常研读侧只收了 Segment Anything(arXiv 2511.16719) |

## 图像、视频与 3D 生成

| 报告 | 公司 | 一句话 |
|---|---|---|
| Qwen-Image-2.0-RL | Alibaba | 在 Qwen-Image-2.0 上做 RLHF + on-policy 蒸馏;组合奖励模型、GRPO 框架与混合 CFG,最后用 OPD 合并 T2I 与编辑两条策略 |
| World-Tracing | WorldLabs | 生成像素对齐的几何,超出可见范围 |
| Seed3D-2.0 | ByteDance | 仿真可用的高保真 3D 生成,统一 PBR 模型(Seed) |
| Causal-Forcing | ShengShu | 自回归扩散蒸馏的正确做法,实时交互视频(生数,与清华合作) |
| MOVA | OpenMOSS | 音视频同步生成,非对称双塔 |
| TRELLIS-2 | Microsoft | 原生紧凑的结构化 3D 潜表示(O-Voxel) |
| Adversarial-Flow-Models | ByteDance | 对抗式流模型(Seed) |
| HunyuanVideo-1.5 | Tencent | 8.3B 视频 DiT,把成本压到单卡跑得动 |
| Kandinsky-5.0 | Sber | 图像与视频的基座模型家族 |
| HunyuanImage-3.0 | Tencent | 原生多模态图像生成 |
| MixGRPO | Tencent | 混合 ODE-SDE 让流模型 GRPO 提速(混元) |
| HunyuanWorld-1.0 | Tencent | 可探索、可交互的 3D 世界生成 |
| AAPT | ByteDance | 自回归对抗后训练做实时交互视频生成(Seed) |
| Self-Forcing | Adobe | 自回归视频扩散的训推差距:用自己的输出做条件 |
| Flow-GRPO | Kuaishou | 在线 RL 训练流匹配模型(可灵,与港中文合作) |
| SpatialLLM | JohnsHopkins | 3D 空间智能的多模态模型设计(CVPR 2025 highlight) |
| Global-Local-Tree-Search | BUPT | VLM 全局-局部树搜索做室内 3D 场景(CVPR 2025) |
| FirePlace | Google | LLM 常识加几何约束做 3D 物体摆放(CVPR 2025 highlight;DeepMind) |
| CAST | ShanghaiTech | 单张 RGB 做组件对齐的 3D 场景重建(SIGGRAPH 2025 Best Paper) |
| Janus-Pro | DeepSeek | 解耦视觉编码:看图和画图不共用一只眼睛 |
| Hunyuan3D-2.0 | Tencent | 高分辨率带纹理 3D 资产生成 |
| LayoutVLM | Stanford | VLM 可微优化 3D 布局(CVPR 2025) |
| TRELLIS | Microsoft | 结构化 3D 潜表示,一套潜变量出多种 3D 格式 |
| Video-3D-LLM | CUHK | 把 3D 场景当视频,位置感知表示(CVPR 2025) |
| Transfusion | Meta | 一个模型同时做 next-token 与扩散 |
| Stable-Diffusion-3 | StabilityAI | MMDiT + 校正流的文生图基模报告 |
| FlexiCubes | NVIDIA | 可微等值面提取,基于梯度的网格优化 |
| Seedance-1.0 | ByteDance | 视频基模的质量、效率、可控性三角与蒸馏;1.5 pro 与 2.0 同系列并入(arXiv 2506.09113) |
| Wan | Alibaba | 开源视频基模的规模、数据与算力配方(arXiv 2503.20314;判定档案只拒过 Wan 2.7 的无原件) |
| Qwen-Image-2.0 | Alibaba | 图像基模的母体,库内只收了它的 RL 续作(arXiv 2605.10730) |
| LongCat-Video | Meituan | 开源视频生成的全流程技术报告(arXiv 2510.22200) |
| Seedream-4.0 | ByteDance | 统一文生图与图像编辑的多模态生成(arXiv 2509.20427) |
| Kling-Omni | Kuaishou | 统一多任务的视频生成框架(arXiv 2512.16776) |
| Movie-Gen | Meta | 视频、图像、个性化与音频四件套媒体基模;Meta 生成线此前零收录(arXiv 2410.13720) |

## 音频

| 报告 | 公司 | 一句话 |
|---|---|---|
| Step-Audio-R1.5 | StepFun | 听觉领域的思维链推理 |
| StepAudio-2.5 | StepFun | 一个骨干带 ASR/TTS/Realtime 三种特化 |
| Step-Audio-2 | StepFun | 带检索增强的音频理解与对话 |
| Kimi-Audio | Moonshot | 同一段音频同时走离散语义 token 与连续 Whisper 特征;并行生成、垫 6 个 blank 延迟起声(依据 arXiv v1,26 页) |
| Moshi | Kyutai | 全双工实时语音对话基模 |
| DAC | Descript | 改进 RVQGAN 的高保真音频压缩 |
| VALL-E | Microsoft | 把 TTS 当编解码语言建模,零样本音色克隆 |
| EnCodec | Meta | 神经音频编解码器 |
| Whisper | OpenAI | 大规模弱监督的鲁棒语音识别 |
| AudioLM | Google | 用语言模型方法生成音频 |
| Voxtral | Mistral | 开源多语语音-文本统一基模;Mistral 音频线零收录(arXiv 2507.13264) |
| MiMo-Audio | Xiaomi | 音频语言模型的 few-shot 学习能力(arXiv 2512.23808) |

## 世界模型与 Agent

| 报告 | 公司 | 一句话 |
|---|---|---|
| JEPA-Anything | CUHK | 正交预测分解把 JEPA 的潜目标拆成互补因子分路学习再合并,同一套框架跑视觉、生物、临床、控制、分子动力学、物理场与天气七个领域 |
| Real-Time-EXPO-FT | Stanford | VLA 推理延迟让观测过时:大 VLA 慢慢出动作块,轻量编辑策略按最新观测快速改动作,在此之上做 RL 微调 |
| Fugu | Sakana | 动态编排 agent 脚手架的编排器模型 |
| AI-Co-Mathematician | Google | 数学研究的 agentic 工作台(DeepMind) |
| HunyuanWorld-2.0 | Tencent | 重建、生成、模拟三合一的 3D 世界模型 |
| pi0.7 | PhysicalIntelligence | 可操控的通用机器人基模,涌现能力 |
| Matrix-Game-3.0 | Skywork | 实时流式交互世界模型,长程记忆 |
| V-JEPA-2.1 | Meta | 解锁视频自监督的稠密特征(FAIR) |
| World-Guidance | ByteDance | 在条件空间做世界建模来生成动作(Seed) |
| DreamDojo | NVIDIA | 大规模人类视频训练的通用机器人世界模型 |
| Infinite-World | Meituan | 无位姿层级记忆,交互世界模型扩到 1000 帧(与南开合作) |
| DreamZero | NVIDIA | 世界动作模型即零样本策略 |
| LingBot-World | AntGroup | 开源世界模型(蚂蚁灵波 Robbyant) |
| HunyuanWorld-1.5 | Tencent | HunyuanWorld-1.0 的可交互续作(HY-WorldPlay) |
| VL-JEPA | Meta | 视觉语言联合嵌入预测架构(FAIR) |
| ToolOrchestra | NVIDIA | 模型与工具的高效编排,把智能从「更大模型」挪到「更会调度」 |
| Dreamer-4 | Google | 在可扩展世界模型里训练 agent(DeepMind) |
| V-JEPA-2 | Meta | 自监督视频模型做理解、预测与规划(FAIR) |
| Cosmos | NVIDIA | Physical AI 的世界基础模型平台 |
| Genie | Google | 无动作标注的纯视频里,学出可交互的潜动作 |
| DreamerV3 | Google | 一套超参掌握多样控制任务(DeepMind,Nature) |
| DexTouch-WM | HKUST | 人手与灵巧手共用一套触觉阵列与动作表示,让人类触觉交互数据监督同一个动作条件世界模型,联合预测未来 RGB 与双手触觉 |

## 自进化系统

本组已发布卡片按首发日从新到旧;未解读排在后面,发布后插到对应日期位置。一句话里仍标明「哪个部件在自进化」。已收录的邻近条目:`AIDE2`(Weco,harness)、`Fugu`(Sakana,编排器)、`AgentEvolver`(Alibaba,任务/课程)、`AReaL-2.0`(AntGroup,演化控制面)。

| 报告 | 公司 | 一句话 |
|---|---|---|
| SoL-Pi | NVIDIA | 在 harness 层递归扩展自动研究循环,筛出动作执行、上下文压缩、观测处理与委托阅读四个机制;EdgeBench 51 题上性能与 Pi 相当,token 流量降 44.7–49.0% |
| AIDE2 | Weco | autoresearch 套 autoresearch,递归自我改进的首份实验证据;**原件是官方博客,`papers/` 下无 PDF** |
| Reward-Free-Self-Evolution | Tencent | 部件:任务/课程。通过世界知识探索做无奖励的自发自进化 |
| Dr-Zero | Meta | 部件:任务/课程。无训练数据的自进化搜索 agent(MSL) |
| R-Zero | Tencent | 部件:任务/课程。challenger 与 solver 共进化,ICLR 2026(腾讯西雅图 AI Lab) |
| GEPA | Berkeley | 部件:Prompt。反思式提示词进化,DSPy 生态(Databricks/Stanford/MIT 合作) |
| SEAL | MIT | 部件:模型权重。让模型自己生成「自编辑指令」来更新权重 |
| Darwin-Godel-Machine | Sakana | 部件:harness。agent 改写自身代码 + 基准存档做开放式进化(UBC 合作) |
| Alita | Princeton | 部件:工具/技能库。最小预定义,自己造 MCP 工具(清华合作) |
| AlphaEvolve | Google | 部件:算子/infra。进化搜索出的 kernel 反过来加速训练它自己的模型;**原件是白皮书,非 arXiv** |
| Absolute-Zero | Tsinghua | 部件:任务/课程。零外部数据,出题者与解题者自博弈(BIGAI 合作) |
| Agent-Workflow-Memory | CMU | 部件:记忆/经验。从轨迹里归纳可复用的工作流 |
| ADAS | UBC | 部件:harness。外层 agent 用代码搜索内层 agent 系统(Meta/Vector 合作) |
| Self-Rewarding-LM | Meta | 部件:模型权重。模型兼任奖励模型,自生成数据再训练 |
| PromptBreeder | Google | 部件:Prompt。DeepMind,自指涉的提示词进化搜索 |
| ScientistTwo | Google | 全自主多 agent 科研框架:建基线、提假设、跑实验与自动消融,再用模拟评审反驳闭环验证结论 |

## 可解释性与对齐

| 报告 | 公司 | 一句话 |
|---|---|---|
| Chain-of-Thought-Monitorability | UK-AISI | 十余家机构联署的立场文:思维链在 RL 阶段是不受直接监督的潜变量,所以读得出来;四条训练与架构路径都能把它磨掉,而作者自认全篇零一手实验数字(依据 arXiv v2,11 页) |
| CoT-Necessary-Evade-Monitors | Google | 把「能不能躲开读思维链的监控器」改写成「要外部给多少帮助才躲得开」:546 题筛出 59 道 CoT 必需的题、三个环境逐个压力测试,前两个躲不开,第 7 节用 10,000 步 RL 真躲开了——办法是外化一个错的中间值(依据 arXiv v1,70 页) |
| CoT-Unfaithful-Reasoning | Anthropic | 把「说出来」量成百分比:四台模型 × 六类提示线索,推理模型的平均忠实性只有 25%(Claude 3.7)与 39%(R1),题一难再掉 44%/32%;结果监督 RL 先抬高再停滞,奖励黑客练到 >99% 会用线索时说出率不跟着涨(依据 arXiv v1,18 页) |
| On-the-Biology-of-a-LLM | Anthropic | 给上线模型画「接线图」:3000 万特征的跨层转码器加归因图,十个案例逐个用干预实验兑现;作者自报只有约四分之一的提示能得到可用的图(**原件是网页**,依据 transformer-circuits.pub 2025-03-27) |
| CoT-Monitoring-Obfuscation | OpenAI | 同一批环境里先量出「读思维链抓作弊」联合召回 95%(只看动作只有 60%),再把监控器接进奖励函数,召回塌到接近零;作者据此主张交一笔可监督性税(依据 arXiv v1,39 页) |
| Alignment-Faking | Anthropic | 把生产模型放进「你正被训练成不再拒绝」的虚构情境:免费用户 14% 顺从、付费几乎从不,隐藏思维链里明写「这次先照做,好让以后还能拒绝」;RL 把这段推理从 12% 抬到 78%,而论文全程只用显现偏好口径(依据 arXiv v2,137 页) |
| Gemma-Scope | Google | 把「训一整套全层全切面 SAE」这件贵到别人做不起的事做完再开源:2B 与 9B 每层每切面、27B 只有选中的三层,权重超过 2,000 个 SAE,连评测口径与宽度梯子一起交出去;不裁决架构之争,也不给任务级结果(依据 arXiv v2,26 页) |
| WildGuard | Ai2 | 7B 的三任务审核器(提问恶意意图 / 回复风险 / 拒答判定):86,759 条训练配方逐桶可回算、5,299 条由三人独立标注的考卷;对抗性提问 85.5 对最佳开源 74.5,拒答判定 88.6 仍低于 GPT-4 的 92.4(依据 arXiv v3,38 页) |
| CoT-Empowers-Serial-Problems | Stanford | 给 CoT 的能力增益称重量:常数深度加立刻舍入的 Transformer 连 O(log n) 步思维链也只能算 AC0(定点数那档是 TC0),而 T 步思维链配 Θ(log n) 嵌入就能算任意规模 T 的电路;唯一的严格分离要假设 TC0 ⊊ NC1,且构造里的思维链是 0/1 门真值、人类读不懂(依据 arXiv v4,38 页) |

## 检索与 RAG

| 报告 | 公司 | 一句话 |
|---|---|---|
| Qwen3-Embedding | Alibaba | 用基模造 embedding 与 reranker 的全流程配方(arXiv 2506.05176) |
| BGE-M3 | BAAI | 多语、多粒度、多功能的自蒸馏嵌入(ACL 2024 Findings,arXiv 2402.03216) |
| jina-embeddings-v4 | Jina | 单模型统一图文多语检索与多向量重排(ACL 2025,arXiv 2506.18902) |

## 评测与 Benchmark

| 报告 | 公司 | 一句话 |
|---|---|---|
| MMLU-Pro | TIGER-Lab | 把 MMLU 失效拆成地板太高、不考推理、数据有噪声三处分别修:干扰项扩到 10 个(83% 的题)、too-easy 靠 8 个小模型投票滤掉 42.23%,12,032 题 14 学科(依据 arXiv v6,24 页) |
| WildBench | Ai2 | 考什么交给真实用户、怎么判交给一张 5 到 10 问的清单:1,024 题、用三个不同水平的基线合成 WB-Reward,与 Arena 人类 Elo 的头部 Pearson 0.984(依据 arXiv v2,19 页) |
| SWE-bench | Princeton | 2,294 道真实 GitHub 缺陷:已合并且自带测试的 PR 同时给出题目、答案与验收标准;判分是 F2P 与 P2P 的与运算,当年最好的模型只解出 1.96%(依据 arXiv v3,52 页) |
| Lessons-from-the-Trenches | EleutherAI | 可复现评测的方法学教训:打分口径怎么会错(arXiv 2405.14782) |
| PosteriorBench | Caltech | 评生成式逆问题求解器要看整个后验而非单个样本:四个物理逆问题配高精度参考后验与五项分布指标 |

# 日常研读库存

按方向分组的材料索引,是 `/readings` 页面分组与排序的唯一数据源;规则见 [docs/10-材料解读流程.md](../docs/10-材料解读流程.md) 第八节。

- `## 方向` 既是页面分组,**也是目录名**——方向标题与 `readings/<方向>/` 逐字一致,测试守双向一致
- 方向名与 [`reports/index.md`](../reports/index.md) 共用同一组,新增方向要两边同时加
- 「材料」等于 `readings/<方向>/<材料>.md` 去掉 `.md`;「机构」只是元数据,做卡片条幅,**不建目录**
- 行序即页面顺序。每个方向的已发布行按首发日从新到旧,同日按材料名字面值升序;未解读排在已发布后面,发布后插到对应日期位置
- 状态由文件推导,不手写:正文存在为已发布,`_<材料>.md` 为施工中,否则未解读;`npm run readings:status` 现场打印
- 网页原件与转录文本原件在 `readings/_src/` 下可能没有对应文件,在一句话里注明并留 URL 线索

## 语言基模

| 材料 | 机构 | 一句话 |
|---|---|---|
| Kaplan-Scaling-Laws | OpenAI | 损失随参数、数据与算力走幂律,跨七个数量级;结论是大模型更省样本,算力该压在模型规模上。arXiv 2001.08361,原件待取 |
| Chinchilla | GoogleDeepMind | 同算力下参数与 token 应等比例放大:70B 配 1.4T token 打赢 280B 的 Gopher,直接修正 Kaplan 的配比结论。arXiv 2203.15556,原件待取 |

## 注意力与长上下文

| 材料 | 机构 | 一句话 |
|---|---|---|
| Topological Trouble With Transformers | GoogleDeepMind | Transformer 用扩张的上下文历史编码结构,这套表示在拓扑上存在固有困难 |
| Attention-Sink-Survey | Tsinghua | **综述**:attention sink 怎么被利用、怎么被解释、怎么被消除,三支各自的代表工作 |
| Latent-Space-Survey | NUS | **综述**:潜空间的基础、演化、机制、能力与展望 |
| TransMLA | PKU | 把已有的 GQA 模型等价转成 MLA,不重训就拿到 DeepSeek 那套 KV 压缩 |
| Mamba-2 | Princeton | 证明 Transformer 与 SSM 是对偶的,并给出结构化状态空间的高效算法 |
| GLA | MIT | 带门控的线性注意力,配一套硬件高效的分块并行训练 |
| Mamba | CMU | 选择性状态空间:让 SSM 的参数随输入变化,线性时间对上 Transformer 的质量 |
| StreamingLLM | MIT | attention sink 的提出者;保住开头几个 token 就能让窗口注意力无限外推,不微调也不崩 |
| YaRN | NousResearch | 分频段插值 RoPE 做窗口扩展,少量微调就能拉长上下文 |
| GQA | Google | 多个 query 头共享一组 KV 头,夹在 MHA 与 MQA 之间换 KV cache |
| O1-Memory-Attention | Google | 注意力的内存可以做到常数级,为 FlashAttention 那条路铺前提 |
| RoPE | Zhuiyi | 旋转位置编码:绝对位置进旋转角,相对距离出现在内积里 |
| Linear-Attention | Idiap | 用核技巧把 softmax 注意力换成线性形式;自回归推理时它等价于一个 RNN |
| Huginn | ELLIS | 循环深度做潜空间测试时算力。arXiv 2502.05171,来自 Latent-Space-Survey |
| Reasoning-by-Superposition | Berkeley | 连续 CoT 叠加态的形式解释。arXiv 2505.12514,来自 Latent-Space-Survey |
| SoftCoT-Plus-Plus | NTU | 连续嵌入里多条并行路径做测试时缩放。arXiv 2505.11484,来自 Latent-Space-Survey |
| Cache-to-Cache | Tsinghua | KV-cache 投影融合,协作从文本信道改到潜信道。arXiv 2510.03215,来自 Latent-Space-Survey |
| Value-State-Gated-Attention | AntGroup | 在 value 上先门,打断注意力–value 抽干循环。arXiv 2510.09017,来自 Attention-Sink-Survey |
| LatentMAS | UIUC | 多智能体共享潜工作记忆。arXiv 2511.20639,来自 Latent-Space-Survey |

## 训练方法与强化学习

| 材料 | 机构 | 一句话 |
|---|---|---|
| On-Policy Self-Distillation | UCSD | 无监督的 on-policy 自蒸馏,后训练不再依赖外部标注 |
| Rethinking-OPD | Tsinghua | 系统查 on-policy 蒸馏的训练动力学:成败取决于师生思维模式是否兼容 |
| Self-Distillation-Zero | Princeton | 让模型自我修订,把 RLVR 的二值奖励变成 token 级的稠密监督,不需要外部教师 |
| SRPO | CASIA | 用样本路由把 GRPO 的组内相对与自蒸馏的 token 级监督合到一起,补上 GRPO 粗粒度信用分配的短板 |
| Graph-of-States | Nankai | 面向溯因推理的状态图框架:CoT 与 ToT 擅长演绎,从不完整观测反推假设是另一回事 |
| GOPD | Renmin | 用奖励外推让学生越过教师,而不是只逼近教师 |
| Exploration-vs-Exploitation | Columbia | 从截断、熵与虚假奖励三个口子重看 RLVR 的探索-利用权衡 |
| EGGROLL | Oxford | 低秩扰动让进化策略在 GPU 上重新变得算得起;naive ES 卡在批量矩阵乘的低算术强度上 |
| Agent-Data-Protocol | CMU | 给 agent 训练数据定一套统一协议,让散落各处的轨迹能互通 |
| FlowRL | SJTU | 用流平衡去匹配完整的奖励分布,而不是最大化奖励;保住少见但正确的推理路径 |
| RL-for-LRM-Survey | Tsinghua | **综述**:RL 如何把 LLM 变成 LRM,以及这条路上的算法、数据与基建全景 |
| RLVR | MSRA | 可验证奖励的强化学习:用规则判对错取代人类偏好,长链推理的主流范式 |
| RM-R1 | UIUC | 把奖励建模本身当成推理任务来做,评分前先写出理由 |
| Criticize-RLVR | Tsinghua | 质疑 RLVR:大 k 下 pass@k 显示它并没有拓宽基座模型的推理边界,只是把分布收窄 |
| Down-Sampling-Rollouts | CMU | rollout 生成易并行、策略更新吃通信,所以多采样再下采样,把这对不对称吃掉 |
| VinePPO | Mila | 用蒙特卡洛回溯做逐步信用分配,替掉学出来的价值网络 |
| Magpie | Washington | 只喂对齐模板的前缀让模型自己续写,把指令数据从对齐模型里「抽」出来 |
| DEITA | ShanghaiTech | 系统研究指令微调的数据选择:复杂度、质量、多样性三个维度怎么自动打分 |
| MiniLLM | Tsinghua | 白盒 LLM 蒸馏:把正向 KL 换成反向 KL,学生不再去覆盖教师分布的长尾 |
| DPO | Stanford | 把 RLHF 的两阶段折叠成一个分类损失,不再需要显式奖励模型与在线采样 |
| QLoRA | Washington | 4-bit NF4 量化底座加 LoRA,单卡微调 65B;配 double quant 与 paged optimizer |
| LLaMA-Adapter | ShanghaiAILab | 零初始化的门控注意力适配器,一小时内把 LLaMA 调成指令模型 |
| Self-Instruct | Washington | 让模型自己造指令数据再训自己,指令微调的数据瓶颈第一次被绕开 |
| IA3 | UNC | 只学三组缩放向量去抑制或放大内部激活,参数量比 LoRA 还小一个量级 |
| STaR | Stanford | 用答对的推理链回头训自己,答错的给出答案再让它补理由 |
| Self-Consistency | Google | 采样多条推理链再投票,比贪心解码稳得多 |
| LoRA | Microsoft | 把权重更新约束成低秩矩阵,微调只训那两个小矩阵;PEFT 的事实标准 |
| Prompt-Tuning | Google | 只训输入端的软提示;模型越大,它与全量微调的差距越小 |
| Prefix-Tuning | Stanford | 冻住模型,只训一段可学的前缀向量接在每层 KV 前面 |
| PPO | OpenAI | 用截断的重要性比率做信赖域近似;RLHF 十年的默认算法 |
| RLHF | OpenAI | 人类偏好训奖励模型再用 RL 优化;InstructGPT 那条主线 |
| Evolution-Strategies | OpenAI | 黑盒进化策略当 RL 的替代:不用反传,极易并行 |
| RL2 | Berkeley | 把 RL 算法本身学进 RNN 的隐状态,元学习式的快速适应 |
| Knowledge-Distillation | Google | 知识蒸馏原始论文:用教师的软标签带温度去教学生 |
| Tulu-3 | AllenAI | 用程序化核对器换奖励模型,综述把 RLVR 这个名字钉在这篇。arXiv 2411.15124,来自 RL-for-LRM-Survey |
| ProRL | NVIDIA | 足够长、足够稳的 RL 能否把推理边界推过基座。arXiv 2505.24864,来自 RL-for-LRM-Survey |

## Agent 训练与工具使用

| 材料 | 机构 | 一句话 |
|---|---|---|
| RAGEN-2 | Northwestern | 多轮 agent 的 RL 训练天生不稳;这一篇盯住「推理坍缩」这个具体失效 |
| HGPO | NTU | 组的层级化:长时程 agentic 任务里逐步分组仍太粗,改成分层的组 |
| Agentic-Reasoning-Survey | UIUC | **综述**:LLM 在封闭世界推理很强、开放动态环境里不行;agentic reasoning 把思考与行动接起来 |
| Agent-Memory-Survey | Renmin | **综述**:AI agent 时代的记忆——存什么、怎么取、怎么忘 |
| Belief-Deviation | CUHK | 主动推理要 agent 边问边收集信息;把信念偏移压下去才不会越问越偏 |
| Tune-the-Environment | Inclusion AI | 高质量 agent 数据太稀缺,SFT 过拟合、RL 冷启动难,那就去调环境而不是调 agent |
| ARPO | Renmin | 面向多轮工具调用的策略优化,快手合作;仓库 dongguanting/ARPO |
| Tool-Star | Renmin | 多工具协同推理的 RL 框架:一次推理里自主调度多个外部工具 |
| GiGPO | NTU | 组中组:在 episode 组之外再按步分组,解决多轮里稀疏延迟奖励的逐步信用分配 |
| When2Call | Harvard | 评测「什么时候**不该**调工具」——已有 benchmark 只看调得准不准 |
| RAGEN | Northwestern | 多轮 RL 下 agent 自进化的系统研究,StarPO 框架 |
| ToolRL | UIUC | 工具学习的关键在奖励设计;SFT 学到的工具能力泛化不出去 |
| TORL | SJTU | 直接从基座模型起步做工具集成 RL,让模型自己发现调用策略 |
| MAST | Berkeley | 1600+ 条标注轨迹跨 7 个框架,归纳多智能体系统的失败模式分类 |
| Search-R1 | UIUC | 用 RL 直接教模型在推理中途调搜索引擎,而不是靠提示 |
| ToolLLM | Tsinghua | 1.6 万真实 API 上的工具调用数据与模型,配 DFSDT 搜索 |
| ReAct | Princeton | 推理与行动交错:想一步、做一步、看结果再想;agent 提示范式的源头 |
| TALM | Google | 工具增强语言模型的早期形态:文本接口调工具,自举扩数据 |

## 推理服务与架构探索

| 材料 | 机构 | 一句话 |
|---|---|---|
| ReQAT | Hanyang | W4A4KV4 下推理精度不掉;发现 FP4 的错集中在数字与运算符这类低熵 token,用轨迹对齐 QAT 加选择性熵最小化补回来 |
| Pair-In-Pair-Out-MTP | Renmin | 一步出一个 token 让长推理链直接等价于长延迟;改成潜空间里成对进出的多 token 预测 |
| KDA | BAAI | **综述**:LLM 写 GPU kernel 先切「训模型」还是「搭 Agent」,不是先切 CUDA/Triton |
| CacheBlend | Chicago | RAG 场景下多段检索文本的 KV 各自缓存再融合,只重算少量交叉位置 |
| S-LoRA | Berkeley | 上千个 LoRA adapter 同时在线服务:统一分页显存加异构批处理 |
| CacheGen | Chicago | 把 KV cache 当流式媒体压缩传输,跨机复用长上下文 |
| H2O | UTAustin | 注意力分数高度集中在少数 token 上,按此淘汰 KV cache |
| AWQ | MIT | 按激活分布挑出关键权重通道加以保护的 4-bit 权重量化 |
| FlexGen | Stanford | 单卡跑大模型的高吞吐离线推理:在 GPU/CPU/磁盘之间解线性规划排布张量 |
| Speculative-Decoding | Google | 小模型起草、大模型一次并行验证,无损加速解码 |
| Orca | SNU | 迭代级调度的提出者:continuous batching 的原始论文,OSDI |
| Sparsely-Gated-MoE | Google | 稀疏门控 MoE 的源头:参数量涨千倍而单样本计算量不变 |
| Clipper | Berkeley | 通用的低延迟预测服务层,NSDI;缓存、自适应批处理与模型选择 |

## 分布式训练与并行

| 材料 | 机构 | 一句话 |
|---|---|---|
| PipeDream | MSR | 1F1B 流水线调度的出处:异步流水并行,气泡与显存的折中 |
| Tectonic | Meta | Meta 的 EB 级分布式文件系统,FAST;把多套专用存储合并成一套 |

## 多模态理解与 Omni

| 材料 | 机构 | 一句话 |
|---|---|---|
| MM-LLMs-Survey | Tencent | **综述**:多模态 LLM 的模块化拆解与近期进展 |
| ViT-Registers | Meta | ViT 特征图里的高范数伪影是模型在借 patch 当寄存器;显式给它几个 register token |
| LLaVA | Wisconsin | 一层投影把视觉特征接进 LLM,再用 GPT 生成的视觉指令数据微调 |
| DINOv2 | Meta | 无监督学出的通用视觉特征,不微调就能直接当下游特征用 |
| Segment Anything | Meta | 可提示分割的视觉基座 SAM,连带 SA-1B 数据引擎;分割任务被重述成 promptable 任务 |
| SigLIP | Google | 把 CLIP 的 softmax 对比损失换成逐对 sigmoid,小 batch 也能训好 |
| BLIP-2 | Salesforce | 冻住图像编码器与 LLM,只训中间的 Q-Former 做桥接 |
| BLIP | Salesforce | 理解与生成统一的图文预训练,配 CapFilt 自举清洗网络图文对 |
| MAE | Meta | 掩掉 75% 的 patch 再重建,视觉侧的自监督预训练 |
| Swin-Transformer | MSRA | 移位窗口的层次化 ViT,把线性复杂度与多尺度一起拿到 |
| CLIP | OpenAI | 四亿图文对的对比预训练,零样本迁移;此后所有多模态的地基 |
| ViT | Google | 把图像切成 16×16 的 patch 当 token 喂给纯 Transformer,视觉侧的架构统一 |
| MoCo | Meta | 动量编码器加队列维护大批负样本,对比学习的代表作 |
| InstructBLIP | Salesforce | 在 BLIP-2 上只更新 Q-Former 做指令感知视觉特征。arXiv 2305.06500,来自 MM-LLMs-Survey |
| NExT-GPT | NUS | 任意模态端到端,轻量对齐对抗工具级联误差。arXiv 2309.05519,来自 MM-LLMs-Survey |
| LLaVA-1.5 | Wisconsin | 投影换成 MLP,再加学术 VQA 与格式提示。arXiv 2310.03744,来自 MM-LLMs-Survey |
| VILA | NVIDIA | 线性投影、交错图文、纯文本指令回混。arXiv 2312.07533,来自 MM-LLMs-Survey |

## 图像、视频与 3D 生成

| 材料 | 机构 | 一句话 |
|---|---|---|
| SceneConductor | NTU | 单图生成完整 3D 场景要从有歧义的证据里推全局几何与物体关系;改用多智能体编排 |
| REST3D | CMU | 单图重建出**物理上站得住**的 3D 场景,直接可进仿真 |
| Points-to-3D | Adelaide | 以点云为先验的结构感知 3D 生成,分阶段采样先补全局几何再修边界 |
| SceneTransporter | Xi'anHiTech | 最优传输引导的组合式隐扩散:已有方法能出部件却分不清开放世界里的实例 |
| SceneMaker | Tsinghua | 开放集 3D 场景生成:去遮挡与位姿估计解耦,并构造了配套的开放集数据 |
| D4RT | GoogleDeepMind | 单个 transformer 从一段视频里同时推深度、时空对应与相机参数;核心是一套查询机制,绕开逐帧稠密解码,让模型按需探任意时空点的 3D 位置 |
| 3DSlim | SKKU | 3D 场景语言理解沿用语言模型的因果解码器并不合适,掩码方式才是关键 |
| GRPO-Guard | SYSU | flow matching 上做 GRPO 会隐式过优化——奖励涨而画质与对齐崩;用规范化截断兜住 |
| SceneGen | SJTU | 单图一次前向出整个 3D 场景:位置头同时给出资产与它们的相对空间位置 |
| OmniPart | HKU | 部件感知的 3D 生成:语义上解耦、结构上仍连成一体 |
| PartCrafter | PKU | 首个从单张 RGB 图直接联合生成多个语义部件网格的结构化 3D 模型 |
| MeshGen | Tsinghua | 带 PBR 纹理的网格生成;渲染增强的自编码器绕开 SDS 优化那套慢与模式坍缩 |
| CausVid | MIT | 把双向视频扩散蒸馏成因果模型,靠 KV cache 单卡流式出 9.4 FPS |
| MIDI | Beihang | 多实例扩散:把预训练的图生 3D 物体模型扩展成一次生成整个场景,不再逐物体分阶段做 |
| 3DTopia-XL | NTU | 用基元扩散扩大 3D 资产生成,针对优化速度与几何保真这两个老问题 |
| Diffusion-Forcing | MIT | 每个 token 独立的噪声水平,把 next-token 预测与全序列扩散接在一起 |
| CLAY | ShanghaiTech | 可控的大规模 3D 资产生成模型,几何与材质分开建模 |
| Holodeck | UPenn | 用自然语言指挥生成具身 AI 的 3D 环境 |
| 3D-Gaussian-Splatting | Inria | 显式的各向异性高斯加光栅化,把辐射场渲染拉到实时 |
| Objaverse-XL | AI2 | 一千万以上的 3D 物体,3D 生成的规模化数据底座 |
| DiT-3D | MBZUAI | 把 DiT 直接搬到 3D 点云形状生成上 |
| HSSD | GeorgiaTech | Habitat 的合成场景数据集,强调与真实房屋分布对齐 |
| Min-SNR | MSRA | 按信噪比给各时间步加权,把扩散训练当多任务优化来平衡 |
| Riemannian-Flow-Matching | Meta | 把 flow matching 推广到黎曼流形上 |
| 3DShape2VecSet | KAUST | 把 3D 形状编码成一组隐向量集合,供神经场与扩散模型共用 |
| DiT | Berkeley | 把扩散的 UNet 换成 Transformer,并证明它照样按 scaling law 走 |
| Flow-Matching | Meta | 直接回归条件概率路径的向量场,比扩散更简洁的连续归一化流训练 |
| Rectified-Flow | UTAustin | 把生成轨迹拉直成直线,少步采样的理论基础 |
| CFG | Google | 不用分类器也能做条件引导:有条件与无条件两次预测做外推 |
| ProcTHOR | AI2 | 程序化生成海量可交互室内环境,给具身智能提供训练场 |
| P2-Weighting | SNU | 按噪声水平重新加权训练损失,让模型把力气花在感知上要紧的那一段 |
| MaskGIT | Google | 并行掩码预测代替逐 token 自回归,图像生成快一个量级 |
| Latent-Diffusion | Heidelberg | 把扩散搬到 VAE 的隐空间里做,Stable Diffusion 的底子 |
| ABO | Berkeley | 来自真实商品的 3D 物体数据与基准,材质与多视角齐全 |
| VQ-GAN | Heidelberg | VQ 码本加对抗损失再接 Transformer,高分辨率图像的离散自回归生成 |
| Score-SDE | Stanford | 用随机微分方程统一 score matching 与扩散,给出连续时间视角 |
| 3D-FRONT | Alibaba | 带布局与语义的室内场景数据集,3D-FUTURE 的场景级配套 |
| DDIM | Stanford | 把扩散反向过程改成非马尔可夫的确定性采样,几十步就够 |
| 3D-FUTURE | Alibaba | 带纹理的家具 3D 形状数据集 |
| DDPM | Berkeley | 去噪扩散概率模型,扩散路线真正跑起来的那一篇 |
| NeRF | Berkeley | 用一个 MLP 表示场景的辐射场,体渲染出新视角;神经场这条路的起点 |
| VQ-VAE | DeepMind | 把连续隐空间量化成离散码本,图像因此能被当作 token 序列建模 |
| PixelCNN | DeepMind | 逐像素自回归建图像,视觉侧的自回归起点 |
| GAN | Montreal | 生成器与判别器对抗训练,生成模型十年的另一条主线 |
| VAE | Amsterdam | 变分自编码器:重参数化技巧让隐变量模型可以端到端反传 |

## 音频

| 材料 | 机构 | 一句话 |
|---|---|---|
| F5-TTS | SJTU | 用 flow matching 做非自回归 TTS,不需要时长模型与音素对齐 |

## 世界模型与 Agent

| 材料 | 机构 | 一句话 |
|---|---|---|
| StereoWorld | HKU | 相机条件的立体世界模型,只在 RGB 模态里同时学外观与双目几何 |
| Solaris | NYU | 现有动作条件视频模型只有单智能体视角;这一套在 Minecraft 里做多人一致的多视角模拟 |
| WoVR | CASIA | 不假设世界模型忠实,从模拟器、交互协议、对齐三层管制想象里的 VLA RL |
| Olaf-World | NUS | 动作标签稀缺逼人去学潜动作,但学出的潜变量常纠缠场景细节、跨情境迁不动 |
| LIVE | CUHK-Shenzhen | 用循环一致性目标压住长时程误差累积,不再需要教师蒸馏 |
| VLA-World-Model-Survey | Tongji | **综述**:按接到策略哪一侧切四刀——规划器、动作模型、合成器、仿真器 |
| Flow-Equivariant-WM | Harvard | 感官流与自身运动耦合成连续对称性;让世界模型对这些流等变,记忆才稳 |
| LeJEPA | Brown | 给 JEPA 一个可证明的目标,去掉那一堆防坍缩的启发式技巧 |
| RLVR-World | Tsinghua | 最大似然与世界模型真正在乎的转移预测指标不对齐,改用 RL 直接优化后者 |
| DINO-WM | NYU | 在冻结的 DINO 视觉特征上建世界模型,零样本规划不用重训策略 |
| JEPA-Position-Paper | ICFO | Les Houches 讲义:用能量模型与潜变量搭出 H-JEPA,不是 OpenReview 立场文本身 |
| WorldVLA | Alibaba | arXiv 2506.21539,来自 VLA-World-Model-Survey;观测与动作串成一条自回归世界模型 |
| DreamGen | NVIDIA | arXiv 2505.12705,来自 VLA-World-Model-Survey;先合成视觉轨迹再反推动作的数据引擎 |
| World-Env | SYSU | arXiv 2509.24948,来自 VLA-World-Model-Survey;把世界模型当 VLA 后训练虚拟环境 |
| Genie-Envisioner | AgiBot | arXiv 2508.05635,来自 VLA-World-Model-Survey;同一套世界基础平台跨规划/合成/仿真复用 |

## 自进化系统

| 材料 | 机构 | 一句话 |
|---|---|---|
| Self-Evolving AI Agents Survey | Glasgow | 自进化的一张分类地图:抽出一个统一的反馈回路框架,再把技术切成单 agent 优化、多 agent 优化、领域优化三支 |
| RSI Survey | UCR | 1250 篇语料切成两轴(改什么 × 回路闭合到什么程度),中心刀是「有界自我精修」对「开放式 RSI」,并指出治理级的度量是最空的一块 |
| Idea2Story | MIT | 把研究概念自动展开成完整科学叙事的流水线 |
| Self-Evolving-Agents-Survey-2 | Shanghai | **综述**:自进化 agent 的四问——进化什么、何时进化、怎么进化、在哪进化 |

## 可解释性与对齐

| 材料 | 机构 | 一句话 |
|---|---|---|
| Scaling Monosemanticity | Anthropic | 稀疏自编码器在 Claude 3 Sonnet 上规模化,抽出可解释、可干预的单义特征 |
| Refusal Direction | ETH | 拒答行为由残差流里的单一方向中介;删掉该方向就能定向解除拒答,加回去能诱发拒答 |

## 深度学习基石

1986–2017 的地基:循环网络、卷积、词向量、归一化、残差、注意力与优化器。它们不是「基模报告」,但今天每一条设计都踩在上面。

| 材料 | 机构 | 一句话 |
|---|---|---|
| ResNet | MSRA | 残差连接让上百层可训,此后所有深网络的默认组件 |
| BatchNorm | Google | 逐 batch 归一化中间激活,深层网络才敢用大学习率 |
| GRU | Montreal | **对照实验**:固定参数量下 tanh/LSTM/GRU 在音乐与语音上比;门控优于 tanh,两种门控无统一赢家 |
| Seq2Seq | Google | encoder-decoder 把变长序列映射到变长序列,机器翻译的范式转换 |
| Word2Vec-NegativeSampling | Google | word2vec 第二篇:负采样与层次 softmax 把训练拉到可负担,并处理短语 |
| Word2Vec | Google | CBOW 与 Skip-gram,词向量的起点 |
| AlexNet | Toronto | ImageNet 上让深度 CNN 一举确立地位;GPU 训练加 ReLU 与 dropout 的组合 |
| LSTM | TUMunich | 用门控与恒定误差流治住 RNN 的梯度消失,长程依赖第一次可训 |
| RNN | UCSD | 多层网的误差反传:隐层表征从任务误差里长出来,同一规则也能训迭代网 |
| Bahdanau-Attention | Montreal | 注意力机制的原始论文:解码时软对齐到编码器的每个位置 |
| Adam | Toronto | 一阶矩与二阶矩自适应的优化器,十年后仍是默认选择 |


## 检索与 RAG

把外部知识接进模型的那一支:稠密检索、后期交互、检索增强预训练与生成,以及长上下文里检索位置的影响。

| 材料 | 机构 | 一句话 |
|---|---|---|
| DPR | Meta | 双塔稠密检索取代 BM25,开放域问答的检索侧起点 |
| ColBERT | Stanford | 后期交互:先各自编码再做 token 级 MaxSim,兼顾双塔的快与交叉编码的准 |
| REALM | Google | 把检索器放进预训练一起端到端学,而不是事后接上去 |
| RAG | Meta | RAG 这个名字的出处:检索器与生成器联合,知识密集任务不再全靠参数记忆 |
| Atlas | Meta | 检索增强的少样本学习:小模型加检索能打过大得多的纯参数模型 |
| Lost-in-the-Middle | Stanford | 长上下文里的信息放在中间就会被忽略,呈 U 形;检索排序因此不是无所谓的 |
| RAG-Survey | TongjiUniversity | **综述**:RAG 从朴素到进阶再到模块化的三代划分,以及检索、增强、生成三段各自的技术谱 |


## 评测与 Benchmark

只评不训的那一类。收它们是因为**评测口径决定了大家在优化什么**——读一份 benchmark 等于读一次「这个方向认为什么算做对了」。

| 材料 | 机构 | 一句话 |
|---|---|---|
| MIND | CSU | 首个开放域闭环回访式基准,测世界模型的记忆一致性与动作可控性 |
| BFCL | Berkeley | 函数调用榜单从单次工具使用一路演进到 agentic 评测的完整记录 |
| tau2-bench | Sierra | 双控环境:用户也能动手用工具,不再只是被动的信息源 |
| AgentIF | Tsinghua | agent 场景下的指令遵循评测:真实应用里的约束又长又杂 |
| Agentic-Function-Calling-Robustness | IBM | 已有研究都在提高调用准确率,这一篇专看鲁棒性这一面 |
| HLE | CAIS | 专家出题的封顶级学术评测,用来对付已经刷穿 MMLU 的模型 |
| WebWalker | Alibaba | 传统搜索只捞到浅层内容;这一套测的是模型能不能在网页里逐层走下去 |
| Robust-Function-Calling | OPPO | Hammer：用 function masking 做端侧函数调用,抗函数名误导 |
| ToolSandBox | Apple | 有状态、可交互的工具评测沙盒,不要求模型显式吐对话状态 |
| Arena-Hard | Berkeley | 从 Chatbot Arena 真实对局里自动挑难题构造的评测集 |
| tau-bench | Sierra | 把用户也模拟进来:agent 要在多轮里遵守领域规则并与用户来回确认 |
| MT-Bench | Alibaba | 细粒度多轮对话基准 MT-Bench-101：三层能力、13 任务；常见对齐没有明显抬分 |
| GAIA | Meta | 人类容易、模型很难的真实助理任务;需要多步工具与网页操作 |
| AgentBench | Tsinghua | 八类环境上的 agent 综合评测,操作系统、数据库到网页购物 |
| Gorilla | Berkeley | 接海量 API 的模型与评测,BFCL 的前身;引入 AST 匹配判对错 |
| API-Bank | Alibaba | 工具增强 LLM 的早期系统性评测:何时调、调哪个、怎么规划 |
| HELM | Stanford | 把评测拆成场景×指标的矩阵,强调覆盖面与多维度而非单一分数 |
| BIG-Bench | Google | 204 个众包任务的超大评测集,用来找模型能力的涌现与断崖 |
| TruthfulQA | Oxford | 专挑人类也常答错的问题,测模型是否跟着复述常见谬误 |
| MMLU | Berkeley | 57 个学科的多选题;十年里被引用最多、也被刷得最狠的通用知识评测 |

# 知识库施工路线图(需求清单 + 进度)

维护规则:每完成/开工一篇就更新状态列;新会话续建前先读本文件 + `docs/question-authoring.md` 的知识库规范。
架构类文章的**权威参考源**:[references/frontier-llm-architecture-handbook-2026.md](references/frontier-llm-architecture-handbook-2026.md)(截至 2026-08 的前沿架构对比,含 arxiv 清单)。

状态:⬜ 待写 · 🚧 施工中 · ✅ 完成 · 🖼️ 完成但待补图

## 文章标准(摘要,全文见 question-authoring.md)

- 一篇 = 一个可独立成文的知识点,**尽量细碎**;同主题另开一篇「XX总览」做对比串联
- 覆盖细节与公式(块级 `$$` 独行)+ 通俗直白的类比讲解
- 适当 mermaid 可视化;mermaid 表达不了的画面留占位符:`> 🖼️ 占位:<想要的图的描述>`(用户后期换图)
- 文末必须有 `## 相关文献`,附 arxiv 链接;**不确定的编号必须联网核实,严禁编造**
- 文件名 = 题目 frontmatter `topic` 第一段(受控词表,新增题目定 topic 时先对齐本清单)

## 批次 1:点名主题(后训练 + Infra 核心)

| 文章(=topic 头) | 分类 | 状态 | 说明 |
|---|---|---|---|
| GRPO | RL | ✅ | 样板,已按新标准补相关文献 |
| PPO | RL | ✅ | clip 目标、GAE、RLHF 四模型工程 |
| DPO | RL | ✅ | BT 模型推导、隐式奖励、IPO/KTO/SimPO |
| SFT | SFT | ✅ | 全流程:数据→模板→loss mask→packing→评估 |
| LoRA | SFT | ✅ | 低秩分解、rank/alpha、QLoRA、DoRA/rsLoRA |
| ZeRO | AI Infra | ✅ | 显存解剖、Stage 1/2/3、通信代价、Offload |
| DeepSpeed | AI Infra | ✅ | 引擎全景、与 FSDP 对比、配置实务 |
| RL训练框架 | AI Infra | ✅ | verl(HybridFlow)/OpenRLHF/TRL 对比与选型 |
| rollout引擎 | AI Infra | ✅ | vLLM(PagedAttention)/SGLang(RadixAttention)、RL 场景集成 |

## 批次 2:模型架构系列(参考架构手册拆分)

| 文章(=topic 头) | 分类 | 状态 | 说明 |
|---|---|---|---|
| 架构总览 | 预训练 | ⬜ | 三条路线之争、KV cache 主战场、判断框架 |
| GQA | 预训练 | ⬜ | MHA→MQA→GQA 谱系 |
| MLA | 预训练 | ⬜ | 潜在压缩,含 CCA 延伸 |
| SWA | 预训练 | ⬜ | 滑动窗口、局部:全局比例、窗口大小谱 |
| 稀疏注意力 | 预训练 | ⬜ | DSA/CSA/HCA |
| 线性注意力 | 预训练 | ⬜ | Mamba-2/GDN/KDA、O(1) 状态 |
| Hybrid注意力 | 预训练 | ⬜ | 混合比例、MiniMax 反例、三路线对照 |
| 注意力配件 | 预训练 | ⬜ | QK-Norm/门控/attention sink/跨层KV共享/逐层预算 |
| RoPE | 预训练 | ⬜ | 含 NoPE/partial RoPE/YaRN、层间分工规律 |
| MoE基础 | 预训练 | ⬜ | 专家粒度趋势、共享专家、dense 前缀 |
| MoE路由 | 预训练 | ⬜ | softmax/sigmoid/ReLU/哈希/分位数、负载均衡、aux-loss-free |
| LatentMoE | 预训练 | ⬜ | 压缩空间专家、通信动机(可并入 MoE路由,施工时定) |
| Norm位置 | 预训练 | ⬜ | PreNorm/PostNorm/Sandwich、RMSNorm 统一史 |
| FFN与激活 | 预训练 | ⬜ | SwiGLU/GELU/ReLU²、维度设计 |
| 残差流 | 预训练 | ⬜ | 超连接/mHC/AttnRes |
| MTP | 预训练 | ⬜ | 训练信号 + 投机解码双用途 |
| 优化器 | 预训练 | ⬜ | AdamW/Muon/MuonClip、优化器不匹配问题 |

## 批次 3:预训练大主题 / 推理加速 / 量化 / 多模态

| 文章(=topic 头) | 分类 | 状态 | 说明 |
|---|---|---|---|
| 预训练流程 | 预训练 | ⬜ | 数据→tokenizer→课程→退火→评估全景 |
| ScalingLaws | 预训练 | ⬜ | Kaplan/Chinchilla、算力最优、数据受限 |
| 数据工程 | 预训练 | ⬜ | 清洗/去重/配比/合成数据 |
| 推理加速 | AI Infra | ⬜ | 总览:投机解码/KV 优化/batch 策略/编译 |
| 量化 | AI Infra | ⬜ | 参数量化:GPTQ/AWQ/FP8/INT4/MXFP4、QAT vs PTQ |
| 并行策略 | AI Infra | ⬜ | DP/TP/PP/EP/SP、Megatron、与 ZeRO 的关系 |
| Diffusion | 多模态 | ⬜ | DDPM/DDIM/噪声调度/引导 |
| FlowMatching | 多模态 | ⬜ | 与 diffusion 的关系、rectified flow |
| VLM结构 | 多模态 | ⬜ | ViT/CLIP/连接器(线性/Q-Former)/原生多模态 |
| 后训练总览 | RL | ⬜ | SFT→RLHF→DPO→RLVR 全景对比串联 |
| RLHF与RM | RL | ⬜ | 偏好数据、BT 训练、reward hacking |
| KL散度 | RL | ⬜ | k1/k2/k3 估计器、前向/反向 KL |
| 解码策略 | SFT | ⬜ | 已有题目(SFT/001),文章补全景 |

## 待议(出现对应题目再排期)

RAG 系列(检索优化/Embedding 已有题)、Agent 系列、蒸馏、长上下文训练、Tokenizer 单篇。

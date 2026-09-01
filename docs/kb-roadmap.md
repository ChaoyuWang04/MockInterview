# 知识库结构与施工进度

页面 `/kb`。**按训练流程编排**:模型结构 → 预训练与微调 → 强化学习 → Infra → 多模态 → 应用。

## 结构约定(2026-08 重构)

- **顺序**:文件夹用 `NN-` 数字前缀控制顺序,页面显示时剥掉前缀;目录内 `00-总览.md` 自然排第一
- **层级**:支持任意层嵌套(如 `05-多模态/03-视频生成/视频VAE.md`)
- **与题库的联动**:题目 frontmatter `topic` 第一段 → **按文章名全库匹配**(不再要求同名分类文件夹),因为分章已不与题库分类一一对应
- **文章名全局唯一**(各章的 `00-总览` 除外,它们是 hub 页不参与匹配),有测试守护
- **占位稿**:正文含 `> 🚧 占位:` 标记,页面显示为虚线灰框;写完删掉该行即可

新增文章:在对应章节建文件 → 写正文 → 删掉占位标记 → `npm test`。撰写标准(粒度、公式、mermaid、相关文献)见 [question-authoring.md](question-authoring.md);架构类参考 [references/frontier-llm-architecture-handbook-2026.md](references/frontier-llm-architecture-handbook-2026.md)。

## 现状

共 113 篇:**38 篇已成文**,75 篇占位待写。占位篇里都写了「计划覆盖」提纲,续写时照着展开即可。

列出所有待写篇目:

```bash
grep -rl "🚧 占位" knowledge/ | sort
```

## 章节地图

### 01-模型结构(18 篇 · 15 已成文)

总览 · KV共享注意力(MHA/MQA/GQA)· MLA · 稀疏注意力 · 线性注意力 · SWA · Hybrid注意力 · 注意力配件 · RoPE · MoE基础 · MoE路由 · Norm位置 · FFN与激活 · 残差流 · MTP · 🚧MagiAttention · 🚧Tokenizer · 🚧新架构追踪

> 新出现的注意力变体各开一篇;还不够成篇的先记进「新架构追踪」。

### 02-预训练与微调(9 篇 · 6 已成文)

🚧总览 · 预训练流程 · ScalingLaws · 数据工程 · 优化器 · SFT · LoRA · 🚧蒸馏 · 🚧长上下文训练

### 03-强化学习(9 篇 · 6 已成文)

总览 · RLHF与RM · PPO · GRPO · 🚧**GRPO变体**(DAPO/Dr.GRPO/GSPO 各修了什么问题,一篇讲透)· DPO · KL散度 · 🚧**多模态RL**(Flow-GRPO/Diffusion-DPO)· 🚧**AgenticRL**(多轮轨迹、工具环境、信用分配、rollout 系统变化)

### 04-Infra(33 篇 · 8 已成文)

**01-原理(20 篇)**——先概述后细节:
总览🚧 · 并行策略 · ZeRO · 量化 · 推理加速 · 解码策略 · 🚧集合通信 · 🚧Roofline与Bound分析 · 🚧访存与算子优化 · 🚧GEMM优化 · 🚧KVCache · 🚧FlashAttention · 🚧PagedAttention · 🚧RadixAttention · 🚧连续批处理 · 🚧投机解码 · 🚧PD分离 · 🚧Prefill与Decode的矩阵形状 · 🚧CudaGraph · 🚧TorchCompile

**02-框架与引擎(13 篇)**——**只讲架构与用法,原理一律引到 01-原理**:
🚧总览 · RL框架对比 · 🚧verl · 🚧slime · 🚧ROLL · 🚧OpenRLHF · 🚧Megatron · DeepSpeed · 🚧FSDP · 推理引擎对比 · 🚧vLLM · 🚧SGLang · 🚧NCCL

### 05-多模态(33 篇 · 3 已成文)

**根目录(共用范式)**:🚧总览 · 🚧VAE · Diffusion · FlowMatching · 🚧DiT
**子领域**(每个含 00-总览):01-视觉理解(VLM结构 + 视觉编码器/任意分辨率)· 02-图像生成(SD架构/条件控制/图像编辑/图像Tokenizer)· 03-视频生成(视频扩散架构/时空注意力/视频VAE/长视频一致性)· 04-音频(语音识别/TTS/音频编解码/音乐生成)· 05-3D生成(NeRF/3DGS/多视角扩散/网格生成)· 06-世界模型(世界模型范式/视频世界模型/动作条件与交互)

### 06-应用(11 篇 · 全部占位)

🚧总览 · **01-RAG**:总览/检索优化/Embedding/切块与索引/RAG评测 · **02-Agent**:总览/规划模式/工具调用/记忆与上下文/多智能体

> 与 AgenticRL 的分工:**这里讲系统怎么搭,强化学习章讲怎么训**。

## 重构时的改名对照(2026-08)

| 原路径 | 新路径 | 说明 |
|---|---|---|
| 预训练/架构总览.md | 01-模型结构/00-总览.md | 章节 hub 页 |
| 预训练/GQA.md | 01-模型结构/KV共享注意力.md | 内容本就是 MHA/MQA/GQA 对比 |
| RL/后训练总览.md | 03-强化学习/00-总览.md | 章节 hub 页 |
| AI Infra/RL训练框架.md | 04-Infra/02-框架与引擎/RL框架对比.md | 单个框架另有独立篇 |
| AI Infra/rollout引擎.md | 04-Infra/02-框架与引擎/推理引擎对比.md | vLLM/SGLang 另有独立篇 |
| SFT/解码策略.md | 04-Infra/01-原理/解码策略.md | 属推理侧 |
| 多模态/VLM结构.md | 05-多模态/01-视觉理解/VLM结构.md | 理解与生成并列 |

其余 31 篇仅移动位置、文件名不变;题目 topic 词表因此**只需改 `GQA` → `KV共享注意力`**(当前题库无题使用该 topic,无需改动)。

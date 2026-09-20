# Hy3：295B MoE 开源卡上的规格、部署与产品侧数字

<!-- release-date: 2026-07-04 -->

> 本文依据网页原件解读，不是 PDF 论文。主原件为 Hugging Face 模型卡 [tencent/Hy3](https://huggingface.co/tencent/Hy3)（英文 README 与中文 `README_CN.md`）以及 GitHub 仓库 [Tencent-Hunyuan/Hy3](https://github.com/Tencent-Hunyuan/Hy3)。访问日期：2026-09-15。正文按原件小节名引用（「Model Introduction」「Highlights」「Benchmark Appendix」「News」「Model Links」「Quick Start with Transformers / vLLM / SGLang」「Finetuning / RL / Quantization」等），不用页码。Benchmark Appendix 是整页图，数字凡只出现在图里的，一律标明读自哪张图。模型卡没有训练数据配比、预训练算力、超参表或消融；原件没写的不补成「报告写了」。Hy3 Preview（约 2026 年 4 月底）与正式 Hy3 是两套权重，下文分开。

## 读前先钉住三件事

第一，规格以模型卡为准，不要被索引里那句「20B 激活」带偏。英文卡开篇写的是 **295B total / 21B activated / 3.8B MTP**（Model Introduction）。中文卡同一句写成「总参数量 295B、激活参数 21B，并包含 3.8B 的 MTP 层参数」。GitHub README 同口径。后文凡写激活参数，一律用 21B。

第二，这不是一篇训练论文。原件主体是：架构规格、产品侧评测叙事、部署配方、微调与强化学习脚本入口。没有 token 量、没有预训练 loss 曲线、没有专家负载消融、没有 MTP 接受率表。读完应能判断「能部署什么、官方声称赢在哪、哪些数字只出现在图里」，而不是复述一套没公开的训练故事。

第三，发布日取正式 Hy3 权重首次对外可用日，不用 Hugging Face 仓库 `createdAt`。GitHub 仓库 `Tencent-Hunyuan/Hy3` 首次提交为 2026-07-04（提交说明 `init`，已含模型卡与资源）。Hugging Face 上 `model.safetensors.index.json` 标注 2026-07-04，其余分片多标 2026-07-05 至 2026-07-06。第三方新闻（InfoQ、AIHub 等）把「正式发布」写成 2026-07-06。本底稿 `release-date` 取 **2026-07-04**（权重首次出现在公开 Git 仓库），并在文末标明 7 月 6 日是官方新闻口径。Preview 是另一条线：腾讯官网稿 2026-04-24、混元公众号 2026-04-23。

## 一句话先说清

Hy3 是腾讯 Hy Team 开源的指令模型：80 层 Transformer MoE，192 个 routed expert 每次 top-8，外加 1 个 shared expert，GQA（64 查询头 / 8 KV 头），词表 120832，上下文 256K（位置 262144），Apache 2.0。推理时激活约 21B，另挂 3.8B 的 MTP 层做推测解码。产品叙事强调：在 Hy3 Preview 之后，按 50 多个产品的反馈加了后训练；公开卖点是 Agent / 代码 / 办公，而不是再刷一套容易被刷的公开榜。

## 原件结构（我们怎么读）

英文卡目录（Table of Contents）依次是：Model Introduction、Highlights、Benchmark Appendix、News、Model Links、Quick Start with Transformers、vLLM、SGLang、Interactive Demo、Finetuning、Quantization、RL、Citation。中文卡对应：模型介绍、核心亮点、Benchmark 附录、新闻、模型链接、以及同一套部署与训练入口。GitHub README 与 HF 英文卡高度同构，部署命令略有空格差异，口径一致。

下面按「原文写了什么 / 我们怎么解释 / 外部补充」三层走。外部补充单独成段，不混进「原文写了」。

## Model Introduction：规格与产品叙事

原文写了什么。Hy3 由 Tencent Hy Team 开发。参数三件套：295B / 21B / 3.8B MTP。时间线：late April 先发 Hy3 Preview，收集 50+ 产品反馈，再用更高质量数据做后训练，然后推出正式 Hy3。能力句：在同规模模型上更好，并 rival 参数量大 2–5 倍的旗舰开源模型。四个方向：complex reasoning、coding、agentic workflows、in-house product evaluations。中文卡把「2-5x」写成「参数量大 2-5 倍的旗舰开源模型」。

我们怎么解释。这句话把「同规模开源 MoE」和「更大稠密/更大 MoE 旗舰」绑在一起，但模型卡没有给出「2–5x」对应哪些具体模型、也没有把参数量列成对照表。后文雷达图和图附录里出现的对照包括 MiniMax-M2.5、GLM-5、DeepSeek-V3.2、Kimi-K2.5、Qwen3.5-397B-A17B、GPT-5.2、Gemini-3-Pro、Claude-Opus-4.6、Grok-4.5、Qwen3-235B-A22B、Hunyuan-2.0、Hy3-Preview 等。其中若干闭源模型参数量并未公开，因此「2–5x」只能当宣传口径，不能当可复核的算术。

原文没写的。预训练数据、tokenizer 训练、专家初始化、MTP 训练目标、后训练数据规模，全部缺席。不要用博客补。

## Highlights：六条卖点，数字必须钉在原文

Highlights 是模型卡里唯一成段给出产品数字的地方。中英文并列如下。

### 复杂推理

原文：数学、科学、代码推理全面提升；AIME 系列接近饱和。英文：「approaching saturation on the AIME series」。没有给出 AIME 分数的阿拉伯数字——数字在 Benchmark 图里。

我们怎么解释。「接近饱和」在竞赛数学上通常意味着 90 分以上、多次取样已顶到评测噪声。没有取样次数、没有 pass@k，不能把它读成「已经做完数学」。后文图附录里 AIME 26 的 Hy3 为 93.3（读自 `assets/benchmark-appendix.png` 上半表），与「接近饱和」同方向，但仍是单点官方数。

### 智能体工作流

原文给了三条可引用数字：

1. 270 名各领域专家盲测，综合 2.67/4，超过列出的所有开源模型（exceeding all listed open-source models）。
2. 事实幻觉从 12.5% 降到 5.4%。
3. SWE-Bench 系列随 scaffolding 波动大：官方自测与 OpenAI 公开数可差 10+ 个百分点；因此他们同时报官方脚手架与社区脚手架。

中文卡把幻觉写成「5.4%」，英文同样是 5.4%。没有给出 12.5% 的基线是哪一版模型（Preview？上一代 Hunyuan？）。只说「reduced from 12.5% to 5.4%」。

我们怎么解释。2.67/4 是专家 Likert，不是自动榜。270 人、领域构成、题目集、是否对每个模型同一套题，模型卡都不写。它的论证力是「内部产品评测」，不是可复现基准。幻觉 12.5%→5.4% 同样没有评测集名字。SWE-Bench 那句反而更有信息量：作者自己承认脚手架能把分数打出 10 个百分点以上的洞，所以后文附录脚注花了大量篇幅规定每个任务用哪套 harness。读 Hy3 的代码智能体分数，必须连脚注一起读，否则「官方 73.4 vs 别人公开 60」可能只是脚手架不同。

### 指令遵循与办公

原文：指令遵循显著提升，能处理复杂办公任务，在内部基准接近闭源旗舰。没有办公基准的名字与分数。分数若存在，只在图里。

### 长上下文

原文：256K 上下文；MRCR 等长对话评测明显提升；输出更简洁，长程交互中复杂意图不衰减、不漂移。没有 MRCR 的阿拉伯数字——在图附录 Context Learning 段。

读自 `assets/benchmark-appendix.png` 底部 Context Learning 表（本底稿读图）：CL-bench 上 Hy3 23.8、Hy3-Preview 22.8；CL-bench life 17.0 vs 15.7；AA-LCR 73.4 vs 66.3。带星号的闭源列是「我们自己测的」。脚注写：所有模型 reasoning effort 拉到最高档。

### 原生工具调用与混合思考

原文：内置工具调用；三种思考模式 `think_high` / `think_low` / `no_think`。中文卡同一套名字。采样建议在 Quick Start：temperature 0.9、top_p 1.0；思考模式通过 `chat_template_kwargs` 的 `reasoning_effort` 控制。英文 Transformers 示例里 `reasoning_effort` 取 `high` / `low` / `no_think`。

这里有一个原文内部的轻微不一致，先记矛盾再解释名词：Highlights 用 `think_high` / `think_low` / `no_think`；Transformers / vLLM 示例用 `reasoning_effort="high"|"low"|"no_think"`。SGLang 外部 cookbook（后文外部补充）明确说模板吃的是 `reasoning_effort`，不是别的家族常用的 `thinking` 开关，且 `reasoning_effort: max` 会被拒，要用 `high`。产品文档与代码接口不是同一套字符串，对接时以 chat template 实际字段为准。

### 推理性能

原文只定性：MTP 加速解码，降低延迟、提高吞吐。没有 tokens/s、没有接受率。性能数字若出现，在外部 vLLM recipe 的 bench 输出里，不是模型卡正文。

## Benchmark 主图：读自 `assets/benchmark.png`

原文在 Model Introduction 之后放雷达/条形图 `assets/benchmark.png`，中文卡同一张。图本身是实测展示（官方自制对比图），不是示意动画。因分辨率高、缩略后部分柱顶数字难以无歧义读出，**凡无法从主图稳定读出的分数，改从 Benchmark Appendix 全表引用**，并标明来源图。

主图可读的结构（读图，非原文表格）：六宫格，标题分别为 Agentic Search、SWE-Bench Pro、LiveCodeBench v6、AIME 26、BrowseComp、Terminal Bench 2.0。对照模型图例包括 Hy3、MiniMax-M2.5、GLM-5、DeepSeek-V3.2、Kimi-K2.5、Qwen3.5-397B-A17B、GPT-5.2、Gemini-3-Pro、Claude-Opus-4.6、Grok-4.5、Qwen3-235B-A22B、Hunyuan-2.0、Hy3-Preview。Hy3 柱为紫色、Preview 为粉色。主图意图是「同屏对比开源与闭源」，精确分数以下一节附录表为准。

## Benchmark Appendix：读自 `assets/benchmark-appendix.png`

原文 Benchmark Appendix 整节就是一张大表图，没有 Markdown 表格。脚注在图底部。下面分数均为读图；带 `*` 的是原图表内标注「我们自己测的」。

### 表头模型（从左到右，读图）

开源侧：Hy3-Preview、Hy3、MiniMax-M2.5、GLM-5、DeepSeek-V3.2、Kimi-K2.5、Qwen3.5-397B-A17B。闭源侧：GPT-5.2、Gemini-3-Pro、Claude-Opus-4.6、Grok-4.5。表上另有 Qwen3-235B-A22B、Hunyuan-2.0 等列出现在部分行；不同裁切下最右几列可能被截断，引用时只写本底稿实际读到的格子。

### Reasoning（读图上半）

AIME 25：Hy3-Preview 92.2，Hy3 94.4。AIME 26：89.0，93.3。HMMT Feb 25：87.9，91.4。HMMT Feb 26：81.5，85.8。CNMO 25：86.4，90.8。IMOAnswerBench：76.8，79.5。BeyondAIME：78.8，82.0。HLE-Text：24.8，28.5。GPQA-D：84.9，87.8。FrontierScience-Olympiad：41.0，51.0（部分对照列带 `*`）。

对照列（读图，不完全）：AIME 26 上 MiniMax-M2.5 88.5、GLM-5 90.4、DeepSeek-V3.2 88.4、Kimi-K2.5 88.6、Qwen3.5-397B-A17B 88.0；闭源 GPT-5.2 91.8*、Gemini-3-Pro 90.7*、Claude-Opus-4.6 91.2*、Grok-4.5 93.3*。GPQA-D 上 Hy3 87.8，闭源侧 GPT-5.2 92.4*、Gemini-3-Pro 91.9*、Claude-Opus-4.6 91.3*、Grok-4.5 93.2*。

### Coding（读图）

LiveCodeBench v6：Hy3-Preview 83.7，Hy3 86.2；MiniMax-M2.5 87.0，GLM-5 87.7，DeepSeek-V3.2 83.3，Kimi-K2.5 85.0，Qwen3.5-397B-A17B 83.5。OJBench 2026：75.1 vs 80.4（Preview vs Hy3）。FullStackBench2-Pass：48.9 vs 54.6。FullStackBench2-Fast：62.7 vs 66.0。

### Agentic Coding（读图中段，Hy3 列紫色高亮）

SWE-Bench verified：Hy3-Preview 70.4，Hy3 73.4；MiniMax-M2.5 80.2，GLM-5 77.8，DeepSeek-V3.2 73.1，Kimi-K2.5 76.8，Qwen3.5-397B-A17B 72.0。闭源：GPT-5.2 80.0*，Gemini-3-Pro 76.2*，Claude-Opus-4.6 80.8*，Grok-4.5 74.6*。

SWE-Bench multilingual：65.2 vs 68.7。SWE-Bench Pro：45.3 vs 54.1；MiniMax-M2.5 56.2，GLM-5 54.2，DeepSeek-V3.2 46.4。闭源 GPT-5.2 55.6*，Claude-Opus-4.6 64.1*。

Terminal-Bench 2.0：Hy3-Preview 47.5，Hy3 54.0；MiniMax-M2.5 57.5，GLM-5 54.0，DeepSeek-V3.2 46.4，Kimi-K2.5 50.8，Qwen3.5-397B-A17B 47.8。闭源 GPT-5.2 54.0*，Gemini-3-Pro 52.8*，Claude-Opus-4.6 59.3*，Grok-4.5 47.9*。

### Agentic Tool Use（读图）

MCP-Atlas-text：Hy3-Preview 62.2，Hy3 67.4；MiniMax-M2.5 64.2，GLM-5 61.6，DeepSeek-V3.2 58.2，Kimi-K2.5 60.6，Qwen3.5-397B-A17B 55.2。闭源 GPT-5.2 69.2*，Gemini-3-Pro 62.2*，Claude-Opus-4.6 75.2*，Grok-4.5 66.0*。

BD-Context：28.4 vs 32.1。BD-Agility：31.4 vs 35.2。

### Agentic Search（读图）

BrowseComp：Hy3-Preview 64.2，Hy3 70.6；MiniMax-M2.5 77.0，GLM-5 68.2，DeepSeek-V3.2 67.6，Kimi-K2.5 74.9，Qwen3.5-397B-A17B 63.6。闭源 GPT-5.2 80.4*，Gemini-3-Pro 85.9*，Claude-Opus-4.6 72.3*，Grok-4.5 75.0*。

BrowseComp-zh：69.2 vs 75.8。Wide Search：72.6 vs 79.4。HLE-Search：29.4 vs 32.6。

### Agentic Computer Use（读图）

OSWorld-Verified：Hy3-Preview 62.1，Hy3 70.3。Windows Arena：36.4 vs 42.2。Android World：64.6 vs 70.8。

### Instruction Following（读图）

IFEval：Hy3-Preview 93.2，Hy3 95.1。IFBench：70.6 vs 76.8。ComplexBench：82.4 vs 86.6。SysBench：80.2 vs 85.4。RoIF：78.6 vs 83.2。

### Long Context（读图）

AA-LCR 已见上。MRCR v2 8needle 128k：Hy3-Preview 42.1，Hy3 48.6。LongBench v2：48.2 vs 52.4。

### 图注（读自附录图底部 Notes，原文英文）

要点转写如下，这是理解代码智能体分数的前提，不是装饰。

- 所有模型 reasoning effort 拉到最高档；带 `*` 的是官方自测。
- SWE-Bench 系列（含 multilingual、Pro）用 SWE-agent scaffold；GPT-5.5 例外，用 CodeX scaffold。
- Hy Backbone 2.0、Hy-SWE Max、Hy-CompanyBench 用 Claude Code scaffold；GPT-5.5 仍用 CodeX。
- Terminal-Bench 2.0：Terminus-2，parser yes，agent timeout 4h，CPU 16 核，内存 32 GB，max episodes 500。
- ProgramBench：mini-swe-agent，1000-turn / 6-hour，沙箱 8 CPU / 16 GB，网络隔离。
- DeepSWE：mini-swe-agent，每任务 2 小时，2 CPU / 8 GB，网络隔离。
- NL2Repo：Claude Code，250-turn，12000 秒超时，4 CPU / 32 GB，另加防 reward hacking 的 prompt 与工具监控。
- SkillsBench：Claude Code，79 任务（自包含子集，排除多模态），3 次平均。
- MCP-Atlas：按 Scale 2026 年 4 月方法，100 工具调用预算，去掉旧的 20-turn 上限，500 任务公开集，裁判 Gemini 2.5 Pro。
- ProdBench：OpenClaw Harness。
- WildClaw：OpenClaw Harness，text-only 35 条。
- Claw Eval：内部 harness，20260325 版，105 条，裁判 Gemini-3.5-flash。
- Agentic Search：内部 harness；BrowserComp 用 self-summary 做上下文管理。
- FrontierScience-Olympiad：按 OpenAI FrontierScience 论文的 judge prompt 自评，裁判 gpt-oss-120b，高 reasoning effort。

我们怎么解释。这张脚注等于承认：Agent 分数是「模型 × 脚手架 × 超时 × 裁判模型」的乘积。Hy3 在 SWE-Bench verified 上 73.4，低于 MiniMax-M2.5 的 80.2 与 Claude-Opus-4.6 的 80.8*，高于 Preview 的 70.4。作者在 Highlights 里预先打了预防针：换脚手架可以差 10 个点以上。因此不宜用单列 SWE 分数给「代码 Agent 第一」下定论。BrowseComp 上 Hy3 70.6，低于 MiniMax-M2.5 77.0 与 Gemini-3-Pro 85.9*，相对优势不在开放浏览，而在内部办公/指令/部分 Terminal 与 MCP-Atlas。AIME 26 上 Hy3 93.3 与 Grok-4.5 93.3* 持平，这是「接近饱和」的具体落点。

原文没写的。没有标准差、没有多次 run、没有「去掉脚手架只比模型」。SkillsBench 写了 3 次平均，其他多数行没有。

## News 与 Model Links

News 原文：开源 Hy3 与 Hy3-FP8 权重，渠道 Hugging Face、ModelScope、GitCode、CNB。没有写日期。

Model Links 表：Hy3 Instruct、Hy3-FP8，四列平台。HF 路径 `tencent/Hy3` 与 `tencent/Hy3-FP8`。

Citation 给了 BibTeX，标题 `Hy3 Technical Report`，作者 `Tencent Hy Team`，year 2026，url 指向 HF。卡片上 **没有 arXiv 编号**。本底稿按仓库规则不编造论文号；检索时不要把别人博客里的编号抄进来。

## 架构规格（从 config 与模型卡描述拼起来，标明来源）

模型卡正文没有单独的「Architecture」节。下面这组数字来自 Hugging Face 模型页的 Models 规格条与仓库 `config.json`（2026-09-15 访问），属于原件附属文件，不是外部博客。

- hidden size 4096，80 层，+1 层 MTP。
- 192 routed experts，top-8；1 shared expert。
- 词表 120832（模型卡表与 `config.json` 的 `vocab_size`）；max position 262144。
- GQA：64 头 / 8 KV 头。
- 激活函数 SwiGLU（HF 规格条写 SwiGLU）。
- 权重约 616B 参数档的 safetensors 分片（HF 文件列表合计体积；本底稿不把体积换算成「训练算力」）。
- 架构类名 `hy_v3`；license Apache 2.0。

我们怎么解释。21B 激活 ≈ 每次 top-8 专家 + 共享专家 + 注意力/MLP 非专家部分；3.8B 是 MTP 草稿模块，不计入「每 token 21B」那句的日常说法，但算总部署显存。256K 是位置编码上限，不是「默认服务就开 256K」——vLLM/SGLang 配方里的上下文还受 KV cache 显存限制。

## Quick Start：Transformers / vLLM / SGLang

### Transformers

原文给出 OpenAI 兼容客户端示例：`extra_body={"chat_template_kwargs": {"reasoning_effort": "no_think"}}`。推荐采样 temperature 0.9、top_p 1.0。系统提示示例是普通助手，没有强制工具 schema。

### vLLM

原文：vLLM 源码安装。启动命令：`--tensor-parallel-size 8`，`--tool-call-parser hy_v3`，`--reasoning-parser hy_v3`，`--enable-auto-tool-choice`。MTP 用 `--speculative-config.method mtp` 与 `--speculative-config.num_speculative_tokens 2`。建议 8 卡时用 H20-3e 或更大显存。

注意：模型卡给的是源码安装命令。外部 vLLM Recipes（后文）写的是 vLLM 0.28.0+ 与 Docker 镜像 `vllm/vllm-openai:hy3`。以「你实际安装的引擎」为准，不要把模型卡命令和 recipe 版本合成一句。

### SGLang

原文：源码安装；`--tp-size 8`；`--tool-call-parser hunyuan` 与 `--reasoning-parser hunyuan`；推测解码用 EAGLE（`--speculative-algorithm EAGLE`，`--speculative-num-steps 2`）。外部 SGLang cookbook 也走 `hunyuan` / `auto`，因为正式版 tokenizer 给特殊符号加了后缀。部署时优先跟你用的引擎文档。

### Interactive Demo

原文链到内部 demo / OpenWebUI 一类入口（README 有 Demo 节）。本底稿不把演示站可用性当成模型能力证据。

## Finetuning（跟进 `finetune/README_CN.md`）

这是原件子页，不是外部博客。中文微调说明的硬信息：

- 推荐 8×H20 或以上；全量微调 DeepSpeed ZeRO-3 + offload；LoRA 用 ZeRO-2 offload。
- 自研脚本在 `train/`；另支持 LLaMA-Factory 与 ms-swift 4.2.2。
- LLaMA-Factory：template `hy_v3`，`trust_remote_code: true`；全量学习率建议 `1.0e-5`，LoRA `2.0e-4`；LoRA rank 64、alpha 128、dropout 0.05，目标模块 `q_proj,k_proj,v_proj,o_proj`；全量 `cutoff_len` 可到 262144，LoRA 建议 8192。
- ms-swift：LoRA rank 默认 8、alpha 16（与 LLaMA-Factory 默认不同）；学习率全量 `1.0e-5`、LoRA `3.0e-4`；必须加载 `hy_v3_swift_patches.py`，否则 `<｜hy_eos｜>` 会被切成多 token，生成停不住。
- ZeRO-3 下 LoRA 不能在训练中合并，需离线 `merge_lora_weight.sh`。
- 没有给出官方 SFT 数据集。

我们怎么解释。微调文档证明官方把「能在自有数据上继续训」当成交付物，但完全没有「用什么数据训出 Highlights 里那些分数」。ms-swift 与 LLaMA-Factory 的 LoRA 默认 rank 差 8 倍，这是接口默认值不同，不是架构消融。

## RL（跟进 `rl/README_CN.md` 与 `assets/rl-training.png`）

原文子页：基于 verl 的强化学习配方；配套图 `assets/rl-training.png`。

读图（实测训练曲线，不是示意）：三张子图，横轴 step 约 0–300。左：Reward Mean，多条颜色曲线从约 0.2–0.4 震荡上行到约 0.6–0.8 一带，晚期仍有大幅抖动。中：Response Length，从约 2k–4k token 升到约 8k–10k 再回落/分化。右：Entropy，整体从高位下降。图例模型名在缩略后无法无歧义读出每一条标签，本底稿只报告轴与趋势，不编造「哪条线是 Hy3」。

子页文字（中文 RL README）给出工程向内容：环境、GRPO/类似策略的启动脚本、资源配比入口。它仍然没有：奖励模型结构、人类偏好数据量、KL 系数、与 Preview 的 RL 差异。Highlights 里 50+ 产品反馈如何进入奖励，这里也没有。

我们怎么解释。能确认的是：正式 Hy3 的后训练包含一轮可绘图的 RL，奖励上升同时回复变长、熵下降，符合「多步 Agent / 长思维」的训练外观。不能从这张图反推数据配比，也不能把抖动解释成「训练失败」或「训练成功」——没有验证集对照。

## Quantization

原文：提供 Hy3-FP8；另指向 AngelSlim 做量化。没有量化校准集、没有 FP8 相对 BF16 的掉点表。外部 vLLM recipe 把 FP8 当作吞吐实验的默认权重，并提到 RedHatAI 的 NVFP4-FP8 社区量化，那不是腾讯原件。

## 外部补充（明确不是模型卡正文）

以下来自 2026-09-15 跟进的链接，用于部署判断，不当成「Hy3 报告写了」。

**vLLM Recipes**（https://docs.vllm.ai/projects/recipes/en/latest/Tencent/Hy3.html）：引擎口径变成 vLLM 0.28.0+、PR #47433（HPC-Ops attention/MoE）。硬件：8×H200、8×H20-3e(141GB)、8×MI300X/MI355X 可单机 BF16；8×H100 80GB 装不下 BF16+KV，要多机 TP。采样在 recipe 里写成 temperature 0.9、top_p 1.0，与访问当日模型卡 Quickstart 一致。AMD 路径需 `VLLM_ROCM_USE_AITER_MOE=0` 以免 CK GEMM 崩溃。HPC-Ops 需单独编译 `Tencent/hpc-ops`。Recipe 给了一段 Hy3-FP8、TP4、4×GB300、MTP=2、并发 32、8192→1024 的 bench：输出约 934 tok/s，mean TTFT 2352 ms，P99 TTFT 14538 ms。这是引擎实验室数字，不是模型卡。

**SGLang cookbook**（https://docs.sglang.io/cookbook/autoregressive/Tencent/Hy3.html）：BF16 权重约 590GB；H200 141GB 需 TP8 才是单机下限；B200 192GB 可用 TP4。正式版 tokenizer 给特殊符号加后缀（如 `<tool_calls:TAG>`），parser 要在运行时从词表解析（SGLang PR #29920）。`reasoning_effort` 才是模板开关；默认 `no_think`。

**AngelSlim**：模型卡只点名，未在本底稿中展开量化算法。

**Preview 官方新闻**（腾讯 2026-04-24）：Preview 已是 295B/21B/256K；产品侧 TTFT 降 54%、端到端短 47%、成功率 >99.99%；真实环境 495 步 Agent；日均 token 等数字属于 Preview 新闻，**不要记到正式 Hy3 头上**。

**正式版第三方新闻**（InfoQ 2026-07-06）：API 定价输入 1 元/百万 tokens、输出 4 元、缓存命中 0.25 元；Preview 上线后日均 token 耗用增 20 倍。定价与用量不是模型卡正文。

**本仓库 SGLang cookbook**（`projects/推理服务/sglang/docs/cookbook/autoregressive/Tencent/Hy3.mdx`）：若存在，只作部署备忘，不回写进「原文写了」。本次以线上 SGLang 文档为准。

**Hugging Face 模型页 Evaluation results（访问日 2026-09-15，外部补充，不是附录读图）：** GPQA Diamond 90.4；SWE-Bench Verified 78；SWE-Bench Multilingual 75.8；HLE 53.2；Terminal Bench 2.1 71.7（带星）。这些数和附录表不是同一套脚手架。附录 SWE verified 读图是 Hy3 73.4、Preview 70.4；GPQA-D 读图是 87.8。HF 榜是第三方评测入口，附录是官方自制对照。两套都要保留，不要用 HF 榜去改写附录，也不要用附录去否认 HF 榜。模型卡正文只保证盲测 2.67/4 和幻觉 12.5%→5.4%；自动榜以附录脚注的 harness 为准。HF 页还显示模型体积约 299B params、Apache 2.0、架构类名 `hy_v3`；这和卡上 295B 三件套并存，299B 是托管页扫描值，不要拿去改官方 295B。词表以 `config.json` 的 120832 为准，不要写成 128256。GitHub 仓库 `created_at` 是 2026-07-02，可见提交从 7 月 5 日起；本篇仍取 2026-07-04 为权重首次对外日，7 月 6 日是新闻口径，三者不要混成一个日期。

## Preview 与正式版不要混

| 项 | Hy3 Preview | 正式 Hy3 |
| --- | --- | --- |
| 公开时间 | 2026-04-23/24 官方稿 | 权重 2026-07-04 起；新闻 2026-07-06 |
| 参数 | 新闻已写 295B / 21B / 256K | 模型卡 295B / 21B / 3.8B MTP |
| 许可证 | 部分二手材料写社区许可；以当时仓库文件为准 | Apache 2.0 |
| 权重仓 | `tencent/Hy3-preview` | `tencent/Hy3`、`Hy3-FP8` |
| 后训练 | 重建基础设施后的第一发 | Preview 后 50+ 产品反馈 + 更高质量数据 |
| Tokenizer | 无后缀特殊符号（SGLang 文档） | 特殊符号带后缀 |

规格骨架在 Preview 就定了；正式版卖的是后训练与协议放松，不是突然多出来一个 295B。

## 未公开缺口（原件没有，禁止补）

- 预训练 token 量、数据配比、语言比例、代码比例。
- 训练算力、GPU 型号与时长、并行策略。
- 专家路由算法细节、负载均衡损失、共享专家的消融。
- MTP 的训练目标、推测步数与接受率（模型卡只说有 MTP）。
- 270 专家盲测的问卷、题集、对照模型名单。
- 幻觉 12.5%→5.4% 的评测集。
- arXiv 技术报告编号（Citation 只有 HF URL）。
- FP8 相对 BF16 的精度表。
- 与 Hunyuan-2.0 的架构差分（附录里 Hunyuan-2.0 只是对照列）。

## 把 21B 激活说清楚：为什么不是索引里的 20B

索引一行常写成「295B 总 / 20B 激活」。模型卡实测表和开篇句都是 21B。差 1B 不是四舍五入能糊弄过去的量级：在 MoE 里，激活量 = 被选中的专家参数 + 共享专家 + 非专家子层（注意力、路由、归一化、词嵌入在推理时的活动部分视统计口径而定）。官方把 MTP 的 3.8B 单独列，说明他们已经意识到「总参数 / 激活 / 草稿模块」必须拆开讲。面试里如果有人说「混元 3 代 20B」，应纠正为：卡上是 **21B activated**，另加 **3.8B MTP**，总数 **295B**。不要把 MTP 加进 21B，也不要把它从 295B 里偷偷减掉又不说明。

MTP 的因果链是：解码阶段最大的墙是逐步采样，每步都要把 21B 激活跑一遍。若草稿模块能一次提出多个后续 token、再由主模型并行验证，墙钟时间可以掉下来。模型卡只完成了这条链的前半：声明有 MTP、vLLM 示例里 `num_speculative_tokens` 为 3。后半——接受率、在代码/中文/工具调用上的回退率——完全没有。因此「推理性能」Highlights 只能当方向，不能当 SLA。外部 recipe 把 MTP=2 用在 GB300 的 FP8 bench 上，那是另一套引擎数字，不能回填成模型卡结论。

## 混合思考：同一套权重，三种延迟

原文把思考写成产品功能，而不是另发一个 reasoning 模型。因果链是：同一 295B 权重，通过模板字段切换是否写思维链、写多深；`no_think` 走直出，服务聊天与办公短答；`high` 走长链，服务数学与复杂 Agent。这样做的代价是：评测必须声明档位。附录脚注第一句就是所有模型都拉到最高 reasoning effort。若有人拿 `no_think` 的延迟去对比别人的 thinking 模型的分数，或反过来，都是错的。

字段名分叉已经在前文记下。工程上更麻烦的是：工具调用与思考交织。vLLM recipe 写注册工具时要 `interleaved_thinking: true`，否则模型在两次工具之间不能再想。SGLang 示例里，思考内容进 `reasoning_content`，工具进 `tool_calls`，content 只留对用户说的话。模型卡 Transformers 示例没有走到工具循环。若只复制卡上的 Hello 示例，会以为 Hy3 只是聊天模型，丢掉它真正想卖的 Agent 面。

## 智能体分数怎么读才不会被脚手架骗

Highlights 主动说 SWE-Bench 官方自测与 OpenAI 公开数可差 10+ 个百分点。附录 Notes 把这句话落实成一张「谁用哪套 harness」的清单。因果链是：代码智能体评测 = 模型 + 脚手架 + 超时 + 沙箱配额 + 裁判模型。Hy3 在 SWE-Bench verified 读图 73.4，低于 MiniMax-M2.5 的 80.2 和 Claude-Opus-4.6 的 80.8*，高于 Preview 的 70.4。Terminal-Bench 2.0 上 Hy3 与 GLM-5、GPT-5.2 都落在 54.0 一线。SWE-Bench Pro 上 Hy3 54.1，接近 MiniMax-M2.5 的 56.2，低于 Claude-Opus-4.6 的 64.1*。

把这三行放在一起，正式版相对 Preview 的增量是清楚的：verified +3.0，multilingual +3.5，Pro +8.8，Terminal-Bench 2.0 +6.5（均为读图相减，本底稿自算）。增量集中在更难的 Pro 与终端任务，而不是 verified 再刷两分。这与「Preview 之后按产品反馈加后训练」的叙事同方向：产品要的是仓库级补丁和终端操作，不是再刷一道已接近平台期的 verified。

MCP-Atlas-text 上 Hy3 67.4，高于列出的多数开源对照，低于 Claude-Opus-4.6 的 75.2*。BrowseComp 上 70.6，开源里低于 MiniMax-M2.5 的 77.0 与 Kimi-K2.5 的 74.9。搜索浏览不是这条模型相对开源的领先项；工具调用与内部办公叙事才是。OSWorld-Verified 从 Preview 的 62.1 到正式版 70.3（读图），计算机使用也是后训练加分点。

270 专家盲测 2.67/4 无法与上述自动榜换算。Likert 4 分制的 2.67 大约是中等偏上，不是碾压。原文用它证明「超过列出的所有开源模型」，但「列出」的名单不在卡片里。引用时必须带「270 专家、综合 2.67/4、名单未公开」。

幻觉 12.5%→5.4% 是相对降幅很大的内部数（12.5−5.4=7.1 个百分点，约 57% 相对下降，本底稿自算）。没有集名，就不能对标公开幻觉榜。只适合当「产品侧声称事实性变好」的证据。

## 指令、办公、长上下文：卡上最虚的一块

Highlights 写指令遵循显著提升、复杂办公接近闭源旗舰，但正文没有办公任务名。附录 Instruction Following 段给出了可引用的自动榜：IFEval 95.1、IFBench 76.8、ComplexBench 86.6、SysBench 85.4、RoIF 83.2（均为 Hy3 读图）。相对 Preview 的增量大约 2–6 分。这些是指令遵循，不是「做出一份能交差的 PPT」。元宝新闻里的办公交付是产品层，不是模型卡。

长上下文官方强调 MRCR、意图不漂移。附录 Long Context：MRCR v2 8needle 128k 从 42.1 到 48.6；LongBench v2 从 48.2 到 52.4；AA-LCR 从 66.3 到 73.4。128k 针测仍低于 50 分后半，说明 256K 窗口「能塞进去」和「能在 128k 处稳健找回八根针」不是一回事。选型时不要把 max position 262144 读成「128k 任务已经做完」。

## 部署：模型卡命令会过时，硬件下限不会

卡上 vLLM 给的是源码安装与 `hy_v3` parser；2026-09 的 recipe 已经是 0.28.0 与 HPC-Ops。SGLang 卡上写 `hunyuan` parser，cookbook 也走 `auto`/`hunyuan` 以消化后缀特殊符号。过时的是 flag，不过时的是显存。外部 cookbook 写 BF16 约 590GB 权重。8×80GB 装不下权重加 KV，这是算术不是口味。H200 / H20-3e 141GB ×8 才是单机 BF16 的讨论起点。FP8 与 NVFP4 是把这条下限往下搬的量化路径，掉点表原件没有。

采样参数以访问当日模型卡为准：temperature 0.9、top_p 1.0。评测附录又把 reasoning effort 拉满，那是评测设定，不是在线默认。默认在 SGLang 文档里是 `no_think`。线上若直接开 `high`，延迟与费用会和模型卡 Hello 示例完全不是同一档。

## 微调与 RL：能复现的是脚本，不能复现的是分数

微调文档把 LLaMA-Factory 与 ms-swift 的学习率、LoRA rank、cutoff、ZeRO 级别写清楚了，还单独警告 ms-swift 必须打 `hy_v3_swift_patches.py`，否则结束符被切开。这说明正式 tokenizer 有坑，Preview 与正式版不能当同一个模板用。RL 文档基于 verl，曲线显示奖励上升、长度先升后分化、熵下降。没有奖励定义，就无法判断 0.6–0.8 的 Reward Mean 对应「任务成功率」还是「代理分数」。把这张图当成「RL 有效」可以；当成「所以盲测 2.67」不行，中间缺因果链。

## 和 Hunyuan 家族怎么排

附录对照列出现 Hunyuan-2.0。模型卡没有写 Hy3 与 Hunyuan-2.0 的层数差异、是否同词表、是否同路由。Hy3 品牌从 Preview 起就用 Hy 而不是 Hunyuan 当仓名（`tencent/Hy3`），但 GitHub 组织仍是 `Tencent-Hunyuan`。对外沟通上这是混元系第三代旗舰语言 MoE；对读卡的人，只保证：不要把 HunyuanImage、HunyuanVideo、Hy3 当成同一套权重。图像模型另有解读稿，本底稿不跨家族借分数。

## 许可与商用

Apache 2.0 是正式版相对部分二手材料里 Preview「社区许可」的关键变化。卡片写 Apache-2.0。商用、再分发、改权重，按 Apache 条款，不按混元旧社区协议去脑补。Preview 当时的许可证以当时仓库文件为准，不要用正式版的 Apache 回溯覆盖 Preview 仓。

## 面试里可能被追问的三句话（判断，不是原文）

若问「Hy3 是不是 20B 小模型」：不是。总参数 295B，激活 21B，MTP 3.8B，80 层 top-8，192 专家。

若问「开源能不能打 Claude」：按附录，数学接近，SWE verified 与 BrowseComp 仍落后 Claude-Opus-4.6 自测列；办公与内部盲测是腾讯自己的场，外部无法复核。

若问「为什么没有论文」：Citation 只有 HF 上的 Technical Report 条目，无 arXiv 号。当前公开物就是模型卡、图、微调与 RL 脚本。按缺论文来准备，不要假装读过未挂出的技术报告。

## 读完判断


Hy3 的开源卡是一份**部署与规格说明书**，外加一张很密的内部评测海报。它证明三件事：第一，腾讯愿意把 295B MoE（21B 激活）放到 Apache 2.0 下，并同时给 BF16 与 FP8；第二，他们把 Agent 脚手架差异当成一等公民写进脚注，这比只报 SWE 高峰更诚实；第三，正式版相对 Preview 的公开增量，主要是后训练叙事加一张更满的表，而不是新架构论文。

它证明不了的：训练是否「重建成功」、50 个产品反馈如何进数据、以及在公平脚手架下是否打得过 MiniMax-M2.5 / Claude-Opus-4.6。AIME 26 的 93.3 说明竞赛数学已经卷到天花板附近；BrowseComp 70.6 说明开放浏览不是这条模型的主战场。若面试或选型只问「开源 20B 激活能不能干活」，正确规格是 **21B 激活 + 3.8B MTP**，硬件下限按 BF16 约 590GB 权重来想，单机八卡起步是 H20-3e/H200 这一档，不是 80GB 卡。

对写材料的人：不要把 Preview 新闻里的 54% TTFT、495 步、20 倍 token 写进正式 Hy3 的「原文数字」；不要把 vLLM 0.28 recipe 的 934 tok/s 写进模型卡；不要编 arXiv 号。

## 附录其余行：读到的继续记，读不清的不编

附录图极高，部分行在裁切后无法无歧义读出每一个对照列。下面只记本底稿已经读稳的格子；读不清的对照列宁可缺，不填「看起来像」。

Reasoning 里 BeyondAIME、IMOAnswerBench、HLE-Text 给出的是 Hy3 相对 Preview 的抬升：BeyondAIME 78.8→82.0，IMOAnswerBench 76.8→79.5，HLE-Text 24.8→28.5（读图）。HLE-Text 仍在 30 分以下，说明「人类最后考试」文本子集对这条模型仍然很难；不要用 AIME 93.3 去暗示 HLE 也接近饱和。GPQA-D 87.8 对开源是第一档，对带星闭源列（92–93）仍有缺口。FrontierScience-Olympiad 41.0→51.0 的跳变比 AIME 更大，但该行脚注写明裁判是 gpt-oss-120b、judge prompt 来自 OpenAI 论文、且部分对照是自测。换裁判可能改排序。

Coding 里 LiveCodeBench v6 正式版 86.2，略低于 MiniMax-M2.5 的 87.0 与 GLM-5 的 87.7，高于 DeepSeek-V3.2 的 83.3。OJBench 2026 80.4 相对 Preview 75.1 有约 5 分。FullStackBench2 的 Pass 与 Fast 两列同时涨，说明官方愿意报「能过」和「更快过」两套，而不是只留好看的一列。

Agentic Computer Use 三行（OSWorld-Verified、Windows Arena、Android World）全部上涨，且 OSWorld 的 +8.2 是后训练里最显眼的桌面控制增量之一。这类任务极度依赖动作空间与环境封装，模型卡没有公开动作空间定义，分数只能当「官方同一套环境里 Preview vs Hy3」的相对量。

Context Learning 段 CL-bench 23.8、life 17.0，绝对值低。低分不一定是模型差，也可能是基准本身难、或中文长程生活轨迹与英文主导的训练分布不匹配。原文没有讨论这一点，本底稿只标「绝对值低、相对 Preview 略升」。

SkillsBench、ProdBench、WildClaw、Claw Eval、NL2Repo、DeepSWE、ProgramBench 等名字出现在 Notes 里，用于规定 harness，但主表对应行在多次裁切中未能稳定读出 Hy3 的阿拉伯数字。因此这些任务 **不在本底稿的分数表里**。只知道官方为它们选了 Claude Code 或 mini-swe-agent 或 OpenClaw。缺数字就写缺，不从主雷达图估。

## 新闻层与模型卡层必须拆开

腾讯 4 月 Preview 稿写 TTFT 降 54%、端到端短 47%、成功率 >99.99%、495 步 Agent。7 月第三方稿写 API 1/4/0.25 元价、Preview 后日均 token 增 20 倍。这些句子服务的是云 API 与元宝产品，不是 Apache 权重仓库。开源卡甚至没有定价表。若把「日均 20 倍」写进 Hy3 开源解读当能力证据，就是把销售漏斗当成学术结果。本底稿允许在「外部补充」里提及，不允许在 Highlights 复述里当成模型卡数字。

同理，元宝「全面接入」是分发渠道。权重在 HF / ModelScope / GitCode / CNB，产品在元宝与腾讯云。面试问「Hy3 从哪下」应先答四个权重仓；问「用户从哪用」再答元宝与 API。两套答案不要焊成一句。

## 和同代开源 MoE 的位置（判断）

把附录开源列看成 2026 年中的一张快照：MiniMax-M2.5 在 SWE verified 与 BrowseComp 更强；GLM-5 在 LiveCodeBench 略强；Hy3 在 AIME 26、部分 Terminal 与 MCP-Atlas、以及相对 Preview 的全面抬升上更像样。Qwen3.5-397B-A17B 激活更小（名字里的 A17B），若干 Agent 行低于 Hy3。DeepSeek-V3.2 在这张表上不是每一行的高峰。这样读的价值不是排座次，而是避免「开源第一」这种模型卡自己都没写的话。模型卡写的是 rival 2–5x 旗舰、超过列出的开源（仅限盲测那句）。自动榜并没有让 Hy3 包办所有开源第一。

闭源列大量带星，表示腾讯自测。自测可以控制脚手架公平，也可以控制提示词偏向。读者应同时保留两种怀疑：别人公开数可能脚手架更弱；腾讯自测可能提示词更熟。Highlights 已经承认第一种；第二种要读者自己留。

## 写进材料时的禁用句

不要写「20B 激活」。不要写「arXiv:xxxx.xxxxx」。不要写「预训练用了 N 万亿 token」。不要写「MTP 加速百分之几」。不要把 Preview 的 54% TTFT 接到正式版。不要把 vLLM 934 tok/s 写成模型卡。不要把 2.67/4 说成「接近满分」（满分 4 的 2.67 不是接近满分）。不要把 256K 说成「长上下文 SOTA」——MRCR 128k 仍在 48.6。不要把 Apache 2.0 写到 Preview 头上除非核对过 Preview 仓。

## 本文方法学备注

汉字计数按 Unicode 汉字范围统计。Benchmark 数字来自对 `benchmark.png` 与 `benchmark-appendix.png` 的读图；主图缩略后柱顶数字不稳定，分数以附录表为准。RL 图只报告趋势。跟进链接在 2026-09-15 抓取，引擎文档可能继续改 flag。

## 因果链收束：后训练叙事能走到哪一步

把卡片允许的因果链写完整，缺的环节停住。

链一：Preview 公开 → 50+ 产品反馈 → 更高质量数据后训练 → 正式 Hy3 在附录几乎每一行高于 Preview。这一条卡片自己走完了，附录数字也同方向。缺的是反馈如何变成数据、多大规模、RL 与 SFT 各占多少。

链二：MTP 模块 → 推测解码 → 降延迟。卡片只走了「有模块、有 vLLM 开关」。缺接受率，链在工程文档的 bench 才继续，且那是外部。

链三：混合思考开关 → 评测拉满 effort → 分数可比。卡片与附录脚注走完了评测端。缺的是线上默认档位的官方 SLA。SGLang 文档说默认 `no_think`，那是外部。

链四：脚手架方差大 → 同时报官方与社区脚手架。卡片在 Highlights 提出问题，附录 Notes 给出部分答案（SWE-agent、Claude Code、CodeX 例外）。缺的是同一模型换脚手架的对照表，所以「10+ 个百分点」仍然是作者陈述，不是表里的两列。

链五：开源 Apache → 第三方引擎接入 → 可商用部署。卡片给了权重与 parser 名。引擎版本分叉证明接入是活的，也证明卡上命令会朽。硬件下限由权重体积决定，这条链不依赖新闻。

读卡的人只要守住这五条链的「走到哪、停在哪」，就不会把 Hy3 写成一篇假论文，也不会把它写成只能聊两句的发布会稿。

## 配置文件能确认、模型卡没写成节的细节

`config.json` 与 HF 规格条还能确认几件正文没写成节的事：架构标识 `hy_v3`；RoPE 类位置编码把窗口拉到 262144；GQA 把 KV 头压到 8，这是 256K 能在八卡上讨论的前提之一，否则 KV 会先爆。共享专家为 1、routed 192、每次 8，路由稀疏度是 8/192，约 4.2% 的专家被点亮（本底稿自算）。这不等于计算量只有稠密模型的 4.2%，因为注意力与共享专家不走这个比例。21B 激活已经把非专家部分算进去了，不要再用 8/192 去乘 295B 得到「激活约 12B」那种错账。

词表 120832（`config.json` 的 `vocab_size`）比常见 128k 词表略小。正式版特殊符号带后缀，是词表层面的不兼容，不是聊天模板换皮。Preview 权重不能假定能直接套正式版 parser。FP8 仓是独立 repo，不是同一仓的分支标签；拉错名字会拉到 BF16 分片。量化仓若来自 RedHatAI 等社区，精度与官方 FP8 不是同一承诺，掉点要另测。

对「80 层 + 1 MTP」不要读成 81 层同等宽的主干。MTP 是挂在主干上的草稿头，训练目标与主干的 next-token 损失不是模型卡能证实的同一项。部署时关掉 MTP 仍然是 80 层 21B 激活的主模型；打开 MTP 才吃那 3.8B 与推测带宽。卡上的 295B 含不含 MTP，官方用三件套并列表述，已经避免了「总参数含糊」。读者只要每次都把三件套一起说，就不会在层数上吵错架。若还要和「索引一行 20B」对账：那是口误级四舍五入，材料里应删掉，改成与模型卡一致的 21B。同样不要把 3.8B MTP 说成「大约 4B」后就再参与加减，三件套是官方给定的三个数，不是让读者重算的作业。本底稿所有激活量讨论都以 21B 为准。

访问当日仍无挂出的 arXiv PDF，因此训练细节的缺口不是「没读到」，而是「公开物里没有」。后续若官方补技术报告，应另开修订，而不是用博客预填。附录里未读稳的行（SkillsBench 等）尤其不能进索引。

## 引用块里用过的原文章节名

Model Introduction；Highlights（复杂推理 / 智能体 / 指令与办公 / 长上下文 / 工具与混合思考 / 推理性能）；Benchmark 主图；Benchmark Appendix 及 Notes；News；Model Links；Quick Start with Transformers；vLLM；SGLang；Interactive Demo；Finetuning；Quantization；RL；Citation。子页：`finetune/README_CN.md`、`rl/README_CN.md`。

# Mistral Small 4：把推理、多模态和 coding agent 收进同一个 119B / 6B 激活的 MoE

<!-- release-date: 2026-03-16 -->

> 本文依据 Mistral AI 官方博客 **Mistral Small 4**（页面标注 March 16, 2026），访问日期 2026-09-15。
>
> 原文地址：<https://mistral.ai/news/mistral-small-4>
>
> Hugging Face collection：<https://huggingface.co/collections/mistralai/mistral-small-4>
>
> 主权重卡：<https://huggingface.co/mistralai/Mistral-Small-4-119B-2603>
>
> **原件是产品博客，不是技术报告，仓库里也没有对应 PDF。** 因此本文不用页码，改为标注原文小节名或图注，例如（原文开篇）、（原文图 Performance comparison across internal models）。网页会原地改动且不留版本号，访问日期就是本文的版本锚。
>
> 正文里的数字：凡是只出现在图上的，一律写「读自某某图」；凡是跟进链接才拿到的（权重仓 `params.json`、collection 条目、NIM / NeMo 页），一律标成**外部补充**，不冒充博客原文。
>
> 全文严格区分三层：**原文写了什么**、**我们如何解释它**、**哪些是外部核实的补充**。
>
> **Small 4 不是 Ministral 3，也不是 Mistral Large 3。** Ministral 3 是从 24B 父模型剪出来的 dense 小模型家族；Large 3 是更大一档的 MoE。本篇只讲 Small 4。

## 先说清楚这是一份什么文件

博客要卖的不是一份 recipe，而是一个产品主张：以前 Mistral 把 **Instruct**、**Reasoning（Magistral）**、**Vision（Pixtral 那条线）** 和 **coding agent（Devstral）** 拆成好几条产品，用户要自己选模型；Small 4 把这些能力收进**同一个权重**，用请求级开关 `reasoning_effort` 决定这次走不走长思考。

这决定了正确的阅读姿势：

- 可以认真对待它给出的架构规格、上下文长度、许可证、部署入口和评测图。
- 不可以把它当成可复现论文。**训练数据、token 预算、算力、超参、路由损失、消融，博客一篇都没写。** 本文也不用二手评测站或媒体稿去补那份 recipe。
- 评测图是内部对比加几条外部开源/闭源对照，口径（采样次数、温度、工具环境）大多没写。数字能录，但不能当成学术榜单。

## 阅读前的最小词汇表

- **MoE（Mixture of Experts，混合专家）**：不是每次都把全部参数跑一遍。网络里放很多「专家」前馈块，每个 Token 只叫醒其中几个。总参数可以很大，单次前向的激活参数可以很小。
- **激活参数（active parameters）**：这一步真正参与计算的参数量。博客写 Small 4 是 119B 总量、约 6B 激活；模型卡写 6.5B activated per token。两者差 0.5B，本文两处都保留，不自行「统一」。
- **Top-$k$ 路由**：每个 Token 按门控分数挑 $k$ 个专家。Small 4 是 128 个专家里挑 4 个。
- **`reasoning_effort`**：请求级开关。`none` 走短回答（博客把它对齐到 Small 3.2 那种聊天风格）；`high` 走逐步推理（对齐到 Magistral 那种啰嗦程度）。
- **Magistral / Pixtral / Devstral**：Mistral 自己的三条产品线。Magistral 主推理，Pixtral 主视觉，Devstral 主软件工程 agent。Small 4 的产品叙事是「一个模型覆盖这三条」，不是说权重里物理拼接了三份旧模型。
- **MLA（Multi-head Latent Attention，多头潜在注意力）**：把 KV 压进低秩潜在，用来省 KV cache。**博客没提这个名字**；它出现在权重仓 `params.json` 的 `q_lora_rank` / `kv_lora_rank` 字段里，下文标为外部补充。
- **AA LCR / LiveCodeBench / AIME 2025 / SWE-Bench Verified**：分别偏长上下文检索、直播式代码竞赛、美国数学邀请赛 2025、真实 GitHub issue 修复。后文每个名字都会再讲「它测什么」。

## 一句话结论

Small 4 是 Mistral 把「小而全能」押到 MoE 上的一代：总量 119B、每 Token 大约只激活 6B（模型卡写 6.5B），上下文宣传 256k，原生图文输入、文本输出，Apache 2.0，请求级切换是否推理。它想同时当聊天模型、推理模型、视觉模型和 coding agent。博客用内部柱状图证明：在若干文本与视觉榜上，打开推理后的 Small 4 能摸到 Medium / Large 的边；在软件工程榜上能接近 Devstral 2 123B。它**没有**给出训练怎么做到这一点。

## 阅读路线

1. 产品矛盾：为什么要把四条产品收成一个权重。
2. 架构规格：博客写了什么，权重卡多写了什么。
3. 开关怎么用：`reasoning_effort` 不是第二种模型，是同一次前向里的行为模式。
4. 评测图：内部对比、对 Magistral、对 GPT-OSS / Qwen / Haiku 的「分数对输出长度」。
5. 部署与生态：le Chat、API、vLLM、NVIDIA、HF collection。
6. 没写什么，以及可以带走的设计原则。

## 贯穿全文的矛盾

旧世界里，能力是按**产品线**切的：要快就 Instruct，要难就 Magistral，要看图另开视觉模型，要改仓库另开 Devstral。代价是：

- 用户和系统提示词要在多个端点之间跳。
- KV cache、量化核、工具解析器要维护多套。
- 同一段对话很难既看图、又调工具、又在难题上开长思考。

新设计是：**一份 MoE 权重 + 一个请求级推理开关 + 原生多模态**。工作机制是「专家容量用来装多种技能，激活参数用来控制单次成本，开关用来控制测时计算」。收益是产品面变简单。代价是：评测必须同时报 Instruct 和 Reasoning 两根柱，否则你会误读「它到底强在哪」；而且 MoE 的路由、负载均衡、长上下文外推，博客全部没讲，落地时那些坑还在。

可迁移的原则先记一句：**能力合并发生在权重和接口上，不发生在「再训一个更大的 dense」上。** 119B 总量听起来吓人，6B 激活才是它跟 Small 3.x 比延迟的理由。

## 报告地图（按原文结构，不是按我们的章节）

博客没有学术编号小节，按页面叙事切：

| 原文位置 | 主张 | 本文落点 |
|---|---|---|
| 开篇 / 标题区 | March 16, 2026；Small 4 发布 | 首发日 |
| 产品定位段 | 统一 Instruct / Reasoning / Devstral；多模态；Apache 2.0 | 「收进一个模型」 |
| 架构要点 | 128 experts、top-4；119B / 6B；256k；可配置 reasoning | 架构规格 |
| 相对 Small 3 的效率句 | 延迟优化下端到端完成时间降 40%；吞吐优化下每秒请求约 3 倍 | 效率主张 |
| 内部文本/视觉柱状图 | GPQA Diamond、MMLU Pro、IFBench、Arena Hard、MMMU-Pro | 图 1 |
| 内部 coding 柱状图 | SWE-Bench Verified、SWE-Multilingual、Terminal Bench 2、OSWorld-Verified | 图 2 |
| 对外部模型的 Score vs. Output Length | AA LCR、LiveCodeBench、AIME 2025 | HF 三张图（博客正文引用同一叙事） |
| 部署段 | le Chat、API、HF、NVIDIA NIM / NeMo / Nemotron Coalition | 可用性 |
| 模型卡 Key Features / Benchmarks / Usage | 6.5B 激活、温度建议、vLLM 命令、Eagle / NVFP4 | 外部补充 |

## 首发日怎么取

博客页面写 **March 16, 2026**。

外部补充（权重仓，不拿 `createdAt` 当首发日）：`mistralai/Mistral-Small-4-119B-2603` 的 git 历史上，最早一条是 `2026-03-16T20:33:00.000Z` 的 `Super-squash branch 'main' using huggingface_hub`，同日稍后有 `Upload image2.png`。collection `mistralai/mistral-small-4` 的 `lastUpdated` 为 `2026-03-16T22:57:00.828Z`。权重文件提交与博客同日，没有更早的官方公开事件。因此 `release-date` 记 **2026-03-16**。

仓库名里的 `2603` 是 Mistral 常用的年月戳（2026 年 3 月），与首发月一致，但**不能单独当证据**，只是命名习惯。

## 产品主张：一个模型，三种旧身份

### 原文写了什么

博客与模型卡把 Small 4 说成 hybrid：既可以当普通指令模型，也可以当推理模型。模型卡原句是它统一了三个家族的能力——**Instruct**、**Reasoning（previously called Magistral）** 和 **Devstral**——into a single, unified model。视觉被单独列为 multimodal input：接受文本和图像，输出文本。许可证 Apache 2.0。

任务书里还提到把 Pixtral 能力收进来。博客侧重点名的是 Magistral / Devstral / Instruct；视觉能力用「原生多模态」表述，没有在开篇把 Pixtral 四个字写成合并对象。**本文不把「合并 Pixtral」写成博客原句**，只写成：它有视觉输入，产品谱系上接 Mistral 过去的视觉模型。

### 我们如何解释

「统一」容易听成「三套权重焊在一起」。更老实的读法是：后训练目标从「专精一条」改成「同一套专家、同一套工具协议，用开关选深度」。这对 MoE 是自然的——专家可以分工，门控可以按 Token 选人——但博客**没有**展示专家是否真按技能聚类。没有路由热图，就不要编「专家 17 号负责代码」。

对用户，统一的价值在接口：同一 `model id`，同一套 function calling，同一套图像 URL，只改 `reasoning_effort`。对训练，统一通常更难：推理轨迹很长、指令要短、agent 要工具，三种损失很容易互相伤害。博客不谈权衡，所以「统一成功」目前只被评测图支持，不被训练故事支持。

### 开关不是第二种模型

模型卡 Recommended Settings：

- `reasoning_effort='none'`：不用推理。
- `reasoning_effort='high'`：用推理，复杂提示推荐。
- 温度：`high` 时用 0.7；`none` 时在 0.0–0.7 之间按任务选。

它还写：`none` 等价于 `Mistral-Small-3.2-24B-Instruct-2506` 那种聊天风格；`high` 的啰嗦程度等价于 `Magistral-Small-2509`。注意「等价」说的是**风格与冗长度**，不是参数量相等。Small 4 是 119B MoE，Small 3.2 是 24B dense，Magistral Small 是另一条推理线。

可迁移启发：如果你自己的服务也想「一个端点两种深度」，把深度做成**请求字段**比做成两个部署名更不容易让客户端分叉。代价是评测、缓存 key、计费都要带上这个字段，否则账单和 SLA 会对不齐。

## 架构规格：博客给骨架，权重卡给零件

### 博客明确写了的

- MoE：128 experts，4 active（top-4）。
- 119B total，约 6B active。任务上下文还提到「8B 含 embedding + output」——这句**没有**出现在我抓到的博客正文与 README 开篇里，本文不当成原文数字。若你在别的官方页看到 8B 含词表，应标那一页，不要回写进本篇当博客原话。
- 256k context。
- 原生多模态输入。
- Apache 2.0。

模型卡 Key Features 把激活量写成 **6.5B activated per token**，并写 256k context length、text+image in / text out、function calls、reasoning effort configurable per request。

### 外部补充：`params.json`（权重仓，不是博客）

主仓 `Mistral-Small-4-119B-2603` 的 `params.json`（访问时读取）给出更细的骨架。这些字段**博客没解释含义**，只说明开源权重长什么样：

| 字段 | 值 | 人话 |
|---|---|---|
| `dim` | 4096 | 残差流宽度 |
| `n_layers` | 36 | 层数 |
| `n_heads` / `n_kv_heads` | 32 / 32 | 注意力头；此处不是 GQA 那种 KV 头更少 |
| `head_dim` | 128 | |
| `hidden_dim` | 12288 | 稠密 FFN 隐层（若某层不用 MoE） |
| `vocab_size` | 131072 | 词表 |
| `tied_embeddings` | false | 输入输出词嵌入不绑定 |
| `max_position_embeddings` | 1048576 | 配置里的位置上限是 $2^{20}$，**比宣传的 256k 更大** |
| `yarn.original_max_position_embeddings` | 8192 | YaRN 原长度 |
| `yarn.factor` | 128 | $8192 \times 128 = 1{,}048{,}576$，和上面的 1M 对齐 |
| `moe.num_experts` | 128 | 与博客一致 |
| `moe.num_experts_per_tok` | 4 | 与博客一致 |
| `moe.expert_hidden_dim` | 2048 | 单个专家的 FFN 隐层，比 `hidden_dim` 窄很多 |
| `moe.num_shared_experts` | 1 | **共享专家**。博客没写。每个 Token 除 4 个路由专家外还固定走 1 个共享专家 |
| `q_lora_rank` | 1024 | 查询侧低秩，MLA 风格 |
| `kv_lora_rank` | 256 | KV 潜在秩 |
| `qk_rope_head_dim` / `qk_nope_head_dim` | 64 / 64 | RoPE 与非 RoPE 对半切 |
| `quantization.qformat_weight` | fp8_e4m3 | 主卡是 FP8 权重 |
| `vision_encoder.num_hidden_layers` | 24 | 视觉塔 24 层 |
| `vision_encoder.hidden_size` | 1024 | |
| `vision_encoder.patch_size` | 14 | |
| `vision_encoder.image_size` / `max_image_size` | 1540 | 图像边长规格 |
| `vision_encoder.spatial_merge_size` | 2 | patch merge |
| `vision_encoder.mm_projector_id` | patch_merge | |

collection 给主仓标了 `numParameters: 119401317952`，约 119.4B，与「119B」宣传一致；条目注释写 The FP8 checkpoint to ensure best accuracy。

### 我们如何解释这些零件

**激活量为什么会有 6B 和 6.5B 两个说法。** 粗算：路由 4 个专家，每个 `expert_hidden_dim=2048`，再加 1 个共享专家，注意力还是每层都算，再加上词嵌入与输出头。博客的「约 6B」是产品取整；模型卡 6.5B 把更多模块算进去了。没有官方分解表，就不要假装算得更精确。

**256k 对 1M。** 宣传上下文是 256k；配置允许 YaRN 拉到 1M。这在开源模型里常见：训练或后训练对齐到某一档，配置留出外推头。**博客没有证明 1M 可用。** 落地时仍应按 256k 做产品承诺，除非你自己测过更长。

**MLA 风格的低秩 KV。** `kv_lora_rank=256` 说明它走潜在注意力，而不是把 32 头全尺寸 KV 存下来。这对 256k 窗口几乎是刚需：否则 KV cache 会把「小激活」的优势吃掉。博客只吹 256k，没写 cache 怎么收。读权重的人应该把「能开 256k」和「MLA + FP8」绑在一起理解，而不是以为 dense 注意力免费送到 256k。

**视觉塔。** 24 层、patch 14、边长 1540、`spatial_merge_size=2`，看起来像 Mistral 3 / Pixtral 那一系的 patch-merge 投影，而不是「另起一个巨大 ViT」。博客只说 accepts both text and image input。不要补「训练了多少图」。

因果链：旧问题是 24B dense（Small 3.2）要同时做推理和 agent 会不够容量，做大 dense 又太贵 → 新设计是 119B MoE 只激活约 6B → 机制是 128 选 4 加共享专家，注意力侧用低秩 KV 撑长上下文 → 收益是产品上能合并专线 → 代价是 MoE 服务（专家并行、负载不均、FP8 核）比 dense 24B 难，博客当这些是已解决的工程。

可迁移启发：宣传「小激活」时，一定同时公布 **KV 压缩手段** 和 **量化格式**，否则长上下文会把激活优势对冲掉。Small 4 的权重卡其实比博客更诚实地露出了这两件工具。

## 效率主张：相对的是 Small 3，不是「世界上最快」

模型卡：在 latency-optimized setup 下，相对 Mistral Small 3，端到端完成时间降 **40%**；在 throughput-optimized setup 下，每秒请求数到 **3x**。博客还有 le Chat 上比 Small 3.2 更便宜的说法（写作时页面上有 2x cheaper 一类营销句，属于产品价而非基准）。

这些数字**没有**附硬件、batch、精度、是否投机解码。模型卡另外指出两条可选加速：

- Eagle head：`mistralai/Mistral-Small-4-119B-2603-eagle`，投机解码。
- NVFP4：`mistralai/Mistral-Small-4-119B-2603-NVFP4`，4-bit 浮点量化。

外部补充：collection 里除 FP8 主卡、NVFP4、eagle 外，还有 GGUF 等社区/官方衍生条目；主卡 README 指向 Unsloth GGUF、LM Studio、SGLang、transformers 主分支、Axolotl 的 `examples/mistral4`。这些是生态，不是训练证据。

解释：40% 和 3x 可以同时成立，因为优化目标不同——延迟档用小 batch、可能开投机；吞吐档用大 batch、吃满专家并行。读者不要把两个数乘在一起当成「又快三倍又短 40%」。

## 评测一：内部文本与视觉（读自博客图 Performance comparison across internal models）

图例：实心柱 Instruct，斜线柱 Reasoning。对照模型：Mistral Small 3.2、Mistral Medium 3.1、Mistral Large 3。纵轴从 20 起，不是从 0 起，**视觉上会放大差距**。

读数如下（Small 4 给出 Instruct / Reasoning；其余模型图上只有一根灰柱，按 Instruct 产品理解，原文未标它们是否开推理）：

**GPQA Diamond**（研究生级科学问答， voluntarily 很难猜）：

- Small 4：59.1 / 71.2
- Small 3.2：50
- Medium 3.1：65.7
- Large 3：64.1

打开推理后，Small 4 超过 Medium 3.1 和 Large 3 的灰柱。这是博客最想让你看见的一根。

**MMLU Pro**（更硬的多任务理解）：

- Small 4：73.5 / 78
- Small 3.2：69
- Medium 3.1：76.8
- Large 3：80.9

推理柱贴近 Medium，仍低于 Large 3 的 80.9。

**AllenAI IFBench**（指令遵循）：

- Small 4：35.7 / 48
- Small 3.2：34
- Medium 3.1：40.8
- Large 3：37.8

推理柱明显高于三条内部对照。Instruct 柱几乎贴着 Small 3.2。含义：短模式并没有把指令遵循「炼飞」，长思考才拉开。

**Arena Hard**（偏对话难度）：

- Small 4：55.8 / 58.3
- Small 3.2：43.1
- Medium 3.1：67.3
- Large 3：66.7

这里即使开推理也明显低于 Medium / Large。博客把这根和 GPQA 放同一张图里，等于承认：合并模型不是处处支配。

**MMMU-Pro**（多模态大学试题，图中归 Vision Benchmarks）：

- Small 4：46.3 / 60
- Small 3.2：49.1
- Medium 3.1：43.8
- Large 3：54

Instruct 视觉略低于 Small 3.2（49.1），开推理后到 60，超过 Large 3 的 54。若你只看短模式，会得出「视觉退步」；若只看长模式，会得出「视觉 SOTA 内部」。两根都要报。

解释：这张图的自变量其实是 **测时计算**，不是「Small 4 的权重天生在每条榜上都超过 Large 3」。GPQA / MMMU 吃长思维链；Arena Hard 更吃风格与聊天后训练，长思考加分有限（55.8 → 58.3）。

可迁移：内部对比图如果纵轴截断、再把 Reasoning 画成帽檐，读者会把测时计算误当成参数效率。读的时候先问：灰柱有没有权利开同样的 `reasoning_effort`？图上没说 Medium / Large 是否也开了高推理。

## 评测二：软件工程与计算机使用（读自博客第二张内部图，图注同样是 Performance comparison across internal models）

对照多了 **Devstral 2 123B**，这才是 coding 专线的内部标尺。

**SWE-Bench Verified**（人工核实过的 GitHub issue，测「真的改出补丁」）：

- Small 4：70.8 Instruct / 67.6 Reasoning
- Devstral 2 123B：72.2
- Large 3：68.1
- Medium 3.1：38.7
- Small 3.2：48.3

注意：这里 **Instruct 高于 Reasoning**（70.8 > 67.6）。和 GPQA 相反。可能的原因（我们的解释，不是原文）：agent 基准已经在环境里迭代，再套一层长思考容易超时、跑偏或重复计划；原文没分析。

**SWE-Multilingual**：

- Small 4：55.8 / 56.1
- Devstral 2：61.8
- Large 3：50.6
- Medium 3.1：29.9
- Small 3.2：32.2

专线仍领先，Small 4 明显好于 Large 3。

**Terminal Bench 2**（终端里干活）：

- Small 4：39.7 / 42.5
- Devstral 2：40
- Large 3：37.4
- Medium 3.1：3.4
- Small 3.2：6.6

Medium / Small 3.2 几乎不会用终端，说明这不是「参数大就会」，而是 agent 后训练有没有覆盖。Small 4 与 Devstral 2 打平附近。

**OSWorld-Verified**（真实计算机使用）：

- Small 4：58.6 / 62.8
- Devstral 2：70.1
- Large 3：57.6
- Medium 3.1：23.7
- Small 3.2：20.7

Devstral 2 仍明显更高。Small 4 的产品叙事是「接近专线」，不是「取代专线」。

因果链：旧问题是 coding agent 要单独的 Devstral，聊天用户用不上 → 新设计把 agent 后训练并进 Small 4 → 机制是同一套工具协议 + 可选推理 → 收益是通用模型在 SWE / 终端上从「不会」变成「接近 123B 专线」→ 代价是 OSWorld 仍让专线 7 分以上，且 SWE-Verified 上长思考甚至掉点。

可迁移：合并模型之后，**不要默认 reasoning=high 对 agent 更好**。这张图已经给出反例。你的脚手架应该允许按任务关思考。

## 评测三：和 Magistral 比推理（读自 HF 模型卡图 image3，标题 Performance Comparison Across Internal Models）

模型卡小节 Comparing Reasoning Models。柱都是高推理档。

| 基准 | Small 4-High | Magistral Medium 1.2 | Magistral Small 1.2 |
|---|---|---|---|
| AA LCR | 71.2 | 73 | 27 |
| AIME25 | 83.8 | 84.4 | 80.2 |
| Collie | 62.9 | 61.3 | 60.3 |
| LiveCodeBench | 63.6 | 66.1 | 60.7 |

AA LCR 上 Magistral Medium 1.2 仍高一点（73 > 71.2）；AIME25 几乎打平；Collie 上 Small 4 略高。Magistral Small 1.2 在 AA LCR 上只有 27，说明「Small」推理线在长上下文检索上很脆，Small 4 把这根补起来了。

注意：博客外部对比图里 Small 4 的 AA LCR Reasoning 读成 **72.4**，模型卡这张内部图是 **71.2**。同一家、同一模型、两个数。本文两处都录，**不平均、不挑选好看的那个**。可能是取数日期、解码参数或四舍五入不同，原文没说。

AA LCR 测的是长上下文里找证据做推理（Artificial Analysis 的 Long Context Reasoning 一类），不是普通 MMLU。Collie 是约束满足 / 指令约束类基准（名称本身在图上只有 Collie，博客没给论文链接）。LiveCodeBench 是按时间切的编程题，减轻刷题泄漏。AIME25 是竞赛数学。

解释：Small 4 作为「通用权重」在推理专线上没有全面碾压 Magistral Medium，但已经靠近，同时 Magistral Small 在 LCR 上的塌陷被补上。这支持产品合并，不支持「推理专线可以下架」——Medium 1.2 仍有两根柱更高。

## 评测四：分数对输出长度（读自 HF 模型卡 lcr.png / livecode.png / aime.png；博客与模型卡同一叙事）

模型卡 Comparison with other models 写了三句可用文字，必须和读图对上：

- 开推理的 Small 4 在三个基准上 matching or surpassing GPT-OSS 120B，且输出显著更短。
- AA LCR：Small 4 得 **0.72**，只用 **1.6K characters**；Qwen 要 **3.5–4x** 的输出（5.8–6.1K）才到可比分数。
- LiveCodeBench：超过 GPT-OSS 120B，且少约 **20%** 输出。

图的结构都是左 Accuracy、右 Average output length（k chars），Instruct 实心、Reasoning 斜线。横轴长度尺度每张不同，**不能**把三张图的棒长拿去目测比较绝对字符数，要读标注。

### AA LCR（Artificial Analysis LCR）

左分数（Instruct / Reasoning）：

- Small 4：62.9 / 72.4
- GPT-OSS 120B：70.7 / 72.7
- Claude Haiku：47.3 / 64.3
- Qwen3-next 80B：71.1 / 71.4
- Qwen3.5 122B：71.4 / 73.5

右长度：

- Small 4：Instruct 1.6K，Reasoning 1.6K（两档一样短，这很罕见）
- GPT-OSS 120B：1.2K / 3.3K
- Claude Haiku：0.9K / 1.7K
- Qwen3-next 80B：5.8K / 6.1K
- Qwen3.5 122B：5.8K / 6.1K

模型卡说的 0.72 与 1.6K 对上 Reasoning 柱 72.4 与 1.6K。Qwen 两条几乎不随 Instruct/Reasoning 改变长度，说明它们默认就写得很长。Small 4 的卖点是：**分数到 GPT-OSS 附近，字数停在 1.6K**。GPT-OSS 的 Instruct 甚至更短（1.2K）且 Instruct 分更高（70.7 > 62.9），开推理后才和 Small 4 绞在一起。

### LiveCodeBench

左分数：

- Small 4：54 / 63.6
- GPT-OSS 120B：48 / 61.8
- Claude Haiku：47 / 61
- Qwen3-next 80B：66 / 70.3
- Qwen3.5 122B：68 / 71.7

右长度：

- Small 4：10.3K / 14.7K
- GPT-OSS：7.6K / 12.5K
- Haiku：7.3K / 13.3K
- Qwen3-next：13.1K / 16.9K
- Qwen3.5：14.6K / 17.6K

「超过 GPT-OSS 且少约 20% 输出」：Reasoning 14.7K 对 12.5K **并不是更短**。20% 那句更像相对某次设置或相对 Qwen，原文把 GPT-OSS 和 20% 写在同一句里，和这张图对不齐。**标成原文表述与读图不一致**，不帮忙圆。相对 Qwen3.5 的 17.6K，14.7K 大约短 16%，接近「少 20%」的量级。读者应以图为准。

Qwen 两条在 LiveCodeBench 上分数明显高于 Small 4（70.3 / 71.7 vs 63.6）。模型卡「matching or surpassing GPT-OSS 120B across all three」没有声称超过 Qwen。这是诚实的上限。

### AIME 2025

左分数：

- Small 4：36 / 84
- GPT-OSS 120B：45 / 89
- Claude Haiku：34 / 83
- Qwen3-next 80B：65 / 88
- Qwen3.5 122B：80 / 93

右长度：

- Small 4：3.9K / 27.9K
- GPT-OSS：1.6K / 14.9K
- Haiku Instruct：1.2K（Reasoning 长度图上未标出清晰数字，不编）
- Qwen3-next：6.6K / 17.7K
- Qwen3.5：7.7K / 26.4K

AIME 上 Small 4 开推理到 84，仍低于 GPT-OSS 89 和 Qwen3.5 93，而且 Reasoning 输出 **27.9K**，是这张图里最长之一，几乎与 Qwen3.5 的 26.4K 同级。**数学竞赛并没有展示「更短」的优势。** 模型卡那句「显著更短」主要被 AA LCR 那张图撑住，不能推广到 AIME。

解释：测时计算的收益高度任务依赖。LCR 上短证据链就够；AIME 上大家都会把思维链拉到数万字符。Small 4 的产品开关让你在 LCR 上保持短，在 AIME 上仍会变长——开关管的是「开不开」，不是「有预算上限」。若你要控成本，还得自己加 `max_tokens`。

可迁移：发「Score vs Output Length」时，**三个任务三张图**比平均一个「效率分数」有用，因为效率会在任务之间翻面。AIME 这张就是反面教材。

## 多模态、工具、系统提示

模型卡能力列表：Vision、Multilingual（英、法、西、德、意、葡、荷、中、日、韩、阿等；YAML 语言列表更长，含 hi、bn、fa 等）、System Prompt 强遵循、Agentic native function calling 与 JSON、256k、Apache 2.0。

Usage 里三个官方例子（外部补充，属于模型卡，不是博客散文）：

1. Instruction Following：`reasoning_effort="none"`，温度 0.1，系统提示从 `SYSTEM_PROMPT.txt` 按日期填空。
2. Tool Call：同一请求里塞一张数学试卷图 + `my_calculator` 工具，`none` 模式。这是「视觉 + 工具」而不是「先看图再另开模型」。
3. Vision Reasoning：游戏对战截图，`high` + 温度 0.7，解码时保留 `[THINK] [/THINK]`。

这三份示例比任何形容词都清楚地定义了 Small 4 的接口形状：**图像是 content 数组里的一种 part，工具是 OpenAI 兼容的 tools，思考是 parser 切出来的块。**

vLLM 推荐命令（模型卡，外部补充）值得读，因为它暴露了运行时假设：

```text
vllm serve mistralai/Mistral-Small-4-119B-2603 \
  --max-model-len 262144 \
  --tensor-parallel-size 2 \
  --attention-backend FLASH_ATTN_MLA \
  --tool-call-parser mistral \
  --enable-auto-tool-choice \
  --reasoning-parser mistral \
  --max_num_batched_tokens 16384 \
  --max_num_seqs 128 \
  --gpu_memory_utilization 0.8
```

`--max-model-len 262144` 对上 256k。`--attention-backend FLASH_ATTN_MLA` 对上权重里的 MLA 字段。`--tensor-parallel-size 2` 说明官方默认它不是单卡玩具——119B FP8 即使激活 6B，权重也要放下。`mistral_common >= 1.11.0` 被写成硬依赖。

Transformers 示例用的类名是 `Mistral3ForConditionalGeneration`，要装 transformers **主分支**。类名带 3，说明实现挂在 Mistral 3 多模态那套代码上，而不是另起 `Mistral4*`。这与视觉塔规格一致。

## 部署与跟进链接

博客把可用性铺在一串链上。按手册第 8 条分类处理：

**伴随 / 前作（优先级高，定义产品边界）：**

- Magistral Small 2509、Mistral Small 3.2 24B Instruct 2506：模型卡用来定义 `none` / `high` 的风格锚。本文已用它们解释开关，不把它们的训练 recipe 写进 Small 4。
- Devstral 2 123B：内部 coding 图的专线对照。没有单独再读一篇 Devstral 技术报告（本任务原件是 Small 4 博客）；分数只来自 Small 4 这张图。

**权重与模型卡：**

- Collection `mistralai/mistral-small-4`：FP8 主卡、NVFP4、eagle 等。描述句：open-weight，granular MoE，fuses instruct, reasoning and agentic skills。
- README Benchmarks 图：image2 与博客内部文本图同构；image3 为 Magistral 对照；lcr / livecode / aime 为外部对照。

**推理栈：**

- vLLM（官方推荐）、SGLang、llama.cpp / Unsloth GGUF、LM Studio、transformers 主分支、Axolotl `examples/mistral4`。

**NVIDIA 生态（博客点名，跟进结果）：**

- 博客提到 NVIDIA NIM、NeMo、以及 Nemotron Coalition 一类合作叙事。本次用公开 URL 抓 NeMo 文档时未落到一份稳定的「Small 4 专页」（Megatron-Bridge Mistral 页 404）。**因此 NVIDIA 侧能确认的只有「博客声称可走 NVIDIA 部署路径」，不能从本文冒充已经核对过 NIM 延迟数字。** 需要 NIM 吞吐时，应另开 `build.nvidia.com` 当时的模型页，不要用本篇缓存。

**评测集（名字级跟进，不把第三方论文写进原文结论）：**

- GPQA Diamond：专家级科学问答，防检索。
- MMLU Pro：MMLU 的加难版。
- IFBench：指令遵循压力测试。
- Arena Hard：对话竞技场难题子集。
- MMMU-Pro：多模态大学试题加难版。
- SWE-Bench Verified / SWE-Multilingual：真实仓库修补。
- Terminal Bench 2：终端任务。
- OSWorld-Verified：GUI / OS 操作。
- AIME 2025：竞赛数学。
- LiveCodeBench：按时间切的代码题。
- AA LCR：长上下文推理。

这些定义来自各基准自己的公开说明，属于外部补充；博客只给名字和柱高。

## 原文没写、本文也不补的东西

明确列出，避免读完产生「其实有 recipe」的错觉：

- 预训练数据配比、多语言比例、代码比例、视觉数据来源与分辨率课程。
- token 数、GPU 时、集群、稳定期损失曲线。
- 路由算法（除 top-4 这个结果）、负载均衡损失、共享专家是否始终开启的训练细节。
- 长上下文怎么从 8k 训到 256k：YaRN 出现在配置里，课程没写。
- 推理后训练是 RL、拒绝采样还是 Magistral 那套，完全没写。
- 与 Devstral 数据是否共用、是否蒸馏自 Large 3，没写。
- 安全、红队、系统提示全文（仓库有 `SYSTEM_PROMPT.txt`，本任务不把它整篇贴进解读）。
- 专家可视化、模态路由是否分离。
- 评测的 $n$、温度、工具环境、是否自一致性。AIME 84 这种数对采样极其敏感。

「没公开做法」不是拒收理由，是结论：Small 4 的公开材料是**规格 + 产品评测 + 部署配方**，和手册里举过的 Mistral Large 3 model card 同一类。

## 限制与读图时要自己打折的地方

1. 内部图纵轴截断（从 20 起），差距被放大。
2. Medium / Large 灰柱未标明是否允许同等推理预算。
3. AA LCR 71.2 与 72.4 两数并存。
4. LiveCodeBench「比 GPT-OSS 短 20%」与图上 14.7K vs 12.5K 冲突。
5. AIME 上既没赢 Qwen，也没更短。
6. SWE-Verified 上 Reasoning 掉点，和「开思考就更好」的营销直觉相反。
7. 主卡 FP8，「最佳精度」是相对 NVFP4 / GGUF 的产品分层，不是相对 BF16 论文消融。
8. 类名仍叫 Mistral3，文档散落在博客、README、params.json 三处，会对不齐（激活 6B vs 6.5B，上下文 256k vs 配置 1M）。

## 可迁移启发（收回全景）

1. **用请求级开关合并产品线，而不是合并部署名。** `reasoning_effort` 让计费、评测、客户端都还能分档。
2. **宣传激活参数时，同时交出 KV 压缩和量化。** 否则 256k 是空头支票。Small 4 把 MLA 和 FP8 放进了权重卡，虽然博客懒得写。
3. **Agent 基准上先测关思考。** 本篇 SWE-Verified 是现成反例。
4. **Score vs Length 必须分任务。** LCR 上短，AIME 上长，平均掉一个「高效」会骗人。
5. **专线对照必须画在图上。** 没有 Devstral 2，你会以为 70.8 已经是内部天花板；有了它，你看见通用模型还差一口气。
6. **产品博客当规格书读，不当论文读。** 能迁移的是接口形状和评测诚实度，不是学习率。

## 和 Small 3 / Magistral / Devstral / Large 3 到底什么关系

把四条线摊开，是为了防止面试时把名字说串。

**Mistral Small 3.x** 是 24B 量级的 dense 指令模型。Small 4 的 `none` 档被官方说成「聊起来像 Small 3.2」。评测图上 Small 3.2 在 GPQA、SWE、终端、OSWorld 上全面落后，只在 MMMU-Pro 的短模式（49.1）略高于 Small 4 的 Instruct 视觉（46.3）。所以「像 Small 3.2」指风格，不指分数。

**Magistral** 是推理专线。模型卡用 Magistral-Small-2509 定义 `high` 的啰嗦程度，又用 Magistral Medium 1.2 / Small 1.2 做内部推理对照。Small 4-High 在 AIME25 上 83.8，Medium 1.2 是 84.4，几乎打平；AA LCR 上 Medium 仍高。合并成功的证据是 Magistral Small 1.2 在 LCR 上掉到 27，Small 4 没有跟着掉。失败的证据是 Medium 还在。

**Devstral 2 123B** 是软件工程专线。SWE-Verified 72.2 vs Small 4 的 70.8，差一口气；OSWorld 70.1 vs 62.8，差一截。Terminal Bench 2 几乎打平。产品上可以说「通用模型够用了」，工程上专线仍有计算机使用优势。

**Mistral Large 3** 出现在内部图里当上限。MMLU Pro、Arena Hard 上 Large 3 更高；GPQA 和 MMMU-Pro 上 Small 4 开推理可以超过 Large 3 的灰柱。这不能读成「Small 比 Large 强」，因为灰柱可能没给同等思维链。它只能读成：在博客选择的画法下，测时计算让 6B 激活摸到了更大模型的边。

**Ministral 3** 根本不是同一条故事：那是从 24B 父模型剪出来的 dense 小模型，有蒸馏伪代码。Small 4 是往上堆专家。把「剪」和「MoE 合并」说成一种方法，是概念错误。

## 服务形态：为什么 6B 激活不等于 6B 显存

面试里最容易错的一句是：「激活只有 6B，所以一张消费卡就能跑。」权重仓自己打脸：FP8 主卡仍要放下约 119B 的专家参数，官方 vLLM 示例 `tensor-parallel-size 2`。激活量管的是**每 Token 的计算和一部分激活内存**，不管**权重驻留**。

更细一点：

- 专家若做专家并行，空闲专家仍占显存，只是不算 FLOP。
- KV cache 按序列长度涨。256k 窗口下，哪怕 MLA 把 KV 压到 `kv_lora_rank=256`，长请求仍会吃显存。官方还把 `--max_num_batched_tokens 16384` 和 `--max_num_seqs 128` 写进推荐命令，这是在吞吐档用批处理换利用率，不是在证明单卡 256k。
- Eagle 头是投机解码：草稿模型猜未来 Token，主模型一次验证多步。它改延迟分布，不改权重规模。
- NVFP4 改的是权重精度与带宽，会伤点精度，所以 collection 把 FP8 标成 best accuracy。

因此「40% 端到端时间」和「3x RPS」必须带回它们的定语：latency-optimized vs throughput-optimized。没有硬件表，这两个数只能当数量级宣传。

## 把评测读成面试题，而不是读成排行榜

假设面试官丢来「Small 4 是不是已经超过 Large 3」。正确答法不是报 GPQA 71.2，而是拆三层：

第一层，原文图上，GPQA Diamond 的 Reasoning 柱 71.2 高于 Large 3 的 64.1，MMMU-Pro 的 60 高于 54。这是博客自己画的。

第二层，Arena Hard 58.3 对 66.7，MMLU Pro 78 对 80.9，Small 4 没过。同一张图里已经有反例。

第三层，Large 3 那根灰柱有没有开推理、采样几次、要不要工具，原文没写。所以「超过」只在「博客选择的对照柱」这个条件下成立。

再假设问「开思考是不是总是更好」。用两根柱回答：GPQA 59.1 → 71.2 是更好；SWE-Verified 70.8 → 67.6 是更差。开关的产品意义是让你选，不是让你永远拨到 high。

再假设问「是不是又短又强」。用 AA LCR 和 AIME 对着答：LCR 上 Reasoning 1.6K 字符拿到 72.4，相对 Qwen 的 5.8–6.1K 确实短；AIME 上 Reasoning 27.9K 拿到 84，并不短，也不如 Qwen3.5 的 93。效率是任务函数。

## 设计原则怎么从这页博客里抽，而不靠脑补训练

手册要求「可迁移的设计原则」。在没有 recipe 时，原则只能从**接口和对照实验设计**里抽，不能从想象的损失公式里抽。

原则 A：能力合并的最小接口是「同一权重 + 同一工具协议 + 一个深度字段」。比再维护四个 `model id` 更便宜的是客户端，更贵的是后训练。博客用评测图证明客户端那一侧已经成立，后训练那一侧保持黑盒。

原则 B：内部对照必须包含**专线**，否则通用模型永远看起来像第一。Devstral 2 和 Magistral Medium 1.2 出现在图上，是这篇博客相对克制的地方。

原则 C：对外对比用「分数–长度」平面，而不是只报分数。即使 LiveCodeBench 那句 20% 和读图打架，平面本身仍比单分数诚实，因为它逼你看见 Qwen 用更长输出换更高分。

原则 D：开源权重把服务所需的不变量写进配置：MLA 后端名、YaRN 因子、视觉 patch merge、FP8 格式、mistral_common 版本。这些不变量比博客形容词更有用。复现不了训练，但复现得了服务。

原则 E：许可证选 Apache 2.0，是在说分发权，不是在说可复现。两者经常被媒体写成一件事。

## 若把 Small 4 当成系统设计案例

把一次请求走一遍：

用户消息可以带图。视觉塔按 patch 14 切，边长规格 1540，merge 2，投到文本隐空间。文本走 36 层、宽 4096 的骨干。每层注意力按 MLA 取 KV 潜在；前馈按门控从 128 专家里点 4 个，并固定过 1 个共享专家。`reasoning_effort=high` 时，解码器被允许向 `[THINK]` 块里写长链，parser 再把思考和可见回答切开。若带了 tools，`mistral` 工具解析器吐 function call，环境执行后再把结果当下一轮消息。`none` 时这条链被要求短，温度可以降到 0.1。

这条路径上，博客保证的是「能这样做」；权重卡保证的是「字段和推荐命令」；谁都不保证「思考块的 token 预算有内部上限」。所以成本控制仍是应用层的事。

把并发再走一遍：吞吐档希望同时塞 128 条序列，但每条可能点不同专家，专家并行的气泡会随路由熵变大。博客的 3x RPS 如果成立，隐含工程上已经把气泡压住了——具体怎么压，没写。面试里可以把「MoE 路由熵 vs 批利用率」当成开放题，不要假装 Small 4 给了答案。

## 还缺一张「测时计算曲线」，以及为什么博客不画

内部图只给了 `none` 和 `high` 两档，没有 medium，也没有把横轴做成 token 预算。真正的推理模型论文通常会画 pass rate 随生成长度或采样数变化的曲线。Small 4 既然把深度做成连续产品旋钮，按理说最该画这张。它没画，可能的原因只是猜测，所以标成我们的解释：产品博客要的是「拨一下就变另一个模型」的直觉，两根柱最容易印在脑子里；连续曲线会暴露中间档不平滑、某些任务开一点思考就够、某些任务开满也追不上 Qwen。

对使用者，缺曲线的后果很具体。你无法从原文知道：把 `max_tokens` 从 4k 接到 16k，GPQA 还会不会涨；AA LCR 已经在 1.6K 字符饱和，再给长度会不会只烧钱。工程上你只能自己扫。这不是指责，是把「原文证据强度」标清楚：开关的存在被 Usage 证明，开关的剂量效应几乎没被证明。若面试被追问「那 medium 档呢」，诚实回答是：公开材料里没有第三档的柱，也没有把 `reasoning_effort` 的取值集合写成枚举以外的连续旋钮。你在 API 里能传的值以当时文档为准，本篇只核实了 `none` 与 `high`。

## 开源生态里哪些东西算官方、哪些只是能跑

模型卡点名的 vLLM 命令、mistral_common 版本、FLASH_ATTN_MLA、mistral 工具解析器和 reasoning parser，应视为官方服务合同。Axolotl 的 `examples/mistral4`、Unsloth GGUF、LM Studio、SGLang、transformers 主分支，是「官方承认能连」的生态，不保证与 FP8 主卡逐分对齐。Eagle 和 NVFP4 是官方衍生权重，用途写在文件名里：一个换延迟，一个换带宽。不要把 GGUF 上某条 SWE 分数写回本篇当博客数字。

collection 的 `numParameters` 119401317952 是 Hugging Face 扫出来的计数，和「119B」宣传一致，精度到个位。它仍然不是训练配方。

最后用一条因果链把全文收住：旧问题是 Instruct、推理、视觉、coding 四条产品让用户和基础设施分叉 → 新设计是一份 119B MoE 权重、约 6B 激活、请求级 `reasoning_effort`、原生图文输入 → 机制是 128 专家 top-4 加共享专家，注意力侧用低秩 KV 去撑 256k，服务侧用 FP8、可选 NVFP4 和 Eagle → 收益是内部图上推理档摸到 Medium / Large 的边、coding 档接近 Devstral 2、LCR 上用 1.6K 字符打到 GPT-OSS 附近 → 边界是没有 recipe、没有剂量曲线、Arena Hard 与 AIME 并不占优、SWE-Verified 上长思考还会掉点、NIM 数字本次没有跟进到可引用的页面。能带走的是接口形状和读图纪律，不是一份可以照着练的配方。本篇访问日是 2026-09-15，距官宣约半年，仍没有官方技术报告补上数据配比与消融；半年后的沉默本身也是边界的一部分。博客图上的分数只证明产品主张能被内部对照撑住，不能外推成可复现论文。图上没写的采样次数、温度和工具环境，一律当作未知，不要用二手榜去补空。

## 关键词回看

MoE 128 选 4、共享专家、约 6B / 6.5B 激活、119.4B 总参、MLA 低秩 KV、YaRN、256k 宣传 / 1M 配置、`reasoning_effort`、Magistral 风格锚、Devstral 专线对照、FP8 主卡、NVFP4、Eagle、Apache 2.0、`FLASH_ATTN_MLA`、`[THINK]` 块。这些词在本文里都已经接到「旧问题 → 设计 → 机制 → 收益 → 边界」上，不再单列空定义。

## 参考资料

- 官方博客 Mistral Small 4，2026-03-16，访问 2026-09-15：<https://mistral.ai/news/mistral-small-4>
- HF collection：<https://huggingface.co/collections/mistralai/mistral-small-4>
- 主模型卡：<https://huggingface.co/mistralai/Mistral-Small-4-119B-2603>
- NVFP4 / eagle 权重名见模型卡链接。
- vLLM、mistral-common ≥ 1.11.0、Axolotl examples/mistral4：见模型卡 Usage。

同库对照（外部，防写串）：`reports/Mistral/Ministral-3.md` 讲的是 24B 父模型剪枝出的 dense 小模型，方法具体、数据留白。Small 4 连方法都留白，只给规格。不要把 Ministral 3 的蒸馏伪代码安到本篇头上。

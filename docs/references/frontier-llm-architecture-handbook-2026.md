# 前沿基础模型架构手册（截至 2026 年 8 月 5 日）

> 覆盖范围：所有公开权重、公开配置的前沿模型。闭源模型（GPT-5.x、Claude、Gemini）单列一节说明「我们不知道什么」。
> 语言风格：尽量说人话。术语第一次出现时都给类比。
> 主要一手来源：Sebastian Raschka 的 LLM Architecture Gallery（93 个模型的 config 对照）、各家技术报告、HuggingFace config.json。

---

## 0. 先给你一句话

**2026 年架构竞争的唯一主战场是：怎么在不变笨的前提下，把 KV cache 和长上下文的注意力开销砍下去。**

其他所有变化（MoE 怎么切、用什么优化器、残差怎么连）都是围绕这个主线的配套动作。理解了这一句，下面所有东西都能挂上去。

**为什么是这个主战场？** 因为模型的用途变了。2023 年模型是「问一句答一句」，2026 年模型是 **reasoning + agent**：一次任务里模型要吐几万 token 的思考链、读几十个工具返回、维持几十万 token 的上下文。这时候瓶颈不再是「参数够不够多」，而是：

- **KV cache 占显存**：上下文越长，缓存越大，能同时服务的用户越少
- **prefill 的注意力是平方复杂度**：$O(L^2)$，L 是序列长度
- **decode 每一步都要把整个 KV cache 从显存读一遍**：这是纯内存带宽开销，跟算力无关

所以你会看到，2026 年几乎每一篇技术报告的架构章节，标题里都有 "efficiency" 或 "long-context"。

---

## 1. 架构变量总清单

先把「到底有哪些旋钮」列全，后面逐个展开。这张表是这份手册的骨架。

| 大类 | 变量 | 典型取值 |
|---|---|---|
| **注意力** | KV 头共享方式 | MHA / MQA / GQA |
| | 每 token 表示压缩 | 无 / MLA / CCA |
| | 看多少 token | 全看 / 滑动窗口 / 稀疏选择（DSA、NSA） |
| | 序列维度压缩 | 无 / CSA / HCA |
| | 有没有 KV cache | 有（softmax 注意力） / 无（线性注意力、SSM） |
| | 混合比例 | 3:1 / 5:1 / 7:1 / 全 attention |
| | 跨层共享 KV | 有 / 无 |
| | 逐层预算 | 均匀 / 逐层不同 query 头数 |
| | 附加件 | QK-Norm、输出门控、attention sink、attention bias |
| **位置编码** | 方式 | RoPE / NoPE / partial RoPE / 混合 |
| | 长度外推 | YaRN / 无 / 只在部分层用 |
| **MoE** | 专家总数 | 8 → 896 |
| | 每 token 激活数 | 1 → 16 |
| | 专家粒度 | 粗（大专家少）/ 细（小专家多） |
| | 共享专家 | 有 / 无 / 变体（always-on SwiGLU） |
| | Router 打分 | softmax / sigmoid / ReLU / hash / 分位数 |
| | 负载均衡 | aux loss / aux-loss-free bias |
| | dense 前缀 | 前 N 层不做 MoE |
| | 专家计算空间 | 原始维度 / 压缩隐空间（Latent MoE） |
| **归一化** | 类型 | LayerNorm → RMSNorm（已统一） |
| | 位置 | pre-norm / post-norm / sandwich norm |
| **FFN / 激活** | 激活函数 | SwiGLU(SiLU) / GELU / ReLU² |
| **残差流** | 结构 | 单流 / 超连接 HC / mHC / Attention Residuals |
| **训练目标** | 多 token 预测 | 无 / MTP-1 / MTP-3 |
| **优化器** | 选择 | AdamW / Muon / MuonClip / SOAP |
| **数值精度** | 训练与权重 | BF16 / FP8 / MXFP4 |
| **嵌入** | 特殊设计 | 常规 / 每层嵌入 PLE / N-gram 嵌入 |

**注意一个反直觉的事实**：这张表里几乎没有一项直接决定模型「聪不聪明」。Raschka 自己的总结是——架构变化主要降低运行成本，而质量表现主要由**数据质量、数据量和训练配方**驱动。架构是效率的战场，不是智力的战场。

---

## 2. 注意力：全谱系拆解

### 2.1 先搞清楚 KV cache 是什么

**类比**：模型生成文本像一个人边写边回顾。每写一个字，他都要回头看一遍前面写过的所有字，决定下一个字写什么。

- **Q（Query，查询）**：我现在这个位置想找什么
- **K（Key，键）**：前面每个位置的「索引标签」
- **V（Value，值）**：前面每个位置的「实际内容」

生成第 100 个 token 时，前 99 个 token 的 K 和 V 已经算过了，没必要重算——**存起来，这就是 KV cache**。

**代价**：缓存大小 = 层数 × KV 头数 × 头维度 × 2（K和V）× 序列长度 × 精度字节数。它随上下文**线性增长**。

一个直观的数：Gemma 3 27B 每个 token 要 496 KiB 的 KV cache。128k 上下文 = 约 62 GB。一张 H100 只有 80 GB——**光缓存就吃掉大半张卡**，模型权重都还没放。

而 DeepSeek V4-Flash 每 token 只要 5.4 KiB，差了近 **100 倍**。这就是过去两年架构进化的全部意义。

**所以记住这个公式，下面五条技术路线各砍其中一项：**

$$\text{KV cache} = \underbrace{L_{\text{layers}}}_{\text{①跨层共享}} \times \underbrace{n_{kv} \times d_h}_{\text{②GQA/MLA/CCA 压这里}} \times 2 \times \underbrace{S}_{\text{③SWA/CSA/HCA 压这里}} \times \underbrace{b}_{\text{④量化}}$$

而 **⑤线性注意力**更狠：直接让这一项变成常数。

---

### 2.2 路线一：减少 KV 头数（MHA → MQA → GQA）

**MHA（Multi-Head Attention，多头注意力）**：原版。每个 query 头配一套自己的 K 和 V。

**MQA（Multi-Query Attention）**：所有 query 头**共用一套** K/V。缓存砍到 1/头数，但质量掉得比较明显。

**GQA（Grouped-Query Attention，分组查询注意力）**：折中。把 query 头分成几组，**每组共用一套 K/V**。

> **类比**：MHA 是每个学生配一个专属图书管理员；MQA 是全校共用一个；GQA 是每个班配一个。

GQA 是 2023–2025 年的绝对主流，现在依然是「保守派」的默认选择。典型配置是 4:1 到 8:1（几个 query 头共用一套 KV）。

**代表**：Llama 3/4、Qwen3 全系、GLM-4.5/4.7、GPT-OSS、MiniMax M2/M2.5/M2.7、Grok 2.5、Gemma 3/4

---

### 2.3 路线二：压缩每个 token 的表示（MLA、CCA）

**MLA（Multi-head Latent Attention，多头潜在注意力）** —— DeepSeek 的招牌。

思路：不存完整的 K 和 V，而是把它们**压缩成一个低维的「潜在向量」** 存起来，用的时候再投影回去。

> **类比**：GQA 是「几个人共用一本书」，MLA 是「把书压缩成摘要存着，要用时再展开」。

**关键权衡**：MLA 省显存，但**增加计算**（多了压缩和解压两次矩阵乘）。在长上下文场景下这笔账划算，因为 decode 阶段是内存带宽瓶颈不是算力瓶颈。

数字说话：DeepSeek V3 671B 有 61 层，KV cache 只要 68.6 KiB/token，比 Gemma 3 27B（496 KiB）少了 7 倍——**参数量大 25 倍，缓存反而小 7 倍**。

**代表**：DeepSeek V3/V3.2、Kimi K2/K2.5/K2.6、Mistral Large 3、Mistral Small 4、GLM-5、Sarvam 105B、LongCat

**CCA（Compressed Convolutional Attention，压缩卷积注意力）** —— Zyphra 的 ZAYA1-8B 用的，和 MLA 是近亲但更激进。

MLA 只是把 KV **存**成压缩形式，算注意力时还要展开到完整头空间。CCA 更进一步：**直接在压缩空间里做注意力运算**，算完再展开。

好处：不仅省 KV cache，还省 prefill 和训练时的注意力 FLOPs。

那个 "Convolutional"（卷积）是补丁：压缩会让 Q 和 K 变窄、表达力下降，所以加一个轻量卷积给压缩后的 Q、K 补充局部上下文。**只对 Q 和 K 做，不对 V 做**——因为 Q、K 决定「看哪里」（需要精细），V 只是「被平均的内容」。

CCA 论文自称在同等压缩率下优于 MLA，但注意这是作者自己的实验，没有第三方复现。

---

### 2.4 路线三：少看一些 token（SWA、DSA）

**SWA（Sliding Window Attention，滑动窗口注意力）**：每个 token 只看前面固定数量的 token，比如 512 或 1024 个。

> **类比**：读书时不再每写一句都翻回第一页，只看最近两页。

代价很明显：**丢失长程信息**。所以没人用纯 SWA，都是**混着用**——大部分层用 SWA，少数层用全局注意力。

**局部:全局的比例是一个真实的设计旋钮**，各家给出的答案很不一样：

| 模型 | 比例 | 窗口大小 | 备注 |
|---|---|---|---|
| Gemma 2 | 1:1 | 4096 | 起点 |
| Gemma 3 / Gemma 4 | **5:1** | 1024 | 窗口比 Gemma 2 缩小到 1/4 |
| GPT-OSS | **1:1** 交替 | — | 最保守 |
| OLMo 3 | 3:1 | — | YaRN 只在全局层用 |
| Step 3.5 Flash | 3:1 | — | — |
| Laguna XS.2 | 3:1 | **512** | — |
| Xiaomi MiMo-V2-Flash | 5:1 | **128** | 窗口小得离谱 |
| Arcee Trinity Large | 3:1 | — | 512k 上下文 |

**MiMo-V2-Flash 那个 128 token 窗口值得单独说**：40 个 SWA 层里每个 token 只能看前面 128 个字，几乎相当于一个 n-gram 模型；全部长程能力都压在 8 个全局层上。而它的 Agent 分是 47.3，很能打。这说明**局部层的窗口可以压得比大多数人以为的小得多**。

**DSA（DeepSeek Sparse Attention，稀疏注意力）**：不是「只看最近的」，而是**用一个轻量选择器动态挑出最相关的 top-k 个历史 token 去看**。

> **类比**：SWA 是「只看最近两页」，DSA 是「用索引挑出最相关的十页来看」。

DSA 保留了访问任意历史位置的能力，代价是多一个选择器。DeepSeek V3.2 首次上线，GLM-5/5.1 跟进。

---

### 2.5 路线四：压缩序列本身（CSA / HCA，DeepSeek V4）

这是 2026 年最新、也最狠的一招。前面所有方法压的都是「每个 token 存多少」，**CSA/HCA 压的是「存多少个 token」**。

- **CSA（Compressed Sparse Attention）**：每 4 个 token 压成 1 个缓存条目（m=4），再用 DSA 风格的选择器稀疏挑选
- **HCA（Heavily Compressed Attention）**：每 **128** 个 token 压成 1 个条目（m'=128），因为条目已经极少，反而可以对它们做**稠密**注意力

> **类比**：CSA 是「把书按四页一组做摘要，然后挑几组细读」；HCA 是「把书按 128 页一章做极简摘要，然后每章都扫一遍」。

**为什么两个都要？** 因为它们互补：CSA 保留细节但只能挑着看，HCA 看得全但极度粗糙。DeepSeek V4 **交替使用 CSA 层和 HCA 层**，两条路都保留一个 128-token 的滑动窗口分支处理最近的未压缩 token。

效果（DeepSeek V4 论文自报，相对 V3.2）：1M 上下文下，V4-Pro 单 token 推理 FLOPs 降到 **27%**、KV cache 降到 **10%**；V4-Flash 更狠，FLOPs **10%**、cache **7%**。

⚠️ **但要注意**：论文没有做消融实验。这些数字是「完整 V4 配方」的结果，里面还混着更好的数据、Muon 优化器、mHC、精度优化和推理系统改动。**不能把收益全算在 CSA/HCA 头上**。

---

### 2.6 路线五：干脆不要 KV cache（线性注意力 / SSM）

这是 **hybrid attention 的核心**，也是你问题的正主。

**问题的根源**：softmax 注意力的 KV cache 随序列长度线性增长，这是它的**结构性缺陷**，压缩只能减小常数，改变不了增长率。

**线性注意力的解法**：不存所有历史，只维持一个**固定大小的状态向量**，新 token 来了就更新这个状态。

> **类比**：
> - softmax 注意力 = 把每一天都写成一篇日记，全部留着，随时可以翻回去查某一天
> - 线性注意力 = 只写一份不断更新的「近况总结」，新事发生就改总结。**总结的长度永远不变**

$$\text{softmax attention: } O(L) \text{ 缓存} \quad\longrightarrow\quad \text{线性: } O(1) \text{ 状态}$$

**代价一目了然**：固定大小的状态装不下所有细节，必须**决定忘掉什么**。这在「大海捞针」类的精确检索任务上会掉分——这是线性注意力最经典的弱项。

主流的四个具体机制：

| 机制 | 出处 | 核心区别 |
|---|---|---|
| **Mamba-2** | 状态空间模型（SSM） | 学一个状态空间滤波器，本质是时间维度上的动态卷积 |
| **Gated DeltaNet (GDN)** | Qwen3-Next / Qwen3.5 | delta 规则写入「快权重记忆」，两个门：alpha 控遗忘、beta 控写入强度。**每个头一个标量遗忘门** |
| **KDA（Kimi Delta Attention）** | Kimi Linear / K3 | GDN 的精细版：**每个特征通道一个独立的遗忘门**，而不是每头一个标量 |
| **Lightning Attention** | MiniMax-01 / Ling 2.5 | 另一种线性注意力实现 |

**GDN 和 KDA 的差别值得记住**，因为这是 2026 年最典型的「精细化」思路：

- GDN：整个头一起决定「这一步忘多少」——粗粒度，一刀切
- KDA：每个特征通道自己决定——细粒度，可以「忘掉时间信息但保留语义信息」

Kimi 的 K3 直接把 KDA 作为整个 2.8T 模型的骨干，说明这条路在超大规模上被验证了。

---

### 2.7 Hybrid Attention：为什么要混，比例怎么定

**核心矛盾**：纯线性注意力检索能力差，纯 softmax 注意力太贵。

**解法**：大部分层用线性（便宜），少数层保留 softmax（保留精确回溯能力）。

> **类比**：一个团队里大部分人只记得「项目大概进展」，但每隔几个人留一个「有完整档案柜」的人。需要查具体某天发生什么时，问他。

有意思的是，研究发现**混合架构不只是「接近」纯 softmax，在检索和外推上甚至能超过它**。原因猜测是：线性层承担了「压缩总结」的工作，让 softmax 层能把有限的注意力预算集中在真正需要精确定位的地方。

**Raschka 的一句关键提醒**：*「层比例」比「hybrid attention」这个标签本身信息量大得多。* 说某个模型「用了 hybrid」等于没说，要问 3:1 还是 7:1。

**各家的比例选择**：

| 模型 | 线性:softmax | 层数拆解 | 线性机制 | softmax 机制 | KV/token |
|---|---|---|---|---|---|
| Qwen3-Next 80B-A3B | 3:1 | 36 GDN + 12 门控注意力 | Gated DeltaNet | Gated Attention | 24 KiB |
| Qwen3.5 397B-A17B | 3:1 | 45 GDN + 15 | Gated DeltaNet | Gated Attention | 30 KiB |
| Qwen3.6 35B-A3B | 3:1 | 30 GDN + 10 | Gated DeltaNet | Gated Attention | 20 KiB |
| Kimi Linear 48B-A3B | ~3:1 | 20 KDA + 7 MLA | KDA | 门控 MLA（NoPE） | **7.9 KiB** |
| Ling 2.5 1T | **7:1** | 70 Lightning + 10 MLA | Lightning Attention | MLA | 11.2 KiB |
| Nemotron 3 Nano 30B | ~4:1 | 23 Mamba-2 + 6 GQA + 23 MoE | Mamba-2 | GQA | **6 KiB** |
| Nemotron 3 Super 120B | 5:1 | 40 Mamba-2 + 8 GQA + 40 MoE | Mamba-2 | GQA | 8 KiB |
| Nemotron 3 Nano 4B | ~5:1 | 21 Mamba-2 + 4 GQA + 17 FFN | Mamba-2 | GQA | 16 KiB |
| Kimi K3 2.8T | 未公开 | — | KDA | 门控 MLA | — |

**几个值得注意的点**：

1. **3:1 是事实标准**。Qwen 全系和 Kimi Linear 独立收敛到同一个比例，这不像巧合。
2. **NVIDIA 走得最极端**。Nemotron 3 Nano 52 层里只有 **6 层**是注意力。它敢这么做，可能和 NVIDIA 有自家推理栈、能吃下 Mamba 内核优化红利有关。
3. **Ling 2.5 用 7:1，激活参数却高达 63B**——它在「线性层多」和「激活参数多」之间做了个不同的取舍：省下的缓存预算换成了更宽的计算路径。
4. **Kimi Linear 换了两半**：不光把线性部分从 GDN 换成 KDA，还把 softmax 部分从普通注意力换成了门控 MLA，并且在 MLA 层用 **NoPE**（不加位置编码）。这个组合把 KV 压到了 7.9 KiB。

**Ling 2.5 宣称在 32k 序列长度下吞吐是 Kimi K2 的 3.5 倍**——但这是厂商自报的**系统级**对比，包含了完整实现和推理栈，不能孤立归因给 Lightning Attention。

---

### 2.8 反方证据：MiniMax 掉头回全注意力

**这是整份手册里最重要的一条反例，别跳过。**

MiniMax-01 是最早的大规模线性注意力混合模型之一（1:7 的 Lightning:MLA）。但从 **M2 开始，MiniMax 掉头回到了全注意力 GQA**，并且 M2.5、M2.7 一路保持。

Raschka 的模型卡里写得很直白：MiniMax M2.5 **刻意避开**滑动窗口和线性注意力混合，就用朴素的 62 层 GQA。

而结果是：**M2.7 的 AA 综合分 38.1，Agent 分 61.5——在所有开源模型里名列前茅**，超过了同期用 hybrid 的 Qwen3.5（32.0）。

**这说明什么？**

1. **hybrid 不是免费午餐**。它买的是长上下文的吞吐和显存，卖的是（部分）精确检索能力和训练/推理栈的复杂度。
2. **如果你的目标上下文是 200k 而不是 1M，全注意力可能依然是更好的选择**。M2 系列上下文是 196k——在这个长度上，GQA 的 248 KiB/token 还扛得住。
3. **架构选择是和产品定位耦合的**，不是纯技术优劣。MiniMax 主打 coding 和 agent，这类任务要在几十万 token 里精确定位代码片段，正是线性注意力的弱项。

**所以「最新的基础模型都用 hybrid attention」这个说法是不准确的。** 准确的说法是：**2026 年出现了三条并行路线**，各家按自己的场景下注：

| 路线 | 代表 | 赌的是什么 |
|---|---|---|
| **Hybrid 线性** | Qwen3.5/3.6、Kimi Linear/K3、Ling 2.5、Nemotron 3 | 上下文会一直变长（→1M），必须干掉线性增长的缓存 |
| **压缩 + 稀疏 softmax** | DeepSeek V3.2/V4、GLM-5/5.1 | 保留 softmax 的表达力，靠压缩和稀疏把成本压下去 |
| **保守全注意力** | MiniMax M2.x、Mistral | 200k 够用了，架构简单换来训练稳定和实现可靠 |

---

### 2.9 其他 KV 优化：跨层共享与逐层预算

**跨层 KV 共享（Cross-Layer Attention）—— Gemma 4 E2B/E4B**

前面的层算好 K/V，**后面的层直接复用**，自己只算 Q。

规则很讲究：滑动窗口层复用前面的滑动窗口层，全局层复用前面的全局层——**同类型才共享**。

- Gemma 4 E2B：35 层里只有前 15 层算自己的 KV，后 20 层复用
- Gemma 4 E4B：42 层里 24 层算自己的，后 18 层复用

省了大约一半 KV cache：E2B 在 128k 上下文下省 2.7 GB，E4B 省约 6 GB。

代价：这是一种近似，**降低了模型容量**。跨层注意力原论文报告在小模型上影响很小——注意「小模型上」这个限定。

**逐层注意力预算 —— Laguna XS.2**

传统做法是每层给一样的注意力预算。Laguna 在 config 里加了 `num_attention_heads_per_layer`，**每层可以有不同的 query 头数**，同时保持 KV 形状兼容。

它的分配很反直觉：**给滑动窗口层更多 query 头（8 个 query 配 1 个 KV），给全局层更少（6 配 1）**。

逻辑是：全局层要看整个上下文，本来就贵，所以少给点头；滑动窗口层便宜，可以多给。**把预算花在便宜的地方**。

这个思路可以追溯到 Apple 2024 年的 OpenELM。

---

### 2.10 注意力上的小配件

这些不改变复杂度，但几乎每家都会调：

**QK-Norm（Query-Key 归一化）**：在算注意力分数前，对 Q 和 K 各做一次 RMSNorm。目的是**训练稳定性**——防止注意力 logits 爆炸。已经接近标配。OLMo 2、Qwen3、GLM、Gemma 3/4、MiniMax、Laguna 都用。

Kimi 的 **QK-Clip** 是同一问题的另一种解法：直接裁剪，配合 Muon 组成 MuonClip。

**Gated Attention（门控注意力）**：给注意力输出加一个逐头的门控。Qwen3-Next 首用，Arcee Trinity、Laguna、Kimi K3 的门控 MLA 都是这个思路。

**Attention Sink（注意力汇聚点）+ attention bias**：GPT-OSS 用。给注意力加一个「垃圾桶」位置，让模型在「什么都不想看」时有地方放注意力权重，避免被迫把权重摊到无关 token 上。

**位置编码的三种玩法**：
- **RoPE**（旋转位置编码）：绝对主流
- **NoPE**（No Positional Encoding，不加位置编码）：部分层完全不加。SmolLM3 每隔 4 层去掉一次；Kimi Linear 在 MLA 层用 NoPE；Tiny Aya、Arcee Trinity、LongCat、Sarvam 105B 都是 RoPE+NoPE 混用
- **partial RoPE / p-RoPE**：只对头维度的一部分加 RoPE。MiniMax M2、Gemma 4 的全局层、Qwen3.5 的门控注意力（256 维头里只有 64 维加 RoPE）

**为什么有的层不加位置编码？** 直觉解释：局部层已经通过窗口隐含了位置信息，全局层反而需要更「位置无关」才能外推到训练时没见过的长度。**OLMo 3 的做法很能说明问题——YaRN 长度外推只在全局层用。**

---

## 3. 各家选择大表

按发布时间排序。KV/token 是 bf16 精度下的每 token 缓存大小，越小越好。

### 3.1 大模型（100B+）

| 模型 | 日期 | 规模（总/激活） | 注意力 | 层结构 | KV/token | 上下文 |
|---|---|---|---|---|---|---|
| DeepSeek V3 | 2024-12 | 671B / 37B (5.5%) | MLA | 61 MLA | 68.6 KiB | 128k |
| DeepSeek R1 | 2025-01 | 671B / 37B | MLA | 61 MLA | 68.6 KiB | 128k |
| Llama 4 Maverick | 2025-04 | 400B / 17B (4.3%) | GQA | 36 分块 + 12 全局 | 192 KiB | 1M |
| Qwen3 235B-A22B | 2025-04 | 235B / 22B (9.4%) | GQA + QK-Norm | 94 GQA | 188 KiB | 128k |
| Kimi K2 | 2025-07 | 1T / 32B (3.2%) | MLA | 61 MLA | 68.6 KiB | 128k |
| GLM-4.5 | 2025-07 | 355B / 32B (9%) | GQA + QK-Norm | 92 GQA | 368 KiB | 128k |
| GPT-OSS 120B | 2025-08 | 117B / 5.1B (4.4%) | GQA + sink | 18 SWA + 18 全局 | 72 KiB | 128k |
| Grok 2.5 | 2025-08 | 270B | GQA | 64 GQA | 256 KiB | 131k |
| MiniMax M2 | 2025-10 | 230B / 10B (4.3%) | GQA + QK-Norm + pRoPE | 62 GQA | 248 KiB | 196k |
| DeepSeek V3.2 | 2025-12 | 671B / 37B | MLA + **DSA** | 61 MLA | 68.6 KiB | 128k |
| Mistral Large 3 | 2025-12 | 673B / 41B (6.1%) | MLA | 61 MLA | 68.6 KiB | 262k |
| MiMo-V2-Flash | 2025-12 | 309B / 15B (4.9%) | SWA 5:1（窗口 128） | 40 SWA + 8 全局 | 144 KiB | 262k |
| GLM-4.7 | 2025-12 | 355B / 32B | GQA + QK-Norm | 92 GQA | 368 KiB | 203k |
| Kimi K2.5 | 2026-01 | 1T / 32B | MLA | 61 MLA | 68.6 KiB | 256k |
| Arcee Trinity Large | 2026-01 | 400B / 13B (3.3%) | 门控 GQA + SWA 3:1 | 45 SWA + 15 全局 | 240 KiB | 512k |
| Step 3.5 Flash | 2026-02 | 196B / 11B (5.6%) | GQA + SWA 3:1 | 36 SWA + 12 全局 | 192 KiB | 262k |
| **GLM-5** | 2026-02 | 744B / 40B (5.4%) | MLA + **DSA** | 78 MLA | 87.8 KiB | 203k |
| MiniMax M2.5 | 2026-02 | 230B / 10B | GQA + QK-Norm | 62 GQA | 248 KiB | 196k |
| **Ling 2.5** | 2026-02 | 1T / 63B (6.3%) | **Lightning 7:1 MLA** | 70 Lightning + 10 MLA | 11.2 KiB | 256k |
| **Qwen3.5** | 2026-02 | 397B / 17B (4.3%) | **GDN 3:1 门控注意力** | 45 GDN + 15 GA | 30 KiB | 262k |
| Nemotron 3 Super | 2026-03 | 120B / 12B (10%) | **Mamba-2 + GQA** | 40 M2 + 8 GQA + 40 MoE | 8 KiB | 1M |
| Mistral Small 4 | 2026-03 | 119B / 6.6B (5.6%) | MLA | 36 MLA | 22.5 KiB | 256k |
| MiniMax M2.7 | 2026-03 | 230B / 10B | GQA + QK-Norm | 62 GQA | 248 KiB | 196k |
| Kimi K2.6 | 2026-04 | 1T / 32B | MLA | 61 MLA | 68.6 KiB | 256k |
| GLM-5.1 | 2026-04 | 744B / 40B | MLA + DSA | 78 MLA | 87.8 KiB | 203k |
| **DeepSeek V4-Flash** | 2026-04 | 284B / 13B (4.6%) | **CSA/HCA + mHC** | 43 层 | **5.4 KiB** | 1M |
| **DeepSeek V4-Pro** | 2026-04 | 1.6T / 49B (**3.1%**) | **CSA/HCA + mHC** | 61 层 | 7.7 KiB | 1M |
| **Kimi K3** | 2026-07 | **2.8T / ~50B (1.8% 专家)** | **KDA + AttnRes + 门控 MLA** | 未公开 | — | 1M |

### 3.2 中小模型（<100B）

| 模型 | 日期 | 规模 | 注意力 | 特点 |
|---|---|---|---|---|
| Gemma 3 27B | 2025-03 | 27B dense | GQA + SWA 5:1（窗口 1024） | 262k 词表 |
| Qwen3 30B-A3B | 2025-04 | 30B / 3B | GQA | 无共享专家 |
| SmolLM3 3B | 2025-06 | 3B dense | GQA + 每 4 层 NoPE | NoPE 节奏实验 |
| GPT-OSS 20B | 2025-08 | 21B / 3.6B | GQA 交替 SWA/全局 | attention sink + bias |
| **Qwen3-Next 80B-A3B** | 2025-09 | 80B / 3B (3.8%) | **GDN 3:1** | 24 KiB，hybrid 的起点 |
| **Kimi Linear 48B-A3B** | 2025-10 | 48B / 3B | **KDA 3:1 MLA** | 7.9 KiB，MLA 层用 NoPE + ShortConv |
| OLMo 3 32B | 2025-11 | 32B dense | GQA + SWA 3:1 | post-norm，YaRN 只在全局层 |
| **Nemotron 3 Nano 30B** | 2025-12 | 30B / 3B | **23 Mamba-2 + 6 GQA** | 6 KiB，最极端的混合 |
| LongCat-Flash-Lite | 2026-01 | 68.5B / ~3B | MLA + RoPE/NoPE | N-gram 嵌入，512+256 专家 |
| Tiny Aya 3.35B | 2026-02 | 3.35B dense | GQA + SWA 3:1 | **并行 transformer 块** |
| Gemma 4 31B | 2026-04 | 30.7B dense | GQA + SWA 5:1 + 全局层统一 KV | 256k 上下文 |
| Gemma 4 E2B/E4B | 2026-04 | 5.1B(2.3B有效)/8B(4.5B有效) | MQA/GQA + **跨层 KV 共享** | **PLE 每层嵌入** |
| Qwen3.6 35B-A3B | 2026-04 | 35B / 3B | GDN 3:1 | 20 KiB |
| **Laguna XS.2** | 2026-04 | 33B / 3B | 门控 GQA + SWA 3:1（窗口 512） | **逐层 query 头预算** |
| **ZAYA1-8B** | 2026-05 | 8.4B / 760M | **CCA + 4:1 GQA** | AMD GPU 训练，**top-1 专家** |

---

## 4. MoE：全部变量与各家配置

### 4.1 MoE 是什么（30 秒版）

把 FFN（前馈网络）拆成 N 个小网络（专家），每个 token 只激活其中几个。

> **类比**：从「一个什么都懂的全科医生」变成「一个分诊台 + 一堆专科医生」。分诊台（router）看一眼病人，挑 2 个专科医生来会诊。

好处：**总参数量和计算量脱钩**。1.6T 参数的 DeepSeek V4-Pro 每 token 只算 49B——**只有 3.1%**。

### 4.2 变量一：专家数量与粒度

这是过去两年最明显的单调趋势——**专家越来越多，每个越来越小**：

| 模型 | 专家总数 | 每 token 激活 | 稀疏度 |
|---|---|---|---|
| Mixtral（2023） | 8 | 2 | 25% |
| Grok 2.5 | 少而大 | — | — |
| Llama 4 Maverick | 128 | 1 + 1 共享 | — |
| DeepSeek V3 | 256 | 8 + 1 共享 | 5.5% 参数 |
| Qwen3 235B | 128 | 8，**无共享** | 9.4% |
| Kimi K2 / K2.5 / K2.6 | **384** | 8 | 3.2% |
| Gemma 4 26B-A4B | 128 | 8 + 1 共享 | 15.1% |
| Mistral Small 4 | 128 | **4** + 1 共享 | 5.6% |
| Qwen3.6 35B-A3B | 256 | 8 + 1 共享 | 8.6% |
| Qwen3.5 397B | **512** | — | 4.3% |
| LongCat-Flash-Lite | **512 + 256 个零/恒等专家** | top-12 | 4.4% |
| DeepSeek V4-Flash | 256 | 6 + 1 共享 | 4.6% |
| DeepSeek V4-Pro | 384 | 6 + 1 共享 | **3.1%** |
| **ZAYA1-8B** | — | **1**（top-1！） | 9% |
| **Kimi K3** | **896** | **16** | **1.8%** |

**为什么细粒度更好？** 直觉是**组合数**。8 选 2 只有 28 种组合，256 选 8 有约 4×10¹⁴ 种。更细的切分让模型能表达更多样的「专家组合」，即使总参数和激活参数不变。

**注意 LongCat 那个「零/恒等专家」**：256 个专家什么也不做（直接输出恒等映射）。这等于给 router 一个「这个 token 不需要额外计算」的选项——**变相实现了每 token 的动态计算量**。

**Kimi K3 的 1.8% 是目前公开的稀疏度纪录**。它明确指出：在这个稀疏度下，**路由和优化本身变成一阶难题**——所以才需要配套的 Stable LatentMoE。

### 4.3 变量二：共享专家（Shared Expert）

**共享专家 = 一个所有 token 都会经过的专家**，不参与路由。

> **类比**：专科医院里的「全科门诊」，所有病人都先过一遍，处理通用问题；专科医生只负责各自领域的特殊情况。

**目的**：让路由专家不必重复学习通用知识，可以更专注地特化。

**但这是一个真正有分歧的设计点**：

| 用共享专家 | 不用 |
|---|---|
| DeepSeek V3/V3.2/V4、Llama 4、GLM-4.5、Qwen3-Next、Qwen3.5/3.6、Gemma 4 MoE、Mistral Small 4、Laguna | **Qwen3 235B-A22B**、MiniMax M2 系列 |

有意思的是 **Qwen 自己在两代之间改了主意**：Qwen3 235B 明确去掉了共享专家（模型卡写的是「优化服务效率」），但 Qwen3-Next 和 Qwen3.5 又加了回来。

**Grok 2.5 是个有趣的中间态**：它没有正式的共享专家，但有一个**常开的 SwiGLU 路径**，功能上等价于共享专家。

### 4.4 变量三：Router（路由器）

**这是 MoE 里最容易出问题的部件**，因为它要做一个离散选择，而离散选择不可导。

**打分函数**：

| 方式 | 代表 | 特点 |
|---|---|---|
| **Softmax** | Mixtral、传统做法 | 分数和为 1，专家之间**竞争**。一个高了别的必须低 |
| **Sigmoid** | DeepSeek V3 起、Laguna | 每个专家**独立**打分。两个专家都很相关时可以都拿高分 |
| **ReLU** | ReMoE、BlockFFN、DECO | 天然产生大量零，可平滑学习激活比例 |
| **Hash** | DeepSeek V4 | 基于哈希的路由 |
| **分位数（quantile）** | Kimi K3 Stable LatentMoE | 明确说是为了**去掉在超大规模下容易崩的脆弱超参** |

**softmax → sigmoid 的迁移值得理解**：softmax 强制竞争，会压制「两个专家都该被激活」的情况；sigmoid 独立打分更灵活，还能缓解极端偏斜的分数分布。DeepSeek V3 还额外加了一个标量缩放因子来平衡共享专家和路由专家的贡献。

**负载均衡**：MoE 有个天然的死循环——某个专家早期碰巧表现好 → 更多 token 被送过去 → 它学得更好 → 更多 token 被送过去。最后少数专家承担全部工作，其余「死掉」。

传统解法是**辅助损失（auxiliary loss）**：给不均衡加惩罚。缺点是这个损失和语言建模目标冲突，会伤害质量。

**DeepSeek 的 aux-loss-free 方案**是 2024 年最漂亮的一个 trick：给每个专家的分数加一个 **bias**，这个 bias **不由梯度更新**，而是用一条简单规则——过载的专家 bias 减小（变得不吸引人），闲置的专家 bias 增大。

> **类比**：不是罚医生「你今天看太多病人了」，而是悄悄调整分诊台的排序，让忙的医生排后面。

这个方案的精妙之处在于：**它不进入损失函数，所以不和语言建模目标打架**。现在已经被广泛采用。

### 4.5 变量四：Latent MoE（潜在空间专家）

**Nemotron 3 Super 和 Kimi K3 的新招**。

思路：在把 token 送给专家之前，**先把它压缩到低维**，专家在低维空间里计算。

**为什么？** 因为 MoE 在多卡部署时，token 要通过网络被「分发」到各个专家所在的卡上，算完再收回来（all-to-all 通信）。**Token 表示越小，通信量越小。**

这是一个典型的「架构为系统服务」的设计——它优化的不是模型质量，而是**分布式部署时的通信开销**。

### 4.6 变量五：Dense 前缀

**GLM-4.5/4.7 的做法**：前 3 层不做 MoE，用普通 dense FFN。DeepSeek V3 也有类似的 dense 前缀。

**为什么？** 前几层学的是最通用的低层特征（token 级别的模式），没什么可特化的，强行路由反而增加不稳定性。

---

## 5. 其他架构变量

### 5.1 归一化

**类型**：LayerNorm → **RMSNorm** 的迁移已经全面完成。手册里 93 个模型，除了 GPT-2 基本全是 RMSNorm。理由很朴素：去掉了均值中心化那一步，更快，效果没差。

**位置**：这里有分歧。

| 方案 | 说明 | 代表 |
|---|---|---|
| **Pre-Norm** | 归一化在子层之前 | 绝对主流 |
| **Post-Norm（残差内）** | 归一化在子层之后 | **OLMo 2、OLMo 3** 坚持这条路，理由是训练稳定性 |
| **Sandwich Norm** | 前后都加 | Arcee Trinity、Gemma 系列的变体 |

### 5.2 激活函数

**SwiGLU（用 SiLU 的门控 MLP）是事实标准**，几乎所有模型的专家和 FFN 都是这个。

例外值得注意：
- **Nemotron 3 Nano 4B 用 ReLU²**（ReLU 的平方）
- **Gemma 4 E2B 用双倍宽的 GELU MLP**
- DECO 提出 NormSiLU，但还没进主流模型

### 5.3 残差流：2026 年的新战场

这是最近才热起来的方向。传统 transformer 的残差是「一条单线」，2026 年有两家在改它：

**mHC（Manifold-Constrained Hyper-Connections，流形约束超连接）—— DeepSeek V4**

把单条残差流换成**多条并行残差流**（V4 用 n=4），加上流之间的学习映射。

> **类比**：原来是一条主干道，所有信息挤在上面；现在是四条并行车道，中间有匝道互通。

结构上需要三个映射：
- **Pre Mapping**：把多条流合并成一个正常宽度的向量喂给注意力/MoE
- **Post Mapping**：把层输出分发回多条流
- **Res Mapping**：跨层混合多条流

**为什么加 m（流形约束）？** 原版超连接的 Res Mapping 是一个自由学习的矩阵，多层堆叠会不可预测地放大或缩小信号。mHC 把它**投影到双随机矩阵流形上**——所有元素非负、每行每列和为 1。这让残差混合变成一种「稳定的信息再分配」而不是可能爆炸的变换。

**成本**：超连接原论文里 7B OLMo MoE 的 FLOPs 从 13.36G 涨到 13.38G，几乎不变（因为映射操作在很小的「流数量」维度上）。DeepSeek 优化实现后，27B 模型上 n=4 只增加 **6.7%** 训练时间。

⚠️ 但 Raschka 提了一个很实在的质疑：**只看 FLOPs 太简单了**。加宽的残差状态还是要存储、搬运、混合，实际开销可能更多来自内存带宽和实现复杂度，而这一点论文没测。

**收益**：超连接原论文报告达到基线性能只需大约**一半的训练 token**。

**Attention Residuals（AttnRes）—— Kimi K3**

Kimi 描述为残差连接的「drop-in 替代」：不是均匀地逐层累加表示，而是**跨深度有选择地检索表示**。声称 2% 的计算成本换 25% 的训练效率。

技术报告尚未完整发布，细节待补。

### 5.4 MTP（Multi-Token Prediction，多 token 预测）

训练时不只预测下一个 token，而是同时预测后面 2–4 个。

**两个用途**：
1. **训练信号更密集**，模型学得更快
2. **推理时做投机解码**——直接用这些额外的预测头当 draft，不需要单独的小模型

**各家配置**：
- MTP-1：DeepSeek V3/R1/V3.2
- MTP-3：GLM-4.7、Step 3.5 Flash、MiniMax M2.5/M2.7（3 个 MTP 模块）
- 共享权重 MTP：Nemotron 3 Super
- 有 MTP 但只在训练用：GLM-4.5、MiniMax M2

**Step 3.5 Flash 特别提到 MTP-3 在训练和推理都用**，这是它吞吐特别高的原因之一。

### 5.5 优化器：AdamW vs Muon

**这是 2025–2026 年最大的一个非架构变化。**

**AdamW** 统治了十年。**Muon**（MomentUm Orthogonalized by Newton-Schulz）是新挑战者。

**Muon 做什么**：对 2D 权重矩阵，取梯度 → 算动量 → 用 Newton-Schulz 迭代把动量矩阵**正交化** → 用正交化后的矩阵更新权重。

> **类比**：AdamW 是给每个参数单独调整步长；Muon 是把整个权重矩阵的更新方向「摆正」，让各个方向的更新幅度均衡，避免更新集中在少数几个方向上。

**三个实际好处**：
1. 报告约 **2× 的计算效率**（达到同样损失需要更少 token）
2. **省显存**——只维护一个动量缓冲，而 AdamW 要两个（一阶和二阶矩）
3. 在**大 batch size** 下比 AdamW 更稳定，AdamW 会退化而 Muon 还能保持

**已采用的模型**：Kimi K2 / K2.5（**MuonClip** = Muon + QK-Clip，在 15.5T token 上预训练无 loss spike）、GLM-4.5 / 4.7 / 5、DeepSeek V4。

**Kimi 的 update RMS matching 框架**是个被广泛引用的副产品：通过保证不同优化器的参数更新 RMS 范数一致，可以在优化器之间**迁移学习率**，省掉重新调参的算力。NVIDIA 的 SOAP/Muon 论文明确借用了这个框架。

⚠️ **两个必须知道的坑**：
1. **Muon 只能优化 2D 矩阵**（Linear 层），一维张量（bias、norm 参数）还是要用 AdamW。所以实际是混合优化器。
2. **优化器不匹配问题**：用 Muon 微调一个 AdamW 预训练的模型，效果**明显更差**，反之亦然。绝大多数开源模型是 AdamW 预训练的，这严重限制了 Muon 在微调场景的实用性。这是 2026 年 ICML 一篇论文的核心发现。

**这条对你直接相关**：如果你要在开源模型上做 RL post-training，**优化器要和预训练时保持一致**。别看到 Muon 好就换。

### 5.6 数值精度

- **BF16**：默认
- **FP8**：DeepSeek V3 起在训练中大量使用
- **MXFP4**：Kimi K3 的权重发布格式。1.4 TB 聚合显存才能加载

### 5.7 嵌入层的特殊设计

**PLE（Per-Layer Embeddings，每层嵌入）—— Gemma 4 E2B/E4B**

给每一层配一个小的、token 特定的嵌入向量，在 FFN 之后作为额外的残差更新加进去。

**关键点**：不是给每层一份完整的嵌入表副本。每层嵌入只查一次，然后切片分给各层。

**为什么？** 因为嵌入表参数是**查表式的、便宜的**，而 attention 和 FFN 参数是**要真算的、贵的**。PLE 把额外容量放在便宜的地方，让昂贵的 transformer 主干保持小尺寸。

这就是 Gemma 4 E2B 「5.1B 参数但 2.3B 有效」的来源——**「有效参数」指的是主干的计算规模**。

**N-gram 嵌入 —— LongCat-Flash-Lite**：把大量参数挪进 N-gram 嵌入表，同样是「把容量放在便宜的地方」。

---

## 6. 为什么这么选：耦合关系

这是你问的核心——**这些选择不是独立的，它们互相配套**。

### 耦合 ①：线性注意力 ↔ 极致稀疏 MoE

**观察**：用 hybrid 线性注意力的模型，几乎全都同时把 MoE 稀疏度推到极致。

- Qwen3-Next：80B/3B（3.8%）
- Kimi Linear：48B/3B
- Nemotron 3 Nano：30B/3B
- Kimi K3：2.8T，16/896 专家（1.8%）

**为什么配套？** 因为两者卡的是**同一个瓶颈的两半**：

- 线性注意力砍的是 **KV cache 显存**
- 稀疏 MoE 砍的是 **每 token 计算量**

如果只做一半，另一半就成为新瓶颈。KV cache 从 250 KiB 降到 20 KiB，你就能在同样显存里跑 12 倍的 batch——但如果计算量没降，你的 GPU 会算不过来。**反过来也一样**。

**这条对你的 GPU 系统方向直接有用**：这两个优化改变的是 roofline 上的位置。线性注意力把 decode 从「内存带宽受限」往「计算受限」推，稀疏 MoE 又把它推回去。真正的调优要看这两者的**净效果**，而不是分别看。

### 耦合 ②：MoE 稀疏度 ↔ Router 稳定性 ↔ 优化器

稀疏度越高，router 越难训。原因很直接：**每个专家能拿到的梯度信号越少**。

Kimi K3 说得最明白：在 16/896 这个稀疏度下，**路由和优化变成一阶难题**。所以它必须配套三样东西：

1. **Stable LatentMoE**：压缩表示降低通信
2. **分位数路由**：去掉在超大规模下崩掉的脆弱超参
3. **Muon 系优化器**：更稳定的大 batch 训练

同理，DeepSeek V4-Pro 的 3.1% 稀疏度配的是**哈希路由 + Muon + mHC**。

**这是一条清晰的因果链：想更稀疏 → router 必须更鲁棒 → 优化器和路由算法都要换。**

### 耦合 ③：局部/全局注意力 ↔ 位置编码

规律非常一致：

| 层类型 | 位置编码 | 长度外推 |
|---|---|---|
| 局部层（SWA / 线性） | 完整 RoPE | 不需要 |
| 全局层 | NoPE 或 partial RoPE | YaRN 只在这里用 |

**为什么？** 局部层的窗口本身就编码了位置信息（「只能看前 512 个」），所以 RoPE 是锦上添花。全局层要跨越几十万 token，**RoPE 在训练长度之外会失效**，所以要么减弱（partial RoPE）、要么去掉（NoPE）、要么补救（YaRN）。

OLMo 3 的做法是这条规律的最好证据：**YaRN 只在全局层用**。

### 耦合 ④：MTP ↔ 投机解码 ↔ agent 场景

MTP 本来是训练技巧，但它顺带产出了投机解码的 draft 头。**在 agent 场景下，decode 步数是主要延迟来源**，所以 MTP 从「训练技巧」变成了「推理必需品」。

这解释了为什么 MTP 从 MTP-1（DeepSeek V3）涨到 MTP-3（GLM-4.7、Step 3.5、MiniMax M2.5/M2.7）——**agent 需求推着它涨**。

### 耦合 ⑤：模型规模 ↔ 敢不敢用新架构

**观察**：激进的新架构总是先在中小模型上试，验证后才进旗舰。

- Qwen3-Next（80B，2025-09）验证 → Qwen3.5（397B，2026-02）采用
- Kimi Linear（48B，2025-10）验证 → Kimi K3（2.8T，2026-07）采用
- mHC 论文在 27B 上做实验（2025-12）→ DeepSeek V4（1.6T，2026-04）采用

**这是一条稳定的规律，可以用来预测**：现在在小模型上出现的东西（CCA、逐层预算、ReLU² 激活），大概率会在 6–12 个月后进旗舰。

---

## 7. 闭源模型：我们不知道什么

**必须说清楚**：GPT-5.x（含 GPT-5.6 Sol）、Claude 系列（含 Fable 5、Opus 4.8）、Gemini 系列，**都没有公开架构细节**。

我们只有间接证据：

- **GPT-4** 被 George Hotz 传为 8 个约 220B 的模型组合（约 1.76T），Soumith Chintala 呼应过。但 MoE 里通常只有 FFN 层按专家复制、注意力层共享，所以真实总量可能在 1.2T–1.76T 之间。**OpenAI 从未确认。**
- **GPT-OSS 是唯一的官方开放窗口**。它用交替的滑动窗口/全局注意力 + attention sink + attention bias。合理推测这反映了 OpenAI 内部的部分设计偏好，但**不能假设 GPT-5 就是放大版的 GPT-OSS**。
- Anthropic 和 Google 的架构信息**基本为零**。Gemma 是 Google 的开源线，但 Google 从未说过 Gemma 和 Gemini 共享架构。

**所以这份手册的准确定位是：开源前沿模型的架构手册。** 任何声称知道 GPT-5 或 Claude 架构的材料，都值得高度怀疑。

---

## 8. 判断框架：以后看到新模型怎么读

给你一套固定的读法，按这个顺序问：

**第一层：它在哪条路线上？**
1. KV/token 是多少？（这一个数字信息量最大）
2. 有没有线性注意力/SSM？比例是多少？
3. softmax 层用的是 GQA、MLA 还是压缩变体？
4. 局部:全局比例？窗口多大？

**第二层：MoE 怎么切？**
5. 总专家数 / 激活数 / 稀疏度百分比
6. 有没有共享专家？
7. Router 打分函数？负载均衡怎么做？

**第三层：配套件**
8. 位置编码是不是分层不同？
9. 有没有 MTP？几个？
10. 优化器是 AdamW 还是 Muon 系？
11. 残差流有没有改？

**第四层：最重要的一问**
12. **有没有消融实验？** 如果技术报告只给「新架构 vs 上一代」的端到端对比，那所有收益都是**数据 + 配方 + 架构的混合结果**，不能归因给架构。DeepSeek V4 的 CSA/HCA 就是这个情况——论文没有消融。

---

## 9. 原始资料清单

**综述与对照（先看这些）**
- Sebastian Raschka, *LLM Architecture Gallery* — https://sebastianraschka.com/llm-architecture-gallery/ （93 个模型的 config 对照，带 KV cache 计算器和两两 diff 工具，本手册主要数据来源）
- Sebastian Raschka, *The Big LLM Architecture Comparison*（2025-07，持续更新）
- Sebastian Raschka, *A Visual Guide to Attention Variants in Modern LLMs*（2026-03）
- Sebastian Raschka, *Recent Developments in LLM Architectures: KV Sharing, mHC, and Compressed Attention*（2026-05）

**注意力机制原论文**
- Gated Delta Networks — arXiv:2412.06464
- Mamba-2 — arXiv:2405.21060
- Gated Attention — arXiv:2505.06708
- Kimi Linear（KDA）— arXiv:2510.26692
- Compressed Convolutional Attention（CCA）— arXiv:2510.04476
- Cross-Layer Attention（KV 共享）— arXiv:2405.12981（NeurIPS 2024）

**残差流**
- Hyper-Connections — arXiv:2409.19606
- mHC: Manifold-Constrained Hyper-Connections — arXiv:2512.24880

**模型技术报告**
- DeepSeek V3 — arXiv:2412.19437
- DeepSeek V3.2 — arXiv:2512.02556
- DeepSeek V4 — HuggingFace `deepseek-ai/DeepSeek-V4-Pro` 仓库内 PDF
- Kimi K2 — arXiv:2507.20534（MuonClip 在这篇）
- Kimi K2.5 — arXiv:2602.02276
- Kimi K3 — https://www.kimi.com/blog/kimi-k3（技术报告随权重发布）
- Qwen3 — arXiv:2505.09388
- GLM-4.5/4.7 — arXiv:2508.06471；GLM-5 — arXiv:2602.15763
- MiniMax M2 系列 — arXiv:2605.26494
- Gemma 3 — arXiv:2503.19786；Gemma 4 — ai.google.dev 模型卡
- Nemotron 3 Nano / Super — research.nvidia.com/labs/nemotron
- ZAYA1-8B — arXiv:2605.05365
- Laguna XS.2 — poolside.ai 技术报告
- Step 3.5 Flash — arXiv:2602.10604
- Arcee Trinity Large — arXiv:2602.17004
- OLMo 3 — arXiv:2512.13961

**优化器**
- Muon 可扩展性研究（Moonlight 模型）— Liu et al. 2025
- SOAP, Muon, and Beyond: Pushing LLM Pretraining Scales（NVIDIA）— arXiv:2607.20548，附开源代码库 github.com/NVIDIA-NeMo/Emerging-Optimizers
- Can Muon Fine-tune Adam-Pretrained Models?（ICML 2026）— arXiv:2605.10468（优化器不匹配问题）

**MoE**
- DeepSeek-MoE（细粒度专家 + 共享专家的奠基）— Dai et al. 2024
- Slicing and Dicing: Configuring Optimal Mixtures of Experts — arXiv:2605.11689

---

## 附录：术语速查

| 缩写 | 全称 | 一句话 |
|---|---|---|
| MHA | Multi-Head Attention | 原版多头注意力，每头独立 KV |
| MQA | Multi-Query Attention | 所有头共用一套 KV |
| GQA | Grouped-Query Attention | 分组共用 KV，主流折中 |
| MLA | Multi-head Latent Attention | 把 KV 压缩成潜在向量存储 |
| CCA | Compressed Convolutional Attention | 直接在压缩空间做注意力，加卷积补偿 |
| SWA | Sliding Window Attention | 只看最近 N 个 token |
| DSA | DeepSeek Sparse Attention | 动态挑选 top-k 历史 token |
| CSA | Compressed Sparse Attention | 序列压缩（m=4）+ 稀疏选择 |
| HCA | Heavily Compressed Attention | 重度序列压缩（m=128）+ 稠密注意力 |
| GDN | Gated DeltaNet | 线性注意力，头级标量遗忘门 |
| KDA | Kimi Delta Attention | GDN 的通道级门控版本 |
| SSM | State Space Model | 状态空间模型，Mamba 属于这类 |
| MoE | Mixture of Experts | 稀疏专家混合 |
| MTP | Multi-Token Prediction | 一次预测多个 token |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| NoPE | No Positional Encoding | 不加位置编码 |
| PLE | Per-Layer Embeddings | 每层独立的小嵌入向量 |
| mHC | manifold-constrained Hyper-Connections | 流形约束的多残差流 |
| QK-Norm | Query-Key Normalization | 对 Q、K 做归一化以稳定训练 |
| YaRN | Yet another RoPE extensioN | RoPE 长度外推方法 |
| SwiGLU | Swish-Gated Linear Unit | 主流 FFN 激活结构 |
| Muon | MomentUm Orthogonalized by Newton-Schulz | 正交化动量的优化器 |

---

*编制日期：2026 年 8 月 5 日。这个领域每两周就有新模型，请以各家最新技术报告为准。*

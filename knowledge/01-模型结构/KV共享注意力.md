# KV 共享注意力(MHA → MQA → GQA)

> 🔴 重点考点:本篇是当前复习重点,文末「面试考点串联」给出问法对照。

一句话:**让多个 query 头共享同一套 K/V 头,把 KV cache 按共享比例直接砍下来**——不改注意力的数学本质、不加新模块,是 2023 年以来开源模型的绝对主流,到 2026 年依然是「保守派」的默认选择。

> **类比**:MHA 是每个学生配一个专属图书管理员;MQA 是全校共用一个;GQA 是每个班配一个。学生(query 头)各有各的问题,但管理员(K/V 头)维护的书架索引大同小异——**所以砍管理员比砍学生划算**。

注意力本身的公式、多头怎么拆、$\sqrt{d_k}$ 从哪来,见 注意力基础 篇;本篇只盯着「K/V 存几套」这一个旋钮。

## 一、先看这笔账:KV cache 里到底有多少字节

自回归生成第 $t$ 个 token 时,前 $t-1$ 个位置的 K、V 与上一步完全相同,没必要重算——存起来复用,就是 KV cache(机制、生命周期与抢占见 KVCache 篇)。它的总字节数是:

$$
\text{KV bytes} = 2 \times B \times L \times S \times n_{kv} \times d_h \times b
$$

即 2(K 和 V 各一份)× 批量 $B$ × 层数 $L$ × 序列长 $S$ × **KV 头数 $n_{kv}$** × 每头维度 $d_h$ × 每元素字节数 $b$(bf16 为 2)。**这个式子里唯一能被结构改动砍掉的因子就是 $n_{kv}$——MQA/GQA 动的就是它。**

这笔账有多痛:Gemma 3 27B 每 token 要 496 KiB——**按所有层都存全长算**,单条 128k 上下文就是约 62 GiB。一张 80 GB 的 H100 光缓存吃掉大半,权重还没放。(这里用上界口径是为了看清「什么优化都不做」时的规模;Gemma 3 自己靠 5:1 滑窗把真正随长度增长的部分压掉了大半,见 SWA 篇。跨模型比 KV/token 时要先统一口径,见 Hybrid注意力 篇。)更要命的是 **decode 每生成一个 token 都要把整份 cache 从 HBM 完整读一遍**——那是纯内存带宽开销,算力大量闲置。

这也解释了共享 K/V 的收益为什么集中在 decode:

| 阶段 | 主要瓶颈 | 减小 $n_{kv}$ 的收益 |
|---|---|---|
| Prefill | 矩阵计算(一次算很多 token) | 少写缓存、省显存;注意力 FLOPs 基本不变 |
| Decode | **内存带宽**(每步读全部历史) | 读取字节按比例下降,直接转化成吞吐与并发 |

## 二、MQA:一步砍到底

**MQA(Multi-Query Attention)**,Shazeer 2019 提出:$h$ 个 query 头**全部共用一套 K/V**,即 $n_{kv} = 1$。

收益很直接——KV cache 变成 MHA 的 $1/h$,decode 的带宽压力同比例下降。代价是**质量掉得比较明显**,训练也更容易不稳:所有头被迫用同一份「索引 + 内容」,头与头之间「检索什么」的多样性没了,信息瓶颈全压在唯一一套 KV 投影上。还是那个类比——全校共用一个图书管理员,他的编目方式不可能同时适合所有人。

## 三、GQA:分组折中

### 机制与谱系

**GQA(Grouped-Query Attention)**把 $h$ 个 query 头分成 $g$ 组,**每组共用一套 K/V**($n_{kv} = g$),缓存按组数等比缩小:

$$
\frac{\text{GQA 缓存}}{\text{MHA 缓存}} = \frac{g}{h}
$$

也就是说,组数越少越省,但组内共享的约束越强。它是一个**谱系而不是单点**:$g = h$ 退化成 MHA,$g = 1$ 退化成 MQA,中间任取。

| 方案 | KV 头数 $n_{kv}$ | 4 个 query 头时的分配 | 缓存相对 MHA |
|---|---|---|---|
| MHA | $h$ | Q1→KV1 · Q2→KV2 · Q3→KV3 · Q4→KV4 | 1 |
| GQA($g=2$) | $g$ | Q1,Q2→KV1 · Q3,Q4→KV2 | $g/h$ |
| MQA | 1 | Q1,Q2,Q3,Q4→KV1 | $1/h$ |

典型配置是 4:1 到 8:1(几个 query 头共用一套 KV),也有更激进的:Qwen3 235B 是 64Q : 4KV = 16:1;Llama 3 全系固定 8 个 KV 头,8B / 70B / 405B 的组比分别是 4:1 / 8:1 / 16:1。

同一代模型内部也不一定统一——**Llama 2 的 7B 和 13B 用的还是 MHA,只有 70B 换成了 GQA**。回答「某某系列用不用 GQA」时不能拿一个尺寸推广到全系列。

### 为什么几乎无损

MHA 训出来的多套 K/V 头**冗余度很高**:不同头的 key/value 投影学到的内容大量重叠。真正需要多样性的是「问什么」(Q),而不是「书架怎么编目」(K/V)。所以保留少数几套不同的 KV 子空间就足以撑住绝大部分表达力——质量随 $g$ 下降的曲线长期平坦,只有逼近 $g = 1$(MQA)才明显掉头。

### uptraining:旧模型也能低成本改造

GQA 论文的另一半贡献:不必从头训。拿现成的 MHA checkpoint,把每组内的 K/V 头**平均池化**成一套作为初始化,再用**原预训练约 5% 的算力**继续训练(uptraining),就能得到接近 MHA 质量、接近 MQA 速度的模型。

但要注意反过来不成立:**不能在推理时把 MHA 直接无损改成 GQA**。共享 K/V 改变了模型函数本身,直接丢掉多余的 KV 头等于换了个模型,必须经过转换加继续训练才能用。

## 四、数字:2026 年在役的 GQA 模型

bf16 口径,KV/token 越小越好:

| 模型 | 规模(总/激活) | 层结构 | KV/token | 上下文 |
|---|---|---|---|---|
| Qwen3 235B-A22B | 235B / 22B | 94 层 GQA + QK-Norm | 188 KiB | 128k |
| GLM-4.5 / 4.7 | 355B / 32B | 92 层 GQA + QK-Norm | 368 KiB | 128k / 203k |
| MiniMax M2 / M2.5 / M2.7 | 230B / 10B | 62 层 GQA + QK-Norm | 248 KiB | 196k |
| Grok 2.5 | 270B | 64 层 GQA | 256 KiB | 131k |
| Llama 4 Maverick | 400B / 17B | GQA,36 分块 + 12 全局 | 192 KiB | 1M |
| GPT-OSS 120B | 117B / 5.1B | GQA + sink,18 SWA + 18 全局 | 72 KiB | 128k |

拿 Qwen3 235B 验一下第一节的公式:$2 \times 94 \times 4 \times 128 \times 2 = 192{,}512$ 字节 ≈ **188 KiB**,对上了。单条 128k 上下文约 23 GiB。

如果它仍是 64 头 MHA,这个数要乘 16,约 2.9 MiB/token,128k 上下文要 **376 GiB**——GQA 一刀省出 16 倍,长上下文服务能不能开起来就取决于这一刀。

对照组:MLA 阵营的 DeepSeek V3(671B,61 层)每 token 只要 68.6 KiB。参数量接近 GLM-4.5 的两倍,缓存却只有它的五分之一——**除了压 KV 头数,还能压每头的表示**,那条路线见 MLA 篇。

## 五、为什么 2026 年仍是「保守派默认」

2026 年有三条并行路线:hybrid 线性、压缩加稀疏 softmax、保守全注意力——GQA 是第三条的基石。它的护城河不在数学而在工程:

- **实现简单、训练稳**:相对 MHA 只改了 KV 投影的形状,没有新算子、没有新超参。
- **生态全兼容**:FlashAttention、vLLM/SGLang 等推理引擎、各类微调栈都把 GQA 当一等公民支持;MLA 和线性注意力都需要专门适配。
- **反例实锤**:MiniMax-01 曾是最早的大规模线性混合模型之一,但从 M2 起**掉头回纯 GQA**(62 层)。在 196k 上下文这个量级上,朴素 GQA 的 248 KiB/token 还扛得住,换来的是训练稳定与精确检索能力不打折。来龙去脉见 Hybrid注意力 篇。

## 六、细节与坑

- **$g$ 要迁就张量并行(TP)**:推理时注意力头按 TP 度切到多张卡,**KV 头数最好能被 TP 度整除**。若 TP 度大于 $n_{kv}$,引擎只能把 KV 头整头复制到多卡,复制掉的正是 GQA 省下的显存。所以 $n_{kv}$ 几乎都取 2 的幂(4/8/16),Llama 3 固定 8 个 KV 头正好对应 TP=8 时一卡一头。
- **组内是「共享」不是「平均」**:前向把共享的 K/V broadcast 给组内每个 query 头,反向梯度从组内所有头汇聚回同一套 KV 权重。实现要用 `repeat_kv` 这类广播视图,**别真的物化复制张量**,否则显存收益归零。
- **QK-Norm 常配套**:上表里 Qwen3、GLM-4.5、MiniMax M2 全是「GQA + QK-Norm」——算分前对 Q、K 各做一次 RMSNorm 防注意力 logits 爆炸,已接近标配,细节见 注意力配件 篇。
- **GQA 改不了增长率**:它把缓存砍成 $g/h$,但总量仍随序列长度**线性增长**。想动增长率得靠滑动窗口、稀疏或线性注意力(见 SWA 篇、稀疏注意力 篇、线性注意力 篇)。
- **组数没有通用最优值**:先满足 $h$ 能被 $g$ 整除、$g$ 能被 TP 度整除,再在目标硬件和上下文长度上实测质量、首 token 延迟、单 token 延迟、吞吐与最大并发。别只按一个固定比例选。

## 七、面试考点串联

| 高频问法 | 本文哪一节 |
|---|---|
| 比较 MHA、MQA、GQA、MLA 的 Q/K/V 组织方式与取舍 | 三(表)+ 四(MLA 对照,详见 MLA 篇) |
| 给一份 config,现场算每 token 的 KV cache | 一(公式)+ 四(Qwen3 验算) |
| 怎样从 batch、层数、长度和 KV 头数估算缓存 | 一 |
| MQA 为什么明显掉质量,GQA 为什么几乎无损 | 二 + 三 |
| GQA 的组数怎么选?KV 头数和每组 query 头数怎么换算 | 三 + 六 |
| 为什么不能在推理时把 MHA 直接改成 GQA | 三(uptraining 反向不成立) |
| 手上有 MHA 旧模型,怎么低成本改造 | 三(uptraining,约 5% 算力) |
| Llama 2 是不是全系列都用 GQA | 三(7B/13B 是 MHA,70B 才是 GQA) |
| 这些机制在 prefill 和 decode 阶段的收益有什么不同 | 一(阶段表) |
| 长上下文服务该选 MHA、GQA 还是 MLA | 四 + 五 |
| 为什么 2026 年还有旗舰模型坚持纯 GQA | 五 |

## 相关文献

- Attention Is All You Need(Transformer 与 MHA 原论文)— [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Fast Transformer Decoding: One Write-Head is All You Need(Shazeer,MQA 提出)— [arXiv:1911.02150](https://arxiv.org/abs/1911.02150)
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints(GQA 与 uptraining)— [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- Llama 2: Open Foundation and Fine-Tuned Chat Models(7B/13B 用 MHA,70B 用 GQA)— [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- The Llama 3 Herd of Models(GQA 大规模实践)— [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

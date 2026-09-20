# ViT：少给视觉先验，用数据和规模把 Transformer 直接接到图上

<!-- release-date: 2020-10-22 -->

**本文依据**：`An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`，arXiv 2010.11929v2（2021-06-03，ICLR 2021 正式版），22 页。作者 Alexey Dosovitskiy 等，Google Research, Brain Team。盘上 PDF 页眉写明 arXiv:2010.11929v2 [cs.CV] 3 Jun 2021，同页写 Published as a conference paper at ICLR 2021。首发日取 arXiv v1 提交日 2020-10-22（[arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929) 的 Submission history：v1 Thu, 22 Oct 2020 17:55:59 UTC；这是外部补充，不来自 PDF 正文）。文中所有数字都标了 PDF 页码；标「外部补充」的段落不来自本文。

## 一句话

卷积把局部性、平移等变这些视觉先验焊进每一层；纯 Transformer 几乎没有这些先验，在 ImageNet 这种中等规模上会输给同体量 ResNet。ViT 只做一件事：把图切成固定大小的 patch，当 NLP 里的 token 用，后面接几乎原封不动的 Transformer encoder。先验不够，就用更大的预训练数据补。JFT-300M 上预训练之后，ViT-H/14 在 ImageNet 上到 88.55%（PDF p. 6），预训练算力明显低于当时的大 ResNet。

它解决的不是「注意力比卷积更懂图」，而是另一句：**当数据够大时，少 inductive bias 的通用序列模型，可以在图像分类上追上甚至超过专门为视觉设计的 CNN，而且预训练更便宜。**

## 一、矛盾：视觉一直不敢把 Transformer 直接接到像素上

NLP 这边的故事已经跑通：先在大语料上预训练 Transformer，再微调到下游（PDF p. 1）。模型可以扩到 100B 以上参数，性能还没有饱和迹象（PDF p. 1）。

视觉这边卷积仍是默认答案。想把自注意力引进来，常见两条路（PDF p. 1）：

1. 注意力当 CNN 的插件，或者只替换卷积网络里的某几块，整体骨架还是卷积；
2. 用专门设计的局部、稀疏、分轴注意力去「近似」全局注意力，因为像素对像素的全局注意力是像素数的平方，现实分辨率算不起。

后一类理论上更干净，但 specialized attention pattern 在当时的加速器上不好扩，大规模图像识别的 SOTA 仍是 ResNet 一类（PDF p. 1）。

作者的判断很干脆：这些工程近似本身可能才是瓶颈。他们要试的是相反方向——**尽量少改 Transformer，只改输入怎么变成序列**（PDF p. 1）。

中等规模上这招看起来会失败。ImageNet 上不强正则时，ViT 比同体量 ResNet 低几个百分点（PDF p. 1）。这并不意外：Transformer 没有卷积那种平移等变和局部性，数据不够时泛化更差（PDF p. 1–2）。

数据一旦到 14M–300M 张图，故事反过来：**大规模训练压过 inductive bias**（PDF p. 2）。这是全文主轴，后面所有设计都围着它转。

## 二、输入：一张图变成一串「词」

标准 Transformer 吃的是一维 token 序列。图像是 $H\times W\times C$ 的二维数组。ViT 的做法是把图切成 $P\times P$ 的 patch，展平，再用一个可学的线性投影打到 Transformer 的隐空间维 $D$（PDF p. 3）。

patch 数

$$
N = HW / P^2
$$

就是有效序列长度。$P$ 越小，$N$ 越大，计算越贵（PDF p. 5）。论文命名约定：ViT-L/16 表示 Large 变体、patch $16\times 16$（PDF p. 5）。标题里的「16x16 words」就是这个意思——一张 224 的图大约 196 个 patch，再加一个分类 token。

和 BERT 一样，序列最前面塞一个可学习的 **[class] token**。Transformer 最深层里这个位置的状态 $z_0^L$ 当作整图表示，接到分类头上（PDF p. 3 式 4）。预训练时分类头是带一层隐层的 MLP，微调时换成一层线性（PDF p. 3）。

位置信息用**可学习的一维位置嵌入**，加在 patch 嵌入上。作者试过更「懂二维」的位置编码，收益不明显，所以主实验就用 1D（PDF p. 3；附录 D.4）。

编码器几乎是 Vaswani et al. 2017 的原版：交替的多头自注意力和 MLP，LayerNorm 在块前，残差在块后；MLP 两层，中间 GELU（PDF p. 3–4）。作者特意强调：这样 NLP 里已经打磨好的可扩展实现可以几乎开箱即用（PDF p. 3）。

用公式把前向写全（PDF p. 4 式 1–4）：

$$
z_0 = [x_{\mathrm{class}}; x_p^1 E; \ldots; x_p^N E] + E_{\mathrm{pos}}
$$

$$
z'_\ell = \mathrm{MSA}(\mathrm{LN}(z_{\ell-1})) + z_{\ell-1}
$$

$$
z_\ell = \mathrm{MLP}(\mathrm{LN}(z'_\ell)) + z'_\ell
$$

$$
y = \mathrm{LN}(z_L^0)
$$

其中 $E$ 把展平 patch 投到 $D$ 维，$E_{\mathrm{pos}}\in\mathbb{R}^{(N+1)\times D}$。

图 1（PDF p. 3）把这条流水线画成：图切成格子 → 线性投影 + 位置嵌入 + 额外 [class] → 标准 Transformer encoder（MSA 与 MLP 交替）→ MLP Head 出类别。视觉读图：左侧是一张图被划成 3×3 的 patch 示意，右上是 encoder 内部的残差块；这是机制示意图，不是实测曲线。

### 少 inductive bias 到底少在哪

CNN 里局部性、二维邻域、平移等变是焊在每一层的。ViT 里只有 MLP 是局部且平移等变的，自注意力是全局的。二维结构几乎只出现两处（PDF p. 4）：

1. 一开始切 patch；
2. 微调到更高分辨率时，按二维位置插值位置嵌入。

初始化时位置嵌入**不含** patch 的二维坐标信息，空间关系全部要从数据里学（PDF p. 4）。这不是疏忽，是实验假设：先验不够，数据来补。

### 混合架构：也可以先卷积再 Transformer

不想从原始像素起步，可以把 CNN 特征图上的「格子」当 patch。极端情况 patch 空间大小 $1\times 1$，等于把特征图空间维展平再投影（PDF p. 4）。分类 token 和位置嵌入照旧加。后文会看到：小模型上 hybrid 略好，大模型上差距消失（PDF p. 7）。

## 三、微调：分辨率变了，位置嵌入怎么跟着变

预训练在大数据上做，下游用更小的任务微调。预训练头丢掉，换成零初始化的 $D\times K$ 线性层，$K$ 是下游类别数（PDF p. 4）。

常见做法是微调分辨率高于预训练（PDF p. 4）。patch 大小不变，序列就变长。Transformer 能吃任意长度（显存允许），但预训练好的位置嵌入对不上新网格。作者按位置在原图中的二维坐标做插值（PDF p. 4）。**分辨率调整和切 patch 是整篇里仅有的两处「人手注入二维先验」**（PDF p. 4）。

这招的可迁移含义很具体：想换输入分辨率，不要重训位置表，按网格插值通常够用——前提是位置嵌入已经学到了「近的 patch 更像」这种拓扑，第四节的可视化会印证这一点。

## 四、规格、数据、训练 recipe（论文写到的部分）

### 三种体量

配置对齐 BERT，再加一个 Huge（PDF p. 5 表 1）：

| 模型 | 层数 | 隐维 $D$ | MLP 维 | 头数 | 参数量 |
|---|---|---|---|---|---|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

### 预训练数据

三档，用来扫「数据量 vs 架构先验」（PDF p. 4）：

- ImageNet（ILSVRC-2012）：1k 类、1.3M 图；
- ImageNet-21k：21k 类、14M 图；
- JFT：18k 类、303M 高分辨率图。

预训练集会相对下游测试集去重，做法跟随 Kolesnikov et al. 2020（PDF p. 4）。下游：ImageNet 原验证标签和 ReaL 清洗标签、CIFAR-10/100、Oxford-IIIT Pets、Oxford Flowers-102，以及 19 项的 VTAB（每项 1000 条训练样本，分 Natural / Specialized / Structured）（PDF p. 4–5）。

CNN 基线不是原版 ResNet，而是 BiT 那套修改：BatchNorm 换成 GroupNorm，再用 standardized convolution，论文记作 ResNet (BiT)（PDF p. 5）。Hybrid 从 ResNet50 的 stage 4 出特征，或把 stage 4 的层挪进加长的 stage 3，序列长 4 倍（PDF p. 5）。

### 优化器与分辨率

预训练（含 ResNet）用 Adam，$\beta_1=0.9$，$\beta_2=0.999$，batch 4096，weight decay 0.1；线性 warmup 再 decay（PDF p. 5）。附录说这个设定里 Adam 预训练 ResNet 略优于 SGD，和常见实践相反（PDF p. 5、p. 16 表 7）。微调一律 SGD+momentum，batch 512（PDF p. 5）。表 2 的 ImageNet：ViT-L/16 微调到 512、ViT-H/14 到 518，并做 Polyak 平均、因子 0.9999（PDF p. 5）。预训练分辨率 224；未特别说明时微调 384（PDF p. 13–14）。

指标两条：微调精度；以及把冻结表示映射到 $\{-1,1\}^K$ 的正则最小二乘 few-shot，闭式可解，用来做便宜的在线评估（PDF p. 5）。

附录表 3 给出预训练超参摘要（PDF p. 13）：JFT 上 ViT 跑 7 或 14 epoch，ImageNet-21k 上 30/90 epoch，ImageNet 从头训 300 epoch 且用 cosine、更强正则（weight decay 0.3、dropout 0.1），ImageNet 从头训还加 global norm 1 的梯度裁剪。Dropout 用在每个 dense 层之后（qkv 投影除外）以及位置嵌入加到 patch 嵌入之后（PDF p. 13）。

这些数字是论文公开的 recipe 边界。JFT 的具体清洗、增强列表、以及 TPU 拓扑，正文没有展开成可复现脚本；开源仓库在脚注给出（PDF p. 1）：`https://github.com/google-research/vision_transformer`。仓库现状是外部补充，本文不追踪。

## 五、主结果：更少预训练算力，打平或超过大 CNN

表 2 把最大的 ViT 和文献里的大 CNN 放在一起（PDF p. 6）。算力单位是 TPUv3-core-days：TPU v3 核心数（每芯片 2 核）乘训练天数（PDF p. 5）。三次微调的均值±标准差：

| | ViT-H/14 JFT | ViT-L/16 JFT | ViT-L/16 I21k | BiT-L R152x4 | Noisy Student EfficientNet-L2 |
|---|---|---|---|---|---|
| ImageNet | 88.55 ± 0.04 | 87.76 ± 0.03 | 85.30 ± 0.02 | 87.54 ± 0.02 | 88.4 / 88.5* |
| ImageNet ReaL | 90.72 ± 0.05 | 90.54 ± 0.03 | 88.62 ± 0.05 | 90.54 | 90.55 |
| CIFAR-10 | 99.50 ± 0.06 | 99.42 ± 0.03 | 99.15 ± 0.03 | 99.37 ± 0.06 | — |
| CIFAR-100 | 94.55 ± 0.04 | 93.90 ± 0.05 | 93.25 ± 0.05 | 93.51 ± 0.08 | — |
| Pets | 97.56 ± 0.03 | 97.32 ± 0.11 | 94.67 ± 0.15 | 96.62 ± 0.23 | — |
| Flowers-102 | 99.68 ± 0.02 | 99.74 ± 0.00 | 99.61 ± 0.02 | 99.63 ± 0.03 | — |
| VTAB 19 项 | 77.63 ± 0.23 | 76.28 ± 0.46 | 72.72 ± 0.21 | 76.29 ± 1.70 | — |
| TPUv3-core-days | 2.5k | 0.68k | 0.23k | 9.9k | 12.3k |

\* 88.5% 来自 Touvron et al. 2020 的略改进结果（PDF p. 6）。

读表时抓住三件事：

1. **同一份 JFT，更小的 ViT-L/16 已经在所有列出任务上超过 BiT-L，预训练核·天是 0.68k vs 9.9k**（PDF p. 5–6）。
2. ViT-H/14 再把 ImageNet、CIFAR-100、VTAB 往上推，预训练仍远少于先前 SOTA（PDF p. 5–6）。
3. 公开的 ImageNet-21k 也能训出能用的 ViT-L/16：大约 8 核 cloud TPUv3、30 天（PDF p. 6）。没有 JFT 时这条路径更现实。

作者自己加了限制：预训练效率不只由架构决定，还受日程、优化器、weight decay 等影响，所以 4.4 节做了受控的性能–算力曲线（PDF p. 6）。

图 2（PDF p. 6）把 VTAB 拆成 Natural / Specialized / Structured。视觉读图：四组柱状图，ViT-H/14（蓝）在 Natural 和 Structured 上高于 BiT-R152x4（紫）、VIVI、S4L；Specialized 上顶尖两名接近。正文结论与柱状形态一致（PDF p. 6）。精确到任务的数字在附录表 9（PDF p. 22），例如 ViT-H/14 (JFT) 均值 77.6，与表 2 的 77.63 同量级。

附录还报了 ObjectNet：ViT-H/14 按 BiT 的评估设置，top-5 82.1%、top-1 61.7%（PDF p. 20）。

## 六、数据量：先验在小数据上值钱，在大数据上变成枷锁

第一组实验：ImageNet → ImageNet-21k → JFT-300M 预训练，小数据上扫 weight decay、dropout、label smoothing，再微调到 ImageNet（PDF p. 6 图 3）。

图 3 视觉读图：横轴三档预训练集，纵轴 ImageNet Top-1。小数据上 ViT 落在 BiT 阴影带下方；数据变大后 ViT 翻上去，更大变体也逐渐超过更小变体。正文：ImageNet 预训练时 ViT-Large 不如 ViT-Base（尽管有中等正则）；ImageNet-21k 上两者接近；**只有 JFT-300M 才让大模型完整兑现**（PDF p. 6）。CNN 在 ImageNet 预训练上仍赢，大数据上被 ViT 反超（PDF p. 7）。

注意脚注：ImageNet 预训练的模型仍会在 ImageNet 上「微调」，因为微调时升分辨率本身就涨点（PDF p. 6）。所以这条曲线不是「冻结表示」，是含分辨率提升的迁移。

表 5 是图 3 对应的全表，微调 384，且**没有**表 2 用的 Polyak 和 512 分辨率（PDF p. 15）。几个锚点：

- 只在 ImageNet 上预训练：ViT-B/16 的 ImageNet 77.91，ViT-L/16 是 76.53，ViT-L/32 掉到 71.16（PDF p. 15）——大模型在小数据上确实更差。
- ImageNet-21k：ViT-L/16 的 ImageNet 85.15，ViT-H/14 85.13（PDF p. 15）。
- JFT：ViT-H/14 的 ImageNet 88.04（PDF p. 15），低于表 2 的 88.55，差在表 2 的额外技巧。

第二组实验更「干净」：从 JFT 里随机抽 9M、30M、90M 和全量，**不额外正则**，超参固定，只用 early-stopping 取最好验证精度；为省算力报线性 few-shot 而不是全微调（PDF p. 7 图 4）。

图 4 视觉读图：横轴 JFT 样本数 10M–300M 量级，纵轴线性 5-shot ImageNet。ResNet 在小子集上更高、更早平台；ViT 小数据过拟合更重，90M+ 后反超。正文例子：ViT-B/32 比 ResNet50 稍快，9M 上差很多，90M+ 更好；ResNet152x2 vs ViT-L/16 同样模式（PDF p. 7）。

**可迁移判断**：卷积先验是小数据的保险；数据过了某个规模，限制假设空间的先验会变成天花板。论文没有声称找到了精确的交叉点，只给出 9M / 90M 这个量级上的翻转（PDF p. 7）。

## 七、算力曲线：同样 FLOP，ViT 通常更划算

4.4 节把数据瓶颈拿掉（全在 JFT-300M 上），只看迁移性能对预训练算力（PDF p. 8）。模型集合写得很具体：7 个 ResNet、6 个 ViT、5 个 hybrid，epoch 7 或 14（PDF p. 8）。Hybrid 名字末尾的数字是 ResNet 骨干的总下采样比，不是 patch 大小（PDF p. 8）。

图 5（PDF p. 7）视觉读图：横轴预训练总计算（exaFLOPs，对数），纵轴分别是五数据集平均迁移精度和 ImageNet。三条族：Transformer、ResNet (BiT)、Hybrid。ViT 整体在 ResNet 左上方；小算力处 Hybrid 略高于纯 ViT，大模型处并到一起。

三条结论（PDF p. 8）：

1. 同样性能，ViT 大约少用 2–4× 计算（五数据集平均）；
2. 小预算 Hybrid 略好，大模型差距消失——作者说这有点意外，因为直觉上卷积局部特征应该在任何尺度都帮得上忙；
3. 试过的范围内 ViT 还没有饱和。

表 6 给出逐模型数字（PDF p. 15），例如：

- ViT-B/32、7 epoch：ImageNet 80.73，55 exaFLOPs；
- ViT-L/16、14 epoch：87.12，1567 exaFLOPs；
- ViT-H/14、14 epoch：88.08，4262 exaFLOPs；
- ResNet200x3、14 epoch：87.22，3306 exaFLOPs；
- R50x1+ViT-L/16、14 epoch：87.12，1668 exaFLOPs。

v2 的 arXiv 注释写明 Figure 5 和 Table 6 的 exaFLOPs 计算有一处错误已修正，相对排序基本不受影响。这是 [arxiv 页面](https://arxiv.org/abs/2010.11929) 上的版本说明（外部补充），盘上这份就是修正后的 v2。

附录 D.5 补充「理论 FLOP ≠ 墙上时间」。图 12（PDF p. 19）视觉读图：左图峰值推理 img/sec/core 随输入边长 64–512 变化，ViT 与相近 ResNet 速度同量级，二次爆炸只在最大模型、最大分辨率上刚刚显出来；右图单核能塞的最大 batch，ViT 明显更省显存。正文：推理与反传速度差一个与模型无关的常数因子（PDF p. 19）。

## 八、模型内部：先验不够，并不等于什么都没学到

### patch 投影像一组局部基

第一层把展平 RGB patch 线性投到 $D$ 维。图 7 左（PDF p. 9）是 ViT-L/32 这些滤波器的前 28 个主成分。视觉读图：一块块小的方向性、纹理状滤波器，像在每个 patch 内部做低维结构分解。论文说它们「像合理的基函数」（PDF p. 8）。

### 位置嵌入自己长出二维拓扑

图 7 中（PDF p. 9）：每个格子是「某个 (row, col) 的位置嵌入」与所有其他位置的余弦相似度。视觉读图：近的格子更亮，同行、同列也更像。正文还提到更大网格上有时出现正弦结构（PDF p. 8）。这解释了为什么手工 2D 位置编码涨不了点（PDF p. 8、p. 17 表 8）。

表 8（ViT-B/16，ImageNet 5-shot linear，PDF p. 17）：无位置 0.61382；1D 默认 0.64206；2D 0.64001；相对位置 0.64032。有无位置差一截，编码方式之间几乎持平。作者猜测：输入已经是 patch 级（例如 14×14 而不是 224×224），空间关系没那么难学（PDF p. 18）。图 10（PDF p. 18）显示具体相似度图案还随超参变，所以「学到了拓扑」不等于「学到了同一张图」。

### 注意力距离 ≈ 感受野

用注意力权重在图像空间上的平均距离当「注意力距离」，类比 CNN 感受野（PDF p. 8）。图 7 右 / 图 11（PDF p. 9、p. 18）：每个点是某一层某一个头、在 128 张例图上的平均距离，图宽 224。浅层有的头已经看全图，有的头很局部；hybrid 浅层那种高度局部的头更少，像是卷积已经代劳了早期局部处理（PDF p. 8）。深度增加，距离整体变大；网络后半多数头跨 token 看很宽（PDF p. 20）。

图 6（PDF p. 8）视觉读图：三对 Input / Attention——狗、飞机、鸟。Attention 是从输出 token 滚回到输入空间的热力图，亮区落在主体而不是背景。做法是 Attention Rollout：ViT-L/16 所有头平均，再把各层权重矩阵递推相乘，计入层与层之间的混合（PDF p. 20）。图 14（PDF p. 21）是随机再抽的 128 张，同一套方法。这些图是定性「看在语义相关区域」，不是定量 SOTA 证据。

## 九、自监督：试过，有用，但还差一大截

NLP 的成功不只来自可扩展，也来自大规模自监督。作者做了一次初步的 **masked patch prediction**，模仿 BERT 的掩码语言模型（PDF p. 8–9）。

具体腐蚀：50% 的 patch 嵌入被破坏——80% 换成可学习 [mask]，10% 换成随机其他 patch，10% 保持原样（PDF p. 14）。预测目标是被腐蚀 patch 的 3-bit 均值颜色（512 色）（PDF p. 14）。他们还试过 4×4 降采样后的 16 个 512 分类、以及整 patch 的 L2 回归；都能用，L2 略差，主结果用均值颜色，因为 few-shot 最好（PDF p. 14）。15% 腐蚀率略差（PDF p. 14）。

自监督模型在 JFT 上 1M step（约 14 epoch），batch 4096，Adam，基学习率 $2\cdot 10^{-4}$，10k warmup，cosine（PDF p. 14）。更小的 ViT-B/16：ImageNet 79.9%，比从头训高 2%，仍比有监督预训练低 4%（PDF p. 9）。作者还观察到：不必把 JFT 和 1M step 用满，约 100k step 之后下游收益递减，ImageNet 上预训练也有相近增益（PDF p. 14）。对比学习路线明确留给未来（PDF p. 9）。

**边界**：这篇论文的自监督是附录级探索，不能当成「ViT 应该用 MAE/对比学习」的原文主张。

## 十、附录里几条会改设计决策的消融

### 分类 token 不是魔法，学习率才是

为了贴近原版 Transformer，主文始终用 [class] token，再经带 tanh 隐层的 MLP 出类别（PDF p. 16）。起初「去掉 extra token、对 patch 做 GAP 再线性分类」表现很差；后来发现既不是 token 的锅，也不是 GAP 的锅，而是 **GAP 需要另一档学习率**（PDF p. 16–17 图 9）。图 9 视觉读图：CLS、lr=$8\cdot 10^{-4}$ 与 GAP、lr=$3\cdot 10^{-4}$ 的 5-shot 曲线能走到一起；GAP 仍用 $8\cdot 10^{-4}$ 则明显偏低。v2 相对 v1 的修改之一就是补上这段 CLS vs GAP 讨论（外部补充：arXiv 页面注释）。

### 形状：深度比宽度更值得先加，缩小 patch 不增加参数也能涨

从图 8 的交点配置出发（8 层、$D=1024$、$D_{\mathrm{MLP}}=2048$、patch 32）分别加深度、宽度、MLP 宽度、缩小 patch（PDF p. 16）。深度改进最大，一直能看到 64 层，但 16 层之后已有收益递减；宽度变化最小；缩小 patch、加长序列「不加参数却相当稳地涨」（PDF p. 16）。作者建议：算力可能比参数量更能预测性能；若必须偏科，偏深度而不是宽度；总体仍是各维按比例一起加更稳（PDF p. 16）。

### Axial 注意力：精度换得到，TPU 上不一定划算

把全局自注意力改成行注意力+MLP 再列注意力+MLP，Axial-ViT-B/16 和 B/32 的 5-shot 高于对应 ViT-B，但更贵——每个全局块变成两个轴向块，还多一个 MLP（PDF p. 19–20 图 13）。AxialResNet 的 FLOP–精度看起来合理，naive 实现在 TPU 上极慢，作者说没能拿它做大规模实验（PDF p. 19）。可迁移点：**「理论上更省的注意力」如果吃不满加速器，墙上时间可以差一个数量级。**

## 十一、论文自己划的未完成工作

结论重申策略：除了最初切 patch，不往架构里塞图像专用先验；图当 patch 序列，用 NLP 那种 encoder 处理；配上大数据预训练，分类 SOTA 且预训练相对便宜（PDF p. 9）。

明确留下的挑战（PDF p. 9）：

1. 检测、分割等其它视觉任务——他们指向 Carion et al. 2020（DETR）说这条路有希望，但本文没做；
2. 自监督仍大幅落后大规模有监督预训练；
3. 继续放大 ViT 很可能还能涨。

没有写的、本文也不补：具体数据增强配方的完整列表、JFT 标注质量、多机通信、以及任何 2021 年之后的 ViT 变体。

## 十二、可迁移启发（区分「能直接用」和「绑在规模上」）

能直接复用的设计动作：

- **先把模态变成 token 序列，再尽可能用标准 Transformer**，而不是先发明一种加速器不喜欢的专用注意力。收益是实现和扩展路径现成；代价是必须接受「中小数据上可能输给带先验的模型」。
- **分辨率变了就插值位置嵌入，不要重训位置表。** 前提是位置嵌入已经学到相对拓扑。
- **分类头：微调时丢掉整个预训练 MLP 头，换成零初始化线性层**，论文觉得比只重初始化最后一层更稳一点（PDF p. 14）。
- **小数据才堆正则，大数据可以几乎不 dropout。** 表 3 里 JFT 的 dropout 是 0.0，ImageNet 从头训是 0.1 且 weight decay 0.3（PDF p. 13）。
- **测模型时把「数据规模」和「算力」拆开。** 图 3/4 回答数据，图 5 回答算力；混在一起会把 BiT 的 9.9k 核·天和架构优劣绑死。

绑在规模 / 硬件上的结论，不要当教条：

- 「纯 Transformer 一定优于 CNN」只在论文扫过的 JFT 量级和受控 FLOP 曲线上成立；ImageNet 从头训时大 ViT 更差（PDF p. 15）。
- Hybrid 在小模型上有用、大模型上消失，不能外推成「永远该先卷积」。
- 自监督 +2% 不能外推成现代 MIM 的数字。
- TPU 上的显存优势（图 12）换到别的硬件要重测。

一条贯穿全文的原则：

> **Inductive bias 是小数据的杠杆，也是大数据的天花板。先问数据够不够，再决定要不要把局部性和等变焊进架构。**

## 关键词回看

- **Patch embedding**：把 $P\times P$ 图像块展平后线性投到 $D$ 维，当作 token。
- **[class] token**：额外可学习向量，其最终状态当整图表示；GAP 也能用，但学习率不同。
- **1D 位置嵌入**：按光栅顺序的可学习向量；二维结构主要靠数据学，微调时按 2D 网格插值。
- **Inductive bias**：这里特指卷积自带的局部性、平移等变、二维邻域；ViT 几乎只在切 patch 和插值位置时用到。
- **JFT-300M / ImageNet-21k**：用来证明「规模压过先验」的两档大数据；前者内部，后者公开。
- **Attention distance**：注意力权重在图像平面上的平均跨度，用来类比感受野。
- **Masked patch prediction**：把一部分 patch 换成 mask/随机/原样，预测颜色统计；本文里是初步自监督，不是主结果。

## 参考资料

- 原论文 PDF（盘上 v2）：`readings/_src/多模态理解与 Omni/ViT.pdf`
- arXiv：https://arxiv.org/abs/2010.11929 （v1 2020-10-22；v2 2021-06-03）
- 作者给出的微调代码与预训练权重：https://github.com/google-research/vision_transformer （PDF p. 1 脚注）

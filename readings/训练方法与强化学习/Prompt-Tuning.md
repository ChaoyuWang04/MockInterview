# Prompt Tuning：冻住语言模型，只在输入端学一段软提示

<!-- release-date: 2021-04-18 -->

**本文依据**：`The Power of Scale for Parameter-Efficient Prompt Tuning`，arXiv **2104.08691v2**（[cs.CL] 2 Sep 2021），15 页，EMNLP 2021。作者 Brian Lester*、Rami Al-Rfou、Noah Constant，Google Research（Lester 为 Google AI Resident）。原件首次公开日取 arXiv **v1** 提交日 **2021-04-18**；本地 PDF 已是 v2，解读依据 v2，不回写首发日。文中数字都标 PDF 页码；标「外部补充」的段落不来自本文。Prefix-Tuning 只作为对照前提简述，细节不展开成专篇。

## 一句话

把整份预训练语言模型冻住，每个下游任务只在输入前头加一小段可训练的连续向量（软提示）。T5 规模越大，这段软提示越能追上全量微调；到十亿参数量级，Prompt Tuning 追上甚至对齐更强的多任务 model tuning，而 T5-XXL 上每个任务只需 **20,480** 个可训练参数（提示长度 5，PDF p.2 图 2）。域迁移时往往比改全网更稳。

## 一、矛盾：大模型要么整份复制，要么靠人写离散提示

预训练成功之后，适配下游任务有两条主流路（PDF p.1–2）。

一条是 **model tuning**（也就是全量微调）：每个任务改一遍全部权重，再各存一份完整模型。T5-XXL 一份就要 **110 亿** 参数（PDF p.2 图 2）。任务一多，存储和推理批次都拆不开。

另一条是 GPT-3 那套 **prompt design**（也叫 priming）：模型冻住，用人写的任务说明和几个例子当离散文本提示。一份通才模型能伺候很多任务，但任务描述容易写错、能塞进输入的条件文本有限，质量仍远落后于微调。论文给的硬对比是：GPT-3 175B 在 SuperGLUE 上的 few-shot 比微调后的 T5-XXL 低 **17.5** 分（**71.8** 对 **89.3**），尽管参数多 **16** 倍（PDF p.2）。

自动化搜离散词（Shin 等 2020）能超过手写提示，但仍填不平和 model tuning 的差距（PDF p.2）。

于是目标换成：既要冻住一份通才模型，又要把整份标注数据的信号压进可微的条件里，而不是只靠几个离散例子。

## 二、做法：只训输入端 $P_e$，其余 $\theta$ 一律不动

任务一律做成 T5 的 text-to-text：分类不再是 $\mathrm{Pr}(y|X)$，而是生成代表标签的 token 序列 $Y$，即 $\mathrm{Pr}_\theta(Y|X)$（PDF p.2–3）。

普通离散提示是把词序列 $P$ 接到 $X$ 前面，最大化 $\mathrm{Pr}_\theta(Y|[P;X])$，而 $P$ 的向量来自冻住的词表。Prompt Tuning 放开这条限制：提示有自己的参数 $\theta_P$，只更新 $\theta_P$。条件生成写成 $\mathrm{Pr}_{\theta;\theta_P}(Y|[P;X])$（PDF p.3）。

落到张量上：输入 $n$ 个 token 先变成嵌入矩阵 $X_e\in\mathbb{R}^{n\times e}$。软提示是 $P_e\in\mathbb{R}^{p\times e}$。拼成

$$
[P_e;X_e]\in\mathbb{R}^{(p+n)\times e}
$$

再原样走 encoder–decoder。梯度只落到 $P_e$（PDF p.3）。参数开销就是 $E\times P$（嵌入维 × 提示长度）。提示越短，新参数越少（PDF p.3）。

人话：这不像改网络内部的函数，而像在输入里塞进一组「假词」。网络怎么算还是预训练那一套；变的是后续输入被怎么读。第四节会把这句话拿来和 adapter 对照。

### 和 GPT-3 离散提示差在哪

| | GPT-3 prompt design | 本文 Prompt Tuning |
|---|---|---|
| 提示是什么 | 词表里的真实 token | 连续嵌入，不必对应任何词 |
| 怎么找 | 人手写或非可微搜索 | 反向传播，吃全量标注 |
| 模型权重 | 冻住 | 冻住 |
| 质量 | SuperGLUE 远落后于微调（PDF p.2） | 大模型上追上 model tuning（PDF p.1 图 1） |

### 和 Prefix-Tuning 差在哪（只记对照，不展开）

Li and Liang（2021）的 Prefix-Tuning 冻住模型，但把可学习前缀接到 **encoder 每一层** 的激活上（含输入层），等于在每层固定一段与样本无关的激活（PDF p.2、p.6）。本文进一步简化：只在 **嵌入后的输入** 前拼一段 $P_e$，中间层仍随当前样本更新（PDF p.6）。

默认配置里提示长度取 **100**，长于 Prefix-Tuning 常用的 **10** token 前缀；但因为只动输入层、不改各层激活，任务专属参数仍更少（PDF p.4）。Prefix-Tuning 还常用再参数化来稳住训练，训练期参数会再胀一截；本文配置不需要这套再参数化（PDF p.6）。用 BART 时 Prefix-Tuning 还要给 decoder 也加前缀；本文只给 encoder 侧加提示（PDF p.6）。

论文自认与 Prefix-Tuning、WARP 同期发展，并强调自己是第一个表明：**只靠输入端软提示、不加中间层前缀、不加任务专属输出层**，就足以和 model tuning 竞争（PDF p.2）。

## 三、T5 的坑：span corruption 不适合当冻住通才

GPT-3 是自回归语言模型，一直在写自然文本。T5 预训练是 **span corruption**：输入里挖洞、插 sentinel，目标是按 sentinel 把挖掉的片段拼回去，而且每个预训练目标都以 sentinel 开头（PDF p.3）。冻住之后，解码器先验几乎改不动。作者怀疑：只靠一段提示，很难把「总想吐 sentinel」的习惯拧过来（PDF p.3）。

所以冻住模型试了三种设定（PDF p.3–4）：

1. **Span Corruption**：T5 原样冻住。
2. **Span Corruption + Sentinel**：下游目标前头补一个 sentinel，让目标长得更像预训练。
3. **LM Adaptation**：用 Raffel 等讨论过的 LM 目标，再继续自监督一小段——给自然前缀，写自然续写。这次适配 **只做一次**，得到一份可复用的冻住模型。

LM 适配最长试到 **100K** step，大约相当于原 T5 预训练步数的 **10%**（PDF p.4、p.6）。作者事先并不确定「预训练末期再改目标」能不能接近从零按 LM 训，所以这本身也是实验问题（PDF p.4）。

## 四、默认配方与怎么训

冻住底座是公开 **T5.1.1** 全尺寸：Small、Base、Large、XL、XXL（相对 T5.1：预训练去掉监督数据、调了 $d_{\mathrm{model}}$ / $d_{\mathrm{ff}}$、激活从 ReLU 换成 GeGLU）（PDF p.4）。

图中绿色叉号的默认配置（PDF p.4）：

- LM 适配再训 **100K** step 的 T5
- 用类别标签嵌入初始化提示（见消融）
- 提示长度 **100**

评测在 SuperGLUE dev；每个提示只训一个任务，训练数据不跨任务混合；text-to-text 格式跟 T5，但输入前 **不加** 任务名（PDF p.4）。

训练 30,000 step，T5 标准交叉熵，恒定学习率 **0.3**，batch **32**。用 dev 集默认指标做 early stopping。JAX + Adafactor：weight decay $1\times 10^{-5}$，$\beta_2$ decay **0.8**，parameter scaling 关。实现用 Flax（PDF p.4）。单任务 model tuning 基线学习率 **0.001**，并扫过 batch，最后用每批 **216** token；多任务 model tuning 基线每批 **$2^{20}$** token，混合物里含 DPR（PDF p.4 脚注 3–4）。

调参与运行规模写在附录：77 次超参搜索（prompt 40 + 单任务 model tuning 37），主结果与消融共 **195** 次训练，域迁移另 **18** 次，ensemble 另 **24** 次（PDF p.12）。Small / Base 的提示在 4 块 TPU v2 上训，更大模型在 16 块 TPU v3 上训（PDF p.12）。

## 五、规模一上来，缺口就合上

图 1 把四条线画在同一张 SuperGLUE 分–模型参数图上（PDF p.1）：单任务 model tuning、多任务 model tuning、Prompt Tuning、GPT-3 的 prompt design。均值和标准差来自 3 次运行。

结论按论文原话（PDF p.4）：

- 模型越大，Prompt Tuning 越能追上 model tuning。
- XXL（**110 亿** 参数）上，Prompt Tuning 对齐更强的 **多任务** model tuning 基线，任务专属参数少 **20,000** 倍以上。
- 对 GPT-3 few-shot：Prompt Tuning 的 T5-Small 对齐 GPT-3 XL（大 **16** 倍以上）；Prompt Tuning 的 T5-Large 超过 GPT-3 175B（大 **220** 倍以上）。

作者也试过把 GPT-3 的手工文本提示直接套到 LM 适配后的 T5 上，同规模远低于 GPT-3，可能和预训练数据、架构、以及 T5 更短的序列长度有关（PDF p.4 脚注 5）。这不是本文主结果，只说明「同一套离散提示不能在两家模型间直接搬家」。

## 六、消融：大模型对配方不挑剔

图 3 四张图，3 次运行的均值和标准差（PDF p.5）。贯穿所有消融的观察：**XXL 对超参最稳**。

**提示长度** $\{1,5,20,100,150\}$（PDF p.5）。多数尺寸必须长过 1 个 token 才像样；XXL 即使用 **单 token** 提示仍然强，作者据此说模型越大，撬动行为所需的条件信号越少。超过 **20** token 之后增益就边际了。脚注 6：大于 100 对更大模型似乎 **略有损害**，Prefix-Tuning 也见过前缀过长反而掉点。

**初始化**（PDF p.5–6）：

- 随机均匀：从 $[-0.5,0.5]$ 采样
- 词表采样：T5 SentencePiece 里按预训练语料频率排序的前 **5,000** 个常见 token
- 类别标签：用各类别字符串的嵌入初始化提示里的若干位置；多 token 标签则平均嵌入；标签用完后用词表采样补齐

类别标签最好。小模型上三种初始化差距大，到 XXL 差距消失。ReCoRD / WSC 要生成短自由文本，没有分类标签，改用任务相关词初始化（ReCoRD：commonsense、reasoning、reading、comprehension；WSC：commonsense、pronoun、resolution）（PDF p.5 脚注 7）。

**预训练目标**（PDF p.6）：LM 适配全面好于 span corruption；给下游目标加 sentinel 的「权宜之计」收益很小。XXL 即使用 span corruption 也还能用。适配越长越好，一直到 100K step；XXL 对短适配也宽容，增益已经不大。

非理想的 span corruption 设定下，尺寸之间不稳定：Small 反而超过 Base、Large、XL。检查发现许多任务上这些中等模型 **从未学会输出合法类别标签**，因而得 0 分。两种常见失败：从输入里抄片段、预测空串。3 次运行方差很低，所以不是偶然。作者的判断：span corruption 冻住模型不可靠，5 个尺寸里只有 2 个好用；LM 适配后则各尺寸都稳（PDF p.6）。

LM 适配 100K step 的 T5.1.1 各尺寸 checkpoint 已随 T5 仓库发布（PDF p.6）。

## 七、参数量：XXL 长度 5 就是 20,480

图 2 的对照数字（PDF p.2）：T5-XXL 每份微调模型 **110 亿** 参数；提示长度 **5** 时，调好的提示每任务只要 **20,480** 参数，少五个数量级以上。

这和附录表 4 对得上：XXL、$d$ 对应可训练参数 $=$ 提示长度 × **4,096**（PDF p.13 表 4）：

| T5 尺寸 | 长度 1 | 长度 5 | 长度 20 | 长度 100 | 总参数（长度 1 时） | 可训练占比（长度 5） |
|---|---:|---:|---:|---:|---:|---:|
| Small | 512 | 2,560 | 10,420 | 51,200 | 76,961,664 | 0.00333% |
| Base | 768 | 3,840 | 15,360 | 76,800 | 247,578,624 | 0.00155% |
| Large | 1,024 | 5,120 | 20,480 | 102,400 | 783,151,104 | 0.00065% |
| XL | 2,048 | 10,240 | 40,960 | 204,800 | 2,849,759,232 | 0.00036% |
| XXL | 4,096 | **20,480** | 81,920 | 409,600 | 11,135,336,448 | **0.00018%** |

表 4 注明：可训练参数就是提示本身；总参数含冻住的 T5（含 SentencePiece 查找表）。论文写 Small 长度 20 为 **10,420**（不是 $20\times 512=10{,}240$），此处按表照录，不擅自改。Large 长度 20 也是 20,480，那是 $20\times 1{,}024$，和 XXL 长度 5 的 20,480 同数不同来源。

图 4 把几种适配方法的任务参数画在一起，架构固定为 T5.1.1，提示 / 前缀长度 1–100（PDF p.6）：超过十亿参数时，Prompt Tuning 任务专属参数低于 **0.01%**。Prefix-Tuning 推理大约 0.1%–1%，训练因再参数化更高。WARP 只调输入和输出层，低于 0.1%。Prompt design 只存 500–2,000 个 token ID，参数最少，但质量代价最大（PDF p.6 及脚注 9）。

## 八、和邻近方法怎么划界

**WARP**（Hambardzumyan 等 2021）：可训练参数加在输入层，但依赖 MLM 的 `[MASK]` 和可学习输出层，把 mask 投到类别 logit，因此只能单输出分类。Prompt Tuning 不改输入格式、不加任务头，质量也更接近 model tuning（PDF p.6）。

**P-tuning**（Liu 等 2021）：可学习连续提示按人工模板插在输入各处。要在 SuperGLUE 上出强结果，P-tuning **必须和 model tuning 一起用**，提示和主干一起更新；本文主干冻住。P-tuning 还要在输入里加「锚点」token（例如 RTE 假设后加问号）；本文输入原样不动（PDF p.6 及脚注 10）。

**Qin and Eisner（2021）** 的 soft words：位置靠手工原型，每层还有 $\Delta\ell_i$，参数随深度涨（PDF p.6）。

**Adapter**（Houlsby 等 2019）：冻住 BERT-Large、加 2%–4% 瓶颈层，GLUE 接近全量微调。机制不同：adapter 改的是作用在表示上的 **函数**（允许改写各层激活）；Prompt Tuning 函数不动，只加新的输入表示（PDF p.6–7）。

## 九、域迁移：冻住通才，少记捷径

作者的解释（不是新公式）：冻住核心 LM 之后，模型不能靠改内部理解去背数据集里的词面线索和虚假相关，只能用提示间接调制输入表示，因此对训练 / 评测输入分布不一致应更稳（PDF p.7）。下面两组实验是证据。

**抽取式 QA（MRQA 2019）**：在 SQuAD 上训，用 SQuAD 验证 F1 选 checkpoint，再到域外 dev 上零样本评 F1（PDF p.7 表 1，3 次均值 ± 标准差）：

| 数据集 | 域 | Model Tuning | Prompt Tuning | $\Delta$ |
|---|---|---:|---:|---:|
| SQuAD | Wiki | $94.9\pm 0.2$ | $94.8\pm 0.1$ | $-0.1$ |
| TextbookQA | Book | $54.3\pm 3.7$ | $66.8\pm 2.9$ | **$+12.5$** |
| BioASQ | Bio | $77.9\pm 0.4$ | $79.1\pm 0.3$ | $+1.2$ |
| RACE | Exam | $59.8\pm 0.6$ | $60.7\pm 0.5$ | $+0.9$ |
| RE | Wiki | $88.4\pm 0.1$ | $88.8\pm 0.2$ | $+0.4$ |
| DuoRC | Movie | $68.9\pm 0.7$ | $67.7\pm 1.1$ | $-1.2$ |
| DROP | Wiki | $68.9\pm 1.7$ | $67.1\pm 1.9$ | $-1.8$ |

多数域外集 Prompt Tuning 更好；TextbookQA 差 **12.5** F1。作者观察：域差越大（生物医学、教科书）赚得越多。Model tuning 更好的集里，DROP 和 SQuAD 同属 Wikipedia，属于最小的域转移（PDF p.7）。

**复述检测（QQP $\Leftrightarrow$ MRPC）**（PDF p.7 表 2）：

| 训 → 评 | 方法 | Accuracy | F1 |
|---|---|---:|---:|
| QQP → MRPC | Model | $73.1\pm 0.9$ | $81.2\pm 2.1$ |
| QQP → MRPC | Prompt | $76.3\pm 0.1$ | $84.3\pm 0.3$ |
| MRPC → QQP | Model | $74.9\pm 1.3$ | $70.9\pm 1.2$ |
| MRPC → QQP | Prompt | $75.4\pm 0.8$ | $69.7\pm 0.3$ |

QQP 训、MRPC 评：提示比全量微调高 **3.2** accuracy、**3.1** F1。反方向接近，accuracy 略好、F1 略差。作者据此认为 model tuning 可能过参数化、更容易过拟合训练任务（PDF p.7）。

## 十、Prompt ensembling：同一份冻住模型，N 份提示

存 N 份 T5-XXL 大约每份 **42 GiB**，推理还要跑 N 次前向（PDF p.7）。Prompt Tuning 对同一任务训 N 个提示，核心 LM 仍共享。处理一条样本时，不必跑 N 个模型：batch 设成 N，复制同一条输入、换不同提示，一次前向即可。存储和推理上的节省，和图 2 的多任务混批是同一逻辑（PDF p.7–8）。

五个提示、一份冻住 T5-XXL、默认超参，简单多数投票（PDF p.8 表 3）：

| 任务 | 指标 | 平均 | 最好单提示 | Ensemble |
|---|---|---:|---:|---:|
| BoolQ | acc. | 91.1 | 91.3 | 91.7 |
| CB | acc./F1 | 99.3 / 99.0 | 100.00 / 100.00 | 100.0 / 100.0 |
| COPA | acc. | 98.8 | 100.0 | 100.0 |
| MultiRC | EM/F1a | 65.7 / 88.7 | 66.3 / 89.0 | 67.1 / 89.4 |
| ReCoRD | EM/F1 | 92.7 / 93.4 | 92.9 / 93.5 | 93.2 / 93.9 |
| RTE | acc. | 92.6 | 93.5 | 93.5 |
| WiC | acc. | 76.2 | 76.6 | 77.4 |
| WSC | acc. | 95.8 | 96.2 | 96.2 |
| SuperGLUE (dev) | — | 90.5 | 91.0 | **91.3** |

所有任务上 ensemble 超过单提示平均，并达到或超过最好的那一个提示（PDF p.8）。

## 十一、可解释性：像词，但不是一段人话

理想情况是提示本身就是清楚的自然语言任务说明（PDF p.8）。软提示在连续空间里，做不到这一点。

做法：对每个学到的提示 token，在冻住词表里找余弦距离最近邻。观察（PDF p.8）：

- 单个 token 的 top-5 近邻往往是紧的语义簇（如 Technology / technology / …，或 entirely / completely / totally / altogether / 100%）。随机从嵌入空间抽样的向量没有这种簇，说明提示确实在学「像词」的表示。
- 类别标签初始化时，标签常常一直留在近邻里。随机或词表初始化时，类别也会出现在近邻中，但更分散。作者猜测模型把期望输出类存在提示里当参照，用类别初始化更集中、更容易。
- 长度 100 时，多个位置会共享同一组近邻：要么提示容量过剩，要么提示没有序列结构，信息难以钉在某个位置。
- 整体序列几乎不可读，类似 AutoPrompt。BoolQ 上常看到 science / technology / engineering 一类近邻，而约 **20%** 的 BoolQ 问题属 Nature/Science 类。作者只把它标成待查的线索：提示的一个角色可能是把模型「扳」到某个领域语境（PDF p.8）。

## 十二、限制与论文没写的

论文写清的边界：

- **规模是前提。** 小 T5 上 Prompt Tuning 明显落后 model tuning；缺口要到十亿以上才合上（PDF p.1、p.4）。不要把 XXL 的结论套到 Small。
- **预训练目标不匹配会直接废掉中等模型。** 纯 span corruption 下 Base / Large / XL 可以稳定地输出非法标签、得 0 分；LM 适配要额外投入最多 100K step（PDF p.6）。
- **不是 few-shot 离散提示。** 它吃的是下游全量标注，和 GPT-3 priming 不是同一资源假设（PDF p.1–2）。
- **提示占序列位置。** 长度要在质量和上下文预算之间折中；超过约 20 token 增益很小，超过 100 对大模型可能略伤（PDF p.5）。
- **软提示读不成自然语言。** 近邻簇不等于一段可部署的文字说明（PDF p.8）。
- **WSC 被改写成生成指代对象** 后，只能用「给定指代正确」的训练例，训练集从常见的 554 条收到 **259** 条（PDF p.13–14 表 7）。和其他 SuperGLUE 论文的 WSC 数字不能直接横比。
- 图 1 的 GPT-3 线是 Brown 等报告的 SuperGLUE **dev** few-shot，不是作者在同一代码栈里复现（PDF p.4）。

论文没写、本文也不补：没有和 LoRA / Adapter 在同一 T5-XXL SuperGLUE 设定下的头对头数字；没有推理延迟墙钟；没有把软提示合并进词表后的部署细节；没有中文或其他语言。

## 十三、可迁移的几条

1. **先问底座是不是「会写自然文本」的冻住模型。** T5 式 span corruption 直接冻住，小中模型可能根本学不会合法输出；先做一次 LM 适配，再在各任务上只训提示，比每个任务微调更划算（PDF p.3–4、p.6）。
2. **参数效率的数字要带长度。** 对外说「XXL 每任务 20,480」时，对应的是长度 5，不是默认实验的长度 100（长度 100 是 409,600）（PDF p.2 图 2、p.13 表 4）。
3. **大了以后超参变钝。** 初始化、长度、适配步数在 XXL 上差异收缩；把调参预算花在小模型上，再把稳的配方直接搬到大模型，是这篇消融的读法（PDF p.5）。
4. **域外比域内更能看出「该不该冻住」。** 域内 SQuAD 几乎打平，域外 TextbookQA 拉开 12.5 F1（PDF p.7）。若产品会换领域，冻住通才、只换提示，值得作为默认对照。
5. **集成可以发生在提示上，而不必发生在模型上。** 同一条输入、batch 维换提示，比存 N 份 42 GiB 权重现实得多（PDF p.7–8）。

## 关键词回看

- **软提示（soft prompt）**：输入前可训练的连续嵌入 $P_e$，不必是词表里的词。
- **Prompt Tuning**：冻住 $\theta$，只更新 $\theta_P$。
- **Prompt design / priming**：冻住模型，用离散文本当条件；GPT-3 few-shot 走这条。
- **Model tuning**：下游改全部权重。
- **Prefix-Tuning**：每层前缀激活可学；本文只动输入层。
- **LM adaptation**：把 span-corruption 的 T5 再按语言模型目标续训一小段，得到更适合被提示控制的冻住底座。
- **Prompt ensembling**：同一冻住 LM 上训多份提示，投票或批内并行。

## 参考资料

- 论文 PDF（本地 v2）：`readings/_src/训练方法与强化学习/Prompt-Tuning.pdf`
- arXiv：https://arxiv.org/abs/2104.08691
- LM 适配 T5.1.1 发布说明（论文脚注 8）：T5 仓库 `released_checkpoints.md` 中 `lm-adapted-t511lm100k` 一节

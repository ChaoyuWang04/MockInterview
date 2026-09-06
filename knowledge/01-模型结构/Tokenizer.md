# Tokenizer(BPE / BBPE · 词表大小与 fertility)

一句话:tokenizer 是**文本和整数序列之间那本可逆的字典**——模型永远看不见字符,只看得见它切出来的整数;这本字典切得好不好,直接决定同一段文字要花多少 token、embedding 与输出层要占多大、以及模型能不能数清一个单词里有几个字母。

## 一、它站在整条链路的哪个位置

一次前向最前面两步是固定的:**tokenizer 把字符串切成 token id 序列**,**embedding 表按 id 查行**,得到一串 $d$ 维向量交给第一层;输出端反过来,在词表上算分布、采样出一个 id,再由 tokenizer 拼回字符串。所以 tokenizer 是全链路里**唯一不可微、不参与训练的一环**——它在预训练开始之前就被冻住,之后模型的每一个参数都是围着它长出来的。这条性质是第五节全部代价的根。

> 顺手划清一条边界:token id 查出来的那张表,和检索里说的「句向量 embedding」不是一回事。前者是逐 token 的查表参数,后者是把整段文本压成一个稠密向量用于相似度检索,属于 Embedding 篇的范围。

### 三种切分粒度,两种死法

| 粒度 | 词表规模 | 序列长度 | 主要问题 |
|---|---|---|---|
| 词级(按空格或词典切) | 十万到百万 | 最短 | **OOV**:没见过的词只能变 `<unk>`,信息当场丢掉;中文还得先配一个分词器 |
| 字符级 / 字节级 | 几十到 256 | **最长** | 没有 OOV,但序列长好几倍,而注意力是长度的平方,算不起 |
| **子词(subword)** | 三万到二十六万 | 折中 | 今天的默认解;坑全在「切在哪」上 |

两个极端各死一次:词级死在**表达不了**,字符级死在**算不起**。子词的思路是让**常见词整块保留、罕见词拆成常见片段**——`unhappiness` 切成 `un` + `happi` + `ness`,信息没丢,长度也没炸。

## 二、BPE 的合并过程

### 训练:数相邻对、合并、记账

BPE(Byte Pair Encoding)原本是个数据压缩算法,Sennrich 等人 2015 年把它搬进了机器翻译。训练过程一句话说得完:**从最小单位出发,反复把语料里出现次数最多的那一对相邻符号并成一个新符号,并把这次合并按顺序记进一张表**。

```python
def train_bpe(corpus, num_merges):
    merges = []                                    # 有序合并表,下标就是优先级
    for _ in range(num_merges):
        pairs = Counter()
        for word, freq in corpus.items():          # word 是已切成基本单位的元组
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq              # 相邻对按词频加权统计
        if not pairs: break                        # 已经合无可合
        best = pairs.most_common(1)[0][0]          # 取全语料出现最多的一对
        corpus = {apply_merge(w, best): f for w, f in corpus.items()}
        merges.append(best)                        # 记账:这次合并的优先级最低
    return merges
```

三个常被追问的点:

- **基本单位是什么**,决定了这是 BPE 还是 BBPE(见本节第三小节)
- **合并表是有序的**。第 1 条的优先级高于第 5000 条,这个顺序在推理时就是全部
- **词表大小 = 基本单位数 + 合并次数 + 特殊 token**。GPT-2 的 50257 正是这么来的:256 个字节 + 50000 次合并 + 1 个 `<|endoftext|>`。所以「词表大小」这个超参**实际控制的是合并次数**

推理侧**完全不看这段文本里谁频率高**:把每个块拆成基本单位,反复挑出当前相邻对里**在合并表中排名最靠前**的那一对合并掉,直到一对都不在表里。**所以同一段文本每次切出来完全一样**——训练期的频次统计已经固化成一张有序表,推理期只是照表执行:确定、无状态、可缓存。(Unigram 那一族是例外,它允许采样出不同切分,见下面的对照表。)

### 先切块再合并:pre-tokenization 才是隐藏的主角

真实的 BPE 从不直接在整段文本上跑。**它先用一条正则把文本切成块(大致相当于「词」),BPE 只在块内部合并,永远不跨块。** 这条正则是 tokenizer 里最被低估的设计:

- OpenAI 的 `cl100k_base`(GPT-3.5/4 那一代)里有一段 `\p{N}{1,3}+`,意思是**数字最多三位一组**。于是 `1234` 被切成 `123` + `4`,同一个数位在不同长度的数里落进不同 token——这是多位数算术容易翻车的直接来源之一
- LLaMA 走了另一条路:**把所有数字拆成单个数字**,每位一个 token,序列更长但每一位对齐
- 块边界还决定了空格归谁。GPT-2 系把前导空格并进词里(` the` 是一个 token),SentencePiece 干脆把空格替换成可见元符号 `▁`;两种做法的目的一样——**让解码时能无歧义地还原原文**

### BBPE:退到字节层,OOV 就此消失

**BPE 的基本单位是字符,BBPE(byte-level BPE)的基本单位是 UTF-8 字节。** 只改这一处,换来一条硬保证:

**UTF-8 里任何字符都是 1–4 个字节,而 256 个字节值全都在词表里,所以任何输入都表示得出来,永远不会有 `<unk>`。** 字符级 BPE 做不到——Unicode 码点有十几万个,基本单位表根本装不下,装不下的只能退化成 `<unk>`。Wang 等人的 BBPE 论文里,在同等翻译质量下词表能压到字符级方案的八分之一。

代价要说准:**罕见字符会变得很碎**。一个汉字是 3 个 UTF-8 字节,若没有任何合并覆盖它,就实打实占 3 个 token——**OOV 变成了「切得碎」,信息没丢,长度涨了**,这正是第四节中文那笔账的源头。还有个常被拿来考的实现细节:GPT-2 并不把原始字节直接塞进词表,而是**先把 256 个字节一一映射到一批可打印的 Unicode 字符**再跑 BPE,于是词表文件里不会出现控制字符和空白,可读、可存、可 diff。

### 近亲三种:判据不同,层次也不同

| 名字 | 它是什么 | 决定「合并/保留谁」的判据 | 切分唯一吗 |
|---|---|---|---|
| **BPE / BBPE** | 合并算法 | 相邻对的**出现频次**最高 | 是(按 merge 表贪心) |
| **WordPiece** | 合并算法 | 合并后**语料似然**提升最大,实践中按 $\frac{f(ab)}{f(a)\,f(b)}$ 打分 | 是(编码时最长匹配) |
| **Unigram LM** | **剪枝**算法 | 从大种子词表出发,先删**删掉后似然损失最小**的 | **否**,可按概率采样多种切分 |
| **SentencePiece** | **不是算法,是工具框架** | 里面既能装 BPE,也能装 Unigram | 取决于装了哪个 |

最高频的混淆点就在最后一行:**「用 BPE 还是用 SentencePiece」这个问法本身错位了**。SentencePiece 解决的是另一件事——**把输入当成一串裸 Unicode,连空格都当普通符号(换成 `▁`)**,于是不需要任何语言相关的预分词器,中日文这类没有空格的语言可以和英文走同一条流水线,并且保证 `decode(encode(x))` 精确还原(它管这叫 lossless tokenization)。Llama 1/2 的原话就是「用 SentencePiece 里的 BPE 实现」。顺带一提,Unigram 那条「切分不唯一」还有个训练期用途——**subword regularization**,每轮给同一句话不同切法,当数据增强用。

## 三、词表大小的三方权衡与 fertility

### fertility:一个词平均被切成几片

$$
\text{fertility} = \frac{\text{切出的 token 数}}{\text{原文的词数(或字符数)}}
$$

这个比值说的是「一个词平均被剁成几块」,越低表示同样的信息用更少 token 装下了,也常被叫作压缩率。**分母用词还是用字符必须先讲清楚,否则两个数字根本没法比。** 给一个论文实测:Llama 3 把 Llama 2 的 32K SentencePiece 词表换成 128K 之后,英文样本上的压缩率从 **3.17 字符/token 提高到 3.94 字符/token**,同一批文本少了约五分之一的 token。

### 词表一变大,三件事同时发生

| 维度 | 词表变大 | 因为 |
|---|---|---|
| **序列长度** | ↓ 变短 | 同样的话切得更少:上下文窗口装得下更多内容,自回归的步数也更少 |
| **参数与显存** | ↑ 变大 | embedding 表和输出投影(lm_head)都是 $V \times d$,**两张表都正比于 $V$** |
| **输出层计算** | ↑ 变大 | 每生成一个 token 都要算一遍 $V$ 维 logits,再在全词表上做一次 softmax |

把账算实。Llama 2 7B 是 $V=32000$、$d=4096$,两张表合计 2.6 亿参数,占 67 亿的约 **4%**;Llama 3 8B 宽度没变、$V$ 涨到 128256,两张表变成 **10.5 亿参数,占 80 亿的约 13%**——多出来的 7.9 亿占了 7B 到 8B 那 13 亿总差额的六成。Gemma 3 4B 更极端:$V=262208$、$d=2560$,单是 embedding 表就 **6.7 亿参数**。训练侧还有一处更疼:**logits 张量的形状是 $[\text{batch} \times \text{seq}] \times V$**,batch 4、序列 8192、$V=128256$ 时,光这一个 FP32 中间张量就 **16.8 GB**,比模型权重还大——这就是大词表模型几乎必上融合交叉熵(不物化完整 logits)的原因。输出层本身怎么加速(分层 softmax、采样 softmax、词表并行),归 Softmax优化 篇管。

### 那到底该多大

**没有普适答案,但方向性结论是有的。** Tao 等人的词表缩放律指出:**最优词表应该随模型规模一起涨**,而现役模型普遍偏小——他们估计 Llama2-70B 的最优词表至少是 **216K,是它实际 32K 的 7 倍**;在固定 2.3e21 FLOPs 预算下,把词表从 32K 抬到 43K,ARC-Challenge 就从 **29.1 提到 32.0**。现役量级(均取自各家公开配置),趋势是**从 3 万量级涨到 13 万至 26 万量级**,涨出来的部分主要买多语言与代码的覆盖:

| 模型 | GPT-2 | Llama 2 / Mistral 7B v0.1 | Llama 3 系 | DeepSeek-V3 | Qwen3 | Gemma 3 |
|---|---|---|---|---|---|---|
| 词表大小 | 50257 | 32000 | 128256 | 129280 | 151936 | 262208 |

### 但 fertility 不是越低越好

这是最容易被追问翻的一条。**fertility 只衡量「切得省不省」,不衡量「切得对不对」**:

- 一味压 fertility 会把大量长串塞进词表,其中不少在真实数据里几乎不出现,**训练时拿不到几次梯度**——这就是 glitch token(未训练 token)的来源。Land 与 Bartolo 做了自动检测,发现这类 token 在各家模型里都相当普遍,一旦被喂进去容易触发离谱输出
- 词表越大,**平摊到每个 token 的训练样本越少**,长尾 token 的表示学不扎实
- Ali 等人在 2.6B 规模上训了 24 个模型,结论很直白:**fertility 和 parity 这两个指标并不总能预测下游表现**

所以口径应该是:**fertility 是成本指标,不是质量指标**——它能解释「为什么贵」,不能单独用来选 tokenizer。

## 四、中文、代码、多语言:坑都埋在哪

### 中文:没有空格,预分词的前提就不成立

英文的 pre-tokenization 靠空格切词,**中文没有词边界,一整句可能落进同一个块**;更糟的是,训练语料里中文占比低的话,合并表里就几乎没有中文的合并,一个汉字直接退化成 3 个字节 token。Chinese-LLaMA 那篇给了最直观的对照:原版 LLaMA 词表覆盖的汉字**不到一千个**,同一句 28 个字的中文,原版切出 **35 个 token**,扩表到 49953 之后只要 **16 个**;论文的说法是,同样的上下文窗口能装下约两倍信息,生成速度也快了约一倍。

### 代码:缩进和符号是纯成本

代码里出现最多的字符是空格。GPT-3 的文本 tokenizer 用在代码上最大的浪费就在这:**连续空格没有对应 token,一层四格缩进要烧掉好几个 token**。Codex 的做法是**专门加一批表示不同长度空白串的 token**,同样的代码少用了约 **30%**。今天的通用模型基本都做了这件事,但它提醒的是一条通则:**tokenizer 的效率完全取决于它训练时见过什么分布**。数字是同一类坑——三位一组还是逐位切,直接影响多位数算术的稳定性。

### 多语言:fertility 高等于变相涨价

低资源语言在语料里占比低 → 合并表里几乎没有它的合并 → 切得极碎。这不是「效果差一点」,而是三笔实打实的账:**上下文窗口缩水**(同样 32K 窗口,fertility 翻倍就只装得下一半内容)、**按 token 计费的 API 直接贵一倍以上**、**推理步数变多**(自回归一步只出一个 token,token 多就是延迟高)。

Petrov 等人量化过这个差距:**同一段文本翻成不同语言,token 数最多能差 15 倍**;Ahia 等人直接算了 API 账单,结论是大量语言的使用者**多付了钱还拿到更差的结果**。训练侧同样疼:Ali 等人测出,拿纯英文 tokenizer 去训多语言模型,额外训练开销最高到 **68%**。

> 🖼️ 占位:同一段文本在若干语言上的 fertility 柱状对比(以英文为基准 1.0,标出中文、日文与一两种低资源语言的倍数)

**根因不在算法,在数据配比**:BPE 只会忠实反映训练语料的频次分布,谁在语料里多,谁就被优待。所以调 tokenizer 的第一步永远是调它的训练语料配比(语料侧怎么配,归 数据工程 篇管)。

**顺带解释一个经典现象**:「模型数不清 strawberry 里有几个 r」也出在这里——这个词被切成两三个 token 之后,模型手里只有这两三个整数对应的向量,**单词内部的字母序列压根不在它的输入里**。它要么在训练数据里见过足够多谈论拼写的文本、把答案背下来,要么就只能猜。这是输入表示的结构性缺陷,不是推理能力不足。

## 五、换 tokenizer:为什么伤筋动骨,怎么迁移

### 要动的东西比想象中多

| 要动什么 | 因为 |
|---|---|
| **embedding 表** | 每一行的含义全变了,旧的 $V \times d$ 参数整张作废 |
| **lm_head** | 输出维度就是 $V$,必须重建;绑权重(tied embedding)的模型两边要一起改 |
| **模型学到的 token 级统计** | 模型学的是「这个 id 后面常跟哪个 id」,换表之后这些共现关系全部对不上 |
| **全部下游资产** | 已缓存的 KV、已建好的向量索引、按 token 计的长度上限与计费口径,都要重算 |

一句话:**tokenizer 是预训练的地基,不是可插拔的前处理**。所以现实里几乎没人「换一下试试」,只有两条路——**扩表(vocabulary extension)**:保留原词表往后追加新 token,embedding 与 lm_head 各接一段新行,**旧 id 的含义完全不变、已经学到的东西全保住**,只需继续预训练把新 token 喂熟(Chinese-LLaMA 的 32000 → 49953 就是这条路),代价是新旧两批 token 的切分会互相打架、整体设计仍被旧词表约束;**换表**:整张换掉等于把地基刨了,后面必须跟大规模继续训练甚至重训。

### 新 token 的 embedding 怎么初始化才不至于白训

随机初始化能用,但收敛慢——这批新行等于从零学起,前期还会扰动已经训好的部分。业界思路一致:**用旧词表给新 token 造一个有意义的起点**。

- **最朴素的一招**:用旧 tokenizer 把新 token 切开,拿这几个旧 embedding 取平均。便宜、好实现,是扩表的默认做法
- **WECHSEL**:借一套同时覆盖两种语言的静态词向量当桥,把新 token 对齐到语义相近的旧 token 上再初始化
- **FOCUS**:把新 token 表示成新旧词表**重叠部分**的加权组合,权重由一个辅助的静态向量空间给出
- **ZeTT**:训一个 hypernetwork,输入一个 tokenizer 直接预测出对应的 embedding,做到近似零样本换表

共同点是同一句话:**别让新行从噪声开始**。共同的边界也是同一句话——它们只解决「起点」,**后面照样要继续训练**才能真正长好。

### 什么时候才值得动

Dagan 等人给了一个可操作的门槛:**领域继续训练的规模超过约 500 亿 token 时,专门为这个领域重训一次 tokenizer 才划算**;低于这个量级,省下的 token 换不回重训的代价与风险。继续训练的配方本身,归 长上下文训练 篇与 SFT 篇管。

最后补三条边界。绕开 tokenizer 的路线确实有(如 Byte Latent Transformer 直接在字节上按熵动态分块),但还没进主流模型主干,进展跟踪归 新架构追踪 篇管;多模态里的「视觉 tokenizer」是完全另一件事——那是把图像离散成码本,归 图像Tokenizer 篇管;切完之后 token 怎么被加工,分别归 注意力基础 篇、FFN与激活 篇和 RoPE 篇管。

## 六、面试考点串联

> 题库目前没有任何真题的 `topic` 指向本篇,**下表全部是补充题**:按正文各节的核心断言反推出来的问法,用于自测覆盖,不代表已被真实面试考过。

| 高频问法 | 本文哪一节 |
|---|---|
| BPE 是怎么训出来的?到了推理阶段,同一段文本为什么每次都切得一模一样 | 二(有序合并表) |
| BBPE 和 BPE 差在哪?为什么说到了字节这一层就不会有 OOV | 二 |
| SentencePiece 和 BPE 是什么关系?WordPiece 又是按什么判据合并的 | 二(四行对照表) |
| pre-tokenization 那条正则会影响什么?数字为什么要特殊对待 | 二 |
| fertility 是什么?它高了具体让谁难受 | 三 |
| 词表开大到底换来了什么?代价压在哪几处?是不是越大越好 | 三(三方权衡表 + 参数账) |
| 同样一句话,中文为什么比英文更费 token?有什么办法救 | 四 |
| 代码场景下 tokenizer 最容易在哪出问题 | 四 |
| 多语言模型里,低资源语言的用户为什么等于多付钱?根因在算法还是数据 | 四 |
| 为什么模型数不清一个单词里有几个字母?这该归咎于什么 | 四 |
| 想给一个开源模型扩词表,要动哪些东西?新 token 的 embedding 怎么初始化才不至于白训 | 五 |
| 什么情况下换 tokenizer 才算划算?不划算的时候一般怎么办 | 五 |

## 相关文献

- Neural Machine Translation of Rare Words with Subword Units(BPE 的提出)— [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- Neural Machine Translation with Byte-Level Subwords(BBPE,词表可压到字符级的八分之一)— [arXiv:1909.03341](https://arxiv.org/abs/1909.03341)
- SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing — [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates(Unigram LM 与采样切分)— [arXiv:1804.10959](https://arxiv.org/abs/1804.10959)
- Fast WordPiece Tokenization(WordPiece 的高效实现;算法原始出处是 Schuster & Nakajima, ICASSP 2012)— [arXiv:2012.15524](https://arxiv.org/abs/2012.15524)
- Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies(Llama2-70B 最优词表估计 ≥ 216K)— [arXiv:2407.13623](https://arxiv.org/abs/2407.13623)
- Tokenizer Choice For LLM Training: Negligible or Crucial?(fertility/parity 不总能预测下游;英文 tokenizer 训多语言最高多花 68%)— [arXiv:2310.08754](https://arxiv.org/abs/2310.08754)
- Getting the most out of your tokenizer for pre-training and domain adaptation(超过 50B token 的领域继续训练才值得重训 tokenizer)— [arXiv:2402.01035](https://arxiv.org/abs/2402.01035)
- Language Model Tokenizers Introduce Unfairness Between Languages(同一文本跨语言 token 数最多差 15 倍)— [arXiv:2305.15425](https://arxiv.org/abs/2305.15425)
- Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models — [arXiv:2305.13707](https://arxiv.org/abs/2305.13707)
- Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca(扩表至 49953,同句 35 → 16 token)— [arXiv:2304.08177](https://arxiv.org/abs/2304.08177)
- Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models — [arXiv:2405.05417](https://arxiv.org/abs/2405.05417)
- WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models — [arXiv:2112.06598](https://arxiv.org/abs/2112.06598)
- FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models — [arXiv:2305.14481](https://arxiv.org/abs/2305.14481)
- Zero-Shot Tokenizer Transfer(hypernetwork 预测新 tokenizer 的 embedding)— [arXiv:2405.07883](https://arxiv.org/abs/2405.07883)
- Evaluating Large Language Models Trained on Code(Codex 加空白串 token,代码少用约 30%)— [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- The Llama 3 Herd of Models(128K 词表 = 100K tiktoken + 28K;英文压缩率 3.17 → 3.94 字符/token)— [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- LLaMA: Open and Efficient Foundation Language Models(SentencePiece BPE、数字逐位切分、未知字符字节回退)— [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- Byte Latent Transformer: Patches Scale Better Than Tokens(按熵动态分块的无 tokenizer 路线)— [arXiv:2412.09871](https://arxiv.org/abs/2412.09871)
- tiktoken(`cl100k_base` / `o200k_base` 的切分正则与特殊 token)— https://github.com/openai/tiktoken
- SentencePiece 官方仓库(`▁` 元符号、`byte_fallback` 与 lossless tokenization)— https://github.com/google/sentencepiece

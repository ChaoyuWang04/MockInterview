# Softmax 优化

一句话:语言模型每生成一个 token 都要在整张词表上打一遍分,词表到十几万、上百万时这一步会同时吃掉计算、显存和通信——本篇讲**分层、自适应、采样 Softmax 与 NCE 各自少算了什么、换来了什么代价**,以及为什么它们大多只能在训练时用;词表大小的参数账与 logits 显存账见 Tokenizer 篇,交叉熵自身的性质与稳定写法见 分类损失 篇,InfoNCE 与负例工程见 对比学习 篇,张量并行的通信规则见 并行策略 篇,本篇不重写。

## 一、先定位瓶颈:贵的是矩阵乘、显存和通信,不是 exp

输出层只有三步:最后一层隐状态 $h\in\mathbb{R}^d$ 先乘输出矩阵得到 $V$ 个分数(logits),$z = W_{\text{out}}h$,$W_{\text{out}}\in\mathbb{R}^{V\times d}$;再对这 $V$ 个分数做一次 softmax;最后取真值那一项算交叉熵。**面试里最常见的误区是把「Softmax 慢」理解成「exp 算得慢」**——把三步的账分开算,结论正好相反。取 $V=131072$(128k)、$d=4096$、一个 micro-batch 共 $B\times S=16384$ 个 token:

| 环节 | 每 token 的量 | 这个算例下的总量 | 性质 |
|---|---|---|---|
| 输出投影 $W_{\text{out}}h$ | $2dV\approx 1.07$ GFLOP | 前向约 **17.6 TFLOP**,反向再翻倍 | 一个 $16384\times4096\times131072$ 的大 GEMM,**算力受限** |
| softmax + 交叉熵 | 约 $3V\approx 4\times10^5$ 次逐元素运算 | 约 6.4 GFLOP,比投影少约 **2700 倍** | 逐元素、访存受限,慢在把 logits 读写几遍 |
| logits 张量 | $V$ 个数 | $16384\times131072$ 个 FP32 = **8.6 GB**;朴素实现还要存概率与梯度副本,峰值 17–26 GB | **显存**,不是计算 |
| 词表并行时的通信 | — | all-gather 完整 logits 要搬 4.3 GB(BF16);融合后只搬 3 次 $16384$ 个标量 ≈ 200 KB | 取决于实现,差四个数量级 |

第一行和第二行相差的正是 $d$ 这个因子:投影是「每个词都要和 $h$ 做一次 $d$ 维点积」,softmax 是「每个词做常数次运算」,所以 **$d$ 越大,exp 在总账里越不值一提**。真正会被词表大小放大的是三样东西:GEMM 的 FLOPs、$[B\times S, V]$ 这块中间张量、以及多卡时怎么把它拼起来。

输出层在整模型里占多大比重,决定了值不值得动它。一个 $N$ 参数的模型前向每 token 约 $2N$ FLOPs,输出层占 $2dV$:Llama 3 8B($d=4096$、$V=128256$)约 **6.5%**;Gemma 3 4B($d=2560$、$V=262208$)约 **16%**;若把词表推到 100 万、仍用 $d=4096$,每 token 8.2 GFLOP,已经接近一个 7B 模型本体的六成。反过来,70B 级模型上输出层只占 1–2%。**这就是为什么大模型时代很少见到分层或采样 Softmax:模型一大,输出层不再是瓶颈,剩下的显存问题用精确方案(第七节)就能解决;真正疼的是小模型配大词表,以及百万级词表。**

## 二、分层 Softmax:把「V 选 1」拆成 log V 次「二选一」

### 机制

把词表挂到一棵二叉树的叶子上,每个内部节点配一个向量 $v_n\in\mathbb{R}^d$,充当一个「往左还是往右」的二分类器。一个词的概率就是从根走到它那片叶子、沿途每一步「走对方向」的概率连乘:

$$
P(w\mid h)=\prod_{j=1}^{L(w)}\sigma\big(\pm\, v_{n_j}^{\top}h\big)
$$

意思是:$L(w)$ 是这个词的路径长度,每个节点算一次 $d$ 维点积再过 sigmoid,正负号表示在该节点该往哪边走。因为每个节点上「左 + 右」的概率天然等于 1,所有叶子的概率加起来也自动等于 1——**它不需要在 $V$ 个词上求和就是一个归一化的分布**。训练时只需要算真值那条路径上的 $L(w)$ 个节点,梯度也只更新这些节点,树平衡时 $L(w)\approx\log_2 V$,128k 词表约 17 次点积,而不是 131072 次。这就是「期望复杂度 $O(\log V)$」的来源。

### 树怎么建,为什么树的质量决定效果

- **Huffman 树**(word2vec、fastText 的做法):按词频建树,高频词路径短。它最小化的是**期望路径长度**——Morin 与 Bengio 在 1 万词表上算过,平衡树的路径是 $\log_2 10000\approx 13.3$,而按词频建树可以压到接近 unigram 熵 9.16,多省约 31%。fastText 做文本分类时也用它:类别数 $k$ 很大时把 $O(kh)$ 压到 $O(h\log_2 k)$。
- **语义 / 聚类树**:Morin 与 Bengio 用 WordNet 的 IS-A 层级、再用 K-means 把多叉节点二分;Chen 等人比过四种建两层树的方式,在 8 亿 token 的 One Billion Word 上困惑度是随机分簇 98.5、按频率分簇 92.0、按词向量 k-means 分簇 85.7。
- **两层树**(Goodman 的 class-based 变体,Mikolov 的 RNNLM 也用):先选 $\sqrt V$ 个类、再在类内选词,复杂度 $O(2\sqrt V)$。它比深二叉树 GPU 友好得多,因为每一层还是一个像样的矩阵乘。

**树质量为什么决定效果**:一个词的概率永远不会超过它所在分支的概率(连乘只会变小)。分簇要是没有语义(随机分),模型根本学不出「给定上下文该走哪支」,分支概率趋于均匀,等于给每个词的概率封了顶。Morin 与 Bengio 的原始结果就是训练快 258 倍、但测试困惑度从 195 涨到 221;Chen 等人在小词表 PTB 上分层版比全 softmax 差(138 对 124),在大词表 billionW 上反而好(85 对 108)——不是树变好了,而是分层版每秒能吃 12650 个 token、全 softmax 只有 510,同样一周它多训了 25 倍数据。

### 它改变了输出结构,推理时反而不省

分层 Softmax 不是「同一个模型换个算法」,它**换了模型**:参数是 $V-1$ 个节点向量而不是 $V$ 行输出矩阵,没有一个 $V$ 维 logits 向量存在,词与词之间也不再有一次统一的打分。这带来三笔代价:

- **GPU 不友好**。同一个 batch 里每个 token 的路径不同,算的是一堆零散的 $d$ 维点积,拼不成一个大 GEMM。Grave 等人在 GPU 上实测,矩阵乘的耗时在词数低于约 50 时几乎是常数——也就是说 GPU 算 1 个词和算 50 个词一样贵,二叉树把工作切到「一次一个节点」正好踩中这个最亏的区间。
- **和现有基础设施打架**。融合交叉熵、词表并行、绑定输入输出 embedding,都建立在「有一个 $[V,d]$ 的输出矩阵」上,分层结构一个都接不上。
- **推理要 argmax、top-k 或完整分布时省不下来**。要算某一个已知词的概率,走一条路径就行,是 $O(\log V)$;但生成时要在全部候选里挑最大的。沿树贪心往下走并不精确——分支概率高不代表叶子的连乘高。精确的做法有两条:要么把所有叶子都算一遍,回到 $O(V)$ 次节点运算,Morin 与 Bengio 自己就写明「要全部词的概率时回到 $O(|V|)$」,Chen 等人实测分层版训练 12650 token/s、测试打分全词表时和 softmax 一样是 510;要么做带剪枝的深度优先搜索——节点概率是它所有后代的上界,低于当前最好叶子的分支整枝砍掉,fastText 在 CPU 上就是这么取 top-T 的,实测接近 $O(h\log_2 k)$。但剪枝是逐样本、数据相关的分支搜索,GPU 上批量做不了,而且分布越平坦砍得越少;要的是完整归一化分布(top-p 采样、蒸馏、RL 里的 logprob)时,无论如何都得回到全部叶子。

所以它的真实定位是:**训练时、以及推理只需要给已知序列打分时(如重打分、语音识别里对少数候选比较)很省;GPU 上做开放生成时不省。**

## 三、自适应 Softmax:按词频分簇,头部全维、尾部降维

### 机制

自然语言的词频是 Zipf 分布——Grave 等人给的数是 PTB 上 20% 的词覆盖了 87% 的出现。自适应 Softmax(Grave 等人,2017)把这一点用到底:按频率把词表切成一个**头部簇**($k_h$ 个最高频词)和 $J$ 个**尾部簇**。头部的 softmax 里除了 $k_h$ 个词,还多放 $J$ 个「簇入口」单元,每个代表一个尾部簇;尾部簇里的词先把 $h$ 用一个投影矩阵从 $d$ 降到 $d/4^i$(PyTorch 的默认 `div_value=4.0`,第 1 簇 $d/4$、第 2 簇 $d/16$),再在簇内做 softmax。头部词的概率直接是头部 softmax 的输出;尾部词的概率是「进入该簇的概率 × 簇内概率」,和两层分层 Softmax 同构,只是头部词**直接挂在根上**而不是也当一个簇——Grave 等人引前人的结果指出,把头部也做成一个簇会掉 5–10% 的效果,因为高频词的概率被多乘了一层簇概率,分布变钝。

### 省了什么

- **头部矩阵乘变小**:$[B,d]\times[d,\,k_h+J]$,$k_h$ 通常几千到几万,而不是 $V$。
- **尾部簇只处理真值落在该簇的 token**:覆盖概率 $p_i$ 的簇平均只看 $p_i B$ 行,维度还降了 4 倍、16 倍。
- **簇的划分按 GPU 的真实耗时来定**。Grave 在 GPU 上实测,矩阵乘的耗时在词数低于约 50 时几乎是常数、之后才线性增长,所以他们用这个耗时模型做动态规划来定每个簇的大小;结论是**簇数 2–5 个就够**,再多不省时间反而伤困惑度,而且最大的簇要装最罕见的词。

实测:训练比全 softmax 快 **2–10 倍**,困惑度接近。Text8(44k 词表)全 softmax 困惑度 144、83 分钟,自适应 147、30 分钟;Europarl 保加利亚语(50k)两者困惑度都是 37,时间 58 分钟对 18 分钟;One Billion Word(约 80 万词表)单卡不到三天训到困惑度 43.9。

### 它给出的是精确归一化分布——与采样类方法的本质区别

自适应 Softmax **没有采样、没有估计**:它算的每个概率都是这个模型精确的归一化概率,损失是这个模型精确的交叉熵,梯度无偏。它「近似」的是**模型本身**——用更少的参数、更低的维度去表示尾部词——而不是梯度。因此训练和推理是同一个模型,推理也能省:

- **argmax**:先算头部;若头部最大项是一个词,它就是全局 argmax,因为任何尾部词的概率 ≤ 它所在簇入口的概率 ≤ 头部最大项;只有最大项是簇入口时才需要进那个簇。PyTorch `AdaptiveLogSoftmaxWithLoss` 的 `predict()` 就是这么做的,官方文档说明它比 `log_prob()` 再取 argmax 更省。
- **完整分布**(top-p 采样、蒸馏、RL 的 logprob):要走所有簇,但尾部簇维度低,总量仍比全 softmax 少。

### 代价

- **结构复杂、部署麻烦**:标签必须按词频排序(id 0 最高频),tokenizer 的 id 要重排或建映射;不能和输入 embedding 绑权重(维度都不一样);词表并行、融合交叉熵、推理引擎的采样 kernel 全部默认「一个 $[V,d]$ 矩阵」,都接不上。`cutoffs`、`div_value` 是超参,换语料要重调。
- **长尾词的能力被压**:尾部词只有 $d/4$、$d/16$ 维的表示。Chen 等人按词频分五段测 entropy,同思路的 D-softmax 在最高频段最好、在 70k–100k 的最低频段最差;对 LLM 来说罕见 token 常是专名、代码符号、多语言字符,恰恰不能牺牲。
- **依赖长尾分布**:分布越平,头部就得开越大,收益越小;Grave 自己也说 44k 这种小词表上「近似方法相对不那么有意思」。

## 四、采样 Softmax:只算真值和少量负样本

### 为什么可以只算一部分

交叉熵对 logits 的梯度是 $p-\text{onehot}$(推导见 分类损失 篇)。换到参数视角:

$$
\nabla_\theta L = -\nabla_\theta z_y + \sum_{w=1}^{V} p(w\mid h)\,\nabla_\theta z_w
$$

意思是:梯度 = 「把真值分数往上推」+「按模型当前概率把所有词的分数往下压」。贵的是第二项——它是对全部 $V$ 个词、按模型分布求的期望。期望都能用采样估:抽一小撮词代替整张词表。这就是 Bengio 与 Senécal 的重要性采样,以及 Jean 等人在 NMT 上用的采样 softmax。

### 做法与 logQ 校正

每个样本(或每个 batch)按**提议分布** $Q$(常用 unigram 或按频率排序后的 log-uniform)抽一个候选集 $S$,只在 $\{y\}\cup S$ 上做 softmax 交叉熵,但每个候选的 logit 要减去 $\log Q(w)$:

$$
L = -z'_y + \log\sum_{w\in\{y\}\cup S} e^{z'_w},\qquad z'_w = z_w - \log Q(w\mid h)
$$

意思是:抽中概率高的词在候选集里「出现得太多了」,减 $\log Q$ 就是按重要性采样把它的份量除回去,这样候选集上的求和才是整张词表求和的无偏估计。不减会怎样:高频词被反复当负样本压制,模型学到的是 $\log\frac{P(w\mid h)}{Q(w)}$ 而不是 $\log P(w\mid h)$——做词向量无所谓,做语言模型就是错的概率。TensorFlow 的候选采样文档把这条校正写得最清楚,`tf.nn.sampled_softmax_loss` 就是这一套。Jean 等人的工程细节值得记:他们把训练语料切成若干区,每区在训练前顺序扫句子、把出现过的目标词累积到一个阈值 $\tau$ 就定成该区的候选子集,让整区样本共享同一组候选,这样输出层仍是一个规整的矩阵乘而不是每个 token 各抽各的;他们的提议分布在子集内均匀,$\log Q$ 项正好抵消成常数,论文自己写明这个选择让估计有偏。

### 为什么梯度仍然有偏

$\sum e^{z'}$ 是配分函数的无偏估计,但损失里取的是它的 $\log$,「先取对数再求期望」不等于「先求期望再取对数」,所以梯度有偏;样本越多偏差越小,$|S|\to V$ 时回到精确。第二个问题是方差:$Q$ 离模型分布越远,重要性权重越不稳——Mnih 与 Teh 复现重要性采样时「几乎所有实验都发散」,Bengio 与 Senécal 要靠自适应提议分布加上百个样本才压住。第三个问题是省得没想象中多:Chen 等人在 gigaword(100k 词表)上要把候选集开到词表的 30% 以上才最准,billionW 上甚至要 50%;最后 billionW 上困惑度 101 好于全 softmax 的 108,只是因为它一周内多训了一倍数据,gigaword 上反而略差(57.6 对 56.5)。

### 和 NCE 的关系

两者原料一样:一个真值加一撮按 $Q$ 抽出的负样本。差别在损失的形状——采样 softmax 是在候选集上做**一次多选一的 softmax**(仍然是「这几个里哪个是真的」),NCE 是对每个候选**各做一次「数据还是噪声」的二分类**。TensorFlow 那份文档把两者列在同一张表里,只差一列。

## 五、NCE:把「V 选 1」改成「数据还是噪声」

### 机制

噪声对比估计(Gutmann 与 Hyvärinen,2010)本来是为**未归一化模型**发明的:不算配分函数,改让模型学会区分「真实数据」和「人造噪声」。Mnih 与 Teh 把它搬到语言模型:给定上下文 $h$,把 1 个真实词和 $k$ 个从噪声分布 $q$(取 unigram)抽出的词混在一起,训练一个二分类器判断每个词来自数据还是噪声:

$$
P(D=1\mid w,h)=\frac{p_\theta(w\mid h)}{p_\theta(w\mid h)+k\,q(w)}
$$

意思是:这个词是数据的后验概率,取决于模型给它的分数和噪声给它的分数之比。关键在于这里的 $p_\theta$ 可以直接用**未归一化**的 $e^{s_\theta(w,h)}$:NCE 把归一化常数当成一个可学的参数,Mnih 与 Teh 发现干脆固定为 1 也不影响效果——模型参数多到自己就能学会近似自归一化。于是每个 token 只算 $k+1$ 个点积,词表多大都无所谓。他们的结果:10k 词表上 25 个 unigram 噪声样本的困惑度 163.1,和最大似然的 163.5 持平,训练 1.5 小时对 21 小时;只抽 1 个样本是 192.5;换成均匀噪声、25 个样本是 195.1。

### 目标为何与最大似然不同,依赖什么

NCE 最大化的是「分辨数据与噪声」这个二分类任务的似然,不是语言模型的似然。Gutmann 与 Hyvärinen 证明它是一致估计,Mnih 与 Teh 写出了 $k\to\infty$ 时 NCE 梯度趋于最大似然梯度;但 $k$ 有限时它就是另一个目标、另一个最优点。两个后果:一是**效果取决于噪声分布和样本数**——噪声越接近数据分布,分辨越难、信号越强(unigram 远好于均匀),样本越多越接近最大似然;Chen 等人在大模型上发现 NCE「实践中很难用」,要把数据噪声比调到 1/50、而且 NCE 损失相近的模型验证熵能差很远,损失值根本不能当训练指标看。二是**推理时分数没归一化**:要概率就得把全 softmax 加回去(Mnih 与 Teh 报困惑度时就是显式归一化的),要 argmax 可以直接比原始分数,但 $W_{\text{out}}h$ 那 $O(dV)$ 一分没省。NCE 的权重 $\frac{p_\theta}{p_\theta+kq}$ 永远在 0 到 1 之间,这是它比重要性采样稳得多的原因。

### word2vec 的负采样(NEG)

Mikolov 等人把 NCE 再简化:去掉 $k\,q(w)$ 那一项,直接对真值取 $\sigma(z_w)$、对 $k$ 个负样本取 $\sigma(-z_w)$;$k$ 小数据取 5–20、大数据取 2–5,噪声分布用 unigram 的 $3/4$ 次方。它连噪声概率的数值都不要、只要能抽样,但代价是学到的是 $\log\frac{P(w\mid h)}{Q(w)}$ 而不是 $\log P$——word2vec 要的只是词向量,从不需要概率,所以无所谓;拿它训语言模型概率就是错的。**与 InfoNCE 的关系一句话**:InfoNCE 是「1 个正例 + $N-1$ 个负例做一次 $N$ 选 1 的 softmax」,形状上是采样 softmax 而不是二分类的 NCE;温度、互信息下界与难负例工程见 对比学习 篇。

## 六、「训练能用、推理不能用」这条线在哪

| 方法 | 训练时省 | 推理:给已知词打分 | 推理:argmax / top-k | 推理:完整分布 | 原因 |
|---|---|---|---|---|---|
| 采样 Softmax | ✅ | ❌ | ❌ | ❌ | 只改了损失,训出来的仍是全 softmax 模型,负样本集是训练期的临时物 |
| NCE / NEG | ✅ | 自归一化可免求和,但 $W_{\text{out}}h$ 仍是 $O(dV)$ | ❌ 仍要 $V$ 个分数 | ❌ 要重新归一化 | 同上 |
| 分层 Softmax | ✅ | ✅ $O(\log V)$ | 剪枝搜索,CPU 可、GPU 批量不可 | ❌ | 换了模型,概率是路径连乘 |
| 自适应 Softmax | ✅ | ✅ | ✅ 头部先判 | 部分 ✅,尾部低维 | 换了模型,但仍是精确归一化分布 |

**为什么近似方法在生成阶段不能无条件沿用**:训练要的只是一个梯度方向的低偏估计,漏掉一部分负样本只是加了噪声,几千步平均下来无所谓;生成时每一步都要从**全部** $V$ 个候选里挑出对的那个,任何「只看一部分词」的做法都可能把正确 token 漏在集合外,而且自回归会把这个错误一路带下去。采样 Softmax 和 NCE 更根本的一点是:它们根本没有给推理留下可复用的东西——模型结构就是全 softmax,推理时不算全部 $V$ 个分数,连 argmax 都无从谈起。

**推理侧真正能做的**,全是不改模型的手段:

- **decode 阶段 $s=1$,输出投影退化成 $[b,d]\times[d,V]$**,小 batch 下它是「把 $W_{\text{out}}$ 读一遍」的访存问题(128k × 4096 的 BF16 矩阵是 1 GB,每步都要读),所以有效的是给 $W_{\text{out}}$ 量化、用张量并行把它切到 $t$ 张卡各读 $1/t$(推理侧 TP 的收益与限制见 并行策略 篇),而不是近似。
- **采样链融合成一个 kernel**:温度、top-k/top-p、抽样在一趟里做完,免全排序(见 解码策略 篇)。第八节讲它和加速的边界。
- **投机解码不省输出层**:验证时草稿的每个 token 都要算完整 logits,输出层的工作只多不少。
- **真要少算词表**只剩近似:把 $W_{\text{out}}$ 的行当向量库做近似最大内积检索取候选,或像早期 NMT 那样按上下文预选一个词表子集。它们会漏词,必须按第九节验收。

## 七、精确方案:保精确,省的是显存与单卡,不省 O(dV)

### 词表并行(Megatron-LM)

把 $W_{\text{out}}$ 按**词表维**切成 $t$ 份放到张量并行组的 $t$ 张卡上,每卡对所有 token 算自己那 $V/t$ 段 logits。朴素做法是 all-gather 拼出完整 logits 再算交叉熵,要搬 $b\times s\times V$ 个数;Megatron 的做法是**把 GEMM 的输出直接和交叉熵融合**,只通信 $b\times s$ 个标量:先各卡求本地最大值、all-reduce 取 MAX 后减掉(数值稳定,同 分类损失 篇第八节);再各卡取真值 token 的 logit——只有持有那个 id 的卡有值、其余卡填 0——all-reduce 求 SUM;然后各卡算本地 $\sum e^{z}$、再 all-reduce 求 SUM;损失 $=\log\sum e^{z}-z_y$。反向也不用通信:每卡拿本地 softmax 减去「真值在我这一片时的 1」就是梯度。三次 all-reduce 搬的都是 $[s,b]$ 大小的张量,和 logits 本身差 $V$ 倍(第一节的算例:4.3 GB 对 200 KB)。每卡的 GEMM 与 logits 显存都变成 $1/t$,**总 FLOPs 一分不少**。工程细节:词表要补齐到 $128\times t$ 的倍数以保证每卡分片对齐,GPT-2 的 50257 在 $t=8$ 时补到 51200。TP 的通信总账与「不出机」规则归 并行策略 篇。

### 融合 / 分块交叉熵

「融合」有两层。浅的一层是把 logits 之后的逐元素步骤(减 max、exp、求和、取 log)合成一个 kernel,少读写几遍 logits——这是 分类损失 篇第八节讲的 log_softmax 与 NLL 融合,Megatron 的 fused cross entropy 选项也是这一层,**logits 本身还是完整落盘**。深的一层是**把输出投影 GEMM 和交叉熵融在一起,让完整 logits 从头到尾不存在**。Liger Kernel 的 fused linear cross entropy 做法:把 $B\times S$ 个 token 切成若干块,块大小取到「一块的 logits 约等于一块输入」($\text{chunk}\times V\approx B S\times d$,128k 词表、$d=4096$ 就是切 32 块);每块算出 logits 后立刻在原地算损失和 $\partial L/\partial \text{logits}$(在线 max 与 logsumexp),马上把 $\partial X$ 和 $\partial W_{\text{out}}$ 累加出来,然后这块 logits 就丢掉。峰值显存从 $BS\times V$ 降到 $BS\times d$ 量级,**FLOPs 不变**——前向、$\partial X$、$\partial W$ 三个 GEMM 一个不少,只是反向的两个提前到前向里做。Liger 自报整套 kernel 在 HF 模型上平均省 60% 显存、提 20% 吞吐。Apple 的 Cut Cross-Entropy 更进一步:只算真值 token 那一个 logit,配分函数用 flash-attention 式的分块在片上算、从不写回显存,反向重算并跳过 softmax 概率低于 BF16 精度的块;论文自报 Gemma 2 2B 的损失显存从 24 GB 降到 1 MB、整个输出头从 28 GB 降到 1 GB,速度与收敛不变。

### tie embedding

输入 embedding 与 $W_{\text{out}}$ 共享一张表,省的是 $V\times d$ 个参数和它们的优化器状态(Llama 3 8B 的量级是 5 亿多参数),但 $z=Eh$ 这个 GEMM 一次都不少——**只省参数,不省计算**;什么时候该绑、什么时候不绑,见 Tokenizer 篇。

三者收成一句:**精确方案解决的是「存不下」和「一张卡算不动」,不是「算得多」;想少算 FLOPs,只能改模型(二、三节)或改目标(四、五节)。**

## 八、Top-k / Top-p 和加速的关系

top-k、top-p 裁的是**采样候选集**,发生在全词表 logits 算完之后——要知道谁是前 $k$ 名,得先给 $V$ 个词全部打分,所以它们**不能单独省掉输出投影**,这是最常被答错的一点。它们能省的,是 logits 之后那一截:全词表排序换成选择(top-k 用 radix select 一类的部分选择;top-p 甚至可以用免排序的拒绝采样直接抽),exp 与归一化只在保留集合上做(softmax 对平移不变,截断后在保留集合内重新归一化本来就是定义),再和温度、抽样合成一个 kernel,一趟扫过 $V$ 个数就结束。三种截断各自的行为、顺序问题与引擎里的实现见 解码策略 篇。

还有一条边界要分清:**截断不是近似**。top-k 作用在精确 logits 上,只是决定「从哪些词里抽」,不引入模型误差;第六节末尾的候选检索是「用近似 logits 决定算哪些词」,可能把正确 token 排除在外。两者一个改抽样、一个改模型输出,面试里被放在一起问时要先把这条线画出来。

## 九、百万级词表怎么选、怎么验收

先把账算出来:$V=10^6$、$d=4096$ 时 $W_{\text{out}}$ 有 41 亿参数,BF16 就是 8.2 GB;16384 个 token 的 FP32 logits 是 **65.5 GB**,任何一张卡都放不下;每 token 输出层 8.2 GFLOP,已经和一个 7B 模型本体同量级。**三样东西同时爆:显存必爆,计算大概率成主项,单卡放不下就还有通信。** 选型顺序:

1. **profiling 定病灶**:时间在 GEMM、在逐元素 pass、在通信,还是直接 OOM(工具与读法见 性能分析与Profiling 篇)。
2. **精确方案先上满**:分块 / 融合交叉熵让 logits 不落盘,词表并行把 $W_{\text{out}}$ 和 GEMM 摊到 $t$ 张卡,tie embedding 省参数。它们不动质量,是免费的。
3. **GEMM 仍是主项时才考虑近似**——先看 token 频率分布:头部 token 覆盖率高、长尾明显,自适应 Softmax(两端可用、精确归一化)优于采样 Softmax(只训练可用、常要 30–50% 的候选才准)。同时回头问一句词表是不是真该这么大(词表缩放律见 Tokenizer 篇)。
4. **推理侧只做精确的事**:融合采样 kernel、$W_{\text{out}}$ 量化、TP 切分;训练时的采样集不带到推理。
5. **最后才是候选检索一类的近似**,而且要带着召回评测上线。

**组合原则**:「训练近似 + 推理精确」是正常搭配(采样 Softmax 训、全 softmax 推);改了模型的方法(分层、自适应)训练推理必须一起验;两个近似不叠加,除非各自单独测过。**验收看五样**:困惑度只在同一 tokenizer 下可比,跨词表要换成每字节或每字符的 bits;按 token 频段分桶看 entropy 或准确率,自适应类方法的损失全藏在最低频段;下游 benchmark;生成质量(人评或 LLM judge,专门看专名、代码、多语言);概率校准——采样 Softmax 和 NCE 改了训练目标,概率可能系统性偏移,要按 分类损失 篇第七节重新校准。速度指标必须是端到端的每秒 token 数,不是单看输出层。

## 十、两张对照表

### Hierarchical vs Adaptive

| | 分层 Softmax | 自适应 Softmax |
|---|---|---|
| 结构 | 深二叉树(或两层类),$V-1$ 个节点向量 | 头部短名单 + 2–5 个按频率切的尾簇 |
| 每 token 训练开销 | $O(d\log V)$ | 头部 $O(d\,k_h)$ + 尾部按命中比例、且降维 |
| GPU 友好 | 差:零散点积,拼不成 GEMM | 好:每簇仍是 GEMM,簇大小按 GPU 耗时模型定 |
| 精确性 | 换了模型,但精确归一化 | 换了模型,但精确归一化 |
| 推理 argmax | 枚举叶子或剪枝搜索 | 头部先判,多数情况一步 |
| 效果依赖 | 树的质量(随机分簇 98.5 对 k-means 85.7) | 词频长尾是否明显 |
| 长尾词 | 罕见词有完整路径参数,Chen 实测罕见词段最好 | 罕见词维度被压,罕见词段最差 |
| 优点 | 理论省得最多;给已知词打分最快 | 训练 2–10 倍、困惑度几乎不掉;训练推理两端可用 |
| 缺点 | 效果看树;开放生成不省;最难部署 | 结构复杂、超参多;绑权重 / 并行 / 融合 kernel 都不兼容 |

### 总表

| 方法 | 训练可用 | 推理可用 | 保精确 | 省什么 | 代价 |
|---|---|---|---|---|---|
| 分层 Softmax | ✅ | 打分 ✅ / 生成 ❌ | 换模型后精确 | 训练 $O(dV)\to O(d\log V)$ | 树质量、GPU 差、结构难接 |
| 自适应 Softmax | ✅ | ✅ | 换模型后精确 | 计算集中在头部,尾部降维 | 长尾能力、结构与超参 |
| 采样 Softmax | ✅ | ❌ | ❌ 梯度有偏 | 训练只算 $\lvert S\rvert+1$ 个词 | 依赖 $Q$ 与样本数,常需 30% 以上 |
| NCE / NEG | ✅ | ❌ | ❌ 目标不同 | 训练只算 $k+1$ 个词、免归一化 | 依赖噪声分布与 $k$,分数未归一化 |
| 词表并行 | ✅ | ✅ | ✅ | 单卡计算与显存 ÷ $t$ | 3 次小 all-reduce、词表补齐 |
| 分块 / 融合交叉熵 | ✅ | 推理无损失项 | ✅ | logits 显存 ÷ $(V/d)$ | 反向 GEMM 提前、kernel 复杂 |
| tie embedding | ✅ | ✅ | ✅ | $V\times d$ 参数 | 不省计算 |
| Top-k / Top-p | 不用 | ✅ | ✅ 裁候选不改 logits | 排序、exp、归一化 | 不省投影 |
| 候选检索 | — | ⚠️ 近似 | ❌ 可能漏词 | 投影只算候选 | 召回损失,需评测 |

## 十一、面试考点串联

| 高频问法 | 本文哪一节 |
|---|---|
| 当语言模型的词汇表非常大(例如超过10万个token)时,传统的Softmax计算会成为瓶颈,请列举并解释几种常见的加速Softmax计算的技术方案。 | 一(先定位瓶颈)· 二至五(四种近似各省了什么)· 七(精确方案)· 十(总表) |
| 当语言模型的词汇表非常大(例如超过10万个token)时,有哪些常用的技术手段可以加速Softmax计算?请说明其原理和适用场景。 | 同上;适用场景看六、九 |
| 训练阶段和推理阶段分别适合哪些Softmax加速方法? | 六(那张表与推理侧能做的事) |
| Top-K/Top-p采样如何与Softmax加速结合? | 八 |
| 如果词表扩大到百万级,还有哪些进一步优化思路? | 九(账 + 五步顺序) |
| 如何在保证生成质量的前提下组合使用这些技术? | 九(组合原则与五样验收) |
| Hierarchical Softmax和Adaptive Softmax的优缺点是什么? | 十(对照表)· 二 · 三 |
| 大词表下 Softmax 到底慢在哪?是 exp 慢吗?(补充题) | 一 |
| 采样 Softmax 为什么要减 $\log Q$?不减会学到什么?(补充题) | 四 |
| NCE 和采样 Softmax 都抽负样本,差在哪?NCE 训完的分数能直接当概率用吗?(补充题) | 四 · 五 |
| 词表并行怎么做到不把 logits 拼起来就算出交叉熵?通信量差多少?(补充题) | 七 · 一 |
| 分块交叉熵省的是什么、不省的是什么?(补充题) | 七 |
| 分层 Softmax 训练快那么多,为什么 GPU 上做开放生成时没人用?(补充题) | 二 |

## 相关文献

- Efficient softmax approximation for GPUs(自适应 Softmax)— [arXiv:1609.04309](https://arxiv.org/abs/1609.04309)
- Hierarchical Probabilistic Neural Network Language Model(Morin 与 Bengio,AISTATS 2005)— https://proceedings.mlr.press/r5/morin05a.html
- Distributed Representations of Words and Phrases and their Compositionality(word2vec 的 Huffman 分层 softmax 与负采样)— [arXiv:1310.4546](https://arxiv.org/abs/1310.4546)
- On Using Very Large Target Vocabulary for Neural Machine Translation(采样 softmax)— [arXiv:1412.2007](https://arxiv.org/abs/1412.2007)
- Noise-contrastive estimation: A new estimation principle for unnormalized statistical models(Gutmann 与 Hyvärinen,AISTATS 2010)— https://proceedings.mlr.press/v9/gutmann10a.html
- A fast and simple algorithm for training neural probabilistic language models(NCE 训语言模型,归一化常数固定为 1)— [arXiv:1206.6426](https://arxiv.org/abs/1206.6426)
- Strategies for Training Large Vocabulary Neural Language Models(全 softmax / 分层 / 采样 / NCE / D-softmax 横评)— [arXiv:1512.04906](https://arxiv.org/abs/1512.04906)
- Bag of Tricks for Efficient Text Classification(fastText 的 Huffman 分层 softmax 与剪枝搜索)— [arXiv:1607.01759](https://arxiv.org/abs/1607.01759)
- Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism(词表并行与交叉熵融合)— [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- Liger Kernel: Efficient Triton Kernels for LLM Training(分块的 fused linear cross entropy)— [arXiv:2410.10989](https://arxiv.org/abs/2410.10989)
- Cut Your Losses in Large-Vocabulary Language Models(CCE,logits 不落盘)— [arXiv:2411.09009](https://arxiv.org/abs/2411.09009)
- TensorFlow Candidate Sampling Algorithms Reference(采样 softmax 的 logQ 校正与 NCE / NEG 对照表)— https://www.tensorflow.org/extras/candidate_sampling.pdf
- PyTorch AdaptiveLogSoftmaxWithLoss — https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html

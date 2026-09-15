# RoFormer：用旋转矩阵把相对位置写进注意力

<!-- release-date: 2021-04-20 -->

**本文依据**：`RoFormer: Enhanced Transformer with Rotary Position Embedding`，arXiv 2104.09864v5（[cs.CL] 8 Nov 2023），14 页。作者 Jianlin Su、Yu Lu、Shengfeng Pan、Ahmed Murtadha、Bo Wen、Yunfeng Liu，封面第一单位 **Zhuiyi Technology Co., Ltd.**（深圳）。封面日期印 November 9, 2023。文中数字标 PDF 页码；标「外部补充」的段落不来自本文。Hugging Face 文档链在摘要末尾：https://huggingface.co/docs/transformers/model_doc/roformer。代码仓 `ZhuiyiTechnology/roformer`（PDF p. 2）。

## 一句话

自注意力本身不认识顺序。旧办法多半是把位置向量**加**到词向量上，再硬拆内积；RoPE 改成把 query / key **旋转**，绝对位置进旋转角，相对距离自然出现在内积里。装上它的 Transformer 叫 **RoFormer**。

## 一、矛盾：自注意力不认识顺序，加进去又拆不干净

词的先后对理解句子有用。RNN 靠时间步递推带上顺序；CNN 一度被认为对位置不敏感，但 padding 也能隐式学到位置（PDF p. 1）。预训练语言模型走 Transformer，自注意力并行、也更能抓较长的 token 关系，但 **self-attention 对位置无感**（Yun et al.，PDF p. 2）。

于是位置必须另外灌进去。论文把当时的做法收成两条线（PDF p. 2）：

- **绝对位置**：正弦函数生成，或一组可学向量 $p_i$，加到词向量上再投影成 Q/K/V。
- **相对位置**：把 $m-n$ 写进注意力权重，典型是把 $q^\top k$ 按加法展开后再改项。

两条线有一个共同动作：**位置信息加到上下文表示上**。论文明确说，这样加出来的形式 **不适合线性自注意力**（PDF p. 2）。相对位置那一支还多半依赖对加法展开式的修补，解释性弱，也难接到线性注意力上。

RoPE 要做的不是再改一项展开，而是从「内积只依赖相对距离」这条约束把编码推出来。

## 二、旧编码怎么写进注意力

记序列 $S_N=\{w_i\}_{i=1}^N$，词向量 $x_i\in\mathbb{R}^d$ 还不带位置。位置通过三个函数灌进 query、key、value（PDF p. 2 式 1）：

$$
q_m=f_q(x_m,m),\quad k_n=f_k(x_n,n),\quad v_n=f_v(x_n,n)
$$

注意力权重是 $q_m^\top k_n/\sqrt{d}$ 过 softmax，输出是对 value 的加权和（PDF p. 3 式 2）。已有工作几乎都在选 $f_q,f_k,f_v$。

**绝对位置的典型写法**（PDF p. 3 式 3）：

$$
f_{t\in\{q,k,v\}}(x_i,i):=W_t(x_i+p_i)
$$

$p_i$ 可以是最长 $L$ 的可学表，也可以是 Transformer 原文的正弦（PDF p. 3 式 4）。RoPE 后面会用到同样的正弦频率，但 **不把正弦加到词向量上**，而是拿它去转。

**相对位置**从 Shaw et al. 开始，给 key/value 加可学的相对向量，距离 clip 到 $[r_{\min},r_{\max}]$（PDF p. 3 式 5）。Transformer-XL 把 $q_m^\top k_n$ 拆成四项（PDF p. 3 式 6），再把绝对 $p_n$ 换成正弦相对 $\tilde{p}_{m-n}$，查询侧绝对位置换成与位置无关的 $u,v$（PDF p. 3 式 7）。T5 进一步收成内容内积加可学偏置 $b_{i,j}$（PDF p. 3 式 8）。DeBERTa 认为相对位置主要靠中间两项，于是用 $\tilde{p}_{m-n}$ 替换 $p_m,p_n$（PDF p. 4 式 10）。

论文的判断：这些变体都在 **Vaswani 加法编码的展开式** 上改（PDF p. 4）。RoPE 要从式 1 加约束直接推相对编码，用旋转而不是再加一项。

## 三、约束：内积只看见相对距离

注意力里真正传位置的是 $q_m^\top k_n$。若要相对位置，这个内积应能写成只吃词向量和相对距离的函数 $g$（PDF p. 4 式 11）：

$$
\langle f_q(x_m,m),f_k(x_n,n)\rangle = g(x_m,x_n,m-n)
$$

目标就是找出满足这条的 $f_q,f_k$。

## 四、二维：旋转一次，相对角就出来

先看 $d=2$。把向量当成平面上的复数，论文在 3.4.1 节推：半径与位置无关（旋转不改模长），转角对位置是等差数列。取初始条件 $f_q(x_m,0)=W_q x_m$、$f_k(x_n,0)=W_k x_n$，并令常数相位 $\gamma=0$，得到（PDF p. 4 式 12；推导在 PDF p. 6–7 式 20–33）：

$$
f_q(x_m,m)=(W_q x_m)e^{im\theta},\quad f_k(x_n,n)=(W_k x_n)e^{in\theta}
$$

$$
g(x_m,x_n,m-n)=\mathrm{Re}\bigl[(W_q x_m)(W_k x_n)^* e^{i(m-n)\theta}\bigr]
$$

$\theta$ 是预先给定的非零常数。矩阵写法就是平面旋转（PDF p. 4 式 13）：

$$
f_{\{q,k\}}(x_m,m)=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}W_{\{q,k\}}x_m
$$

人话：**先仿射，再按位置下标转一个角。** 两个位置相减，相对角自动出现在内积里。这就是 Rotary Position Embedding 这个名字的来历。

## 五、任意偶数维：切成 $d/2$ 个二维子空间

$d$ 为偶数时，把空间切成 $d/2$ 个二维子空间，利用内积线性性拼回去（PDF p. 5 式 14）：

$$
f_{\{q,k\}}(x_m,m)=R_{\Theta,m}^d W_{\{q,k\}} x_m
$$

$R_{\Theta,m}^d$ 是分块对角旋转矩阵：第 $i$ 块是转角 $m\theta_i$ 的 $2\times 2$ 旋转，其余为 0（PDF p. 5 式 15）。频率表论文写成

$$
\Theta=\{\theta_i=10000^{-2(i-1)/d},\ i\in[1,2,\ldots,d/2]\}
$$

（PDF p. 5）。3.3 节讨论长期衰减时又写成 $\theta_i=10000^{-2i/d}$，并说这是跟随 Vaswani et al.（PDF p. 5）。两处下标写法不完全相同，正文不替它们对齐。

接到自注意力上（PDF p. 5 式 16）：

$$
q_m^\top k_n=(R_{\Theta,m}^d W_q x_m)^\top(R_{\Theta,n}^d W_k x_n)=x_m^\top W_q^\top R_{\Theta,n-m}^d W_k x_n
$$

其中 $R_{\Theta,n-m}^d=(R_{\Theta,m}^d)^\top R_{\Theta,n}^d$。$R_\Theta^d$ 是正交阵，编码过程模长稳定。矩阵本身很稀疏，直接乘不划算；实现用逐对 Hadamard 形式（PDF p. 7 式 34）：偶数维一对 $(x_{2i-1},x_{2i})$ 与 $(\cos m\theta_i,\sin m\theta_i)$ 做

$$
(x_{2i-1},x_{2i})\mapsto (x_{2i-1}\cos m\theta_i-x_{2i}\sin m\theta_i,\ x_{2i-1}\sin m\theta_i+x_{2i}\cos m\theta_i)
$$

相对加法位置，RoPE 是 **乘法**；相对位置来自旋转矩阵的乘积，而不是改加法展开里的某一项（PDF p. 5）。

机制示意如下（根据 PDF p. 5 图 1 重画，非实测）：

```mermaid
flowchart LR
  x["词向量 x"]
  W["仿射 Wq 或 Wk"]
  rot["按位置 m 分块旋转"]
  qk["带位置的 query 或 key"]
  x --> W --> rot --> qk
```

每个二维块用自己的 $\theta_i$，位置 $m$ 只决定转多少度，不另加一个向量。

## 六、论文强调的两条性质

### 长期衰减

跟随 Transformer 的频率设定后，内积会随相对距离增大而衰减（PDF p. 5）。直觉是离得远的 token 关联应当变弱。3.4.3 节把内积写成二维对的复数乘再取实部（PDF p. 8 式 35），用 Abel 变换把和写成 $|S_{i+1}|$ 的加权（PDF p. 8 式 36–37）。$\frac{1}{d/2}\sum |S_i|$ 随 $|m-n|$ 增大而下降，图 2 画出这条上界随相对距离从约 50 到 250 走低（PDF p. 8 图 2）。这是上界论证，不是注意力权重的实测曲线。

### 接到线性注意力

一般注意力（PDF p. 6 式 17）里，原版 softmax 要对每对 token 算内积，复杂度 $O(N^2)$。线性注意力把相似度写成 $\phi(q)^\top\phi(k)$，先算 key 与 value 的结合（Katharopoulos et al.；PDF p. 6 式 18）。旋转不改隐状态模长，于是可以把旋转矩阵乘在非负映射 $\phi,\varphi$ 的输出上（PDF p. 6 式 19）。分母保持不转，以免除零；分子求和可以含负项，权重不再严格概率归一。论文说仍可表达 value 的重要性，并 **kindly argue** 先前相对位置做法与线性自注意力不兼容（PDF p. 2）。

## 七、实验：翻译、BERT 预训练、GLUE、Performer、中文长文

全部实验跑在两台云服务器、各 4 张 V100 上（PDF p. 9）。

### 机器翻译

WMT 2014 英德，约 450 万句对；在 Vaswani Transformer-base 的自注意力上改 RoPE；词表 37k 联合 BPE；评测对最后 5 个 checkpoint 取平均，beam 4，长度惩罚 0.6；fairseq；Adam $\beta_1=0.9$，$\beta_2=0.98$，学习率从 $1\times 10^{-7}$ 线性升到 $5\times 10^{-4}$ 再按步数平方根倒数衰减；label smoothing 0.1（PDF p. 9）。

| 模型 | BLEU |
|---|---|
| Transformer-base | 27.3 |
| RoFormer | 27.5 |

（PDF p. 9 表 1。）提升 0.2 BLEU。论文只说 better，没有显著性检验。

### 替换 BERT 的正弦位置做 MLM

数据 BookCorpus + Wikipedia（Hugging Face Datasets），训练/验证 8:2；基线 `bert-base-uncased`；batch 64，最大长度 512，100k step，AdamW，学习率 $1\times 10^{-5}$（PDF p. 9）。图 3 左：RoFormer 的 MLM loss 比 vanilla BERT 降得更快（PDF p. 10 图 3）。正文没有给出最终 loss 数字。

### GLUE 微调

MRPC、SST-2、QNLI、STS-B、QQP、MNLI；MRPC/QQP 用 F1，STS-B 用 Spearman，其余用准确率。Hugging Face Transformers，3 epoch，最大长度 512，batch 32，学习率 $2,3,4,5\times 10^{-5}$，按 Devlin et al. 在验证集上报最好平均值（PDF p. 10）。表题印的是 GLEU，任务列表是 GLUE。

| 模型 | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI(m/mm) |
|---|---|---|---|---|---|---|
| BERT | 88.9 | 93.5 | 90.5 | 85.8 | 71.2 | 84.6 / 83.4 |
| RoFormer | 89.5 | 90.7 | 88.0 | 87.0 | 86.4 | 80.2 / 79.8 |

（PDF p. 10 表 2。）论文写「six 个数据集里三个显著更好，且改进可观」（PDF p. 10）。按表，赢的是 MRPC、STS-B、QQP；SST-2、QNLI、MNLI 低于 BERT。QQP 从 71.2 到 86.4 跳得很大，论文未解释量级。

### Performer + RoPE

Performer 用线性注意力避免随长度二次增长（PDF p. 10）。Enwik8；12 层、768 维、12 头的字符 Performer；学习率 $1\times 10^{-4}$，batch 128，最大长度 1024（PDF p. 11）。图 3 右：加上 RoPE 后同 step 下收敛更快、loss 更低。实现用 `lucidrains/performer-pytorch`（MIT，PDF p. 11 脚注）。

### 中文长文

在 WoBERT 上把绝对位置换成 RoPE，对照 BERT / WoBERT / NEZHA 的切词级别与位置类型（PDF p. 11 表 3）：BERT 字级 + 绝对；WoBERT 词级 + 绝对；NEZHA 字级 + 相对；RoFormer 词级 + RoPE。

预训练约 34GB（中文维基、新闻、论坛），多阶段改最大长度与 batch（PDF p. 11 表 4）：

| 阶段 | 最大长度 | batch | 步数 | Loss | Accuracy |
|---|---|---|---|---|---|
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

论文说准确率随序列长度上界升高而升高，并把它归因于 RoPE 的泛化（PDF p. 11）。表里阶段 3、4 把长度降到 256/128 后准确率掉到 64.6%、63.4%，阶段 5 回到 1536 才到 67.4%。这是同一条训练轨迹上的阶段数字，不是受控消融。

下游用 CAIL2019-SCM：8964 组中国裁判文书三元组 $(A,B,C)$，判断 $(A,B)$ 是否比 $(A,C)$ 更相似；多数文本超过 512 字；划分 6:2:2（PDF p. 11–12）。BERT 与 WoBERT 用同一预训练数据。

| 模型 | Validation | Test |
|---|---|---|
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | 66.07% | 69.79% |

（PDF p. 12 表 5。）512 截断时 RoFormer 与 WoBERT 相当、略高于 BERT；拉到 1024 后测试集比 WoBERT **绝对高 1.5%**（68.10 → 69.79，论文原话 PDF p. 12）。表中没有 WoBERT-1024：1024 只跑了 RoFormer，和「超过 512 才拉开」的叙事并不构成同长度对照。

## 八、作者自己写的限制

（PDF p. 12 节 4.5.5）

- 二维子空间里把相对位置写成旋转，**没有透彻解释**为什么比其他位置编码收敛更快。
- 证明了 token 间内积的长期衰减，也看到长文更好，但 **没有可信解释** 为什么强于同类。
- RoFormer 仍建立在 Transformer 基础设施上，预训练需要硬件。

结论重申：相对位置可以写成自注意力里的向量乘积，绝对位置由旋转矩阵编码；英、中基准上预训练收敛更快，长文任务更好（PDF p. 12）。

## 九、可迁移的三条

**① 位置约束写在内积上，而不是写在加法展开上。** 先规定 $\langle q_m,k_n\rangle=g(x_m,x_n,m-n)$，再求解 $f_q,f_k$，比在四项展开里增删更干净。后面若要换注意力核，这条约束仍然可检查。

**② 正交旋转保模长，才能接到线性注意力。** 加法位置会改向量范数与核函数假设；旋转不改 $\|q\|,\|k\|$，才能乘在 $\phi(q)$ 上还保持线性注意力的结合律。分母故意不转，是实现细节，不是理论必然。

**③ 「长度灵活」在这篇里是编码形式，不是外推实验。** 旋转矩阵对任意整数位置都有定义，所以不必先建一张长度为 $L$ 的可学表。中文实验把最大长度从 512 调到 1536 再调回来，并在 1024 截断上跑了 CAIL；论文 **没有** 做「训短测长」的系统外推表。把 RoPE 理解成位置表可以无限延长，需要另文的证据。

## 十、它之后发生了什么

> 以下为 **外部补充**，不来自本文。

摘要已写 RoFormer 进入 Hugging Face Transformers 文档。v5 PDF 仍是 14 页方法文，不讨论后续的位置插值、频率缩放或长上下文外推配方。那些工作若要读，应单独对照原件，不要回写进本节的机制。

## 读完该留下的判断

- **被实验支持的**：英德翻译 +0.2 BLEU（表 1）；BERT 替换正弦后 MLM 收敛更快（图 3 左，无终值）；Performer 加 RoPE 同设置下 loss 更低（图 3 右）；CAIL2019-SCM 上 RoFormer-1024 测试 69.79%，比同数据 WoBERT-512 高 1.5 个百分点（表 5）。
- **表与文字不完全同向的**：GLUE 六项里三项低于 BERT（表 2），论文仍写 three out of six 更好。
- **是推导不是因果证明的**：二维旋转是式 11 加初始条件后的一组解，不是唯一解；长期衰减是 $|S_i|$ 上界随距离下降，不是实测注意力。
- **论文承认没解释的**：更快收敛、长文更强，都还没有机制层面的忠实说明（节 4.5.5）。
- **没写的**：会议录用信息；训短测长的外推曲线；与 ALiBi 等后世相对偏置的对比。

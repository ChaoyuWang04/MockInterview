# VAE:重参数化之后,隐变量模型终于能端到端反传

<!-- release-date: 2013-12-20 -->

**本文依据**:`Auto-Encoding Variational Bayes`,arXiv 1312.6114v11(2022-12-10),14 页。作者 Diederik P. Kingma、Max Welling,Universiteit van Amsterdam Machine Learning Group。盘上 PDF 页眉写明 arXiv:1312.6114v11 [stat.ML] 10 Dec 2022。首发日取 arXiv v1 提交日 2013-12-20。PDF 正文未写会议录用信息,本文不补。文中所有数字都标了 PDF 页码;标「外部补充」的段落不来自本文。

## 一句话

有连续隐变量、后验积不出来、数据又大到不能整批算时,变分下界本身也对变分参数不好求导。这篇论文把隐变量写成「确定性变换 + 与参数无关的噪声」,于是下界变成可反传的随机目标;再用一个识别网络把每个样本的近似后验一次算出来。识别网络是神经网络时,整套东西就是变分自编码器。

它解决的不是「怎么画更像」——2013 年这篇原论几乎不谈画质——而是**怎么让带连续隐变量的有向概率模型,能用普通随机梯度在大数据上联合学习生成参数与推断参数**。

## 一、矛盾:积不出来,也采样不起

有向生成模型的故事很短。先从先验 $p_{\theta^*}(z)$ 抽出隐变量 $z$,再从条件 $p_{\theta^*}(x \mid z)$ 抽出观测 $x$。数据 $X=\{x^{(i)}\}_{i=1}^N$ 是 i.i.d. 的,$\theta^*$ 和每条样本的 $z^{(i)}$ 都看不见(p2)。

想学 $\theta$,最干净的目标是边缘似然 $p_\theta(x)=\int p_\theta(z)p_\theta(x \mid z)\,dz$。想表示数据,最干净的对象是后验 $p_\theta(z \mid x)=p_\theta(x \mid z)p_\theta(z)/p_\theta(x)$。两条路在一般非线性似然下同时断掉(p2):

1. **积不出来。** 边缘似然积不出来,后验就写不出,EM 用不上;均值场变分贝叶斯要的那些对近似后验的期望,解析解也没有。论文点名:似然只要稍复杂一点,比如带非线性隐层的神经网络,这三件事一起 intractable。
2. **数据太大。** 整批优化太贵,希望用小 minibatch 甚至单点更新。采样路线(例如 Monte Carlo EM)每个点都要跑一轮昂贵的采样环,规模上不来。

作者因此同时要三件事(p2):

- 对全局参数 $\theta$ 做高效的近似极大似然或 MAP,顺带能按学到的过程造假数据;
- 给定 $x$ 和选定的 $\theta$,对 $z$ 做高效近似后验推断,用来编码和表示;
- 对 $x$ 做高效近似边缘推断,用来去噪、补全、超分这类需要 $x$ 上先验的任务。

全文方法都围着这三件事转。图 1 把模型画成:实线是生成模型 $p_\theta(z)p_\theta(x \mid z)$,虚线是对不可算后验 $p_\theta(z \mid x)$ 的变分近似 $q_\phi(z \mid x)$;$\phi$ 与 $\theta$ 一起学(p2 图 1)。正文只对隐变量做变分、对全局参数做 ML/MAP;对 $\theta$ 也做变分的版本放在附录 F,实验没做(p2、p12–14)。

从编码角度看,$z$ 就是码。于是 $q_\phi(z \mid x)$ 叫**概率编码器**(给定 $x$,给出码的分布),$p_\theta(x \mid z)$ 叫**概率解码器**(给定码,给出 $x$ 的分布)(p3)。识别模型不必因子分解,$\phi$ 也不再从某个闭式期望里解出来(p3)。

## 二、下界能写,梯度却不能用

边缘似然对样本可加。单点可以拆成(p3 式 1):

$$
\log p_\theta(x^{(i)}) = D_{\mathrm{KL}}\big(q_\phi(z \mid x^{(i)}) \,\|\, p_\theta(z \mid x^{(i)})\big) + \mathcal{L}(\theta,\phi;x^{(i)})
$$

KL 非负,所以 $\mathcal{L}$ 是单点边缘似然的变分下界(ELBO)。两种写法后面都会用到(p3 式 2、式 3):

$$
\mathcal{L}(\theta,\phi;x^{(i)}) = \mathbb{E}_{q_\phi(z \mid x)}\big[-\log q_\phi(z \mid x)+\log p_\theta(x,z)\big]
$$

$$
\mathcal{L}(\theta,\phi;x^{(i)}) = -D_{\mathrm{KL}}\big(q_\phi(z \mid x^{(i)}) \,\|\, p_\theta(z)\big) + \mathbb{E}_{q_\phi(z \mid x^{(i)})}\big[\log p_\theta(x^{(i)} \mid z)\big]
$$

第一种把联合密度和识别密度捆在一起;第二种把「靠近先验」和「重建」拆开。两种都要对 $\theta$ 和 $\phi$ 求导。对 $\phi$ 的朴素蒙特卡洛梯度是(p3):

$$
\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}\big[f(z)\nabla_\phi \log q_\phi(z)\big] \simeq \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)})\nabla_\phi\log q_\phi(z^{(l)})
$$

其中 $z^{(l)}\sim q_\phi(z \mid x^{(i)})$。这就是 score-function / REINFORCE 那一类估计。论文的判断很干脆:方差极高,不实用,并指向 [BJP12](p3)。

**旧问题到这里钉死了:下界是对的,但对识别参数求导的那条常用路走不通。**

## 三、重参数化:把随机性从参数里拆出去

### 做法

只要近似后验满足 2.4 节那些温和条件,可以把 $z\sim q_\phi(z \mid x)$ 写成对辅助噪声的可微变换(p3 式 4):

$$
\tilde{z}=g_\phi(\epsilon,x),\qquad \epsilon\sim p(\epsilon)
$$

$\epsilon$ 的分布不依赖 $\phi$。于是对 $q_\phi$ 的期望变成对 $p(\epsilon)$ 的期望,蒙特卡洛估计对 $\phi$ 可微(p3 式 5):

$$
\mathbb{E}_{q_\phi(z \mid x^{(i)})}[f(z)] = \mathbb{E}_{p(\epsilon)}\big[f(g_\phi(\epsilon,x^{(i)}))\big] \simeq \frac{1}{L}\sum_{l=1}^{L} f\big(g_\phi(\epsilon^{(l)},x^{(i)})\big)
$$

一维高斯是最清楚的例子:$z\sim\mathcal{N}(\mu,\sigma^2)$ 时取 $z=\mu+\sigma\epsilon$,$\epsilon\sim\mathcal{N}(0,1)$(p5)。位置–尺度族都可以写成「位置 + 尺度 $\cdot$ 标准噪声」(Laplace、Student's t、Logistic、Uniform、三角、高斯等)(p5)。另两条路是:可逆 CDF(指数、Cauchy、Gumbel 等),以及复合变换(对数正态、Gamma、Dirichlet、Beta 等)(p5)。三条都失败时,可以用与 PDF 同量级复杂度的近似逆 CDF,文献指向 [Dev86](p5)。

机制上:确定性映射 $z=g_\phi(\epsilon,x)$ 把测度从 $q_\phi(z \mid x)\,dz$ 推到 $p(\epsilon)\,d\epsilon$,所以 $\int q_\phi(z \mid x)f(z)\,dz=\int p(\epsilon)f(g_\phi(\epsilon,x))\,d\epsilon$(p4)。噪声从哪里抽与 $\phi$ 无关,$\phi$ 只出现在可微的 $g$ 里——梯度可以穿过采样这一步。

### 两个估计器:SGVB-A 与 SGVB-B

把技巧套到式 2,得到通用估计器 $\tilde{\mathcal{L}}^A$(p3 式 6):

$$
\tilde{\mathcal{L}}^A(\theta,\phi;x^{(i)})=\frac{1}{L}\sum_{l=1}^{L}\Big(\log p_\theta(x^{(i)},z^{(i,l)})-\log q_\phi(z^{(i,l)} \mid x^{(i)})\Big)
$$

其中 $z^{(i,l)}=g_\phi(\epsilon^{(i,l)},x^{(i)})$。

很多模型里 $D_{\mathrm{KL}}(q_\phi(z \mid x^{(i)}) \| p_\theta(z))$ 能解析积掉(附录 B),只需对重建项采样。KL 这时是对 $\phi$ 的正则,把近似后验往先验推。这给出方差通常更低的 $\tilde{\mathcal{L}}^B$(p4 式 7):

$$
\tilde{\mathcal{L}}^B(\theta,\phi;x^{(i)})=-D_{\mathrm{KL}}\big(q_\phi(z \mid x^{(i)}) \| p_\theta(z)\big)+\frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x^{(i)} \mid z^{(i,l)})
$$

全数据下界用 minibatch 放大(p4 式 8):

$$
\mathcal{L}(\theta,\phi;X)\simeq \tilde{\mathcal{L}}^M(\theta,\phi;X^M)=\frac{N}{M}\sum_{i=1}^{M}\mathcal{L}(\theta,\phi;x^{(i)})
$$

实验里只要 $M$ 够大(例如 $M=100$),每点采样数可以取 $L=1$(p4)。算法 1 就是:抽 minibatch、抽 $\epsilon$、对 minibatch 估计器求 $\nabla_{\theta,\phi}$、用 SGD 或 Adagrad 更新,直到 $(\theta,\phi)$ 收敛(p4)。

式 7 让自编码器的类比变得直接:第一项是正则,第二项是期望负重建误差;$g_\phi$ 把 $(x^{(i)},\epsilon^{(l)})$ 映成该点的近似后验样本,再送进 $\log p_\theta(x^{(i)} \mid z^{(i,l)})$(p4)。

**收益**:变分参数的梯度不再走高方差 score function,可用现成随机梯度上升。**边界**:$z$ 必须是连续的,且存在对 $\phi$ 可微的重参数化;离散隐变量走不通——相关工作里作者自己把这一点写成 wake-sleep 相对 AEVB 的优势(p6)。

## 四、AEVB:识别模型一次算出后验

SGVB 是估计器,AEVB 是把它用在「每点一个连续隐变量、数据 i.i.d.」上的算法:用 SGVB 训练识别模型 $q_\phi(z \mid x)$,近似后验推断变成一次祖先采样,不必对每个点做 MCMC 式迭代(p1、p4)。识别模型学成之后,也可用于识别、去噪、表示和可视化(p1)。

当识别模型是神经网络,整套装置就是 **variational auto-encoder**(p1)。名字容易让人以为目标是重建误差;真正被最大化的是 ELBO,重建只是其中一项。

可迁移的一点:只要后验对每个观测是局部的,就可以把「推断」参数化成从 $x$ 到 $q(\cdot \mid x)$ 的前馈映射,用同一个下界同时学生成与推断。这比「每个点现场跑推断」便宜一个数量级的循环。

## 五、正文里的那个 VAE:高斯先验、对角高斯后验、解析 KL

第 3 节给出具体例子(p5)。先验取无参数的各向同性高斯 $p_\theta(z)=\mathcal{N}(z;0,I)$。解码器 $p_\theta(x \mid z)$ 是多元高斯(连续数据)或伯努利(二值数据),分布参数由单隐层全连接网络(MLP)从 $z$ 算出,细节在附录 C。真后验 intractable。变分近似取对角协方差的多元高斯——脚注写明这只是简化选择,不是方法的限制(p5 式 9、脚注 2):

$$
\log q_\phi(z \mid x^{(i)})=\log\mathcal{N}\big(z;\mu^{(i)},\sigma^{2(i)}I\big)
$$

$\mu^{(i)}$、$\sigma^{(i)}$ 是编码 MLP 的输出。重参数化为 $z^{(i,l)}=\mu^{(i)}+\sigma^{(i)}\odot\epsilon^{(l)}$,$\epsilon^{(l)}\sim\mathcal{N}(0,I)$(p5)。先验与 $q$ 都是高斯,走式 7,KL 不用采样(附录 B)。单点估计器是(p5 式 10):

$$
\mathcal{L}(\theta,\phi;x^{(i)}) \simeq \frac{1}{2}\sum_{j=1}^{J}\Big(1+\log((\sigma_j^{(i)})^2)-(\mu_j^{(i)})^2-(\sigma_j^{(i)})^2\Big)+\frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x^{(i)} \mid z^{(i,l)})
$$

$J$ 是 $z$ 的维数。第一项把每个隐维度的均值往 0 推、方差往 1 推;某一维 $\sigma_j\to 0$ 且 $\mu_j$ 非零会受罚,某一维 $\sigma_j$ 过大也会受罚。这就是后来说的「后验对准先验」在对角高斯下的闭式样子。论文原论没有 $\beta$-VAE,没有把 KL 再乘一个可调系数。

### 附录 B:KL 从哪来

设 $p_\theta(z)=\mathcal{N}(0,I)$,$q$ 为对角高斯。分别积 $\int q\log p$ 与 $\int q\log q$(p10–11):

$$
\int q\log p(z)\,dz=-\frac{J}{2}\log(2\pi)-\frac{1}{2}\sum_{j=1}^{J}(\mu_j^2+\sigma_j^2)
$$

$$
\int q\log q(z)\,dz=-\frac{J}{2}\log(2\pi)-\frac{1}{2}\sum_{j=1}^{J}(1+\log\sigma_j^2)
$$

相减得到(p11):

$$
-D_{\mathrm{KL}}(q_\phi(z)\|p_\theta(z))=\frac{1}{2}\sum_{j=1}^{J}\big(1+\log(\sigma_j^2)-\mu_j^2-\sigma_j^2\big)
$$

识别模型里 $\mu$、$\sigma$ 是 $x$ 与 $\phi$ 的函数。正文式 10 第一项就是这一式。

### 附录 C:编码器与解码器的 MLP

伯努利解码器(MNIST 这类二值/当作二值的像素)(p11 式 11):

$$
\log p(x \mid z)=\sum_{i=1}^{D}\big(x_i\log y_i+(1-x_i)\log(1-y_i)\big),\quad y=f_\sigma\big(W_2\tanh(W_1 z+b_1)+b_2\big)
$$

$f_\sigma$ 是逐元 sigmoid,$\theta=\{W_1,W_2,b_1,b_2\}$。

高斯 MLP 作编码器或解码器时,对角协方差(p11 式 12):

$$
\mu=W_4 h+b_4,\quad \log\sigma^2=W_5 h+b_5,\quad h=\tanh(W_3 z+b_3)
$$

当网络作 $q_\phi(z \mid x)$ 时,把 $z$ 与 $x$ 对调,权重记入 $\phi$。实验里 Frey Face 连续,解码器用高斯输出,均值再用 sigmoid 压到 $(0,1)$(p6–7)。

## 六、和当时几条路差在哪

第 4 节不是展览馆,是在划边界(p6)。

**Wake-sleep**[HDFN95]:作者认为当时文献里,唯一能覆盖同一类连续隐变量模型的在线算法。它也用识别模型。缺点是要同时优化两个目标,合起来并不对应(下界的)边缘似然;优点是离散隐变量也能用。每点计算复杂度与 AEVB 相同(p6)。实验主对照就是它。

**随机变分推断**: [BJP12] 用控制变量压朴素梯度的方差,对象是指数族近似;[RGB13] 是更一般的控制变量(Black Box VI);[SK13] 对指数族自然参数用了与本文类似的重参数化(p6)。AEVB 不把近似后验限制在「对每个点解闭式指数族」。

**自编码器**:线性 AE 与线性高斯模型的联系很老;[Row98] 指出 PCA 是 $p(z)=\mathcal{N}(0,I)$、$p(x \mid z)=\mathcal{N}(x;Wz,\epsilon I)$ 且 $\epsilon\to 0$ 时的极大似然(p6)。去噪/收缩/稀疏自编码器说明:光最小化重建不够学有用表示[BCV13]。SGVB 的正则项由变分下界规定(例如式 10),**没有**通常那种为了学表示而另调的正则超参(p6)。还提到 PSD[KRL08]、GSN[BTL13]、DBM 上的识别模型[SL10]——后几类对着无向模型或稀疏编码,不是这篇要的一般有向模型(p6)。

**DARN**[GMW13]:也是自编码结构的有向模型,但隐变量是二值的(p6)。

**独立同期工作**:[RMW14](Rezende 等)也把自编码器、有向模型、随机变分推断和重参数化连在一起,独立于本文,给 AEVB 另一视角(p6)。参考文献条目写的 arXiv 是 1401.4082(p9)。

## 七、实验:下界、边缘似然、二维流形

数据:MNIST 与 Frey Face(p6)。编码器与解码器隐单元数相同。注意第 5 节开头把「生成模型」写成 encoder、「变分近似」写成 decoder,与第 3 节术语对调,是原文笔误;对照第 3 节与后文「500 hidden units」「encoder and decoder」的用法,实验用的仍是第 3 节那对网络(p6–7)。

优化:对下界估计器的梯度再加小权重衰减,对应先验 $p(\theta)=\mathcal{N}(0,I)$,等价于用下界梯度近似 MAP 里的似然梯度(p7)。所有变分与生成参数从 $\mathcal{N}(0,0.01)$ 初始化,Adagrad 全局步长从 $\{0.01,0.02,0.1\}$ 里按前几轮训练集表现选;minibatch $M=100$,$L=1$(p7)。Wake-sleep 用同一套识别模型(p7)。

### 下界(图 2)

MNIST 隐层 500 单元,Frey Face 200 单元(数据集更小,防过拟合)。隐单元数参考先前自编码器文献,相对排名对这个选择不敏感(p7)。MNIST 的隐空间维 $N_z\in\{3,5,10,20,200\}$,Frey Face 为 $\{2,5,10,20\}$(p7 图 2)。纵轴是每点平均变分下界的估计,估计方差小于 1,图上略去;横轴是已评估的训练点数(p7)。

结论:所有实验里 AEVB 比 wake-sleep 收敛明显更快、解更好。多余隐变量没有带来更多过拟合,作者归因于下界的正则(p7)。计算量:Intel Xeon、有效约 40 GFLOPS 时,每一百万训练样本大约 20–40 分钟(p7)。图是曲线不是表,正文没有给出各 $N_z$ 的最终下界标量。

### 边缘似然(图 3)

隐空间很低维时,可用 MCMC 估学生成模型的边缘似然,细节在附录 D;维数再高估计就不稳,所以编码器、解码器改用 100 隐单元、**3 维**隐变量,数据仍是 MNIST(p7)。对照增加 Monte Carlo EM + Hybrid Monte Carlo[DKPR87](p7)。两种训练集大小:$N_{\mathrm{train}}=1000$ 与 $N_{\mathrm{train}}=50000$(p8 图 3)。MCEM 不是在线算法,不能高效用在完整 MNIST 上(p8)。图上可见小集与大集两条曲线族,正文同样没有把最终 $\log p(x)$ 写成表。

附录 D 的估计器分三步(p11–12):用基于梯度的 MCMC(如 HMC)从后验抽 $L$ 个 $z$;给这些样本拟合密度 $q(z)$;再抽 $L$ 个新后验样本,代入

$$
p_\theta(x^{(i)})\simeq\left(\frac{1}{L}\sum_{l=1}^{L}\frac{q(z^{(l)})}{p_\theta(z^{(l)})p_\theta(x^{(i)} \mid z^{(l)})}\right)^{-1}
$$

作者写明:采样空间维数低于 5、样本足够时,估计才好(p11)。

附录 E 的 MCEM:不用编码器,用 $\nabla_z\log p_\theta(z \mid x)=\nabla_z\log p_\theta(z)+\nabla_z\log p_\theta(x \mid z)$ 采样。流程是 10 步 HMC leapfrog,步长自动调到接受率 90%,然后用得到的样本做 5 次权重更新;参数同样走 Adagrad(p12)。边缘似然用训练/测试集各自前 1000 个点估计,每点用 HMC 抽 50 个后验样本、4 步 leapfrog(p12)。

### 可视化(附录 A)

二维隐空间时,识别模型可把高维数据投到二维流形(p8)。图 4:因为先验是高斯,在单位正方形上均匀取点,经高斯逆 CDF 变成 $z$,再画 $p_\theta(x \mid z)$ 的均值(Frey Face 与 MNIST 各一张流形)(p10)。图 5:MNIST 生成模型在 $N_z=2,5,10,20$ 时的随机样本(p10)。这是定性图,没有数值表。

## 八、附录 F:对 $\theta$ 也做变分(实验没跑)

正文把对全局参数的变分留给附录。引入超先验 $p_\alpha(\theta)$,边缘 $p_\alpha(X)$ 同样拆成 KL + 下界 $\mathcal{L}(\phi;X)$(p12 式 13–14)。$\theta$ 与 $z$ 都重参数化:$\tilde{z}=g_\phi(\epsilon,x)$,$\tilde{\theta}=h_\phi(\zeta)$(p13 式 17–19)。把各项收进 $f_\phi$,蒙特卡洛估计只依赖与 $\phi$ 无关的 $\epsilon,\zeta$,因此可对 $\phi$ 求导(p13 式 21–22)。算法 2 是对 $L$ 次抽样累加 $\nabla_\phi f_\phi$(p14)。

高斯例子里 $p_\alpha(\theta)$ 与 $p_\theta(z)$ 都是 $\mathcal{N}(0,I)$,变分后验对角高斯,于是 $\theta$ 与 $z$ 两侧 KL 都能解析,得到式 24 那种低方差估计器(p13–14)。**正文明确:这类实验留待未来**(p2)。不要把附录 F 读成「VAE 默认对权重做全贝叶斯」。

## 九、作者自己列的下一步

第 7 节四条,都还没做(p8):(i) 用深度网络(例如卷积)作编码器解码器的分层生成结构,用 AEVB 联合训练;(ii) 时间序列 / 动态贝叶斯网;(iii) 把 SGVB 用到全局参数;(iv) 带隐变量的监督模型,用来学复杂噪声。2013 年这篇原论里没有卷积 VAE、没有时序 VAE、没有条件 VAE 的实验。

## 十、可迁移的三条

**① 把「采样」改写成「噪声 + 可微变换」,梯度才能穿过随机节点。** 不是所有分布都能这样写,连续、可重参数化是前提。离散隐变量要另找出路——论文把这条写成与 wake-sleep 的对比,而不是后来的 Gumbel-Softmax。

**② 能解析的 KL 不要拿去采样。** 式 7 / 式 10 把方差从正则项里拿掉,只对重建项采样。先验与近似后验同属高斯时,附录 B 那一行几乎零成本。换先验或换后验族,首先问 KL 还闭不闭式。

**③ 识别模型把「每点迭代推断」换成一次前向。** AEVB 的效率主张建立在 i.i.d. 加每点局部隐变量上。全局隐变量、强耦合的 $z$,不能直接套「一个编码器扫过去」。

对自己项目的直接用法:实现对角高斯 VAE 时,编码器输出 $\mu$ 与 $\log\sigma^2$,$z=\mu+\sigma\odot\epsilon$,损失是解析 KL 减重建对数似然(或加重建误差,符号看你最大化还是最小化);$L=1$、$M$ 上百,是论文自己验证过的设置,不是后世口口相传。

## 十一、它之后发生了什么

> 以下为**外部补充**,不来自本文。

这篇原论停在 MNIST / Frey Face 的下界曲线和二维流形。后来图像生成里常说的「VAE」往往已经换了骨干、换了先验、换了目标,不能回写进 2013 年的主张。本库同一方向下,`VQ-VAE` 把连续隐空间量化成离散码本;`Latent-Diffusion` 把扩散搬到(另一类)VAE 的隐空间里做——都是后续工作,机制与这篇的连续对角高斯后验不是同一件事。ICLR 等会议信息 PDF 未写,本文不补。

## 读完该留下的判断

- **被实验支持的**:在所述 MLP + 对角高斯设定下,AEVB 优化变分下界时,相对 wake-sleep 在图 2 的全部 $N_z$ 上都更快更好;多余隐维未表现为更强过拟合。小隐维 MNIST 上,图 3 把 AEVB、wake-sleep、MCEM 的边缘似然收敛画在一起,并标明 MCEM 不适合完整 MNIST。二维流形与随机样本是定性证据。
- **是作者观察而非单独证明的**:「下界的正则阻止了过拟合」是对图 2 的解释,没有拆掉 KL 再对照的消融。$L=1$ 只要 $M$ 够大,是实验观察,不是定理。
- **没有公开的**:图 2、图 3 没有数值表,最终下界 / 边缘似然不能从正文读成精确标量。没有离散隐变量实验。附录 F 的全变分贝叶斯没有实验。没有卷积、没有标准图像生成指标(FID 等——那是论文之后的评测,本文原论不用)。
- **原文笔误**:第 5 节把 generative model 写成 encoder、variational approximation 写成 decoder,与第 2–3 节相反;读实验时以第 3 节定义为准。

## 关键词回看

- **变分下界(ELBO)**:$\log p(x)$ 减掉 $q$ 与真后验的 KL 之后剩下的那一项,是可优化的代理目标。
- **SGVB**:把 ELBO 里对 $q$ 的期望换成对 $\epsilon$ 的蒙特卡洛,得到可反传的随机目标;A 版通用,B 版把解析 KL 留在外面。
- **重参数化**:$z=g_\phi(\epsilon,x)$,$\epsilon$ 与 $\phi$ 无关。
- **识别模型 / 概率编码器 $q_\phi(z \mid x)$**:用前馈网络逼近 $p_\theta(z \mid x)$。
- **概率解码器 $p_\theta(x \mid z)$**:从码生成观测的条件分布。
- **AEVB**:用 SGVB 联合学 $\theta$ 与 $\phi$ 的 minibatch 算法。
- **变分自编码器**:识别模型为神经网络时的 AEVB 实例;正文例子是 $\mathcal{N}(0,I)$ 先验加对角高斯 $q$。

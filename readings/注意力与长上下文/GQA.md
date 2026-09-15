# GQA:从多头检查点 uptrain 出分组查询注意力

<!-- release-date: 2023-05-22 -->

**本文依据**:`GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints`,arXiv 2305.13245v3(`[cs.CL]` 23 Dec 2023),7 页。作者 Joshua Ainslie、James Lee-Thorp、Michiel de Jong(共同一作)以及 Yury Zemlyanskiy、Federico Lebrón、Sumit Sanghai;封面机构为 **Google Research**(de Jong 脚注 University of Southern California,文中写明工作完成于 Google Research)。封面**没有会议名**。首发日取 arXiv v1 提交日 2023-05-22,v3 不回写。文中数字标 PDF 页码;标「外部补充」的段落不来自本文。

## 一句话

解码每一步都要把 Key、Value 从显存里搬出来,头数一多就搬不动。论文做两件事:用原预训练 **5%** 的步数把已有多头检查点 **uptrain** 成少 KV 头的结构;再把「全体 query 共用一对 KV」放松成「一组 query 共用一对 KV」。后一种叫 **grouped-query attention(GQA)**。在 T5-XXL 上,GQA-8 的平均分几乎贴上多头,推理时间几乎贴上多查询。

## 一、矛盾:解码慢在搬 KV,不在算 FLOPs

自回归解码每生成一个 token,都要把解码器权重、以及**到目前为止全部 Key 与 Value** 再加载一遍。论文把这件事明确写成 **memory bandwidth overhead**(PDF p. 1),并指向 Shazeer 2019 的 MQA、Pope 等人的大规模推理、以及 de Jong 等人的 FiDO。

**多查询注意力(multi-query attention,MQA)** 的做法很干脆:query 仍是多头,Key 和 Value **各只留一头**,所有 query 头去共享这一对。KV cache 体积按头数 $H$ 砍一刀,带宽跟着下来。

论文立刻列出 MQA 的三笔账(PDF p. 1):

1. 质量会掉。
2. 训练可能不稳。
3. 为了「质量版」和「速度版」各训一个完整模型,往往不划算。

当时已有 PaLM 这类从一开始就用 MQA 的模型,也有 T5、LLaMA 这类公开检查点仍是多头注意力(MHA)的模型。后一类已经花掉了预训练预算,问题就变成:**能不能在不重训的前提下,把多头检查点改成推理更快的结构?**

全文两条贡献对应这两个缺口(PDF p. 1):

- **Uptraining**:把已有 MHA 检查点转成 MQA(或后面的 GQA),再用原预训练配方补训一小段。
- **GQA**:KV 头数介于 1 与 $H$ 之间,质量接近 MHA,速度接近 MQA。

## 二、Uptraining:先平均池化,再补 5% 步数

转结构分两步(PDF p. 1 图 1,§2.1):

1. **改检查点。** 各头的 Key 投影矩阵、Value 投影矩阵分别 **mean pool** 成单头(MQA)或按组 pool(GQA)。论文写明:平均池化好过「只留第一头」和「随机初始化新 KV 头」。
2. **再预训练。** 在**同一套预训练配方与数据**上,再跑原训练步数的比例 $\alpha$。正文后来说 $\alpha=0.05$ 时大约 **600 TPUv3 chip-days**(PDF p. 2)。

摘要写的是「**5% of original pre-training compute**」(PDF p. 1)。实验里这 5% 具体落地为 **$\alpha=0.05$ 的原预训练步数**(PDF p. 2–3),表 1 标题也写「5% uptrained」。两边说的是同一次设置,不是另一笔 FLOPs 账。

消融里三种转法的排序正好对应「从原模型里保住了多少信息」(PDF p. 3–4 图 4,T5-Large $\to$ MQA,$\alpha=0.05$):Mean 最好,First 次之,Random 最差。图是条形图,横轴大约 54.4–55.6,论文**没有把三个精确分数写成表**。

$\alpha$ 扫过 0 到 0.1(PDF p. 4 图 5):

- **GQA 转完就能用**,MQA 几乎必须再训一阵才像样。
- 两者都从 **5%** 再训里明显受益;**10%** 收益递减。

## 三、GQA:一组 query 共用一对 KV

把 $H$ 个 query 头切成 $G$ 组,每组共享一对 Key / Value 头。记号(PDF p. 2):

- **GQA-$G$**:$G$ 个组。
- **GQA-1** $=$ MQA(全体共用一对 KV)。
- **GQA-$H$** $=$ MHA(每头一对 KV)。

机制示意见 PDF p. 2 图 2:MHA 是 $H$ 套 Q/K/V;MQA 是 $H$ 个 Q 对着 1 对 K/V;GQA 是每组若干 Q 对着 1 对 K/V。从 MHA 检查点转过来时,组内原头再做一次 mean pool。

为什么不满足于 MQA?论文给了三条缩放理由(PDF p. 2),都是论证,不是新实验:

1. 大模型头数通常跟着涨。MHA$\to$MQA 等于把 $H$ 对 KV 压成 1 对,cache 与带宽都除以 $H$。模型越大,这一刀越狠,容量也砍得越狠。GQA 让「KV 相对 query 的缩减比例」可以跟着规模一起调。
2. KV cache 随模型维度线性涨,FLOPs 与参数随维度平方涨,所以更大的模型里,注意力带宽占比相对没那么极端。
3. 大模型常用切分时,MQA 那唯一的 KV 头会被 **replicate 到每个 model partition**(引 Pope et al.,2022)。GQA 去掉这笔浪费。因此作者**预期** GQA 对更大模型更划算。

**编码器自注意力不用 GQA。** 编码器并行算完表示,内存带宽通常不是主瓶颈(PDF p. 2)。实验里 MQA/GQA 只加在 **解码器自注意力和交叉注意力**(PDF p. 2)。

组数怎么选:图 6 扫 GQA 组数对 T5-XXL 每样本时间的影响,输入长 2048、输出长 512(PDF p. 4)。从 1 组(MQA)加到 **8 组**,变慢不多;再往 MHA 靠,代价越来越大。作者把 **8 组** 选成中间点。纵轴大约 0–2+ 秒量级,图上没有逐点表。

## 四、实验怎么摆

全部基于 **T5.1.1**,JAX / Flax / Flaxformer,Adafactor 与 T5 同一套超参和学习率日程(PDF p. 2)。主实验:T5 Large 与 XXL 的 MHA,以及从公开 T5.1.1 检查点 uptrain 出来的 XXL MQA 与 GQA。

数据(PDF p. 2–3):摘要 CNN/Daily Mail、arXiv、PubMed、MediaSum、Multi-News;翻译 WMT 2014 En$\to$De;问答 TriviaQA。**不做 GLUE 一类分类**,理由是自回归推理在那些任务上不太适用。

微调:学习率常数 **0.001**,batch **128**,dropout **0.1**(PDF p. 3)。CNN/Daily Mail 与 WMT:输入 512、输出 256。其余摘要:输入 2048、输出 512。TriviaQA:输入 2048、输出 32。训到收敛,按 dev 最优选点,**greedy decoding**。

计时:每样本、每 **TPUv4** chip 的时间,xprof 量;8 块 TPU,每卡能放下的最大 batch、上限 32,并行按模型分别调(PDF p. 3)。

消融子集:CNN/Daily Mail、MultiNews、TriviaQA(PDF p. 3)。

## 五、主结果:表 1 对摘要「接近 MHA、接近 MQA」

表 1 是全文唯一完整数字表(PDF p. 3)。$T_{\mathrm{infer}}$ 单位是秒。Average 是各任务 dev 的平均。

| 模型 | $T_{\mathrm{infer}}$(s) | Average | CNN R1 | arXiv R1 | PubMed R1 | MediaSum R1 | MultiNews R1 | WMT BLEU | TriviaQA F1 |
|---|---|---|---|---|---|---|---|---|---|
| MHA-Large | 0.37 | 46.0 | 42.9 | 44.6 | 46.2 | 35.5 | 46.6 | 27.7 | 78.2 |
| MHA-XXL | 1.51 | 47.2 | 43.8 | 45.6 | 47.5 | 36.4 | 46.9 | 28.4 | 81.9 |
| MQA-XXL | 0.24 | 46.6 | 43.0 | 45.0 | 46.9 | 36.1 | 46.5 | 28.5 | 81.3 |
| GQA-8-XXL | 0.28 | 47.1 | 43.5 | 45.4 | 47.7 | 36.3 | 47.2 | 28.4 | 81.6 |

读法:

- MQA-XXL 比 MHA-Large **又快又高**(0.24 s vs 0.37 s;46.6 vs 46.0)。
- GQA-8-XXL 平均 **47.1**,相对 MHA-XXL 的 **47.2** 只差 0.1;时间 **0.28 s**,相对 MQA 的 **0.24 s** 只多 0.04 s,相对 MHA-XXL 的 **1.51 s** 大约快 5 倍量级。
- 单任务并非处处第一:PubMed 与 MultiNews 上 GQA-8 高于 MHA-XXL;WMT 上 MQA 略高(28.5 vs 28.4);TriviaQA 仍是 MHA-XXL 最高(81.9)。

图 3 把平均分对平均推理时间画成散点(PDF p. 3)。图注与正文口径一致:uptrain 后的大 MQA 相对 MHA-Large 是更好的速度–质量点;GQA 再把质量拉近 MHA-XXL。图轴标注写成 **Time per sample (ms)**,表 1 是秒——论文自己两处单位不一致,本文按表 1 的秒引用时间,不把图轴的 ms 改写成另一种测量。

## 六、相关工作里它把自己放在哪

目标是 **decoder 质量 vs 推理时间**,手段是少加载 KV(PDF p. 4)。Shazeer 2019 提出 MQA;后续指出长输入时 MQA 尤其有用。**Rabe 2023** 被写成独立做出过 GQA,并有公开实现(Flaxformer `memory_efficient_attention.py`,文中访问日 2023-05-23)。另有把注意力头分组以省计算的工作,但论文强调那些**不是专门砍 KV 头**(KV 头才决定带宽)。

并列的减带宽手段只点名、不对比数字:FlashAttention、量化、蒸馏、layer-sparse cross-attention、speculative sampling。Uptraining 的灵感来自 Komatsuzaki 等人把稠密 T5 **upcycle** 成 MoE(PDF p. 4)。

## 七、限制:作者自己划的边界

Limitations 一节四条(PDF p. 5),都该当结论读,不要补实验:

1. 带宽问题在**生成较长序列**时最重,而长生成的质量本身就难评。
2. 摘要用 **ROUGE**,作者承认它「不讲完整故事」,因此**不能确信**速度–质量权衡在真实偏好上仍然成立。
3. 算力不够,**没有**拿 XXL GQA 去比「从零训一个同结构模型」,uptrain 相对 scratch 的差距未知。
4. 只评了 **encoder-decoder**。Decoder-only 没有分开的自注意力与交叉注意力,作者**预期** GQA 相对 MQA 的优势会更强——这是预期,不是测量。

附录 A **Training Stability**(PDF p. 7):从零训的 T5-Large MQA,预训练频繁 loss spike,长输入微调立刻发散。Uptrain 的 MQA 稳一些,但仍高方差,不稳定任务上他们报 **三次微调平均**。Uptrain 的 GQA **看起来稳**,所以没有继续追 MQA 不稳的根因。

## 八、可迁移的三条

**① 结构可以后改,贵的是预训练过的容量,不是头的拓扑。** Mean pool 保住信息,再花 5% 步数让网络适应新 KV 布局。要另训一个「推理专用」完整模型之前,先问能不能从现成 MHA 检查点 uptrain。

**② KV 头数是连续旋钮,不是 1 或 $H$ 的开关。** GQA-8 在这张 T5-XXL 表上已经够用;图 6 说明 1$\to$8 几乎不涨延迟。组数应跟模型规模和切分方式一起选,而不是默认抄 MQA。

**③ 编码器不是同一场带宽战争。** 论文明确不改 encoder self-attention。套到 encoder-decoder 时,先动 decoder 自注意力和 cross-attention。

## 读完该留下的判断

- **被表支持的**:$\alpha=0.05$ 的 XXL 上,GQA-8 平均 47.1 vs MHA 47.2,时间 0.28 s vs MQA 0.24 s vs MHA 1.51 s(PDF p. 3 表 1)。Mean pool 优于 First 与 Random(图 4)。5% 有用、10% 递减(图 5)。GQA 转完即有可用质量,MQA 更依赖再训。
- **是作者观察/预期、不是因果证明的**:大模型上 GQA 更划算;decoder-only 上 GQA 相对 MQA 优势更大;MQA 不稳的根因。
- **论文没写的**:从零训 GQA-XXL 的对照;decoder-only 数字;ROUGE 以外的生成质量;图 4/5/6 的精确逐点表。

## 参考资料

- 原件:`readings/_src/注意力与长上下文/GQA.pdf`(arXiv 2305.13245v3)
- Flaxformer(文中脚注):`https://github.com/google/flaxformer`
- Shazeer 2019 MQA、Komatsuzaki 2022 sparse upcycling、Rabe 2023 独立 GQA 实现:见 PDF 参考文献,本文不展开

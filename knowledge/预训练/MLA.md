# MLA(Multi-head Latent Attention)

一句话:**不存完整的 K/V,把它们压成一个低维「潜在向量」存进缓存,用的时候再投影回去**——DeepSeek 的招牌注意力,出自 DeepSeek-V2,V3/R1 一路沿用,如今已是「压缩派」的事实标准。

> **类比**:GQA 是「几个人共用一本书」,MLA 是「把书压缩成摘要存着,要用时再展开」。更妙的是,靠矩阵吸收技巧,读者后来学会了直接读摘要——连「展开」这一步都省了。

## 一、动机:KV cache 是 decode 的第一瓶颈

自回归生成时,每个历史 token 的 K/V 都要缓存,缓存随上下文线性增长(公式全景见「架构总览」篇):

$$
\text{KV cache} = L_{\text{layers}} \times \underbrace{n_{kv} \times d_h}_{\text{MLA 压这里}} \times 2 \times S \times b
$$

直观的数:Gemma 3 27B 每 token 要 496 KiB,128k 上下文约 62 GB——一张 80 GB 的 H100 光缓存就吃掉大半。GQA 的思路是减小 $n_{kv}$(减头数,见「GQA」篇);MLA 换了个方向:**把 $n_{kv} \times d_h$ 这一整块低秩压缩成一个远小于它的 $d_c$**(压维度)。

> 🖼️ 占位:DeepSeek-V2 论文中 MHA / GQA / MQA / MLA 四种方案「缓存里存什么」的对比示意图

## 二、核心机制:压缩、缓存、解压

### 1. 下投影(压缩):缓存里只存潜在向量

对每个 token 的 hidden state $h_t \in \mathbb{R}^{d}$,先做一次低秩下投影:

$$
c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{d_c}, \qquad d_c \ll n_h d_h
$$

**进缓存的只有这个 $c_t^{KV}$**。DeepSeek-V2/V3 里 $d_c = 512$,而完整的 K、V 各是 $n_h d_h = 128 \times 128 = 16384$ 维——一个 512 维向量同时顶替了 K、V 两份 16384 维的存储。

### 2. 上投影(解压):用时再还原出各头的 K/V

$$
k_t^C = W^{UK} c_t^{KV}, \qquad v_t^C = W^{UV} c_t^{KV}
$$

query 不进缓存,但也做同样的低秩压缩($c_t^Q = W^{DQ} h_t$,$q_t^C = W^{UQ} c_t^Q$)——目的不是省缓存,是**省训练时的激活显存**。

### 3. 矩阵吸收:解码时根本不用真的解压

注意力分数里,上投影矩阵夹在中间,可以预先合并:

$$
(q_t^C)^\top k_s^C = (W^{UQ} c_t^Q)^\top (W^{UK} c_s^{KV}) = (c_t^Q)^\top \underbrace{(W^{UQ})^\top W^{UK}}_{\text{可预先合并进 query 侧}} c_s^{KV}
$$

对 value 同理:注意力输出要过输出投影 $W^{O}$,而 $v = W^{UV} c^{KV}$,所以 $W^{UV}$ 可以吸收进 $W^{O}$。于是 decode 时**完整 K/V 从头到尾不需要被展开**,注意力直接在潜在空间里算——数学上等价于一个「所有 query 头共享同一份潜在 K/V」的超宽 MQA,但每个头经由各自的 $W^{UK}$ 切片读出不同的有效 K,逐头表达力没有丢。

## 三、RoPE 的麻烦与 decoupled RoPE(面试高频)

RoPE 是位置相关的旋转矩阵 $R_t$,恰好插在吸收要合并的两个矩阵中间:

$$
q_t^\top k_s = (c_t^Q)^\top (W^{UQ})^\top R_t^\top R_s W^{UK} c_s^{KV} = (c_t^Q)^\top (W^{UQ})^\top R_{s-t} W^{UK} c_s^{KV}
$$

中间那坨 $(W^{UQ})^\top R_{s-t} W^{UK}$ 随相对位置 $s-t$ 变化,**没法预先合并成一个固定矩阵**——旋转和吸收不兼容。

DeepSeek 的解法是 **decoupled RoPE(解耦旋转位置编码)**:潜在部分完全不加位置编码,另外单独留一小段「专门携带 RoPE 的 key」,与潜在部分拼接:

$$
q_{t,i} = [\,q_{t,i}^C;\ q_{t,i}^R\,], \qquad k_{s,i} = [\,k_{s,i}^C;\ k_s^R\,], \qquad k_s^R = \mathrm{RoPE}(W^{KR} h_s)
$$

- $k^R$ 维度很小(V2/V3 里 $d_h^R = 64$),且**所有头共享一份**(MQA 式),直接进缓存;
- 注意力分数 = 潜在部分的内积(可吸收)+ RoPE 部分的内积(本来就小,不用吸收),两段相加;
- 于是每层每 token 缓存 = $d_c + d_h^R = 512 + 64 = 576$ 个数。

## 四、算账:省多少、亏多少

**省的账**:DeepSeek V3 671B 有 61 层,每 token 缓存 $61 \times 576 \times 2\,\text{B} \approx 68.6$ KiB;对比 Gemma 3 27B 的 496 KiB——**参数量大 25 倍,缓存反而小 7 倍**。

**亏的账**:多了压缩、解压两次矩阵乘,prefill/训练的注意力 FLOPs 不降反微升。为什么仍然划算?因为 **decode 是内存带宽瓶颈,不是算力瓶颈**:每生成一个 token 都要把整份 KV cache 从 HBM 搬一遍,算力大量闲置。MLA 把要搬的字节数砍到约 1/7,多出来的矩阵乘吃的是本来就闲着的算力——**拿富余的 FLOPs 换稀缺的带宽**。上下文越长,这笔交易越赚。

## 五、数据流全景

```mermaid
flowchart TB
    subgraph S1["压缩与缓存(训练/推理共用)"]
        H["hidden state h_t"] -->|"W^DKV 下投影"| C["潜在向量 c^KV(512 维)"]
        H -->|"W^KR + RoPE"| R["解耦 RoPE key k^R(64 维,全头共享)"]
        C --> KV[("KV cache:只存 c^KV 与 k^R<br/>每层 576 个数 / token")]
        R --> KV
    end
    subgraph S2["训练 / prefill:展开路径(算力富余)"]
        KV -->|"W^UK 上投影"| K["完整 K"]
        KV -->|"W^UV 上投影"| V["完整 V"]
        K --> A1["标准多头注意力"]
        V --> A1
    end
    subgraph S3["decode:吸收路径(带宽受限)"]
        KV --> A2["不展开 K/V,直接在潜在空间算<br/>W^UK 吸进 query 侧,W^UV 吸进输出投影"]
    end
```

## 六、谁在用

| 模型 | 配置 | KV/token |
| --- | --- | --- |
| DeepSeek V3 / R1 / V3.2 | 61 层 MLA(V3.2 叠加 DSA 稀疏) | 68.6 KiB |
| Kimi K2 / K2.5 / K2.6 | 1T 参数,61 层 MLA(与 V3 同骨架) | 68.6 KiB |
| GLM-5 / 5.1 | 78 层 MLA + DSA | 87.8 KiB |
| Mistral Large 3 | 61 层 MLA | 68.6 KiB |
| Mistral Small 4 | 36 层 MLA | 22.5 KiB |
| Kimi Linear / K3 | KDA 线性层与门控 MLA 混合(约 3:1) | 7.9 KiB / 未公开 |
| Ling 2.5 | 70 Lightning + 10 MLA(7:1) | 11.2 KiB |
| 其他 | Sarvam 105B、LongCat 系(RoPE/NoPE 混用) | — |

最有意思的组合是 **Kimi Linear:MLA 层干脆用 NoPE**(不加位置编码)——位置信息全交给 KDA 线性层承担,连 decoupled RoPE 这块补丁都省了,吸收路径更干净(NoPE 与层间分工见「RoPE」篇,KDA 见「线性注意力」篇,混合比例见「Hybrid注意力」篇)。

## 七、延伸:CCA——干脆在压缩空间里算注意力

MLA 有件「没做完的事」:它只把 KV **存**成压缩形式,算注意力时还是要展开回完整头空间——缓存省了,prefill/训练的 FLOPs 一点没省,还因上投影略贵。

**CCA(Compressed Convolutional Attention,压缩卷积注意力)**,Zyphra 提出,更激进:Q/K/V 全部下投影,**整个注意力运算直接在压缩空间里做**,算完再展开。缓存、参数、注意力 FLOPs 一起省。

- 名字里的「卷积」是补丁:压缩让 Q、K 变窄、表达力下降,于是加一个轻量卷积给压缩后的 Q、K 补充局部上下文;
- **卷积只加在 Q、K 上,不加在 V 上**——Q、K 决定「看哪里」,需要精细;V 只是「被平均的内容」,精度要求低;
- 与头共享(GQA)正交,可组合成 CCGQA,进一步收紧「算力-带宽」的帕累托前沿;
- 采用者:ZAYA1-8B(AMD GPU 上训练的 8.4B MoE)。论文自报同压缩率下优于 MLA/GQA,**但没有第三方复现**,当成值得关注的方向而非定论。

## 八、细节与常见坑

### MLA vs GQA:压维度,不是减头数

| 维度 | GQA | MLA |
| --- | --- | --- |
| 思路 | 减少 KV 头数(组内共享 K/V) | 低秩压缩整个 KV 表示 |
| 缓存内容 | 若干组完整 K/V 头 | 一个 $c^{KV}$ + 一小段 RoPE key |
| 逐头表达力 | 组内共享,打折 | 每头读出不同的有效 K/V,保留 |
| 额外计算 | 无 | 多两次投影(decode 可吸收) |
| 实现门槛 | 标准 FlashAttention 即可 | 需专用 kernel |

同等缓存预算下 MLA 质量通常更好(DeepSeek-V2 自报甚至略优于原版 MHA),代价是实现复杂度。

### 其他坑

- **两种计算模式要切换**:prefill 算力受限,走「展开路径」用标准注意力;decode 带宽受限,走「吸收路径」。推理引擎要维护两套逻辑;
- **对推理栈有硬要求**:吸收后的注意力形状特殊(单份 576 维共享 KV),需要专用 kernel——DeepSeek 开源了 FlashMLA,vLLM/SGLang 均有 MLA 后端;换用小众引擎前先确认支持;
- **低秩不是免费的**:$d_c$ 是信息瓶颈,压得太狠伤容量,512 这个数是训练配方的一部分;也不能把训好的 GQA 模型零成本改造成 MLA(需要转换加继续训练);
- **RoPE 部分别忘了算**:decoupled 那 64 维是缓存里的「隐藏行李」,算账时漏掉它得不出 68.6 KiB。

## 九、面试考点串联

高频问法(与题库联动的切片点):

1. MLA 和 GQA 都省 KV cache,本质区别是什么 →「八、对比表」
2. MLA 的缓存里到底存了什么、每 token 多大 →「二、机制」+「三」结尾的 576 账
3. 矩阵吸收怎么做、为什么 decode 不需要展开 K/V →「二、机制 3」
4. RoPE 为什么和吸收不兼容、decoupled RoPE 怎么设计 →「三」(最高频细节)
5. 省显存但多两次矩阵乘,为什么划算 →「四、算账」(带宽瓶颈论证)
6. 比 MLA 更进一步的方向 →「七、CCA」;稀疏化路线见「稀疏注意力」篇
7. 哪些模型在用、Kimi Linear 的 NoPE 组合妙在哪 →「六」

## 相关文献

- DeepSeek-V2(MLA 首次提出)— [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- DeepSeek-V3(MLA + MoE 大规模实战)— [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Compressed Convolutional Attention(CCA/CCGQA,Zyphra)— [arXiv:2510.04476](https://arxiv.org/abs/2510.04476)
- Kimi Linear(KDA + 门控 MLA + NoPE 组合)— [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)

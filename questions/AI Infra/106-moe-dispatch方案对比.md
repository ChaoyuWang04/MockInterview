---
difficulty: 困难
topic: MoE并行与DeepEP/dispatch方案对比
summary: MoE dispatch 的四种方案、通信量与各自优劣
tags: [真题, 待校对, MoE, EP, 集合通信]
company:
mastered: false
highfreq: false
---

## 题目

MoE 的 dispatch 有哪些方案?流程是咋样?分别有哪些优势和劣势?通信量是多少?(AllGather、AllReduce、all2all、DeepEP)

## 要点

- 先立两条通信量基准,再把"冗余倍数 $E/k$"推出来
- 四方案的取舍轴不是只有通信量,还有**规则性**
- AllReduce 方案在 TP 布局下 dispatch 是免费的,这是它没淘汰的原因
- combine 侧和 dispatch 同阶,只优化一半是常见想当然
- 表里的量级只到阶,别把系数当精确值

## 答案

### 通信量怎么估

只看**每张卡进出的字节数**。本卡这层 $T$ 个 token、隐藏维 $h$、top-$k$、专家摊在 $E$ 张卡上、每元素 $b$ 字节:

$$
V_{\text{All2All}} \;\le\; k\,T\,h\,b, \qquad V_{\text{AllGather}} \;\approx\; E\,T\,h\,b \quad\Longrightarrow\quad \frac{V_{\text{AllGather}}}{V_{\text{All2All}}} \;\approx\; \frac{E}{k}
$$

左式:All2All 只寄该寄的,本卡最多把每个 token 复制 $k$ 份送出(落在本卡的专家不用送,所以是上界)。中式:AllGather 要求每张卡都拿到**全部** $E$ 张卡的 token,和 top-$k$ 无关。相除就是那句该记的话:**AllGather 的冗余倍数就是「EP 规模 ÷ top-k」**。256 选 8 的配置下,EP=32 要多搬 4 倍,EP 上到几百卡就是几十倍。

### 四方案对照

| 方案 | dispatch 每卡量 | combine 每卡量 | 规则性 | 适用 | 主要缺点 |
|---|---|---|---|---|---|
| **AllGather + ReduceScatter** | $\approx E\,Thb$ | $\approx E\,Thb$ | **完全规则**:形状静态、不用交换计数表、CUDA Graph 友好 | EP 小(单节点量级),默认与兜底路径 | 通信量随 EP 线性涨,$E/k$ 倍冗余;还多一块全量 token 的显存 |
| **AllReduce** | 同上;TP 布局下**免费** | $\approx 2E\,Thb$ | 完全规则,实现最省事 | 正确性基线、极小规模 | 最贵的一档;非本卡专家的位置**填零后照样参与求和**,大半带宽在搬 0 |
| **朴素 All2All** | $\le k\,Thb$ | $\le k\,Thb$ | **不规则**:要先排序打包、先交换 split sizes | 通用;EP 一大就只剩这条路 | 对延迟敏感;排序打包本身有开销;负载一歪整层跟着慢;动态形状难上 CUDA Graph |
| **DeepEP** | 与 All2All 同阶,**常数更小** | 同阶 | 仍不规则,但封进库里 | 高吞吐 kernel → 训练 / prefill;低时延 kernel → decode | 依赖 NVSHMEM 与 RDMA,对硬件与网络配置挑剔,运维门槛高 |

### 三条容易被追问的补充

- **AllReduce 方案为什么至今没淘汰**:如果 attention 侧走的是 TP,每张 TP rank 本来就持有全量 token,**dispatch 这一步根本不需要发生**,整层只剩最后一次 AllReduce。写起来最短、最不容易错,所以长期是"先跑通再说"的那条路
- **combine 侧不能省**:dispatch 送的是输入、combine 送的是输出,两边字节数同阶。只优化 dispatch 是常见的想当然,实际两趟一样痛
- **表里的量级只到阶**:真实系统还要加路由元数据、padding、FP8 的 scale 这些零头,别把系数当精确值

## 知识点

$E/k$ 冗余倍数、规则性(静态形状 / 免计数表 / CUDA Graph 友好)是与通信量并列的取舍轴、TP 布局下 dispatch 免费、AllReduce 在搬 0、combine 同阶。

## 追问

- All2All 通信量最省,为什么还有人用 AllGather 或 AllReduce?
- 用 AllGather 做 dispatch 浪费在哪、浪费多少?怎么推?
- DeepEP 的常数小在哪几处?

## Note

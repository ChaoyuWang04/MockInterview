# DistServe：把 Prefill 和 Decode 拆开，才谈得上每卡 goodput

<!-- release-date: 2024-01-18 -->

> 本文依据 Peking University、UC San Diego 与 StepFun 的 **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**，即 OSDI 2024（18th USENIX Symposium on Operating Systems Design and Implementation，July 10–12, 2024, Santa Clara, CA, USA，ISBN 978-1-939133-40-3），arXiv:2401.09670，共 19 页。本地件是 USENIX camera-ready，USENIX 页码 193–211；本文 `(PDF p. N)` 按这份 19 页 PDF 从 1 起编。全文把三件事分开标注：**报告明确写了什么**、**我们如何解释或验算它**、**哪些是外部资料补充**。
>
> 封面第一作者是 Yinmin Zhong 与 Shengyu Liu（Peking University）。其余作者：Junda Chen（UC San Diego）、Jianbo Hu（Peking University）、Yibo Zhu（StepFun）、Xuanzhe Liu 与 Xin Jin（Peking University）、Hao Zhang（UC San Diego）。通讯作者 Xin Jin。代码仓库封面脚注写 [`github.com/LLMServe/DistServe`](https://github.com/LLMServe/DistServe)（PDF p. 2）。按主要归属，目录放在 Peking。
>
> `release-date` 取 **2024-01-18**。这是一篇系统论文，对象从未作为产品对外可用，按「该技术首次官方公开日」取值，即 arXiv v1。OSDI 开会周（2024-07-10）不用于回写。

本站已有两篇相邻工作，边界先钉死，本文不重写：

- [Sarathi-Serve](/reports/Microsoft/Sarathi-Serve) 讲的是**同一张 GPU 上**把 Prefill 切块、无停顿组批。DistServe 明确说 chunked-prefill with piggyback 只能拿 TTFT 换 TPOT，消不掉干扰（PDF p. 3–4）。
- [Mooncake](/reports/Moonshot/Mooncake) 讲的是**生产调度以 KVCache 为中心、过载先拒绝**。DistServe 几乎不谈前缀复用和拒绝策略，它盯的是 **per-GPU goodput** 与并行方案解耦。

## 读前先认识几个词

这篇论文的门槛不在公式，在于它同时用了 GPU 算子和在线服务两套词汇。前四个是通用背景，不全是报告自己定义的。

- **Token（词元）**：模型读写文本的最小单位。
- **Prefill（预填充）**：一次请求的第一段。把整段提示词一次性喂进去，算出所有位置的中间结果，并吐出第一个输出 Token。
- **Decode / Decoding（解码）**：第一段之后的自回归生成。每一步只吃上一个刚生成的 Token，再吐出下一个。
- **KV Cache（键值缓存）**：每个已经处理过的 Token 留下的 Key 和 Value。Decode 每一步都要回看全部历史，所以这些向量被留在显存里。
- **TTFT（Time To First Token，首 Token 延迟）**：请求到达，到第一个输出 Token 出现。报告用它衡量 Prefill 侧 SLO。
- **TPOT（Time Per Output Token，每输出 Token 时间）**：除第一个 Token 外，每个输出 Token 的平均时间。报告用它衡量 Decode 侧 SLO。它和 Sarathi 里的 TBT 接近，但口径是「平均」，不是相邻间隔。
- **SLO attainment（SLO 达成率）**：满足 TTFT 与 TPOT 约束的请求比例。报告默认看 **90%**，附录再报 99%（PDF p. 10、p. 19）。
- **per-GPU goodput**：在给定 SLO 达成率下，**每张已配置 GPU** 能持续扛住的最大到达率。报告把它当成成本指标——每卡 goodput 越高，单次查询成本越低（PDF p. 2）。
- **Instance（实例）**：恰好管一份完整模型权重的资源单元。开了模型并行时，一个实例可以对应多张 GPU。拆开之后就有 Prefill 实例和 Decode 实例，各管一份权重（PDF p. 4）。
- **Intra-op / Inter-op parallelism**：算子内并行（张量并行一类，切矩阵乘，降执行时间，吃带宽）与算子间并行（流水线，切层，近似线性扩吞吐）（PDF p. 3）。

如果你只想记一句话：

> **Prefill 吃算力、Decode 吃带宽。塞进同一张 GPU，两边互相拖。拆到不同 GPU 上，各自配并行、各自扩副本，每卡 goodput 才上得去。**

## 一句话先说清

DistServe 的主张只有一句：**不要把 Prefill 和 Decode 绑在同一组 GPU 上组批。** 拆开之后，干扰没了，资源分配和并行方案也可以按 TTFT、TPOT 各自拧（PDF p. 1–2）。

它不是新的注意力算法，也不是新的显存分配器。它是一层编排：给定模型、负载、延迟目标和达成率，自动给出（PDF p. 7）

- Prefill / Decode 各自的并行方案；
- 两类实例各部署多少份；
- 怎么放到物理集群上，让 KV 传输别把收益吃掉。

评测句写在摘要里，后文实验再拆：相对当时的 SOTA，最多 **7.4× 请求率** 或 **12.6× 更紧的 SLO**，同时 **>90%** 请求仍落在延迟约束内（PDF p. 1）。

## 报告地图：19 页里各写了什么

先摊开篇幅。这是一篇推理服务论文，没有模型配方，也没有训练。密度最高的是第 2–4 节的矛盾、拆分分析和放置算法。

| 报告章节 | PDF 页 | 讲了什么 | 密度 |
|---|---|---|---|
| USENIX 封面 | p. 1 | 标题、作者、OSDI 2024 | 低 |
| 摘要 + 图 1 | p. 1–2 | 7.4× / 12.6×；1.6 vs 5.6 / 10 rps | 高 |
| §1 引言 | p. 2 | goodput；干扰与资源耦合 | 高 |
| §2.1–2.2 | p. 3 | Prefill 算力墙、Decode 带宽墙；连续组批与模型并行 | 高 |
| 图 2 + §2.3 | p. 3–4 | 混一批就两边都慢；chunked-prefill 消不掉 | 高 |
| §3.1 图 3–4 | p. 4–6 | Prefill 批策略；$L_m$；M/D/1；intra vs inter | 高 |
| §3.2–3.3 图 5 | p. 6–7 | Decode 大批；KV 1.13 GB；90 Gbps | 高 |
| §4 算法 1–2 | p. 7–8 | 高/低节点亲和放置；模拟器 | 高 |
| 图 6 + §4.3–§5 | p. 8–9 | 运行时；FCFS；pull KV；6.5K+8.1K 行 | 高 |
| 表 1 图 7 + §6.1 | p. 9–10 | 工作负载与 SLO；4 节点 32×A100 | 高 |
| 图 8–9 + §6.2 | p. 10–12 | Chatbot / 代码补全 / 摘要端到端 | 高 |
| 图 10–12 表 2 + §6.3–6.5 | p. 12–13 | 传输 <0.1%；模拟器误差 <2%；搜索时间 | 高 |
| §7–§9 | p. 13–14 | 吞吐优先、资源不够、长上下文；相关工作；结论 | 中 |
| 参考文献 | p. 14–16 | 引用 | 低 |
| 附录 A–C 表 3 | p. 17–19 | 延迟模型；放置表；99% 达成率 | 高 |

## 先看矛盾：同一张卡上，两个 SLO 在抢

一次请求被切成两段，分界线是第一个输出 Token（PDF p. 2）。整体延迟等于 TTFT 加上 TPOT 乘上后续生成 Token 数（PDF p. 2 脚注）。

不同应用对两边的口味不一样（PDF p. 2）：

- 实时聊天：TTFT 要低，人才能感觉「马上回了」；TPOT 只要快过阅读速度就够——报告写的是约 **250 词/分钟**。
- 文档摘要：更在乎 TPOT，摘要要尽快吐完。

现有系统把两阶段放在同一组 GPU 上，用连续组批把所有用户的 Prefill 和 Decode 揉进同一步，去最大化「全系统每秒吐出多少 Token」（PDF p. 2）。SLO 一紧，就只能超配 GPU。

图 1 把这件事画死（PDF p. 1）。设定是 13B、合成负载、输入 512、输出 64、一张 80GB A100，去模拟「给文章写短摘要」。90% 分位、同时守住两条延迟时，现有共置系统大约只能到 **1.6 rps**。若 Prefill 独占一张卡，能到 **5.6 rps**；Decode 独占一张卡，能到 **10 rps**。报告做了一笔理想账：Prefill 配 2 张、Decode 配 1 张，整体 10 rps，折合每卡 **3.3 rps**，大约是共置的 **2.1×**（PDF p. 2）。

我们怎么读这张图：它不是在说「拆开 magically 变快」，它是在说 **1.6 rps 是被更紧的那条 SLO 卡住的**。共置迫使你用同一套并行、同一池 GPU 去同时伺候两条相反的曲线。

## 第二层矛盾：干扰，加上资源耦合

图 2 用 13B 把混批的代价画出来（PDF p. 4）。对照是「纯 Decode 批」对「再塞进一条 Prefill」。输入 128 和 1024 两档。Decode 被 Prefill 拖长，Prefill 越长拖得越狠；GPU 已经打满时，往 Prefill 里塞 Decode 也会把 TTFT 顶上去。

报告把共置的问题收成三条（PDF p. 3–4）：

**1. Prefill–Decode 干扰。** Prefill 一步常常比 Decode 一步长得多。混在一批里，Decode 的 TPOT 被拉长；Decode 也会给 TTFT 添一截。就算不混批、改成排队，两边仍在抢同一张卡：Prefill 占着，Decode 排队；反过来也一样。给某一侧优先级，另一侧的 SLO 就先破。

**2. Chunked-prefill with piggyback 消不掉干扰。** 报告点名的是把长 Prefill 切块、再捎上若干 Decode（SARATHI / DeepSpeed-MII 那条路，PDF p. 3）。它能减轻 Decode 被整段 Prefill 堵住，但：

- 块太小，Prefill 自己吃不饱 GPU，还要和 Decode 抢；
- 块大到快打满 GPU，留给 Decode 的槽就没了，捎带机会变少；
- 切成 $N$ 块时，前面各块的 KV 要从 HBM 反复搬进 SRAM，总加载量是 $O(N^2)$，不切是 $O(N)$。上下文越长，这个税越重（PDF p. 4）。

**3. 资源与并行耦合。** Prefill 偏算力，紧 TTFT 时更想用 intra-op 把执行时间压下去。Decode 的最优并行跟当前批大小有关。共置只能按「两条 SLO 里更刁的那条」来配，另一条往往被超配（PDF p. 4）。

拆开之后，报告给出的机会是（PDF p. 4）：每类实例一份权重；Prefill 做完把 KV 发给 Decode；Decode 利用率低时，可以多配几份 Prefill 实例对着一份 Decode，把 Decode 批做大。

```mermaid
flowchart LR
    REQ[请求] --> CTL[中心控制器 FCFS]
    CTL --> P[Prefill 实例<br/>盯 TTFT]
    P -->|KV Cache| D[Decode 实例<br/>盯 TPOT]
    D --> OUT[流式输出]
```

这是根据论文图 6 重画的**机制示意**（PDF p. 8），不是实测拓扑。

## Prefill 实例：算力一饱和就别再加批

拆开之后，Prefill 的目标变成：给定到达率，用最少资源把 TTFT 守住（PDF p. 4）。

图 3(a) 是 13B 的 Prefill 吞吐随批大小、输入长度的变化（PDF p. 5）。报告的观察是：一条 **512 Token** 的序列已经能把一张 A100 推到算力墙附近。再往批里加请求，效率不再涨，总时间近似按比例变长，所有人一起慢。于是要先对模型和 GPU 扫出临界长度 $L_m$——超过它，Prefill 就算力受限。只有请求短于 $L_m$ 才考虑组批。实践里提示词常见几百 Token，Prefill 实例的批通常很小（PDF p. 5）。

并行怎么选？图 4 用 **66B、两张 A100**，输入统一 512、泊松到达，对比 intra-op 与 inter-op 的平均 TTFT（PDF p. 5）。低到达率时 intra-op 更好；到达率升高后 inter-op 反过来占优。

报告把拆开后的 Prefill 近似成 **M/D/1 队列**（PDF p. 5）。单卡、FCFS、不组批、执行时间恒为 $D$、到达率 $R$ 且 $RD < 1$ 时：

$$\mathrm{Avg\_TTFT} = D + \frac{RD^{2}}{2(1-RD)}$$

前一项是执行，后一项是排队。两路 inter-op 时，层间激活通信可忽略，$D \approx D_s \approx 2 D_m$，平均 TTFT 变成（PDF p. 5）：

$$\mathrm{Avg\_TTFT_{inter}} = D + \frac{RD^{2}}{4(2-RD)}$$

两路 intra-op 引入加速系数 $K$（$1 < K < 2$，通信让加速不完美），执行时间变成 $D/K$（PDF p. 6）：

$$\mathrm{Avg\_TTFT_{intra}} = \frac{D}{K} + \frac{RD^{2}}{2K(K-RD)}$$

低负载时第一项主导，intra-op 压执行时间更值；高负载时第二项主导，inter-op 更值。TTFT SLO 越紧，越偏向 intra-op；$K$ 掉下去，intra-op 的优势就没了（PDF p. 6、图 4(b)）。

真实提示词长度不均，流水线会起泡，M/D/1 会偏。报告把这件事交给第 4 节的搜索和调度，而不是硬套闭式（PDF p. 6）。

**Takeaway：** Prefill 侧先判断「这条请求有没有打满 GPU」，没打满才组批；并行方案跟着到达率和 TTFT 松紧变，不能和 Decode 共用一把尺子。

## Decode 实例：批要大，紧 TPOT 才上 intra-op

Decode 单条极度带宽受限，组批才是每卡 goodput 的来源（PDF p. 6、图 3(b)）。共置时到达率一高，Prefill 作业变多，Decode 批就做不大——两边在抢 GPU 时间。

拆开之后，可以把多份 Prefill 实例对着一份 Decode，Decode 在专用 GPU 上把批做大，而不必拿 TPOT 去换（PDF p. 6）。批再大，就会撞显存墙（所有在途请求的 KV）。PagedAttention、GQA 以及模型并行，是把 Decode 批继续推向算力墙的手段（PDF p. 6）。报告点名这些工作，本文不重写它们的机制。

图 5 是 13B、批 128、输入 256，看并行度对 Decode 延迟和吞吐（PDF p. 6）。intra-op 能降延迟，但收益递减（通信 + 切完利用率掉）。inter-op 几乎线性扩吞吐。所以：**TPOT SLO 很紧时必须上 intra-op；过了这条线，用 inter-op 线性加吞吐。** 模型单卡装得下时，复制一份权重也是竞争选项——把到达率摊到 $N$ 个副本上，相当于式 (1) 里的 $R$ 换成 $R/N$，代价是多占一份权重显存（PDF p. 6）。

## KV 传得动吗？

OPT-66B、单条 512 Token 的 KV 大约 **1.13 GB**。到达率 10 rps 就要每秒搬 **11.3 GB**，约 **90 Gbps**，才谈得上「传输看不见」（PDF p. 6–7）。报告说现代 LLM 集群常有 InfiniBand（文中举例 **800 Gbps**）；跨节点带宽不够时，就靠节点内 NVLINK——A100 峰值 **600 GB/s**，传输开销仍可忽略（实验在 §6.3）。这件事直接变成放置约束：Prefill 和 Decode 不能随便扔到两台机器上。

## 放置：先按集群带宽分两种算法

目标是最大化 per-GPU goodput（PDF p. 7）。输入包括模型、负载特征、延迟要求和达成率目标。输出叫一份 **placement**：两类实例的并行、副本数、以及怎么落到物理节点。

负载的秒级到达不可预测，但小时到天的模式往往可拟合。DistServe 从历史轨迹拟合分布，再重采样轨迹喂给模拟器，用二分搜索找「刚够达成率」的最大速率（PDF p. 7）。真实打点测 SLO 太贵，所以才上模拟器。

### 高节点亲和：跨节点带宽够，两类实例各自最优再复制

算法 1（PDF p. 7）。节点内 GPU 数记为 $M$，单实例节点上限记为 $N$。枚举可行的 `(inter_op, intra_op)`，权重切完还要装进容量 $C$。对每个配置分别跑 `simu_prefill` / `simu_decode`，留下「每卡 goodput」最高的 Prefill 配置和 Decode 配置，再按目标流量 $R$ 复制：

$$n = \left\lceil \frac{R}{\mathrm{goodput}_p} \right\rceil,\quad m = \left\lceil \frac{R}{\mathrm{goodput}_d} \right\rceil$$

复杂度 $O(NM^{2})$。现代节点常见 $M=8$。最大设定下求解 **不到 1.3 分钟**（PDF p. 7，对应 §6.5）。

模拟器按 Prefill / Decode 各自的 FLOPs 和访存量建延迟模型，细节在附录 A（PDF p. 17）。DNN 推理可预测性高，§6.4 用真机对过。

### 低节点亲和：KV 只能走 NVLINK

跨节点带宽差时，朴素做法是 Prefill 和 Decode 永远放同一节点。大模型会装不下：报告举 **175B ≈ 350 GB** 权重，一对实例就是两份，**8×80 GB = 640 GB < 700 GB**（PDF p. 8）。

算法 2 的关键观察：KV 只在**对应层**之间传。用 inter-op 把层切成 stage，实例切成 segment，**同一 stage 的 Prefill segment 和 Decode segment 放进同一节点**，传输就只能走 NVLINK（PDF p. 8）。节点内，同一实例的各 segment 用同一套并行。节点 GPU 通常 8 张，枚举得完。先枚举 inter-op，再对每个 segment 调 `get_intra_node_configs`，模拟选优，按 goodput 复制到目标流量。

评测集群跨节点只有 **25 Gbps**，所以主实验走低节点亲和算法（PDF p. 9）。正文把引用写成「§2」，按结构应是 §4.2——这是原文笔误，我们按算法编号读。

**Takeaway：** 拆开不是「随便两台机器」。带宽不够时，对齐的是 **stage**，不是整份模型。

## 线上调度：FCFS，再补三刀

图 6：请求进中心控制器，派到队列最短的 Prefill，再派到负载最轻的 Decode（PDF p. 8）。策略本身是 FCFS，针对真实负载加了几刀（PDF p. 8–9）。

**削流水线气泡。** 新 Token 数是批执行时间的可靠代理。Prefill：先扫出打满 GPU 的最短提示词长度 $L_m$，让每批总序列长度靠近 $L_m$——短请求拼在一起，长于 $L_m$ 的单独发。Decode：把 $L_m$ 设成最大批大小。

**扛突发。** 突发会让 KV 洪峰砸向 Decode，显存可能先爆。DistServe 用 **pull 而不是 push**：Decode 按需来 Prefill 取 KV，Prefill 侧 GPU 内存当排队缓冲，Prefill 自己还能继续接活。

**重规划。** 负载画像变了（平均输入输出长度、到达率等），按近期历史重跑放置。算法秒级到分钟级，重载权重「数分钟」，都短于真实负载按小时变的尺度（PDF p. 8）。

明确没做的：抢占、容错。FCFS 在 Prefill 上会有车队效应（长请求堵住短请求）。拆开之后，一份 Decode 挂多份 Prefill，Decode 挂了可能拖垮整片——报告把这两件事都标成未来工作（PDF p. 9）。

## 实现：编排层，不是从零写推理核

四块：放置算法、RESTful 前端、编排层、并行执行引擎（PDF p. 9）。算法 + 前端 + 编排 **6.5K 行 Python**；引擎 **8.1K 行 C++/CUDA**。

前端兼容 OpenAI API，客户端可指定最大输出长度和温度。编排负责派发、传 KV、回结果：跨节点用 NCCL，节点内用异步 `CudaMemcpy`，避免传输堵住计算。每个实例用 Ray actor 当 GPU worker，KV 分布式管理。引擎里集成了连续组批、FlashAttention、PagedAttention，并声称支持 OPT 与 LLaMA（PDF p. 9）。这些集成是报告自己的实现清单，不是对那几篇论文的重写。

## 实验怎么证明

### 集群与负载

4 节点、**32** 张 SXM **A100-80GB**，节点内 NVLINK，跨节点 **25 Gbps**（PDF p. 9）。主实验因此用低节点亲和放置；消融里的高带宽设定走模拟。

模型用 OPT 13B / 66B / 175B，FP16。选 OPT 的经典 MHA，是为了**把 KV 传输压力做大**；报告认为 GQA/MQA 上 KV 更小，DistServe 只会更好看（PDF p. 9）。这是作者方向性判断，不是测出来的数。

到达时间按泊松合成，因为三个数据集都没有时间戳（PDF p. 10）。SLO 是作者按应用经验设的，文中承认当时没有公开的标准 SLO 表（PDF p. 9）。

| 应用 | 模型 | 权重 | TTFT | TPOT | 数据 | 输入/输出均值（图 7） |
|---|---|---|---|---|---|---|
| Chatbot | OPT-13B | 26 GB | 0.25 s | 0.1 s | ShareGPT | 755.5 / 200.3 |
| Chatbot | OPT-66B | 132 GB | 2.5 s | 0.15 s | ShareGPT | 同上 |
| Chatbot | OPT-175B | 350 GB | 4.0 s | 0.2 s | ShareGPT | 同上 |
| 代码补全 | OPT-66B | 132 GB | 0.125 s | 0.2 s | HumanEval | 171.3 / 98.2 |
| 摘要 | OPT-66B | 132 GB | 15 s | 0.15 s | LongBench | 1738.3 / 90.7 |

表 1 与图 7（PDF p. 9）。LongBench 输入被截到 OPT 绝对位置嵌入上限 **2048**（PDF p. 10 脚注）。Chatbot 在三个规模上都测；另外两个应用只测 66B。

基线（PDF p. 10–11）：

- **vLLM**：连续组批 + PagedAttention，共置。只支持 intra-op，按前作把 13B / 66B / 175B 的 intra-op 设成 **1 / 4 / 8**。
- **DeepSpeed-MII**：chunked-prefill，按 token 预算拼块。13B/66B 的 intra-op 与 vLLM 对齐。**175B 跑不了**：`vocab_size=50272`，内核要求 `vocab_size/intra_op` 是 8 的倍数，intra-op=8 不满足；改成 4 会 OOM。

主指标是 90% 达成率下的每卡 goodput，以及把表 1 的两条延迟一起乘 **SLO Scale**（越小越紧）时系统还能撑住的最紧档（PDF p. 10）。

### 端到端：图 8–9 的倍数

Chatbot / ShareGPT（PDF p. 11–12、图 8）：

- 相对 vLLM：**2.0×–4.6×** 请求率；**1.8×–3.2×** 更紧 SLO。
- 相对 DeepSpeed-MII：**1.6×–7.4×** 请求率；**1.7×–1.8×** 更紧 SLO。
- 摘要里的 7.4× 对上的是 DeepSpeed-MII 这一侧。
- vLLM 多数请求 TTFT 还过得去，整体达成率被 **TPOT 违规**拖垮。
- MII 在更大模型上相对好看一些，因为 Prefill 作业更大，切块多少能缓一点干扰；但切块 Prefill 慢于整段 Prefill，TTFT 会先破。

175B 的放置被单独点出来（PDF p. 11）：Prefill **inter-op=3, intra-op=3**，Decode **inter-op=3, intra-op=4**。附录表 3 把 intra-op 写成 TP、inter-op 写成 PP，同一组数是 Prefill TP=3 PP=3、Decode TP=4 PP=3（PDF p. 19）。报告认为这种非对称很难手调，用来证明搜索有用。

代码补全 / HumanEval、OPT-66B（PDF p. 12、图 9(a)）：相对 vLLM **5.7×** 率、**1.4×** 更紧 SLO；相对 MII **1.6×** 率、**1.4×** 更紧 SLO。作为实时助手，两边最终都卡在 **TTFT**。DistServe 靠去掉 Decode 干扰、搜索时给 Prefill 加 intra-op，把 Prefill 平均延迟压下去。

摘要 / LongBench、OPT-66B（PDF p. 12、图 9(b)）：相对 vLLM **4.3×** 率、**12.6×** 更紧 SLO；相对 MII **1.8×** 率、**2.6×** 更紧 SLO。输入很长，但 TTFT SLO 松（15 s），矛盾转到 TPOT。vLLM 共置时长 Prefill 把 Decode 打毛。摘要里的 12.6× 对上的是 vLLM 这一侧。

附录 C、99% 达成率（PDF p. 19、图 13–14）：相对 vLLM 仍有 **3×–8×** 率、**1.24×–6.67×** 更紧 SLO；相对 MII **1.32×–8×** 率、**1.20×–1.58×** 更紧 SLO。正文说更严的达成率目标下 DistServe「可以更好看」，附录是证据。

附录表 3 其余放置（PDF p. 19）：13B ShareGPT 是 Prefill TP=2 PP=1、Decode TP=1 PP=1；66B 的 ShareGPT / LongBench / HumanEval 都是 Prefill TP=4 PP=1、Decode TP=2 PP=2。

### 传输是不是暗税

图 10 把 OPT-175B / ShareGPT 的生命周期拆成五段：Prefill 排队、Prefill 执行、传输、Decode 排队、Decode 执行（PDF p. 12）。即便 175B，KV 传输仍占系统总时间 **不到 0.1%**。绝对时间的 CDF：三个 OPT 上都有 **超过 95% 请求传输延迟 < 30 ms**——尽管跨节点只有 25 Gbps。原因正是算法 2：同一 stage 放同一台机器，走 NVLINK（PDF p. 12）。

### 消融：拆开本身比「给 vLLM 搜并行」更重要

表 2 用真机对照模拟器的 SLO 达成率（PDF p. 13）。vLLM 与 DistServe-Low 在 1.0–4.0 req/s 多档上，误差 **小于 2%**。例如 1.0 req/s：vLLM 真机 97.0% / 模拟 96.8%；DistServe-Low 两边都是 100.0%。2.0 req/s：vLLM 52.8% / 51.0%；DistServe-Low 两边都是 99.3%。

图 11 在模拟里比较四个系统、OPT-66B / ShareGPT（PDF p. 13）。**vLLM++** 枚举并行后仍等于默认 vLLM——intra-op=4 已经是每卡 goodput 最好的。报告据此说：共置下的干扰，会把「换并行」的空间吃掉。**DistServe-High**（算法 1，少约束、假设跨节点带宽高）还能再好一截，因为不再要求同一节点上 Prefill/Decode 共享同一 model stage。

注意：§6.5 正文有一处把算法编号和 High/Low 写反了（PDF p. 13 写「Alg. 1 (DistServe-Low)」）。按 §4 的定义，**High = 算法 1，Low = 算法 2**。图 12 横轴是提供给单实例的 GPU 数 $N \times M$，搜索时间随 GPU 数涨、与模型大小无关；算法可并行。GPU 变多时 Low 比 High 更慢，因为要枚举节点内 Prefill/Decode 组合。即便如此仍在「分钟级」，每次重部署跑一次可接受。§4.1 的「最大设定 <1.3 分钟」与这里一致。图 12 纵轴最高刻度是 80 s 量级，精确秒数正文没写。

## 附录 A：模拟器在算什么

附录把软融合之后的 GEMM 当成延迟主体（PDF p. 17）。符号：$h$ 隐层、$n$ 头数、$s$ 头维（$h=n\cdot s$）、$m$ FFN 中间维；张量并行时 $h,n,m$ 要除以并行度。批侧：$B$ 批大小，$t$ 批内 Token 总数，$t_2$ 各请求长度平方和，$b$ 是 FlashAttention 的 block size。

Prefill 里 QKV / Attn Output / FFN 的算术强度是 $O(t)$。A100-80GB 上 AI 超过 **156** 就算力受限；真实 $t$ 常到几百，这四类 GEMM 按 FLOPs 建模：

$$T_1 = C_1 \cdot (4th^{2} + 2thm)$$

FlashAttention 的 Prefill 注意力按请求分别 launch。报告推得单头接近访存墙（$b=16$ 时 AI≈10.677，$b=32$ 时 ≈21.333），整层写成 $T_2 = C_2 \cdot 3ht_2 / b$。合并：

$$T_{\mathrm{Prefill}} = C_1\cdot(4th^{2}+2thm) + C_2\cdot\frac{3ht_2}{b} + C_3$$

$C_3$ 吃 Python 运行时和噪声。$C_1,C_2,C_3$ 靠 profiling 和插值。

Decode 侧那四类 GEMM 的 AI 是 $O(B)$，$B$ 被显存和 TPOT 卡住，按访存建模 $T_3 = C_4\cdot(4h^{2}+2hm)$。Decode 注意力也是访存墙：$T_4 = C_5\cdot 3ht$。合并时不再另加噪声项，因为 $4h^{2}+2hm$ 已是常数，开销进 $C_4$（PDF p. 17）。

这些公式是模拟器用的，不是线上调度器每步在解的式子。

## 报告自己划的边界

§7（PDF p. 13–14）写了三条「这时别用 DistServe」：

- **吞吐优先、不太在乎延迟的离线任务。** goodput 不是目标，chunked-prefill with piggyback 可能把每步填到算力墙，利用率更高。
- **资源极少（几张甚至一张 GPU）。** 拆开的设计空间几乎没了，共置系统部署更简单。
- **超长上下文。** KV 随长度线性涨，传输绝对值会变大；但 Prefill 计算按平方涨，传输相对 Prefill 的占比反而可能下降。两边算力画像差距会更大，干扰更狠。报告判断拆开在长上下文里「仍然有希望」——这是论证，不是 1M 上下文实验。

相关工作里，Orca / vLLM / SARATHI / FastServe 被归成共置因而有干扰。并发的 Splitwise、TetriInfer、DéjàVu 也被写成同类拆分，报告自称更强调 goodput 和网络带宽（PDF p. 14）。AlpaServe 被写成非自回归 multiplexing。这些比较是作者定位，不是我们的评测。

## 论文之后发生了什么（外部补充）

下面不是 2024-01 这份 PDF 的内容。

拆开 Prefill/Decode 后来成了推理集群的主流叙事之一。vLLM、SGLang 以及多家厂商的生产栈都出现了 PD 分离或类似的实例角色。Mooncake 把调度中心换成 KVCache，并加上过载拒绝——那是下一站，不是 DistServe 已经做了的。Sarathi 的切块组批也进入了共置引擎。两边后来常被**组合**：分离的实例内部仍可以切块。PDF 里把切块写成「消不掉干扰」的共置修补；今天的系统不一定还把这两招当成互斥。

代码仓封面写的是 `LLMServe/DistServe`。开源生态后来的主路径并不是大家都来跑这份研究原型，而是把「拆开 + 按 SLO 搜并行」写进各自的编排层。仓库现状要以当时/现在的 GitHub 为准，不要把论文实现细节当成今天的生产默认。

## 可迁移启发

1. **先问优化目标是 goodput 还是吞吐。** 没有 TTFT/TPOT 约束时，拆开可能亏利用率；有双 SLO 时，共置的「系统 Token/s」会骗人。
2. **干扰和耦合是两件事。** 就算调度上把 Prefill、Decode 错开，只要还在同一组 GPU、同一套并行，紧的那条 SLO 仍会逼你超配。
3. **并行方案按阶段分开搜。** Prefill 低负载偏 intra-op，高负载偏 inter-op；Decode 先靠批，TPOT 紧再上 intra-op。vLLM++ 消融说明：不拆开，搜并行也救不了。
4. **KV 传输是放置问题，不是「RDMA 口号」。** 先算 GB/s，再决定跨节点还是对齐 stage 吃 NVLINK。25 Gbps 的集群上，对齐 stage 仍能把 175B 的传输压到 <0.1%。
5. **用 pull 吸收突发。** Prefill 侧显存当缓冲，比把 KV 推爆 Decode 更稳。
6. **一张 GPU 或纯离线吞吐，别硬套这篇。** 报告自己把这两类场景让给共置和切块。

## 关键词回看

- **per-GPU goodput**：90%（或 99%）双 SLO 下每卡最大 rps，直接对应单次查询成本。
- **Prefill–Decode 干扰**：混批或抢同一卡，TTFT 与 TPOT 互相抬。
- **资源耦合**：共置迫使两阶段共用并行与 GPU 数。
- **$L_m$**：Prefill 打满 GPU 的临界提示词长度；调度用它拼批、削气泡。
- **High / Low node-affinity**：跨节点带宽够则两类实例独立最优；不够则按 pipeline stage 共节点，KV 走 NVLINK。
- **Pull 传输**：Decode 按需取 KV，Prefill 内存当队列。

## 参考资料

- 原件：`papers/Peking/DistServe.pdf`（USENIX OSDI 2024 camera-ready，19 页）
- arXiv：<https://arxiv.org/abs/2401.09670>
- USENIX：<https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin>
- 代码（封面脚注）：<https://github.com/LLMServe/DistServe>
- 本站相邻、本文不重写：[Sarathi-Serve](/reports/Microsoft/Sarathi-Serve)、[Mooncake](/reports/Moonshot/Mooncake)、[PagedAttention](/reports/Berkeley/PagedAttention)

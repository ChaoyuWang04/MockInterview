---
difficulty: 困难
topic: MoE并行与DeepEP/DeepEP原理
summary: DeepEP 的原理,为什么 MoE 下比朴素 all2all 快
tags: [真题, 待校对, MoE, DeepEP, 集合通信]
company:
mastered: false
highfreq: false
---

## 题目

DeepEP 了解吗?大概说下原理?为啥在 MoE 下用 DeepEP 更快?

## 要点

- 定位一句话:不是更快的通用 all-to-all,是**把 MoE 的先验写进了通信实现**
- 两套 kernel 对应 prefill 与 decode 两种瓶颈,不能用同一套
- 非对称域带宽转发:跨节点只发一份,到岸再用快链路转发
- 转发能成立的前提是 node-limited routing,架构与系统是配套的
- 不占 SM 的重叠对 decode 比带宽更值钱

## 答案

DeepEP 是 DeepSeek 开源的专家并行通信库,提供的就是 MoE 的 **dispatch 与 combine 这一对 GPU kernel**。定位要说准:**它不是一个更快的通用 all-to-all,而是把 MoE 的先验知识写进了通信实现里。**

### 两套 kernel,对应两个阶段

| | 高吞吐 kernel | 低时延 kernel |
|---|---|---|
| 面向 | 训练与推理 prefill | 推理 decode |
| 关键设计 | 非对称域带宽转发;可控制占用多少 SM | 纯 RDMA 路径,把延迟压到最短 |
| 重叠方式 | 与计算抢 SM,靠手工划分 SM 配额 | 基于 hook 的重叠,**不占用任何 SM** |
| 数据布局 | 连续 layout:token 紧密排布、每专家一段 | masked layout:每专家固定槽位 + 掩码,**形状静态,可被 CUDA Graph 捕获** |

分两套的理由:prefill 一批几千个 token,包大、看的是带宽;decode 一步只有几十上百个 token,包小到带宽根本用不满,唯一重要的是**这一跳要多少微秒**。一个求吞吐一个求延迟,不可能同一套 kernel 都做到最好。

### 非对称域带宽转发

集群里两级链路:节点内快、跨节点慢。朴素 all-to-all 完全无视这个差异——一个 token 要去某节点上的 3 张卡,它就**跨节点发 3 份**。

DeepSeek-V3 的做法是:跨节点**只发一份**,先沿慢链路送到目标节点里"同槽位"的那张卡,到岸后立刻沿节点内快链路转发给真正的目标 GPU。慢链路流量从"每个目标 GPU 一份"降到"**每个目标节点一份**"。

这套转发要成立,得有人保证"目标节点数"不失控——那就是 **node-limited routing**:每个 token 最多落到 4 个节点。V3 报告给的配套数字是,在这个约束下每 token 平均仍能在每节点选到 3.2 个专家,几乎不损失路由自由度;并称只需 **20 个 SM** 就足以打满两级链路的带宽。**架构侧限扇出,系统侧才敢做转发**,两件事是配套的。

### 相对朴素 all-to-all 好在哪

1. **跨节点去重**:慢链路上的字节按"节点"而非"GPU"计
2. **FP8 dispatch**:送进低精度 GEMM 的输入降到 FP8,字节直接减半(combine 保持 BF16)
3. **decode 段不占 SM**:hook 式重叠把搬运交给网卡 RDMA 引擎,decode 本来 SM 就吃紧,这一条比带宽更值钱
4. **masked layout**:形状静态,decode 段能进 CUDA Graph——而动态形状正是朴素 all-to-all 上不了图的原因

通信量与 all-to-all **同阶**,赢的是常数和可重叠性,不是复杂度。

## 知识点

两套 kernel 的分工、非对称域带宽转发、node-limited routing 与转发的配套关系、FP8 dispatch + BF16 combine、hook 式不占 SM 重叠、masked layout 与 CUDA Graph。

## 追问

- deepep 相对 all2all 的好处?通信量是同阶还是更低?
- FP8 dispatch 配 BF16 combine,为什么 combine 不也降精度?
- 它依赖 NVSHMEM 与 RDMA,线上运维要额外操心哪些配置?

## Note

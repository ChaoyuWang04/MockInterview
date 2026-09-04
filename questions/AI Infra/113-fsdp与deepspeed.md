---
difficulty: 简单
topic: FSDP/与DeepSpeed对比
summary: FSDP与DeepSpeed怎样比较并按训练需求选型
tags: [面经, 待校对, FSDP, DeepSpeed, 分布式训练]
company: 阿里云
mastered: false
highfreq: false
---

## 题目

比较PyTorch FSDP与DeepSpeed在架构、显存、通信、卸载、并行组合、易用性和生态上的差异。什么情况下优先选FSDP，相关wrap、prefetch与CPU offload怎样调？

## 要点

- FSDP是PyTorch原生参数分片组件，DeepSpeed是包含ZeRO、卸载和并行能力的训练系统
- 二者都能分片模型状态，性能差异来自配置、拓扑、版本和模型形状
- wrap粒度过细通信频繁，过粗峰值显存高
- prefetch用额外在途显存换通信隐藏，CPU offload用PCIe带宽换GPU显存
- 先按现有训练栈和必需能力选，再用同一工作负载压测

## 答案

**没有“FSDP一定更快”或“DeepSpeed只适合超大模型”的通用结论。** FSDP更像PyTorch原生的状态分片组件；DeepSpeed提供ZeRO、卸载、优化器及多种并行能力，系统范围更大。具体能力随版本变化，选型要以当前软件栈为准。

| 维度 | FSDP | DeepSpeed |
|---|---|---|
| 集成 | 与PyTorch module、autograd和state dict贴近 | 由engine和配置驱动，改造面通常更大 |
| 显存 | FULL_SHARD等策略分片参数、梯度和优化器状态 | ZeRO-1/2/3逐级分片，并有多种offload路径 |
| 并行组合 | 可与TP、PP组合，但需训练栈协调 | 常见于ZeRO加TP/PP的完整方案 |
| 调试与生态 | 原生工具链通常更直接 | 功能多，配置和版本组合也更多 |

如果团队已经使用原生PyTorch、模型结构规则、无需特殊优化器或复杂流水线，通常先试FSDP；若必须使用成熟的CPU/NVMe卸载、DeepSpeed优化器或既有3D并行平台，DeepSpeed迁移成本可能更低。

### 三个关键旋钮

- **auto wrap**：以Transformer block为常见边界。过细会产生大量all-gather/reduce-scatter；过粗会提高峰值显存。用每个FSDP单元大小和通信trace压测。
- **forward/backward prefetch**：提前取下一单元参数以覆盖通信，但增加同时驻留的参数与显存；只有执行顺序稳定且通信确为瓶颈时才开。
- **CPU offload**：两者都可把部分状态移到CPU，但生命周期、拷贝时机和checkpoint接口不同。它解决容量，不保证更快；PCIe与CPU内存带宽常成为新瓶颈。

混合TP/PP时先满足总卡数与张量形状约束，让高频TP通信尽量留在快速互联域，再用FSDP/ZeRO分片数据并行副本。最终用相同模型、batch、checkpoint策略比较峰值显存、每step时间和恢复成本。

## 知识点

FSDP、ZeRO、auto wrap、参数预取、CPU/NVMe offload、TP/PP/FSDP混合并行。

真实面经：[B002-G01-Q011](../../docs/references/面经原题.md#b002-g01-q011)、[B002-G01-Q029](../../docs/references/面经原题.md#b002-g01-q029)。

老师参考：[P005-Q011](../../docs/references/平台题/P005-Infra-001-030.md#p005-q011)、[P005-Q029](../../docs/references/平台题/P005-Infra-001-030.md#p005-q029)。

## 追问

以下均为平台页面参考追问，不作为面经原话：

- 参考追问：auto_wrap_policy怎样避免通信过碎或峰值显存过高？
- 参考追问：forward/backward prefetch怎样隐藏通信，代价是什么？
- 参考追问：FSDP和DeepSpeed的CPU offload有何实现边界？
- 参考追问：超大集群中哪些DeepSpeed能力值得保留？
- 参考追问：TP、PP与FSDP混合时怎样安排并行域？

## Note

---
difficulty: 困难
topic: 访存与算子优化/分块大小与occupancy
summary: 分块开大开小、shared memory 占用多少分别有什么影响
tags: [面经, 待校对, occupancy, 分块, 寄存器]
company:
mastered: false
highfreq: false
---

## 题目

分块分大了会有什么影响?分小了会有什么影响?shared memory 占用多了、少了又分别有什么影响?

## 要点

- 答案不是「越大越好」或「越小越好」,而是两个方向都有明确代价
- 核心机制:资源占用 ↑ → 驻留 warp ↓ → 延迟掩盖能力 ↓
- 能报出寄存器堆和 shared 的具体额度,把「跷跷板」算给面试官看
- 正确答法是先分 bound 再给结论
- 实操上靠 occupancy 计算器 + profiler,不靠脑补

## 答案

### 一、跷跷板

| | 分块开大 / shared 占用多 | 分块开小 / shared 占用少 |
|---|---|---|
| 数据复用 | 好,每个元素被用更多次 | 差,同一份数据反复从 HBM 读 |
| HBM 访存量 | 少 | 多 |
| 寄存器 / shared 占用 | 高 | 低 |
| 驻留 warp 数(occupancy) | **低** | **高** |
| 延迟掩盖能力 | 弱,一停就没人补位 | 强,总有别的 warp 顶上 |
| 尾部效应 | 明显,块少了不够填满所有 SM | 轻微 |
| 典型风险 | 寄存器溢出到 local memory,反而暴慢 | 访存量爆炸,带宽打满但算力闲置 |

### 二、机制串起来

每 SM 的寄存器堆是 65536 个 32 位寄存器,要让 2048 个线程全驻留,**每线程平均只能用 32 个寄存器**;shared 每 SM 约 164 KB,一个 block 用掉 48 KB 就只能同时驻留 3 个 block。

一句话:**分块开大 = 每个线程扛更多数据 = 寄存器和 shared 都吃得更多 = 能同时驻留的 warp 变少 = 一个 warp 卡在等数据时没人补位。**

### 三、正确答法:先看是哪种 bound

- **访存受限**的算子(elementwise、LayerNorm、attention 的 decode 阶段):瓶颈是带宽,要的是「很多 warp 同时排队取数把带宽打满」→ 倾向**小块、高 occupancy**
- **计算受限**的算子(大 GEMM、prefill 阶段):瓶颈是算力,要的是「每个线程在片上复用足够多数据、别让 Tensor Core 停」→ **50% 甚至 30% 的 occupancy 反而更快**,因为双缓冲已经在指令级把延迟藏住了

**occupancy 不是越高越好**,它只是「掩盖延迟的能力」,不是性能本身。

### 四、实操

不靠脑补:先用 occupancy 计算器算资源上限,再用 profiler 看真实的 achieved occupancy、带宽利用率和 warp stall 原因,两头对齐了再改代码。

## 知识点

occupancy 的定义与计算、寄存器堆与 shared memory 额度、资源占用与驻留 warp 的跷跷板、按 bound 分类给结论、尾部效应、寄存器 spill。

## 追问

- 对于 GEMM 怎么 tune?tune 哪些参数?
- achieved occupancy 低是不是一定有问题?
- 寄存器 spill 到 local memory 为什么这么致命?
- 什么是尾部效应 / wave quantization,怎么规避?

## Note

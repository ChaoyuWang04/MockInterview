---
difficulty: 简单
topic: 性能分析与Profiling/工具分工
summary: Nsight 是什么、能干什么,和 torch.profiler 的区别
tags: [面经, 待校对, Nsight, profiling, 性能分析]
company:
mastered: false
highfreq: false
---

## 题目

nsight 工具是什么?能用来干什么?和 torch.profiler 的区别是什么?

## 要点

- Nsight 是两个工具:Systems(时间线)和 Compute(单 kernel 计数器),别混为一谈
- 三者是望远镜、显微镜、体温计的关系,不是竞品
- 一句话概括三者:时间去哪了 / 这一刀为什么钝 / 这笔账记在哪个算子头上
- 采集开销差异很大,ncu 不能对整个训练开
- 关键区别是系统语义 vs 框架语义

## 答案

### 一、三件工具的分工

| | **Nsight Systems**(nsys) | **Nsight Compute**(ncu) | **torch.profiler** |
|---|---|---|---|
| 看哪一层 | 整条时间线:CPU 线程、CUDA 流、kernel、内存拷贝、NCCL | **单个 kernel 内部**的硬件计数器 | 框架算子(op)层 |
| 回答什么 | 时间去哪了?谁在等谁? | 这个 kernel 为什么慢? | 哪个算子 / 哪行 Python 最贵? |
| 关键产出 | gap、CPU-GPU 重叠、launch 间隙、通信与计算是否并行 | SM 吞吐、DRAM 吞吐、occupancy、warp stall、L1/L2 命中率 | 算子耗时排行、CPU vs CUDA 时间、chrome trace |
| 采集开销 | 低(采样 + API 拦截),可跑真实训练 | **很高**(同一 kernel 重放多次凑计数器),只能挑单个 kernel | 中(每个 op 打点),会放大 CPU 侧开销 |
| 能归因到 Python | 要手工打 NVTX 标记 | 不能 | **原生支持**(`with_stack=True`) |
| 什么时候用 | **第一步,永远先跑它** | 已知某个 kernel 是热点之后 | 想知道「是哪个算子」而不是「哪个 kernel」 |

一句话概括:**nsys 看「时间去哪了」,ncu 看「这一刀为什么钝」,torch.profiler 看「这笔账记在哪个算子头上」。**

### 二、区别的本质:系统语义 vs 框架语义

torch.profiler 和 nsys 的时间线其实来自同一套底层(CUPTI)。差别在于:torch.profiler 多了**框架语义**(知道这个 kernel 属于哪个算子、哪一行 Python),nsys 多了**系统语义**(知道 CPU 线程在干什么、NCCL 什么时候动)。

**卡在框架里就用 torch.profiler,卡在系统里就用 nsys。**

### 三、nsys 的时间线上找四类信号

1. **大段空白(gap)**:GPU 在白烧钱,优先级最高。往上看同一时刻 CPU 在干嘛
2. **密密麻麻的短 kernel**:方块之间的缝隙和方块本身差不多宽 = launch bound
3. **同步点**:CUDA API 行上出现很长的 `cudaStreamSynchronize`
4. **通信没重叠**:NCCL kernel 和计算 kernel 首尾相接排成一条

> **一个必须知道的坑**:`nvidia-smi` 的 GPU-Util **不是利用率**,它只统计「采样窗口内有 kernel 在执行的时间比例」——哪怕只用了 1 个 SM 也显示 100%。真正的利用率要看 ncu 的 SM / DRAM Throughput。

## 知识点

nsys / ncu / torch.profiler 的分层定位、CUPTI 底座、系统语义与框架语义、trace 上的四类信号、GPU-Util 的误判。

## 追问

- ncu 的 SM Throughput 和 DRAM Throughput 都低,说明什么?
- warp stall reason 有哪些,分别怎么对症?
- 怎么只 profile 训练的第 N 个 step,而不是把整个训练都录下来?
- achieved occupancy 低是不是一定有问题?

## Note

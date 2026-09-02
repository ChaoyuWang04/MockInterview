---
difficulty: 简单
topic: GPU架构与执行模型/SIMD与SIMT
summary: SIMD 与 SIMT 的差异,以及 warp divergence
tags: [面经, 待校对, SIMT, warp, GPU架构]
company:
mastered: false
highfreq: false
---

## 题目

SIMD 和 SIMT 的差异是什么?

## 要点

- SIMT 的定义:一条指令,32 个线程各自拿自己的数据执行
- 四个维度的对比:编程视角、数据地址、遇到分支、灵活性
- 一句话概括:SIMT 是「带了自动掩码和地址自由度的 SIMD」
- 代价是 warp divergence,两个分支串行相加
- 能说清 thread / warp / block / grid 与硬件的对应

## 答案

### 一、先把线程组织讲清楚

| 软件概念 | 硬件对应 | 关键性质 |
|---|---|---|
| thread | CUDA Core 上的一条执行流 | 有自己的寄存器 |
| **warp(32 线程)** | **调度的最小单位** | 32 个线程锁步执行同一条指令 |
| block | 被分配到**一个 SM** 上 | 块内可用 shared memory 通信、可同步 |
| grid | 整个 GPU | block 之间不能直接通信 |

**SIMT(Single Instruction, Multiple Threads)**:一条指令,32 个线程各自拿自己的数据去执行。

### 二、和 SIMD 的四点差异

| | SIMD(CPU 向量指令) | SIMT(GPU) |
|---|---|---|
| 编程视角 | 要显式操作「向量」 | 写的是**单个线程**的标量代码 |
| 数据地址 | 必须连续、对齐 | 每个线程可访问**任意地址**(但连续时最快) |
| 遇到分支 | 靠掩码,程序员/编译器处理 | 硬件自动处理,但会**串行执行两个分支** |
| 灵活性 | 低 | 高,代价是硬件复杂 |

一句话概括:**SIMT 是「带了自动掩码和地址自由度的 SIMD」**——你可以像写单线程一样写代码,硬件替你并行。

### 三、代价:warp divergence

因为一个 warp 的 32 个线程**必须执行同一条指令**,当它们走进不同的 if 分支时,硬件只能先让走 A 分支的干活、其余关掉,再让走 B 分支的干活。**两个分支串行执行,时间相加。**

```cuda
// 坏:同一个 warp 里的线程走不同分支,耗时 = 两个分支之和
if (threadIdx.x % 2 == 0) { do_a(); } else { do_b(); }

// 好:让分支在 warp 之间发生,而不是 warp 之内
if ((threadIdx.x / 32) % 2 == 0) { do_a(); } else { do_b(); }
```

这就是优化清单里「减少分支」的真实含义:**不是不能有 if,而是别让 if 在一个 warp 内部劈叉**。

## 知识点

SIMT 定义、warp 是调度最小单位、SIMD/SIMT 四点差异、warp divergence 与改写手法、thread/warp/block/grid 层级。

## 追问

- 为什么 SIMT 的地址自由度是「可以任意,但连续时最快」?
- 一个 warp 里只有 1 个线程活跃,这个 warp 的开销是多少?
- Volta 引入独立线程调度之后,warp 内还是严格锁步的吗?
- block 一旦被分配给某个 SM 会迁移吗?这带来什么后果?

## Note

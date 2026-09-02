---
difficulty: 简单
topic: GPU架构与执行模型/计算单元
summary: Tensor Core 与 CUDA Core 是什么,能不能并行、怎么并行
tags: [面经, 待校对, TensorCore, CUDACore, GPU架构]
company:
mastered: false
highfreq: false
---

## 题目

Tensor Core 是什么?CUDA Core 是什么?两者可以并行吗?怎么并行?

## 要点

- 四个维度对比:干什么、粒度、精度、算力差距
- 「能并行」——它们是 SM 内两套独立的执行流水线,指令可交错发射
- 高性能 GEMM 的常规操作:CUDA Core 干搬运和后处理,让 Tensor Core 不停
- 但两者共享寄存器堆和访存带宽,所以不是算力简单相加
- 加分项:Tensor Core 的形状是硬约束

## 答案

### 一、两者的分工

| | CUDA Core | Tensor Core |
|---|---|---|
| 干什么 | 通用标量运算(FP32/INT32 的加减乘除) | **专做小矩阵乘加**($D = A \times B + C$) |
| 粒度 | 一次一个数 | 一次一小块矩阵(如 16×16) |
| 精度 | FP32/FP64 为主 | FP16/BF16/FP8/INT8 输入,FP32 累加 |
| 算力 | 基准 | **同代下高一个数量级** |

### 二、能并行吗?能

它们是 SM 内**两套独立的执行流水线**,指令可以交错发射:一个 warp 在 Tensor Core 上算矩阵乘的同时,另一个 warp 可以用 CUDA Core 做地址计算或激活函数。

这正是高性能 GEMM kernel 的常规操作 —— **用 CUDA Core 干搬运、地址计算和后处理,让 Tensor Core 一刻不停地算**。CUTLASS 里的 warp specialization(生产者 warp 只搬数、消费者 warp 只算)就是把这件事做到了极致。

### 三、但「并行」是有条件的

两者**共享同一份寄存器堆和访存带宽**。所以并行是**指令层面的重叠,不是算力简单相加**;真正的瓶颈往往还是喂数据的速度——Tensor Core 算得再快,数据搬不上来一样空转。

### 四、加分项:Tensor Core 的形状是硬约束

它只做固定尺寸的小矩阵乘加,由一个 warp 的 32 个线程协同发射一条指令完成:`wmma` 在 FP16 下是 16×16×16,`mma.sync` 的 FP16 主力形状是 m16n8k16(INT8/FP8 上有 m16n8k32)。

**后果:$M$、$N$、$K$ 必须补齐到指令粒度。** decode 阶段 $M=1$ 时也得按 16 行算,**理论利用率上限只有 6.25%** —— 所以很多 decode kernel 干脆放弃 Tensor Core,用 CUDA Core 写专门的 GEMV。

## 知识点

CUDA Core 与 Tensor Core 的分工、两套独立流水线与指令交错、共享寄存器堆与带宽、mma 形状约束、decode 阶段的 Tensor Core 利用率。

## 追问

- 为什么 Tensor Core 的输入是 FP16 而累加用 FP32?
- 什么是 ldmatrix,为什么喂 Tensor Core 需要它?
- 一个 kernel 同时跑满 Tensor Core 和 CUDA Core,瓶颈会出现在哪?
- decode 阶段放弃 Tensor Core 用 CUDA Core,这个取舍怎么算账?

## Note

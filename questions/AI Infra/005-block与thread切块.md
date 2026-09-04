---
difficulty: 中等
topic: 访存与算子优化/切块方法论
summary: block 与 thread 的切块方法论,block 开多大合适
tags: [真题, 待校对, CUDA, 切块, occupancy]
company:
mastered: false
highfreq: false
---

## 题目

写 kernel 时 block 和 thread 的切块方法论是什么?block 开多大合适?怎么把线程映射到数据上?

## 要点

- 三条硬约束先划定可选范围(32 倍数、每 SM 驻留上限、每 block 1024 线程)
- 经验值 128/256 要能说出三条理由,不能只报数字
- 映射原则:让 `threadIdx.x` 走在数据最内层的连续维度上
- grid-stride loop 的写法与三个好处
- 块开太小的代价:occupancy 被驻留 block 数卡死

## 答案

### 一、三条硬约束(A100 / 计算能力 8.0)

- **必须是 32 的整数倍**。warp 是 32 线程的调度单位,block 开 100 个线程会被凑成 4 个 warp(128 线程),最后一个 warp 里 28 条通道全程空转——白扔 21% 的执行槽
- **一个 SM 最多驻留 32 个 block、64 个 warp**。所以 block 只有 32 线程时,32 个 block 也才 32 个 warp,**occupancy 卡死在 50% 上不去**——这是「块开太小」的第一个代价
- **一个 block 最多 1024 线程**,且必须整个装进一个 SM

### 二、经验值 128 或 256

它们同时满足三点:

1. 是 32 的倍数
2. 2048(每 SM 最大线程数)能被整除,不会在最后一批留下余数
3. 数量适中,块内 `__syncthreads()` 的等待代价还不大(线程越多,最慢的那个拖累的人越多)

### 三、怎么映射到数据

原则一句话:**让 `threadIdx.x` 走在数据最内层的连续维度上**。因为 `threadIdx.x` 相邻的线程属于同一个 warp,只有它们读到相邻地址才能触发合并访存。矩阵按行主序存储时,就该让 x 方向对应列号、y 方向对应行号,而不是反过来。

### 四、grid-stride loop:让 kernel 和数据规模脱钩

朴素写法是「一个线程处理一个元素」,grid 大小必须随 n 变;改成「一个线程处理一串元素」:

```cuda
__global__ void add(int n, const float* a, const float* b, float* c) {
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;     // 步长 = 整个 grid 的线程总数
    for (int i = idx; i < n; i += stride)    // 每个线程处理多个元素
        c[i] = a[i] + b[i];
}
```

三个好处:

1. **grid 大小可以固定成「刚好填满 GPU」**(如 SM 数 × 每 SM 能驻留的 block 数),不随 n 抖动,启动开销和尾部效应都稳定
2. **访存仍然连续**——步长是线程总数,同一个 warp 每一轮读的还是相邻的 32 个元素
3. 线程复用带来的循环体外初始化(累加器、常量)只做一次

## 知识点

warp 是 32 线程的调度单位、每 SM 的 block/warp 驻留上限、block size 经验值、线程到数据的映射原则、grid-stride loop。

## 追问

- 分块分大了、分小了分别有什么影响?
- 为什么 `threadIdx.x` 一定要走连续维度?反过来会怎样?
- grid 开多大算「刚好填满 GPU」,这个数怎么算?
- 什么是尾部效应 / wave quantization?

## Note

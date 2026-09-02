---
difficulty: 中等
topic: CUDA流与异步执行/异步拷贝机制
summary: cudaMemcpyAsync 背后的 copy engine 与 stream 语义
tags: [面经, 待校对, CUDA流, DMA, 异步传输]
company:
mastered: false
highfreq: false
---

## 题目

`cudaMemcpyAsync` 背后的机制是什么?

## 要点

- CPU 只是把任务塞进队列就返回,GPU 何时执行 CPU 一概不知
- 真正干活的是独立的 DMA copy engine,不占 SM,所以能和计算重叠
- 数据中心卡通常 H2D / D2H 各一个引擎
- stream 的两条规则决定顺序:流内按序、流间无序
- 两个坑:pinned 是前提;默认流是 blocking stream
- 并发是「允许」不是「承诺」

## 答案

### 一、CPU 只是「把菜单贴到挂单栏」

调用时 CPU 把这次传输的描述塞进目标 stream 的队列就**立刻返回**,GPU 什么时候真开始搬、搬没搬完,CPU 一概不知道。天生异步的还有 kernel launch、`cudaMemsetAsync`;天生同步的是 `cudaMemcpy`(不带 Async)、`cudaDeviceSynchronize`、`cudaStreamSynchronize`。

这么设计是因为:**下发一次任务的 CPU 侧开销在几微秒量级**,而很多 kernel 本身也只跑几微秒。每下发一个就等它跑完,时间会被往返吃掉一大半。

### 二、真正干活的是 copy engine,不是 SM

GPU 上除了跑 kernel 的 SM 阵列,还有专门的 **DMA 引擎(copy engine)**,专职在 host 内存和显存之间搬数据。**搬数据这件事从头到尾不占用 SM**,所以传输和计算在物理上就是可以同时发生的。

数据中心卡通常配**两个 copy engine,H2D 和 D2H 各一个方向**(`cudaDeviceProp::asyncEngineCount` 报告数量),所以「上传下一批数据 + 取回上一批结果 + SM 算当前这批」三件事可以真正同时跑。

### 三、stream 语义决定顺序

1. **同一条 stream 内,操作严格按下发顺序执行**(in-order 队列)
2. **不同 stream 之间没有任何顺序保证**,硬件资源允许时就并发

所以想让传输和计算重叠,**两者必须在不同的 stream 上**,并且 host 内存必须是 pinned。

### 四、两个坑

- **pinned 是前提**:从可分页内存发起时驱动要先拷到暂存区,这个调用**可能退化为同步**,重叠直接消失
- **默认流是 blocking stream**:不指定 stream 时进的是 legacy NULL stream,往它里面下发会先等所有其它 blocking stream 干完、反之亦然。开了几条 stream 做重叠,中间夹一个没指定 stream 的操作,**所有流被这一下全串起来**,而代码看不出任何毛病。出路是 `cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`,或编译时加 `nvcc --default-stream per-thread`

### 五、并发是「允许」,不是「承诺」

开 4 条 stream 不等于快 4 倍——如果每个 kernel 本身就把 SM 占满了,它们照样排队串行。多流真正有价值的场景是:**每个任务都吃不满 GPU**(小 batch decode、大量小算子),或者**任务用的是不同硬件单元**(计算 vs 拷贝 vs 通信)。

## 知识点

异步下发与队列深度、DMA copy engine 独立于 SM、asyncEngineCount、stream 的两条规则、legacy NULL stream、并发的前提条件。

## 追问

- 异步传输一定要用 pinned memory 吗?不用会怎样?
- 开了多流为什么没变快,怎么在 profiler 上确认?
- 怎么表达跨流依赖而不阻塞 host?
- 传输和计算重叠的理论上限是多少?

## Note

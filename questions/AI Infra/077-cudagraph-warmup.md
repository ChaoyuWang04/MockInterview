---
difficulty: 中等
topic: CudaGraph/warmup
summary: 推理为什么要 warmup,warmup 在干什么,不做会有什么后果
tags: [面经, 待校对, warmup, CUDA Graph]
company:
mastered: false
highfreq: false
---

## 题目

推理时为啥一般会做 warmup?warmup 在干啥?

## 要点

- warmup 不只是「跑热身让性能稳定」,它在干三件必须发生在图外面的事
- 三件事:触发一次性初始化、让显存分配进入稳态、稳定时钟频率
- 不做的后果分两档:一次性动作被录进图白跑,或直接让捕获失败
- 官方要求 warmup 跑在**侧流**上,别污染将被捕获的那条流

## 答案

### warmup 在干三件事

1. **触发一次性初始化**:cuBLAS / cuDNN handle 创建、GEMM 算法启发式选择与 autotune 选核、Triton 或 `torch.compile` 的 JIT 编译、NCCL 通信域建立
2. **让显存分配进入稳态**:第一次走到的路径会向 allocator 要新块(可能触发真正的 `cudaMalloc`);warmup 之后这些块已在缓存里,捕获时直接命中
3. **稳定时钟频率**:GPU 从低功耗态爬到稳定频率,前几次的耗时数据本来就不可信

### 不 warmup 会怎样

上面这些动作会被**录进图里**:

- **轻则**图中多了一堆只该做一次的操作,每次重放都白跑
- **重则**它们内部含有 host 侧同步(autotune 要测时间、handle 创建要等驱动),而捕获模式下遇到同步会**直接让捕获失败**

所以 warmup 的本质是:**把所有「只该发生一次」和「需要 host 参与」的动作,赶在捕获开始前做完**。

### 一个容易漏的细节

PyTorch 官方额外要求 warmup 必须跑在**侧流**上(建一条 stream、`wait_stream` 对齐后在其中跑几步,结果丢弃),就是为了不污染将被捕获的那条流。跑几步则看场景,单卡几步即可,DDP 场景官方要求 11 次。

即使不上 CUDA Graph,warmup 也仍有意义——JIT 编译与 autotune 的耗时不应该算进第一批线上请求的延迟里,否则 TTFT 的首个样本会异常高。

## 知识点

一次性初始化(handle / JIT / autotune / NCCL)、allocator 稳态、时钟爬频、侧流 warmup、捕获期禁止同步。

## 追问

- warmup 要跑多少步够?怎么判断已经进入稳态?
- 为什么必须在侧流而不是当前流上做?
- 服务启动时的 warmup 和捕获前的 warmup 是同一件事吗?

## Note

---
difficulty: 困难
topic: GEMM优化/调参空间
summary: GEMM kernel 的调参空间有哪些,各参数调大调小的后果
tags: [真题, 待校对, GEMM, CUTLASS, 调参]
company:
mastered: false
highfreq: false
---

## 题目

对于 GEMM 怎么 tune?具体 tune 哪些参数?

## 要点

- 调参空间要成体系:三层 tile 尺寸 + 流水线级数 + K 方向切分 + block 调度顺序
- 每个参数都能说出「调大什么代价、调小什么代价」
- thread tile 超过 255 寄存器会 spill,这是硬边界
- 两类 quantization(tile / wave)必须提
- 知道现实中是库在 tune,以及为什么没有一套参数通吃

## 答案

**调参空间 = 三层 tile 尺寸 + 流水线级数 + K 方向切分 + block 调度顺序。**

| 参数 | 调大的后果 | 调小的后果 |
|---|---|---|
| **block tile BM/BN** | 算术强度 ∝ tile 边长而升,HBM 流量下降;但 shared/寄存器占用上升、occupancy 下降,尾块浪费变大,矩阵小时 block 数不足填不满 SM | 访存量成倍上升,容易退回访存受限;好处是 block 多、负载均衡好、尾块浪费小 |
| **BK(K 方向厚度)** | 每次 mainloop 干的活更多,`__syncthreads` 被摊薄,单次传输更宽更高效;但 shared 用量 ∝ Stages×(BM+BN)×BK,会挤掉流水级数 | 省 shared、能开更多 stage;但同步更频繁、每次传输太窄,带宽利用率下降 |
| **warp tile WM/WN** | 每 warp 复用更好、重复读 shared 更少;但块内 warp 数减少,调度弹性差、寄存器压力大 | warp 多、延迟掩盖好;但同一份 shared 数据被更多 warp 重复读,shared 带宽成瓶颈 |
| **thread tile TM/TN** | 计算访存比 $\frac{TM \cdot TN}{TM+TN}$ 上升,寄存器复用更充分;但累加器 ∝ TM×TN,**超过 255 寄存器就 spill 到显存,性能崩塌** | 寄存器宽松、occupancy 高;但外积摊不开,shared 读放大 |
| **Stages(流水级数)** | 更能掩盖 global→shared 延迟;但 shared 占用线性上升、驻留 block 数下降,延迟盖满后**收益饱和** | shared 省、occupancy 高;=1 时退化为无双缓冲,搬和算串行 |
| **split-K** | M、N 小时把 K 切给多个 block,并行度上升填满 SM;但要额外 workspace + 归约 kernel,用原子加时结果**不可复现** | 无归约开销、数值确定;但 M、N 小时 grid 太小,大量 SM 闲置 |
| **swizzle(block 调度顺序)** | 分组越大,并发 block 在 M/N 上越聚集,越可能命中 L2 里同一条 A/B 条带;过大则工作集超出 L2 反而下降 | 退化成朴素 row-major 遍历,并发 block 分散,L2 命中率低 |

### 两个必知的坑

- **tile quantization(尾块量化)**:$M$ 或 $N$ 不是 tile 尺寸整数倍时,边缘 block 大部分线程在算无效数据。$M=257$、$BM=128$ 要开 3 行 block,最后一行只有 1/128 有用
- **wave quantization(波次量化)**:block 总数不是 SM 数整数倍时,最后一「波」只用到少数 SM。A100 有 108 个 SM,**109 个 block 的耗时约等于 216 个**

### 现实中谁在 tune

cuBLAS 内部有预编译 kernel 集合 + 启发式选择;CUTLASS 用 profiler 对模板实例穷举;Triton 用 autotune 声明候选 config 实测挑选;`torch.compile` 的 max-autotune 会把两类候选一起 benchmark。**没有一套参数通吃所有 shape,所以「选 kernel」本身才是库的核心竞争力。**

## 知识点

三层分块尺寸、multistage 流水级数、split-K、threadblock swizzle、tile / wave quantization、autotune。

## 追问

- 为什么 swizzle 能提高 L2 命中率?
- split-K 用原子加为什么结果不可复现?怎么办?
- Stages 加到多少就饱和了,怎么判断?
- decode 阶段 M=1 的瘦长矩阵该怎么调?

## Note

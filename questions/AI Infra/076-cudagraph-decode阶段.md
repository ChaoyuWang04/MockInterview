---
difficulty: 中等
topic: CudaGraph/推理中的使用阶段
summary: 大模型推理里哪个阶段、哪部分操作会用 CUDA Graph
tags: [面经, 待校对, CUDA Graph, decode]
company:
mastered: false
highfreq: false
---

## 题目

大模型推理时,一般哪个阶段、哪部分操作会使用 cudagraph?

## 要点

- 只用在 decode,三条理由分别对应 launch 占比、可分档、prefill 形状多变
- 天然可捕获的算子:线性层、norm、激活、RoPE、残差
- 最难纳入的是 attention,要 backend 专门改造
- 实践是**整个 decode step 一起捕获**,不是只捕获一部分

## 答案

### 阶段:只用在 decode

三条理由正好是前面各考点的收口:

1. **launch 占比最高**——decode 每步只产 1 个 token,单个 kernel 极小、数量极多,CPU 下发时间可能比 kernel 自己还长
2. **形状可以冻结**——decode 的形状只由 batch 决定,能按 batch 分档,每档一张图
3. **prefill 不合适**——输入长度千变万化,分档要么档位爆炸、要么 padding 到 max_len 白烧算力,所以通常直接不上图

### 操作:哪些天然能进图

**线性层(QKV 投影、MLP)、LayerNorm / RMSNorm、激活、RoPE、残差加**——这些算子的形状**只由 batch 决定**,天然满足冻结要求。

最难纳入的是 **attention**:它的行为依赖每条序列的实际长度和 block 表,必须由 backend 改造成「形状固定 + 元数据全走设备端张量」才能被捕获。现代引擎的图友好 backend 已经做到这点,所以实践中是**整个 decode step 一起捕获**,而不是只捕获前后几段——只捕获一部分反而会在图与 eager 的交界处重新引入下发开销和同步。

顺带一提采样:top-p / top-k 这类含动态 shape 或 host 分支的实现同样会破图,要么改成定长掩码写法留在图内,要么放到图外单独下发。

## 知识点

decode 阶段的 launch-bound 性质、按 batch 分档、图友好 attention backend、整步捕获、采样路径的图兼容性。

## 追问

- attention backend 要改造哪些地方才能被捕获?
- prefill 上图有没有可能?chunked prefill 呢?
- 一次只捕获一层和捕获整步,差别在哪?

## Note

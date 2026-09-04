---
difficulty: 中等
topic: vLLM/PagedAttention 与连续批处理
summary: vLLM 怎样联合管理 KV、调度请求和优化执行
tags: [面经, 待校对, vLLM, PagedAttention, 连续批处理]
company: 美团、蚂蚁
mastered: false
highfreq: false
---

## 题目

请解释 vLLM 的核心加速机制，并说明如何联合 PagedAttention、连续批处理、CUDA 执行优化和配置调优提高吞吐、控制 TTFT/TPOT；若目标是每秒 2000 个 token，应怎样验证和排查？

## 要点

- KV Cache 负责避免重复投影，PagedAttention负责分配和寻址
- 连续批按迭代加入、退出请求，提高批次有效工作量
- token 预算、KV 空间、CUDA Graph/kernel 和并行共同决定性能
- 吞吐目标必须绑定模型、硬件、输入输出长度、并发和 SLO

## 答案

**vLLM 的主要收益来自“把有限 KV 显存分得更灵活，再让调度器持续用新请求填满每轮执行”。**

PagedAttention把每条序列的逻辑 KV 块映射到显存池中的物理块，按需分配，避免为最大长度预留整段连续空间；共享前缀或并行候选时才可能共享块并触发写时复制。连续批处理在每次模型迭代后移出已完成请求、加入新请求，减少短请求等待长请求的空转。CUDA Graph、融合 kernel 和合适 dtype 则降低每轮执行开销。

调优顺序是：先固定模型、GPU数、输入/输出长度分布、并发和 TTFT/TPOT SLO；再观察排队、每轮 token 数、KV 使用率、prefill/decode 比例、GPU 利用率、kernel 间隙和多卡通信。吞吐不足若伴随 GPU 空闲，多半在调度、CPU 准备或请求不足；GPU 已满则看算子、量化或扩卡；KV 接近耗尽则限制长请求、调整 token 预算或启用可验证的前缀缓存。

块太小会增加块表和管理开销，太大会增加尾块浪费，需随模型、head size 和 workload 测。长序列要限制单请求占用，避免挤掉大量短请求。TP/PP只有模型或吞吐确需多卡时才用，通信可能增加延迟。FP8权重量化与 KV cache dtype 是两项独立设置；版本参数会变化。TGI、TensorRT-LLM 与 vLLM 应在同一负载下比较，不能给固定倍数结论。投机解码和 Prefix Caching 只在接受率或前缀复用率足够时收益明显。

## 知识点

PagedAttention、逻辑/物理块、连续批处理、token budget、TTFT、TPOT、CUDA Graph、Prefix Caching。

- 真实面经：[B002-G01-Q040](../../docs/references/面经原题.md#b002-g01-q040)、[B002-G01-Q057](../../docs/references/面经原题.md#b002-g01-q057)、[B002-G01-Q067](../../docs/references/面经原题.md#b002-g01-q067)、[B002-G01-Q115](../../docs/references/面经原题.md#b002-g01-q115)、[B002-G01-Q174](../../docs/references/面经原题.md#b002-g01-q174)
- 老师答案参考：[P005-Q040](../../docs/references/平台题/P005-Infra-031-060.md#p005-q040)、[P005-Q057](../../docs/references/平台题/P005-Infra-031-060.md#p005-q057)、[P005-Q067](../../docs/references/平台题/P005-Infra-061-090.md#p005-q067)、[P005-Q115](../../docs/references/平台题/P005-Infra-091-120.md#p005-q115)、[P005-Q174](../../docs/references/平台题/P005-Infra-151-180.md#p005-q174)

## 追问

- 页面参考追问：如果压测达不到每秒 2000 token，排查顺序是什么？
- 页面参考追问：vLLM 的调度器与 TGI、TensorRT-LLM 如何比较？
- 页面参考追问：长文本占满显存时，PagedAttention 和连续批怎样调度？
- 页面参考追问：PagedAttention 的块大小如何影响浪费和执行开销？
- 页面参考追问：PagedAttention 相比静态分配节省多少显存，何时收益最大？
- 页面参考追问：vLLM 还有哪些优化手段，如投机解码和 Prefix Caching？
- 页面参考追问：vLLM 多卡 TP/PP 的通信代价是什么？
- 页面参考追问：实际部署中怎样排查 OOM 和长尾延迟？
- 页面参考追问：PagedAttention 与 FlashAttention 有什么区别和关系？
- 页面参考追问：怎样共同优化 TTFT 和 TPOT？

## Note

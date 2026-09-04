---
difficulty: 困难
topic: 连续批处理/CascadeAttention
summary: CascadeAttention与连续批处理怎样衔接并复用公共前缀
tags: [真题, 待校对, CascadeAttention, 连续批处理, KVCache, 推理引擎]
company:
mastered: false
highfreq: false
---

## 题目

Cascade Attention 应该在 Continuous Batching 之前还是之后执行?请说明推理调度、公共前缀识别、前缀/后缀 attention 计算和结果合并之间的关系。

## 要点

- Continuous Batching 是调度层,Cascade Attention 是本轮模型执行中的 kernel 路径
- 调度器先确定本轮请求集合,才能判断哪些请求共享公共前缀
- 公共前缀只算一次,各请求后缀分别计算,再用 LSE 正确合并 attention state
- 是否启用取决于公共前缀长度、batch、后端支持和数值/图捕获约束

## 答案

**逻辑顺序是先由 Continuous Batching 形成本轮 batch,再在模型执行时决定是否走 Cascade Attention。** 两者不在同一层:前者每个 iteration 选择哪些请求和 token 上卡,后者优化这些已选请求的 attention。

当本轮多个请求共享一段很长的前缀时,普通 batch decode 会为每个请求重复读取和计算同一段 prefix KV。Cascade Attention 把 attention 拆成两部分:公共前缀对整个子批只计算一次,每个请求的私有后缀分别计算;两边都返回输出和 log-sum-exp(LSE),再按各自归一化权重合并。这个合并在实数数学上与对完整 KV 一次做 softmax attention 等价,不是删掉前缀信息。

所以若问题只问“之前还是之后”,回答“调度之后、attention kernel 内部”最准确。Continuous Batching 会不断改变 batch 成员,公共前缀长度与是否值得走 cascade 也要每轮重算或更新元数据。只有前缀足够长、共享请求足够多时,少读的 KV 才能覆盖额外的拆分和合并开销。

具体支持是版本和后端相关的:还要检查 attention backend、滑窗或混合 attention、KV dtype、CUDA Graph 兼容性和数值误差。不能把某个引擎当前启发式当成 Cascade Attention 的算法定义。

## 知识点

Continuous Batching、共享前缀、attention state、log-sum-exp、递归注意力。

- 依据:[FlashInfer Recursive Attention](https://docs.flashinfer.ai/tutorials/recursive_attention.html)、[vLLM FlashAttention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)。

## 追问

- 为什么前缀 attention 和后缀 attention 的输出不能直接相加?
- batch 里只有一个请求,或公共前缀很短时,Cascade Attention 为什么可能变慢?
- Continuous Batching 改变 batch 成员后,公共前缀元数据怎样更新?

## Note

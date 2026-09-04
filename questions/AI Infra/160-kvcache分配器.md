---
difficulty: 困难
topic: PagedAttention/KVCache分配器
summary: 怎样设计KVCache块分配器并保证复用、回收与抢占正确
tags: [真题, 待校对, KVCache, PagedAttention, 内存管理, 推理引擎]
company:
mastered: false
highfreq: false
---

## 题目

如果让你设计一个推理引擎的 KV Cache allocator,需要哪些核心数据结构和操作?请说明请求进入、增量生成、前缀复用、分叉写入、结束回收与显存不足时怎样保持正确性。

## 要点

- 物理块池与每请求逻辑块表分离,分配单位按 token block 而非整条序列
- 空闲队列、引用计数和前缀哈希分别负责快速分配、共享生命周期和查重
- 共享块写入前要 copy-on-write,完成或抢占时按引用释放
- 分配失败必须交给调度器延期或抢占,不能覆盖仍被引用的块

## 答案

**allocator 管“哪些物理 KV 块归哪个请求”,attention kernel 再按块表读写。** 最小设计包含四类状态:固定大小的物理块池、空闲块队列、每个请求的逻辑块表,以及块的引用计数;做前缀缓存时再加“前缀哈希 → 物理块”的索引和淘汰顺序。

请求 prefill 或 decode 前,先根据新增 token 与 lookahead 计算所需块数。最后一块还有槽位就续写,不够才从空闲队列取块并追加到请求块表;新 K/V 通过 slot mapping 写到具体位置。命中完整前缀块时只增加引用并复用,请求分叉后若要改写共享块则执行 copy-on-write。

请求完成时逐块减少引用计数,归零后才回空闲池。前缀缓存可让“可复用但当前无人占用”的块继续留在哈希表中,真正缺块时按 LRU 等策略淘汰其缓存身份并重新分配。显存不足时 allocator 应返回“无法满足”,由调度器延期准入或抢占请求;不能偷偷覆盖活跃块。被抢占请求可以把 KV 换出,也可以丢弃后在恢复时重新 prefill。

实现要守住三个不变量:同一物理块不能被两个写者无保护修改;引用计数与块表更新必须一致;失败分配不能留下半更新状态。并发调度下可由单一调度线程串行修改元数据,或使用锁与事务式回滚。测试应覆盖尾块、重复释放、共享后写入、抢占恢复和块池耗尽。

## 知识点

物理块池、逻辑块表、slot mapping、引用计数、copy-on-write、前缀缓存、抢占。

- 参考实现:[vLLM KVCacheManager](https://docs.vllm.ai/en/stable/api/vllm/v1/core/kv_cache_manager/)、[Hybrid KV Cache Manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/)。

## 追问

- block size 过大或过小分别怎样影响碎片、块表和 kernel 访存?
- 前缀缓存中的块引用归零后,为什么不一定立刻从哈希表删除?
- 如果改成手撕代码,你会怎样定义 allocator 的最小接口和不变量测试?

## Note

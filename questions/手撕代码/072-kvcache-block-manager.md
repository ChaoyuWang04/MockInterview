---
difficulty: 困难
topic: 推理引擎/KVCache块管理器
summary: 手写 PagedAttention 的块管理器,含分配复用与回收
tags: [真题, 待校对, 手撕代码, PagedAttention, KVCache, 引用计数]
company:
mastered: false
highfreq: false
---

## 题目

实现一个 KV Cache 的 paged attention block manager。要求支持分配、追加、前缀复用、分叉写入和回收,并说明每个不变量为什么必须成立。

## 要点

- 四类状态:物理块池、空闲队列、每请求逻辑块表、块引用计数
- 追加只在尾块槽位耗尽时才取新块,否则续写
- 前缀命中靠哈希,复用时只加引用计数不拷贝
- 共享块被写入前必须 copy-on-write
- 分配失败要返回失败交调度器处理,不能覆盖活跃块

## 答案

**最小实现只需四类状态:固定大小的物理块池、空闲块队列、每个请求的逻辑块表、以及每块的引用计数。** 前缀缓存再加一张"前缀哈希 → 物理块"的索引。

```python
class BlockManager:
    def __init__(self, num_blocks, block_size):
        self.bs = block_size
        self.free = list(range(num_blocks))      # 空闲物理块
        self.ref = [0] * num_blocks              # 引用计数
        self.tables = {}                         # req_id -> [物理块号]
        self.lens = {}                           # req_id -> 已用 token 数
        self.hash2block = {}                     # 前缀哈希 -> 物理块

    def _acquire(self):
        if not self.free:
            raise OutOfBlocks                    # 交给调度器,不抢占活跃块
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def append(self, req, n_tokens):
        """为新增 token 扩容;尾块还有槽位就续写,否则取新块"""
        table, used = self.tables[req], self.lens[req]
        need = (used + n_tokens + self.bs - 1) // self.bs - len(table)
        new = [self._acquire() for _ in range(need)]   # 先全拿到再提交
        table.extend(new)
        self.lens[req] = used + n_tokens

    def reuse_prefix(self, req, prefix_hashes):
        """命中的完整前缀块只加引用,不拷贝"""
        for h in prefix_hashes:
            b = self.hash2block.get(h)
            if b is None:
                break
            self.ref[b] += 1
            self.tables[req].append(b)

    def cow(self, req, idx):
        """写共享块前先复制,避免污染其他请求"""
        b = self.tables[req][idx]
        if self.ref[b] == 1:
            return b                             # 独占,直接写
        nb = self._acquire()
        copy_block(b, nb)
        self.ref[b] -= 1
        self.tables[req][idx] = nb
        return nb

    def free_req(self, req):
        for b in self.tables.pop(req):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)
        self.lens.pop(req)
```

**三个不变量,每个都对应一类线上事故:**

1. **同一物理块不能被两个写者无保护修改** —— 所以有 `cow`。前缀复用后两个请求生成的内容不同,直接写会互相污染。
2. **引用计数与块表必须同步更新** —— 计数漏减会永久泄漏块,漏加会导致块被提前回收给别人,产生串话。
3. **失败的分配不能留下半更新状态** —— 所以 `append` 里先把需要的块全部拿到再提交到块表;中途失败要把已取的块还回去。

`_acquire` 失败时抛出而不是就地抢占,是因为**该抢占谁是调度器的策略**:延期准入、换出 KV、还是丢弃后重新 prefill,块管理器没有信息判断。

设计层面的完整讨论见 [KVCache 分配器](../AI%20Infra/160-kvcache分配器.md),固定分块的取舍见 [PagedAttention 固定分块](../AI%20Infra/146-pagedattention固定分块.md)。

## 知识点

物理块池与逻辑块表分离、引用计数生命周期、前缀哈希与复用、copy-on-write、分配的原子性、块管理器与调度器的职责边界。

## 追问

- 前缀哈希怎么算才能避免碰撞导致串话?
- 可复用但无人占用的块,按什么策略淘汰?
- 多线程并发调用时,这些状态怎样保护?
- 尾块只用了一半就结束,这部分浪费怎么估算?

## Note

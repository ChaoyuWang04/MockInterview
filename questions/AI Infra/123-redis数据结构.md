---
difficulty: 简单
topic: 缓存系统/Redis数据结构
summary: Redis MGET与ZSET怎样实现并处理高并发
tags: [面经, 待校对, Redis, 缓存系统, ZSET]
company: 百度
mastered: false
highfreq: false
---

## 题目

Redis常用数据结构有哪些？重点说明MGET的执行和集群边界，以及ZSET在Redis 7中的内部编码、复杂度和高并发优化，并与B+树范围查询比较。

## 要点

- Redis的逻辑类型与底层编码需分开回答
- 单实例MGET用一次协议往返完成多个字典查找，复杂度约O(K)
- Redis Cluster多key命令通常要求key位于同一slot，跨slot不能保证原子MGET
- Redis 7小ZSET用listpack，大ZSET通常用dict加skiplist
- ZSET点查、插入和范围查询的复杂度不同，不能统一写O(log N)

## 答案

**Redis常用逻辑类型有String、Hash、List、Set、ZSet、Stream和Bitmap/HyperLogLog等；同一种逻辑类型会按元素数量和大小选择不同底层编码。**

### MGET

单实例收到 `MGET k1 ... kK` 后，在一个命令执行周期内依次对每个key做字典查找并组装回复，时间约为 $O(K)$。它的收益主要是**一次网络往返和一次协议解析**，不是把K次查找变成一次哈希。

Redis Cluster会按slot路由。标准多key命令通常要求所有key位于同一slot，可用hash tag让相关key同槽；跨slot时客户端只能拆成多次请求并汇总，这不是一个原子MGET。高并发下应限制单次key数量，避免大回复长期占用事件循环。

### ZSET

Redis 7中，小且紧凑的ZSET可用 **listpack** 顺序保存member与score；超过配置阈值后通常转为：

- `dict`：member到score的近似 $O(1)$ 查找；
- `skiplist`：按score和member排序，插入/删除及定位约 $O(\log N)$，范围输出再加返回元素数 $M$，即 $O(\log N+M)$。

旧版本资料常写ziplist；Redis 7应使用listpack口径。listpack省指针和对象开销，适合小集合；skiplist+dict占内存更多，但大集合更新、点查和有序范围更稳定。

B+树分支因子大、节点适合页式存储，磁盘/SSD范围扫描更友好；Redis skiplist实现简单、内存更新灵活，适合内存数据库。高并发优化还包括分片热点ZSET、缩小成员、限制大范围返回、批量pipeline，以及避免在单个热点key上执行重操作。

## 知识点

Redis逻辑类型与底层编码、MGET、Cluster hash slot、listpack、dict、skiplist、范围查询复杂度。

真实面经：[B002-G01-Q025](../../docs/references/面经原题.md#b002-g01-q025)。

老师参考：[P005-Q025](../../docs/references/平台题/P005-Infra-001-030.md#p005-q025)。

## 追问

以下均为平台页面参考追问，不作为面经原话：

- 参考追问：Redis Cluster中跨slot的MGET能否保证原子性？
- 参考追问：Redis 7中listpack何时转换为skiplist+dict？
- 参考追问：ZSET与B+树在内存更新和磁盘范围扫描上怎样取舍？

## Note

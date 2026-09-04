---
difficulty: 简单
topic: 数据系统/Spark广播Join
summary: Spark什么时候广播小表,怎样避免大表shuffle并控制内存风险
tags: [面经, 待校对, Spark, SQL, Broadcast Join, AQE]
company: 哔哩哔哩
mastered: false
highfreq: false
---

## 题目

在Apache Spark中执行JOIN操作时，当一张表较大而另一张表较小时，系统通常会采用何种优化策略（如广播小表）来提升性能？请说明该策略的触发条件、底层实现原理及其对数据倾斜和网络传输的影响。

## 要点

- Broadcast Hash Join 把小表发送到执行查询的各 executor
- 每个 executor 构建本地哈希表，大表分区直接 probe，避免大表按键 shuffle
- 是否触发取决于统计信息、广播阈值、连接类型、hint 与内存
- 广播仍有 BroadcastExchange、序列化和每 executor 一份内存成本
- AQE 可依据运行时统计调整计划，但不是无条件切换

## 答案

**一大一小两张表 JOIN 时，常用 Broadcast Hash Join：广播小表，让大表留在原分区本地扫描。** 它省掉的是大表按连接键重分区的 shuffle，不是让网络传输完全消失。

### 底层过程

1. Driver 侧执行小表对应的子计划并生成广播关系。
2. 小表经过序列化和 BroadcastExchange，被分发到参与查询的 executors。
3. 每个 executor 在本地为小表构建哈希表。
4. 大表的每个分区本地扫描，用连接键 probe 这张哈希表并输出结果。

大表不用跨节点重排，所以当小表确实足够小时，网络、排序和磁盘 spill 都可能明显减少。代价是小表要传到多个 executor，并在每个 executor 的内存里保留一份。

### 什么时候触发

Spark 会参考统计信息和 `spark.sql.autoBroadcastJoinThreshold`，也可由 SQL hint 或 DataFrame 的 `broadcast()` 指定。能否广播还取决于连接类型、哪一侧能作为 build side、估算是否可靠以及 executor 内存。

默认阈值和具体行为会随 Spark 版本、配置与 AQE 设置变化。面试里应说“检查当前版本配置和 `EXPLAIN` 物理计划”，不要背一个永久有效的 MB 数字。

### 倾斜会怎样

广播小维表后，大表不再按 JOIN key shuffle，因此由大表热点键引起的 shuffle 分区倾斜通常会缓解。但它解决不了两件事：

- 热点键本身产生海量匹配结果，单个 task 仍要做很多 probe 和输出；
- 小表同一键有多行时，JOIN 结果仍会乘法膨胀。

这时还要从数据粒度、热点键拆分、预聚合或业务过滤上处理，不能只换 JOIN 算法。

### 超阈值能不能强制广播

可以使用 `BROADCAST` hint 或 `broadcast(df)`，但先估算**广播后的真实对象大小**和 executor 并发内存，而不是只看落盘压缩大小。强制后用 `EXPLAIN` 确认物理计划，并观察广播耗时、超时和 executor 峰值。若每个 executor 放不下，强制广播会把 shuffle 问题变成 OOM。

当两边都大、广播侧不稳定，或 JOIN 输出本身很大时，Sort-Merge Join 往往更稳，因为它可以分区处理并借助磁盘 spill。最终选择要用实际分布和计划验证。

### AQE 做了什么

AQE 在运行时拿到 shuffle 分区统计后，可以合并小分区、处理部分倾斜，并在满足条件时把原计划转换为广播连接。它仍受统计、广播阈值、连接语义和内存限制；“开了 AQE”不等于所有 JOIN 都会自动变成最优计划。

## 知识点

Broadcast Hash Join、build/probe、BroadcastExchange、shuffle、数据倾斜、AQE 与物理计划。

- 真实面经：[B002-G01-Q090](../../docs/references/面经原题.md#b002-g01-q090)
- 老师参考：[P005-Q090](../../docs/references/平台题/P005-Infra-061-090.md#p005-q090)

## 追问

- 参考追问：小表略超广播阈值但内存充足时，如何强制触发广播 JOIN？
- 参考追问：广播 JOIN 与 Sort-Merge JOIN 在数据倾斜场景下如何选择？
- 参考追问：Spark 3.0 的自适应查询执行（AQE）如何动态优化 JOIN 策略？

## Note

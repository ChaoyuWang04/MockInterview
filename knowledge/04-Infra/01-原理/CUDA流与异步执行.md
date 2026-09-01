# CUDA流与异步执行

> 🚧 占位:本篇待撰写。写作标准见 docs/04-知识库写作契约.md,样板见「GPU架构与执行模型」。

## 计划覆盖

- stream 是什么、stream 之间怎么并行
- 同步点的危害;什么场景消同步点收益最大;怎么消
- 异步传输一定要 pinned memory 吗?不用的后果
- cudaMemcpyAsync 背后发生了什么
- 通信与计算 overlap 的底层依赖

## 相关文献

- 待补(arxiv 编号必须联网核实,严禁编造)

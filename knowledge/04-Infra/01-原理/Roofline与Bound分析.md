# Roofline与Bound分析

> 🚧 占位:本篇待撰写。写作标准见 docs/04-知识库写作契约.md,样板见「GPU架构与执行模型」。

## 计划覆盖

- Roofline 模型:算术强度 vs 峰值算力/带宽
- 怎么判断一个算子是 compute-bound 还是 memory-bound
- 量化之后 bound 会不会迁移?(权重变小、访存变少,可能从访存受限转计算受限)
- prefill 与 decode 分别落在 roofline 的哪一段
- 实测方法:profiler 指标怎么读

## 相关文献

- 待补(arxiv 编号必须联网核实,严禁编造)

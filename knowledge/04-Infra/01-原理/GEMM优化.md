# GEMM优化

> 🚧 占位:本篇待撰写。写作标准见 docs/04-知识库写作契约.md,样板见「GPU架构与执行模型」。

## 计划覆盖

- 朴素实现 → 分块 → 双缓冲 → Tensor Core 的演进
- 为什么 GEMM 是 compute-bound 的典型
- cuBLAS / CUTLASS 的分层抽象
- 小 batch / 瘦长矩阵为什么效率骤降(与 decode 阶段的关系)

## 相关文献

- 待补(arxiv 编号必须联网核实,严禁编造)

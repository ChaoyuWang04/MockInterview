# Prefill与Decode的矩阵形状

> 🚧 占位:本篇待撰写。写作标准见 docs/05-知识库写作契约.md,样板见「GPU架构与执行模型」。

## 计划覆盖

- prefill:大矩阵乘,compute-bound
- decode:瘦长矩阵/GEMV,memory-bound
- attention 在两阶段的形状差异与 kernel 选择
- batch 大小如何改变两阶段的 bound
- 这一篇是 Roofline 分析的具体案例

## 相关文献

- 待补(arxiv 编号必须联网核实,严禁编造)

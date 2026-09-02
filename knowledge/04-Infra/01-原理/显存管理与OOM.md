# 显存管理与OOM

> 🚧 占位:本篇待撰写。写作标准见 docs/05-知识库写作契约.md,样板见「GPU架构与执行模型」。

## 计划覆盖

- 显存占用构成:权重 / 激活 / KVCache / 碎片
- PyTorch caching allocator 与显存碎片
- CUDA Graph 的显存池为什么不能和常规显存池共用
- 开 CUDA Graph 为什么更容易 OOM
- decode 阶段 OOM 的典型场景与排查顺序

## 相关文献

- 待补(arxiv 编号必须联网核实,严禁编造)

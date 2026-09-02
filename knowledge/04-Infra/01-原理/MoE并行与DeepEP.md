# MoE并行与DeepEP

> 🚧 占位:本篇待撰写。写作标准见 docs/05-知识库写作契约.md,样板见「GPU架构与执行模型」。

## 计划覆盖

- MoE 推理流程:dispatch → 专家计算 → combine
- 四种 dispatch 方案(allgather / allreduce / all2all / DeepEP)的流程、通信量与优劣
- 切分方式:tp 切 attention + ep 切专家 等组合
- 并行的必备条件:拆 batch、dispatch-gemm 重叠、combine-gemm 重叠
- 负载不均衡的后果、在 trace 上的表现与解法

## 相关文献

- 待补(arxiv 编号必须联网核实,严禁编造)

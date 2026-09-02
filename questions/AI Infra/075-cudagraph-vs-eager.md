---
difficulty: 简单
topic: CudaGraph/与Eager差异
summary: eager 逐个下发与 CUDA Graph 重放在形状、控制流、显存等维度的差异
tags: [面经, 待校对, CUDA Graph, eager]
company:
mastered: false
highfreq: false
---

## 题目

eager 和 cudagraph 有啥差异?

## 要点

- 差异的根子只有一条:eager 每步现场组装,图是提前组装好一次性提交
- 要覆盖 CPU 开销、形状、控制流、显存、首次开销、调试六个维度
- 能说出各自的适用阶段(prefill/训练 vs decode)
- 记得点出图的典型翻车:换指针导致静默算旧数据

## 答案

| 维度 | Eager 逐个下发 | CUDA Graph 重放 |
|---|---|---|
| CPU 侧工作量 | 每个 kernel 一次 launch,几微秒 | 整图一次提交,每 kernel 摊薄到亚微秒 |
| 形状 | 每步可以不一样 | **冻结**,一张图只服务一种形状 |
| 控制流 | 可以有 host 侧 if、`.item()` | **不允许**,捕获失败或静默走错分支 |
| 显存 | 常规 caching allocator,用完即还 | 图专属池,地址常驻不释放 |
| 首次开销 | 无 | warmup + capture + instantiate,每档一次 |
| 调试 | 报错栈直指出错那行 | profiler 里只看到一个图节点,难定位 |
| 典型翻车 | — | 换指针而非原地写 → 静默算旧数据 |
| 适用阶段 | prefill、训练、形状多变 | **decode**、形状规整、kernel 碎 |

一句话概括这张表:**eager 用「每步都能变」换灵活,图用「什么都不许变」换下发成本**。所以它们不是替代关系,而是同一个推理引擎里按阶段分工——decode 走图,prefill 与形状异常的请求回退 eager。

还要分清图与 `torch.compile`:后者优化的是 **kernel 本身**(融合、选核、去掉 Python 开销),CUDA Graph 优化的是 **提交方式**。PyTorch 用 `mode="reduce-overhead"` 把两者串起来,在编译产物之上自动套一层图。

## 知识点

launch 开销摊薄、形状冻结、图私有显存池、捕获期一次性成本、compile 与 graph 的分工。

## 追问

- 同一个服务里怎么决定某个请求走图还是回退 eager?
- 图模式下线上出了精度问题,怎么定位?
- `mode="reduce-overhead"` 背后做了什么?

## Note

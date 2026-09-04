---
difficulty: 简单
topic: 显存管理与OOM/内存泄漏诊断
summary: 怎样区分活跃张量、CUDA缓存与真正泄漏并定位持续引用
tags: [面经, 待校对, PyTorch, 显存, 内存泄漏, Profiling]
company: 科大讯飞、阿里
mastered: false
highfreq: false
---

## 题目

在开发或调试深度学习模型过程中，如果遇到内存泄漏问题，您会采取哪些步骤进行诊断和解决？请结合具体工具（如PyTorch的memory profiler）或实践经验说明。

## 要点

- 先分 CPU 内存、GPU 活跃张量、CUDA 分配器缓存和其他库占用
- `allocated` 持续增长更像引用未释放，`reserved` 增长后稳定可能只是缓存池扩容
- 常见根因是保留计算图、容器积累 Tensor、hook/闭包引用和 DataLoader worker
- 用最小复现、分阶段打点、memory summary/snapshot 与短窗口 profiler 定位
- 多卡逐 rank 记录，不能只看一张卡或只看 `nvidia-smi`

## 答案

**先确认“哪一类内存在长”，再找“是谁还持有引用”。** `nvidia-smi` 数字上涨不等于 PyTorch 泄漏，因为 CUDA context、通信库和缓存分配器也占显存。

### 第一步：建立稳定基线

固定 batch、序列长度和训练步骤，先 warmup，再每隔若干步记录：

- 进程 RSS 与系统内存，用来判断 CPU 侧；
- `torch.cuda.memory_allocated()`：仍被活跃 Tensor 使用的显存；
- `torch.cuda.memory_reserved()`：PyTorch 从 CUDA 申请后保留在缓存池中的显存；
- `max_memory_allocated()`：阶段峰值；
- `nvidia-smi`：整个进程和非 PyTorch CUDA 占用。

如果 `allocated` 每步都涨，通常有活跃 Tensor 或计算图被持续引用。如果 `allocated` 回落而 `reserved` 先涨后稳定，通常是分配器为了复用而扩容，不是泄漏。如果进程 RSS 涨但 GPU 指标稳定，就去查 DataLoader、缓存、日志对象和 pinned memory。

### 第二步：按训练阶段切开

在数据加载后、前向后、反向后、`optimizer.step()` 后分别打点，找出第一次出现持续增长的阶段。然后用二分法逐块关闭日志、验证、缓存、hook 或某个 loss 分支，尽快得到最小复现。

常见错误包括：

- 把带计算图的 `loss`、`output`、hidden states 直接 append 到列表；需要数值时用 `.item()`，需要张量时先 `.detach()`；
- 不必要地设置 `retain_graph=True`；
- forward/backward hook 没有移除，闭包又抓住大 Tensor；
- 梯度累计边界写错，或把每步输出一直放在 GPU；
- DataLoader worker、prefetch、持久 worker 和 `pin_memory` 组合让 CPU 内存持续增长。

### 第三步：用工具找引用和分配栈

`torch.cuda.memory_summary()` 适合快速看活跃、保留和分配重试。memory snapshot 可以看分配历史、块和调用栈，更适合定位哪类 Tensor 一直活着。`torch.profiler` 的 memory 记录、shape 和 stack 也有帮助，但这些选项本身开销很大，应只抓短窗口，不能整晚常开。

CPU 对象可用 `tracemalloc`、对象计数和引用检查辅助定位，但 `tracemalloc` 看不到 CUDA 设备分配，因此不能拿它单独证明 GPU 无泄漏。

### `empty_cache()` 为什么不是修复

`empty_cache()` 只能把**没有活跃 Tensor 占用的缓存块**还给 CUDA。仍被 Python 容器、计算图或 hook 引用的 Tensor 不会消失。它有时让 `nvidia-smi` 下降，只说明清了缓存，根因仍在；频繁调用还会破坏复用并增加同步与重新分配。

### 多卡怎么查

每个 rank 分别记录设备号、step、allocated、reserved 和峰值，先找最早偏离的 rank。再检查它是否拿到异常长样本、额外日志、不同控制流或先发生 OOM。其他 rank 后续出现集合通信超时，往往只是某个 rank 更早出错的连锁反应。

## 知识点

活跃显存、缓存池、进程 RSS、引用链、计算图、memory snapshot、短窗口 profiler 与逐 rank 排查。

- 真实面经：[B002-G01-Q103](../../docs/references/面经原题.md#b002-g01-q103)
- 老师参考：[P005-Q103](../../docs/references/平台题/P005-Infra-091-120.md#p005-q103)

## 追问

- 参考追问：怎样区分真正泄漏和 PyTorch CUDA 缓存池的正常行为？
- 参考追问：多卡训练时怎样定位是哪张卡泄漏？
- 参考追问：`pin_memory` 等“伪泄漏”通常怎样产生？

## Note

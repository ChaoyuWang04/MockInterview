---
difficulty: 中等
topic: CUDA流与异步执行/pinned memory
summary: 异步传输是否必须 pinned memory,不用会有什么后果
tags: [面经, 待校对, pinnedmemory, 异步传输, DMA]
company:
mastered: false
highfreq: false
---

## 题目

异步传输一定要使用 pinned memory 吗?不用会有什么后果?

## 要点

- 结论:功能上不强制,性能上等于必须
- 机制根因:DMA 绕过 CPU 按物理地址搬数,应付不了页被换走
- 不用的后果是**两条**:多一次 CPU 拷贝 + 函数失去异步性
- pinned 也不是越多越好,它拖慢的是整台机器
- PyTorch 里 `pin_memory=True` 和 `non_blocking=True` 必须成对出现

## 答案

**结论:功能上不强制,性能上等于必须。**

### 一、为什么需要页锁定

操作系统的普通内存是**可分页的(pageable)**——OS 随时可能把某一页换到磁盘,或搬到别的物理地址。而 DMA 引擎干活时**绕过 CPU 直接按物理地址搬数据**,它没法应对「我正在搬的这页突然被挪走了」。

**pinned memory(页锁定内存,`cudaHostAlloc` / `cudaMallocHost` 分配)** 就是告诉 OS:这块内存钉死在物理内存里、永远不许换页,DMA 才能拿到一个稳定的物理地址直接搬。

### 二、不用会怎样

从可分页内存发起异步拷贝时,驱动没法直接 DMA,只能**先把数据拷到自己内部的一块 pinned 暂存区(staging buffer)**,再从那里 DMA 到显存。后果是两条:

1. 多了一次 **CPU 参与的内存拷贝**,有效带宽明显下降
2. **这个函数会失去异步性**——官方文档的措辞是:涉及可分页内存时该函数「可能对 host 是同步的」;如果需要经 pinned 暂存,驱动可能与该流同步

翻译成人话:**你写了 Async,但它退化成了同步行为**——CPU 卡在这里等,重叠没了,还白白多付一次拷贝。这是「明明用了异步 API 却没有任何加速」的头号原因。

| | 可分页内存(pageable) | 页锁定内存(pinned) |
|---|---|---|
| DMA 能否直接搬 | ❌ 要先拷到驱动暂存区 | ✅ 直接搬 |
| 异步拷贝的行为 | **可能退化为同步** | 真异步,立刻返回 |
| 有效带宽 | 明显更低(多一次 CPU 拷贝) | 打满 PCIe / NVLink |
| 分配开销 | 快(就是普通 malloc) | 慢,要走驱动做页锁定与注册 |
| 代价 | 无 | **占住物理内存,不可换页** |

### 三、pinned 也不是越多越好

页锁定的内存 OS 收不回去,分配过量会挤压系统可用内存、让别的进程频繁换页,**拖慢的是整台机器**,严重时直接触发 OOM。实践做法是**分配一块固定大小的 pinned 缓冲区反复复用**,而不是每次传输都新分配一块。

### 四、PyTorch 里必须成对出现

DataLoader 开 `pin_memory=True`(让 host 侧张量落在 pinned 内存),搬运时写 `.to('cuda', non_blocking=True)`。**只写 `non_blocking=True` 而源张量不是 pinned,等于白写。**

## 知识点

可分页内存与页锁定内存、DMA 与物理地址、staging buffer、异步退化为同步、pinned 缓冲区复用、`pin_memory` 与 `non_blocking` 的配对。

## 追问

- `cudaMemcpyAsync` 背后的机制是什么?为什么传输能和计算重叠?
- 怎么验证一次拷贝真的异步了?
- 一台机器上 pinned 内存开到多大算过量?
- 除了 H2D,D2H 和 P2P 拷贝也需要 pinned 吗?

## Note

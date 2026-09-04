---
difficulty: 中等
topic: FSDP/前反向与参数更新
summary: FSDP前向反向怎样收集参数、分片梯度并更新状态
tags: [真题, 待校对, FSDP, 分布式训练, AllGather, ReduceScatter]
company: 字节
mastered: false
highfreq: false
---

## 题目

FSDP 的参数、梯度和优化器状态分别怎样分片?请按一次前向、反向和 optimizer step 说明 AllGather、ReduceScatter、reshard 与预取发生在哪里。

## 要点

- 计算外参数保持分片,计算某个 FSDP 单元前才 AllGather 完整参数
- FULL_SHARD 前向后释放完整参数,反向前因此还要再次收集
- 反向完成后对梯度 ReduceScatter,不是对完整梯度再 AllReduce
- optimizer 在本地参数、梯度和状态分片上更新,wrap 粒度决定通信边界

## 答案

**FSDP 的核心是“参数只在计算它的那一小段时间临时完整”。** 以 FSDP2 的逐参数分片为例,计算外每个 rank 只持有参数 DTensor 的一片,优化器状态也跟随这片参数本地保存。

1. 前向进入某个 FSDP 单元前,hook 对该组参数做 AllGather,得到临时完整参数并执行这一层。
2. `FULL_SHARD`/`reshard_after_forward=True` 会在前向后释放完整参数并恢复分片态,把峰值压到“一两层完整参数”;代价是反向计算该层前还要再 AllGather 一次。若选择前向后保留完整参数,可省第二次收集,但显存更高。
3. 反向计算出该单元的完整梯度后做 ReduceScatter:一边跨 rank 求和,一边只把对应梯度分片留给各 rank。随后释放完整参数和完整梯度缓冲。
4. optimizer 直接用本地参数分片、梯度分片和状态分片更新;不需要让每张卡先拿到完整优化器状态,也不是“最后 AllReduce 再全量更新”。

按 Transformer 层自底向上应用 `fully_shard` 通常最合适:每层形成一个通信组,计算第 $k$ 层时可预取相邻层参数,并把 ReduceScatter 放到独立流上重叠。只包根模块会一次收集全模型且几乎没有重叠;切得过细又会产生大量小通信。

## 知识点

FSDP2、DTensor、AllGather、ReduceScatter、reshard、prefetch、wrap 粒度。

- 依据:[PyTorch FSDP2 `fully_shard`](https://docs.pytorch.org/docs/main/distributed.fsdp.fully_shard.html)。

## 追问

- `FULL_SHARD` 和 `SHARD_GRAD_OP` 为什么会差一次反向前 AllGather?
- FSDP 与 ZeRO-3 的核心通信流程有什么相同和不同?
- wrap 粒度太粗或太细分别怎样影响显存与通信?

## Note

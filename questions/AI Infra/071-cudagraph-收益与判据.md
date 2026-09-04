---
difficulty: 中等
topic: CudaGraph/好处与生效条件
summary: CUDA Graph 省的是什么开销,一定奏效吗,哪些场景优势明显
tags: [真题, 待校对, CUDA Graph, 推理优化]
company:
mastered: false
highfreq: false
---

## 题目

CUDA Graph 的好处是什么?CUDA Graph 一定能奏效吗?什么场景下 CUDA Graph 的优势比较明显?

## 要点

- 省的是 CPU 侧的 kernel launch 开销,不是 GPU 的计算时间
- 能给出判据:比较「下发一个 kernel 的时间」和「这个 kernel 自己跑的时间」
- 知道量级是个位数到十几个百分点,不是数量级提升
- 建图 + 实例化本身有固定成本,靠重复重放摊薄
- 能各举一类高收益场景与零收益场景

## 答案

### 省掉的是什么

每写一次 `kernel<<<...>>>`,CPU 都要校验参数、打包描述、写进流队列、通知驱动,这套动作约几微秒。异步下发只是把它**藏起来**、没有**消掉**——当下发比 kernel 自身还慢时,GPU 追上 CPU 就只能干等队列被填。CUDA Graph 把这堆下发提前做完一次、存成可执行对象,之后每步只提交这一个对象。

NVIDIA 官方实测(V100,20 kernel × 1000 步,单 kernel 执行 2.9 μs):

| 下发方式 | 每 kernel 实测耗时 |
|---|---|
| 每次 launch 后同步 | 9.6 μs |
| 纯异步重叠下发(eager 常态) | 3.8 μs |
| CUDA Graph 重放 | 3.4 μs |

两点要说破:3.8 → 3.4 **只快约 12%**;kernel 自己才 2.9 μs,说明重放**仍有约 0.5 μs 残余开销**。建图 + 实例化约 400 μs,要靠重放几千次才摊薄。

### 判据:什么时候白干

异步下发下每 kernel 的墙钟约为下发与执行两者取大,把下发压到近 0 则

$$
\text{加速比上限} \approx \frac{\max(t_{\text{launch}},\ t_{\text{kernel}})}{t_{\text{kernel}}}
$$

kernel 一旦比下发慢,分子分母相等,收益就是 1 倍——一点也没有。

| 场景 | 收益 | 为什么 |
|---|---|---|
| 小 batch decode(1–8) | 最高 | 每层十几个 kernel、各几微秒,launch 占比可过半 |
| 大量逐元素小算子(norm、激活、RoPE、残差) | 高 | 典型 launch-bound |
| CPU 弱 / Python 开销大 | 高 | 下发端本来就跟不上 |
| 长 prefill、大 batch 训练 | 几乎为零 | 单个 GEMM 就几毫秒,几微秒被摊到 0.1% |

一句话:**它治的是「CPU 喂不动 GPU」,不治「GPU 自己算得慢」**。上图前先看 profiler 里有没有大片 gap,没 gap 就别指望它。

## 知识点

kernel launch 开销、队列深度、launch-bound、加速比上限判据、建图与实例化的一次性成本。

## 追问

- 和 `torch.compile` 是什么关系?两者能叠加吗?
- profiler 上怎么判断当前是不是 launch-bound?
- 图重放为什么还剩约 0.5 μs 的残余开销?

## Note

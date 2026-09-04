---
difficulty: 中等
topic: PD分离/KV传输
summary: PD 分离下 KV cache 怎么传、走什么硬件、怎么做异步
tags: [真题, 待校对, PD分离, RDMA]
company:
mastered: false
highfreq: false
---

## 题目

PD 分离下怎么处理 KV cache 的传输?怎么传输比较快,用什么硬件,以及怎么做异步传输?

## 要点

- 先估传输量,才知道这件事有多贵
- 硬件优先级:NVLink > RDMA / IB > PCIe > 以太网 TCP
- 快的两个抓手:**单边 RDMA**(不打扰对端 CPU)+ **内存预注册**
- 异步的核心答案是**逐层流水**,把传输藏进后续层的计算里
- 配套四条:独立 CUDA stream、非阻塞提交、粒度折中、接收侧提前备好落地

## 答案

### 先看要传多少

$$
\text{KV 字节数} = 2 \times L \times H_{kv} \times d_h \times S \times \text{bytes}
$$

2 是 K 和 V 两份,$L$ 层数,$H_{kv}$ 是 KV 头数(GQA 下远小于 Q 头数),$d_h$ 头维度,$S$ 序列长度。代入 Llama-3-8B($L=32$、$H_{kv}=8$、$d_h=128$、FP16):每 token 每层 $2\times8\times128\times2 = 4$ KiB,乘 32 层 = **每 token 128 KiB**;一个 8k 的 prompt 就是 **1 GiB**。

### 走什么硬件

| 通道 | 单向带宽量级 | 传 1 GiB 约需 | 用在哪 |
|---|---|---|---|
| **NVLink / NVSwitch** | 300–450 GB/s | **2–4 ms** | 机内 GPU 之间,最快 |
| **RDMA / InfiniBand 400G** | ~50 GB/s | ~21 ms | 跨机主力 |
| InfiniBand 200G | ~25 GB/s | ~43 ms | 跨机 |
| PCIe Gen4 x16 | ~21 GB/s 有效 | ~50 ms | 退化路径,尽量避开 |
| 以太网 TCP | 更低且抖动大 | — | 一般不用于数据面 |

结论很直接:**P 和 D 尽量放在同一机内或同一 NVLink 域**;必须跨机就上 RDMA,靠**单边 RDMA**(WRITE / READ 不打扰对端 CPU)加**内存预注册**(pin 住并注册给网卡,避免每次传输重建映射)把延迟压下去。裸 TCP 走内核协议栈、要多次拷贝,数据面基本不用。传之前还要先把打散的分页 KV **gather 成连续 buffer**,否则会退化成几千次小块散拷。

### 异步怎么做:逐层流水是核心答案

朴素做法是「prefill 全部算完再一次性传」,串行相加 $T_{\text{prefill}} + T_{\text{传输}}$,那 21 ms 赤裸裸暴露在 TTFT 里。

真实系统的做法是**逐层流水(layer-wise pipelining)**:第 $i$ 层算完,它那一层的 KV 就已经定稿、不会再变,**可以立刻开始传**,同时 GPU 接着算第 $i+1$ 层。传输被藏进后续层的计算里:

$$
T_{\text{prefill}+\text{传输}} \approx T_{\text{prefill}} + \frac{T_{\text{传输}}}{L}
$$

(成立条件:每层传输时间 ≤ 每层计算时间。)只有**最后一层**的传输还露在外面,暴露成本从「整份 KV」降到「1/L 份」。$L=32$ 时 21 ms 的暴露量降到 0.7 ms 左右——这就是为什么 PD 分离的传输开销实践中可以做到近乎不可见。

四个配套要点:

1. **独立 CUDA stream**:传输发在拷贝流、计算发在主流,靠 event 做依赖。DMA 引擎和 SM 是两套硬件,本来就能并行。
2. **非阻塞提交 + 轮询完成**:提交后立刻返回,后台查状态,不做同步等待。
3. **粒度折中**:太细则描述符与握手开销占比上升,太粗又藏不住;按层或按若干层一组是常见折中。
4. **接收侧提前备好落地**:D 侧的注册 buffer 与物理块必须在数据到达前分配好,否则 P 传不出去、块被占住,反过来把 P 拖慢。

## 知识点

KV 字节数公式、NVLink / RDMA 带宽量级、单边 RDMA、内存预注册、逐层流水、拷贝流与计算流重叠。

## 追问

- 为什么要先 gather 成连续 buffer 再传?
- 逐层流水的前提条件不满足(每层传输慢于每层计算)会怎样?
- KV 做了量化之后,传输量和 TTFT 会怎么变?

## Note

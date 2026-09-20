# SGLang 09|并行:TP / PP / CP 与 DCP

六种并行各切哪一维、TP 为什么先列后行、PP 的气泡怎么填、上下文并行的 PCP 与 DCP 分管两个阶段,vLLM 解读的「分布式并行」一章已经讲透。这一页只说 SGLang 遇到的问题、它的解法、代价、以及和 vLLM 不一样的地方。01 章讲过每个 TP × PP rank 一个调度器进程、rank 0 收请求后 gloo 广播,02 章讲过流水线并行会关掉零开销重叠,本章只回指不重讲。DP attention 和专家并行归 10 章,数据并行副本的分发在 01 章。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

单卡装不下,权重切到多张卡上,每层就多 2 次 all-reduce。decode 一步的 message 只有 16 KB 到 4 MB,通用通信库贵的不是搬数据,是每次多轮同步的固定成本,80 层就是 160 次固定成本,TPOT 被通信按住。

TP 跨机器就撞上机间带宽,于是把层切成几段跨机器接力(PP)。但一条 128K 的 prompt 按固定块切,后面的块前缀更长、算得更慢,前一级等后一级,气泡随流水线深度一级级传下去,4 级 PP 的 TTFT 降不到 1/4。

长上下文的 KV 一张卡放不下。MLA 的潜向量只有 1 份,TP 切得动权重却切不动它,8 张卡各存一份完整 KV,并发被这份冗余卡死;prefill 的注意力随序列长度平方增长,单卡的 TTFT 线性上不去。

## 二、解法:小 message 自己搬,长 prompt 动态切,KV 按位置条带化

**TP** 的做法和 vLLM 一样,不一样的是 all-reduce 有 6 条加速路加 NCCL 兜底:自写的 custom all-reduce、AMD 的 quick all-reduce、MSCCL++、torch 对称内存、NCCL 对称内存、FlashInfer 的融合 all-reduce。每次调用按顺序查条件,第一个满足的接管。custom all-reduce 默认开,只接管 8 MB 以下的小 message(ROCm 16 MB),只在单机 2、4、6、8 卡、超过 2 卡要 NVLink 全连接、字节数是 16 的倍数时才接;CUDA 上默认是 JIT 编译的 v2 版,多机只在同一个 NVLink 域(GB200 那种)里才接。条件不满足就退回 NCCL,启动时打一行 warning。逐个下发和录图是两套:逐个下发时 PyNCCL、MSCCL++、对称内存都关着,走 torch.distributed;录图时反过来。KV 头按注意力 TP 切,总头数除以「注意力 TP 除以 DCP」,不够分时每卡留 1 个,也就是复制。

**PP** 每一级一个调度器进程,各跑自己的微批循环,微批数是 pp_size 加异步深度。请求从第 0 级异步转发到下一级;隐状态和残差作为代理张量异步发到下一级;前向在单独的流上跑,CPU 趁 GPU 算这一微批时处理上一微批的结果。层数不能整除时多出来的层放到最后几级。**动态分块** 是给长 prompt 的:第 0 级启动时跑 128 次合成 prefill(从 1.25 倍块大小递减),拟合一条「累计耗时对序列长度」的二次曲线,目标耗时是第一块的耗时;之后每块解方程让「前缀 L 再算 x 个 token」的耗时等于目标,前缀越长块越小,各级每块耗时对齐,气泡不再传播。算出来的值乘 0.75 的平滑系数,下限是初始块的 1/4,向下对齐到 max(page_size, 64)。

**上下文并行(CP)** 管 prefill:序列按 zigzag(切成 2 倍 cp 块,每个 rank 拿一头一尾)或 interleave(token 下标模 cp)分到 attn_cp 组的各 rank,只对 extend 和 mixed 批、每请求至少 2 倍 cp 个 token 才切;各 rank 算自己那段 Q 的注意力,K/V 算完 all-gather 回全序列顺序写进各自的 KV 池,所以 KV 仍然每卡一份,省的是 prefill 的算力和激活。decode 不切,注意力权重在 CP 组内是复制的。

**DCP** 管 decode:在 TP 组内每 dcp 个 rank 一组,token 位置 p 归 rank p 模 c;所有 rank 看同一份「虚拟」清单(容量乘 c、页乘 c),写入时只留自己那些位置压进本地池,每卡存和读 1/c 的 KV。注意力各算各的分片,输出和 LSE 打成一个包做一次 all-to-all,按 LSE 精确合并。通信后端默认:Blackwell 且 DCP 组在一个 MNNVL 域用 FlashInfer 的 all-to-all,否则 NCCL all-to-all,别的平台 all-gather 加 reduce-scatter。`--dcp-replicate-q-proj` 把 Q 投影权重复制到每个 rank,每层省一次 all-gather。主线是 MLA 模型(DeepSeek、Kimi),Qwen3.5 的 GQA 层也按同一套接了。

为什么实际不吃亏:custom all-reduce 只接小 message,大的还给 NCCL;PP 的异步 P2P 让 GPU 不等 CPU;动态分块让每级每块耗时相等;DCP 加的是一次与上下文长度无关的集合通信,省的是随上下文线性增长的 KV 存与读。

## 三、代价

- **TP** :每层 2 次 all-reduce;custom all-reduce 常驻 8 MB 缓冲加 IPC 句柄,`Init torch distributed ends` 那行的 `mem usage` 就是这笔;多机 TP 没有 custom all-reduce。
- **PP** :零开销重叠关掉(02 章);prefill 的 CUDA graph 默认关,显式开也只录到 8192 个 token;推测解码默认拒绝;每微批的并发上限是 `max_running_requests` 除以 pp_size;动态分块启动多跑 128 次 prefill;层数不均时最慢的一级决定吞吐。
- **CP** :KV 不省;短请求不切;不能和 PP 同开(moe_dp 路径);不能开 AITER 的 all-reduce 融合;HIP 和 MUSA 直接拒。
- **DCP** :每层 1–3 次 DCP 集合通信;页宽变成 c 倍,radix 树按宽页命中;extend 要先 all-gather 前缀分片,随前缀长度增长;草稿模型的 KV 不切;HiCache 只到 L1/L2 且只 MLA;PD 分离要 Mooncake 或 NIXL,两边 DCP 相等或 1 到 c;统一显存池只认 4 种注意力后端。

## 四、和 vLLM 不一样的七处

1. **进程模型。** vLLM 按拓扑三选一(单进程、多进程、Ray),PP 只是把 step 换成带队列的版本;SGLang 每个 rank 一个调度器进程,PP 每级各自一个事件循环,请求逐级转发;多机就是每台机器跑同一条命令加 `--nnodes` `--node-rank` `--dist-init-addr`,`--use-ray` 只管拉进程。
2. **all-reduce 有几条路。** vLLM 1 个自写加 NCCL;SGLang 6 条加速路按顺序查,custom all-reduce 默认 JIT 的 v2 版,多机同一个 NVLink 域也接管。
3. **PP 的定位。** vLLM 说推理里 PP 很少用,4 个理由;SGLang 拿它当长上下文 TTFT 的招牌(4 机 PP4 跑 128K),动态分块专为此而做,代价是关重叠调度。
4. **PP 的微批数。** vLLM 队列长度是 PP 或 PP+1;SGLang 是 pp_size 加 `--pp-async-batch-depth`,默认 0,末级把输出缓冲起来。
5. **prefill 的 CP。** vLLM 的 PCP 只走 MLA 路径、与 DP 互斥;SGLang 有 zigzag 和 interleave 两种切法,支持 fa3、fa4、flashinfer、DSA、trtllm_mha,可以和 DP attention 叠(tp 要能被 dp 乘 cp 整除),但要显式 `--enable-prefill-cp` 加 `--cp-strategy`。
6. **DCP。** vLLM 的 DCP 对 GQA 有上限、MLA 能到 TP;SGLang 主线是 MLA,页宽乘 c 改变前缀命中粒度,3 种通信后端,Q 投影可复制。两边都复用 TP 的 rank。
7. **序列并行。** vLLM 是编译 pass 自动判定;SGLang 是显式开关 `--enable-layernorm-sp`,只 prefill、只 Qwen3 dense、要 NVLink。

## 五、调参与观测

**参数在哪调。** 并行度、通信实现、多机拓扑全是启动参数,改了要重启;通信相关的内部开关是环境变量。运行时只有一件事能改:`/set_internal_state` 改 `pp_max_micro_batch_size`,范围 1 到 `max_running_requests` 除以 pp_size。`/server_info` 能读回 `attn_tp_size` 这些派生出来的宽度。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--tp-size` · 张量并行度,也是每个 PP 级的卡数 | 启动 | 1 | 调高:每卡权重和 KV 变少、并发上去;每层多 2 次 all-reduce,跨机器后 TPOT 明显变长;tp 乘 pp 要能被 `--nnodes` 整除 | `Init torch distributed ends. elapsed=… mem usage=…` |
| `--pp-size` · 流水线级数 | 启动 | 1 | 大于 1:重叠调度关、prefill 图关、推测解码拒;长 prompt 的 TTFT 靠分块并行降下来,短请求小批下净亏 | `Pipeline parallelism is incompatible with overlap schedule.` |
| `--pp-max-micro-batch-size` · 每微批最多几个请求 | 启动或 `/set_internal_state` | `max_running_requests` 除以 pp_size | 调高:每级并发上去,显存吃紧;调低:并发封顶 | Decode batch 行的 `#running-req` |
| `--pp-async-batch-depth` · 末级多缓冲几个微批的输出 | 启动 | 0 | 大于 0:CPU 处理结果和 GPU 计算重叠,末级不拖后腿;微批总数变成 pp 加它 | Decode batch 行的 `#running-req` 与 `gen throughput` |
| `--enable-dynamic-chunking` · PP 下按拟合曲线动态定块大小 | 启动 | 关 | 开:启动多 128 次 prefill 拟合;`--chunked-prefill-size` 变成初始块,设成固定最优的 2–3 倍;pp 为 1 时无效 | `[PP Dynamic Chunk] [PP0] Predictor ready (quadratic). Target latency: …ms` |
| `SGLANG_DYNAMIC_CHUNKING_SMOOTH_FACTOR` · 块大小跟随曲线的比例 | 环境变量 | 0.75 | 1 严格跟曲线;0.6–0.85 是文档推荐;0 退回固定块;调高块缩得更快、尾块更小 | 长 prompt 的 TTFT |
| `SGLANG_PP_LAYER_PARTITION` · 每级分几层,逗号分隔 | 环境变量 | 均分,余数给最后几级 | 手指:和数不对直接报错;文档建议大的分区放高 rank(15,15,15,16) | 启动报错 `does not match` |
| `--attn-cp-size` · 注意力上下文并行度 | 启动 | 1 | 大于 1:注意力 TP 变成 tp 除以它;只建组,不切序列;要能整除 tp | 启动断言 `tp_size must be divisible by attn_cp_size` |
| `--enable-prefill-cp` 加 `--cp-strategy` · 真正把 prefill 序列切开 | 启动 | 关;策略无默认 | 开:长 prompt 的 TTFT 降,KV 不省;不给策略直接报错;DeepSeek V3.2 只能 interleave,MiMo V2 只能 zigzag | Prefill batch 行的 TTFT |
| `--enable-cp-decode-attn-tp` · CP 下 decode 也切注意力权重 | 启动 | 关 | 开:decode 少算冗余 GEMM;只 DeepSeek-V4 和 GLM-5 | TPOT |
| `--dcp-size` · MLA 的 KV 按位置切到几张卡 | 启动 | 1 | 大于 1:每卡 KV 变 1/c,长上下文并发上去;每层多 1–3 次集合通信;页宽乘 c;只 CUDA 和 HIP | `DCP enabled, dcp_size=…, tp_size=…` |
| `--dcp-comm-backend` · DCP 合并输出用哪种通信 | 启动 | 自动 | `fi_a2a` 只 Blackwell 加 MNNVL;`a2a` 是 CUDA/ROCm 的常规路;`ag_rs` 兜底但多 1 次集合通信 | `DCP (dcp_size=N) selects communication backend …` |
| `--dcp-replicate-q-proj` · 每个 rank 复制 Q 投影权重 | 启动 | 按模型 | 开:每层少 1 次 all-gather,权重显存多一点;只 BF16/FP16 未量化层;只配 a2a 系后端 | `dcp_replicate_q_proj: prepared full-head Q weights for N MLA layers` |
| `--nnodes` `--node-rank` `--dist-init-addr` · 几台机器、我是第几台、大家在哪会合 | 启动 | 1、0、无 | 多机每台跑同一条命令只改 `--node-rank`;每台起 tp 乘 pp 除以 nnodes 个调度器;不给会合地址就用本机 `--nccl-port` | 启动卡住不动 |
| `--nccl-port` `--dist-timeout` · 单机会合端口、初始化超时 | 启动 | 随机、torch 默认 | 同机多实例端口撞了起不来;超时调长只在慢网络上有用 | 启动报端口占用或超时 |
| `--disable-custom-all-reduce` · 关掉自写 all-reduce 走 NCCL | 启动 | 关 | 开:小 message 的 all-reduce 变慢,TPOT 长;用来排查通信错或让多机 warning 安静 | `CustomAllreduce is disabled because …` |
| `--enable-mscclpp` `--enable-torch-symm-mem` `--enable-symm-mem` · 三种备选 all-reduce | 启动 | 都关 | MSCCL++ 只 8、16、32 卡且只在图里;torch 对称内存只 SM90 以上、4/6/8 卡、64–128 MB 以下;NCCL 对称内存要 pynccl | TPOT 与 `mem usage` |
| `--flashinfer-allreduce-fusion-backend` · all-reduce 和 RMSNorm 融合 | 启动 | 不开 | `auto` 在 Blackwell 选 mnnvl、SM90 单机选 trtllm;开了后注意力 TP 组和 MoE 组的 all-reduce 优先走它 | TPOT |
| `--pre-warm-nccl` · 启动时先跑一次 all-reduce | 启动 | 关 | 开:首请求的 P99 TTFT 不再含建连 | `NCCL/RCCL/HCCL warmup completed in …s` |
| `--enable-p2p-check` · 启动时真测 GPU 间 P2P | 启动 | 关(默认当作可 P2P) | 开:PCIe 拓扑异常时提前发现 | 启动耗时 |
| `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` · 各卡空闲显存差 10% 以上就拒绝启动 | 环境变量 | 开 | 关:只 warning,继续起;KV 池按最小那张卡算 | `The memory capacity is unbalanced` |
| `SGLANG_ENABLE_PP_SPEC` · PP 配推测解码的实验路径 | 环境变量 | 关 | 开:只单层 EAGLE、要 `--disable-overlap-schedule`、不能 PD、不能自适应步数、不能 DP attention | 启动断言信息 |
| `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` · custom all-reduce 用 JIT 的 v2 | 环境变量 | 开 | 关:退回旧版;多机 NVLink 域不再接管 | `[AR] Using CustomAllReduceV2` 只在 debug 级 |

**怎么看。** 启动日志三段:`Init torch distributed begin.` 到 `Init torch distributed ends. elapsed=… s, mem usage=… GB` 之间是建组和通信器,custom all-reduce 被关会在这里打 `CustomAllreduce is disabled because …`,句尾是原因(跨节点、卡数不支持、PCIe 超过 2 卡);DCP 有两行 `DCP enabled` 和 `selects communication backend`;PP 有 `Pipeline parallelism is incompatible with overlap schedule.` 和 `Disabling breakable prefill CUDA graph by default for pipeline parallelism`,动态分块有进度条 `Profiling prefill latency for dynamic chunking` 和 `Predictor ready`。多机对不齐的报错就两种:`world_size (…) is not equal to tensor_model_parallel_size (…) x pipeline_model_parallel_size (…)` 和 `tp_size must be divisible by number of nodes`。稳态还是 03 章那两行 `Prefill batch` 和 `Decode batch`;PP 下每一级各打各的。

| 症状 | 先查 | 然后 |
|---|---|---|
| 多机启动卡在建组不动 | 各节点 `--dist-init-addr` 是否一字不差、`--node-rank` 有没有重复、tp 乘 pp 能否被 nnodes 整除 | 文档建议先加 `--disable-cuda-graph` 排除死锁;再看 `--dist-timeout` |
| 启动报 `The memory capacity is unbalanced` | 哪张卡被别的进程占了 | 清掉;确实要带病起就关 `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK` |
| 单机 TP 日志有 `CustomAllreduce is disabled` | 句尾原因:卡数不在 2/4/6/8、PCIe 超过 2 卡 | 换卡数;PCIe 机器接受走 NCCL,加 `--disable-custom-all-reduce` 让它安静 |
| 开了 PP 吞吐反而降 | 有没有 `Overlap schedule is disabled`;`#running-req` 是否卡在 max_running 除以 pp | 短请求负载换 TP;并发不够抬 `--pp-max-micro-batch-size` |
| 长 prompt 的 TTFT 在 PP 下没随级数降 | 是否固定块、`--chunked-prefill-size` 多大 | 先扫固定块找最优,再开动态分块把初始块设成 2–3 倍 |
| 动态分块启动时 `Failed to profile prefill latency` | 显存够不够跑 1.25 倍块的 prefill | 降块大小或比例;失败时静默退回固定块 |
| PP 加推测解码起不来 | 断言 `not compatible with overlap schedule, speculative decoding` | 去掉推测解码;实验路径开 `SGLANG_ENABLE_PP_SPEC` 并满足 5 条限制 |
| CP 开了 KV 容量没变 | 这是设计:KV all-gather 回全量 | 要省 KV 用 DCP(MLA)或 DSA 的 `--enable-dsa-cache-layer-split` |
| `--attn-cp-size` 给了但 prefill 没变快 | 有没有 `--enable-prefill-cp` 和 `--cp-strategy` | 补上;每请求至少 2 倍 cp 个 token 才切 |
| DCP 下 `cached_tokens` 变成大整数的倍数 | 页宽是 page_size 乘 dcp | 这是设计;命中只到宽页边界 |
| DCP 配 DP attention 启动了但结果错 | attn_tp 能否被 dcp 整除 | 启动只查 tp 能否被 dcp 整除,自己保证 DCP 组不跨副本 |
| 确定性模式下 TP 变慢 | NCCL 被钉成树、通道数钉死、custom all-reduce 关 | 这是 13 章的设计,不是通信故障 |

## 六、常见误区

- **把 `--tp-size` 当成注意力 TP。** 名字像。它是每个 PP 级的卡数,注意力 TP 是它除以 DP 再除以 CP,MoE 的 TP 是它除以 EP 再除以 moe_dp,DCP 也从它里面切。8 卡开 DP attention 4 副本时注意力 TP 只有 2,KV 头就按 2 切。
- **多机 TP 看到 custom all-reduce 被关的 warning,以为配错了。** 它只管单机,跨节点的组永远关;多机走 NCCL 是正常路径。想让它安静加 `--disable-custom-all-reduce`。
- **开了 PP 吞吐降了,去查 GPU 利用率。** 第一层原因是重叠调度被关了,02 章那一行 warning 最常被略过;第二层是每微批并发上限被除以了 pp_size。PP 是给「TP 用满整机还装不下」和「128K 级 prompt 的 TTFT」的,短请求高并发换 TP。
- **以为动态分块单机也能用。** 只有 pp 大于 1 才会构造它,pp 为 1 时旗标静默无效,日志里一个字都没有。
- **以为 `--chunked-prefill-size` 在动态分块下还是「每块多大」。** 它变成了初始块和目标耗时的定义,后面的块从它往下缩到最少 1/4。按固定块的最优值直接开动态分块,块会太多太碎;文档说设成 2–3 倍。
- **以为 `SGLANG_PP_LAYER_PARTITION` 是启动参数。** 它不在环境变量注册表里,是直接读进程环境的,`/server_info` 看不到;写错层数和直接报错,不会静默均分。
- **以为 prefill CP 省 KV 显存。** 名字里有「上下文」。它切的是 prefill 的计算,K/V 算完 all-gather 回每张卡;省显存的是 DCP,而 DCP 只切 MLA 的 KV。两个功能两个旗标。
- **只给 `--attn-cp-size` 不给 `--enable-prefill-cp`。** 前者只建组、只影响 MoE 前的 token 共享;真正切序列要开关加策略,不给策略直接报错。
- **以为 PP 和推测解码能一起开。** 默认断言拒绝;NPU 例外;CUDA 上只有实验环境变量,而且限单层 EAGLE、非重叠、非 PD、非自适应、非 DP attention。
- **DCP 开了以为前缀命中不受影响。** 页宽变成 page_size 乘 dcp,radix 树按宽页劈,命中只到宽页边界;PD 和 HiCache 的组合也跟着缩窄。
- **DCP 配 DP attention 时以为启动过了就对了。** 启动只查 tp 能否被 dcp 整除,不查注意力 TP 能否被整除;文档自己写着这条检查还没修,跨副本的 DCP 组会起来但算错。
- **启动报显存不均衡以为是 warning。** 默认是直接报错退出,不是提示;要带病起要显式关那个环境变量。

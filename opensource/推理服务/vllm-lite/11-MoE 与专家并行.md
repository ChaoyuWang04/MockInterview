# vLLM 11|MoE 与专家并行

大 MoE 模型的专家分到很多张卡上以后,每层 token 都要被送去别的卡、算完再收回来,vLLM 怎么让这两次通信可以换实现、可以藏进计算,又怎么不让一张热卡拖住全组。

说法都在源码基准 `94f4170df3`(tag `v0.30.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

先是**放不下**。DeepSeek-V3 这类模型每层有 256 个路由专家,官方专家并行部署文档的例子里,它单机 8 张 H200 起得来,8 张 H100 就不行。专家只能分到很多张卡、很多台机器上;注意力部分又通常按数据并行各放一份(10 章),于是进 MoE 层时每张卡手里是自己那批 token,要用的专家却散在全组。

再是**怎么切都要付通信**。把每个专家切成 N 份,每张卡都得见到全组的每个 token,通信量随卡数线性涨。把专家整份分到各卡,token 只去持有它那几个专家的卡,但这就成了一次形状每步都变的 all-to-all:每层两次,派出去一次、收回来一次,派多少、派给谁由路由当场决定。

然后坏在三处,都落在大规模 MoE 部署的 decode TPOT 与吞吐上:

- **通信时 SM 空转。** 小批 decode 每层要搬的数据不多,固定延迟占大头;两次 all-to-all 在网上的时候,整张卡没有活干。
- **最慢的卡决定整层。** 路由不均,热专家所在的卡要算的 token 比别人多,两次通信都是集合通信,全组都等它。这种不均还随流量漂,昨天均衡的摆位,今天换一批请求就不均了。
- **规模是启动时定死的。** 一张卡出问题或者流量涨落,专家分几份、放在哪,只能整套重启再来。

## 二、解法:一层 MoE 拆成准备、专家、收尾三件,通信与计算各自可换

最天然的直觉是专家整份分卡,token 按路由送过去,算完送回来。vLLM 开了专家并行以后就是这么做的:专家层用「数据并行 × 预填充上下文并行 × TP」那么多张卡组成的一个组(10 章),每张卡拿 1/N 个专家的完整权重,默认按编号连续放。不开专家并行时,同一批卡把每个专家切成 N 份,按 TP 的路子算。

![一层 MoE 在 4 卡专家并行下的数据流:每张卡先对自己的 2 个 token 做路由、各选 2 个专家,再经分发 all-to-all 只发给持有所选专家的卡,同卡两个专家都被选中也只发一份;各卡只算自己那 2 个专家,其中 E1 被故意做热,卡 0 要算 7 份活而卡 1 只有 2 份;算完经合并 all-to-all 原路送回出身的卡,加权求和后再加上本卡自己算、不过网的共享专家输出;底部注明默认后端改用 all-gather 与 reduce-scatter,以及异步后端在合并通信期间算共享专家](/opensource/vllm-lite/11a-moe-dispatch-combine.svg)

**一层 MoE 被拆成三件。** 准备:按路由结果量化激活,分发出去。专家计算:把收到的 token 按专家排好,做两次 GEMM,再排回原序。收尾:乘上路由权重求和,合并回出身的卡。通信只在头尾两件里,计算只在中间一件。头尾和中间之间只约定一件事:分发出来的激活是连续的一整块,还是按专家分桶、每桶定长的批格式。于是 all-to-all 的实现和专家 kernel 可以各写各的,格式对得上就能拼。量化方法决定用哪一族专家 kernel(13 章);每族有一张按硬件排好序的候选表,启动时按 all-to-all 要的格式,挑第一个能用的。

**all-to-all 的实现有 10 种能用。** 参数的取值有 13 个名字,1 个是旧名的别名,2 个已经删除,填了会告警、回落到默认。默认那种其实不是 all-to-all:分发是 all-gather,每张卡收下全组的全部 token;合并是 reduce-scatter。它不用装任何额外的库,代价是通信量随卡数涨:16 张卡、256 个专家、每个 token 选 8 个、路由均匀时,一张卡收下的 token 里只有约 41% 真用得上本卡的专家。其余 9 种只把 token 送到用得着的卡。主力是 DeepEP 的两种模式:高吞吐出连续的一整块,一次吞下大批,给 prefill;低延迟出按专家分桶的定长批,形状固定,能进 CUDA graph,给 decode。NIXL EP 和 DeepEP 低延迟是同一种格式,还能在服务中增删卡;MoRI 是 AMD 上的对应物,也分高吞吐、低延迟两种。另外还有 DeepEP v2、两种走跨机 NVLink 的 FlashInfer 实现,以及 MoonEP:它每一步在线挑几个专家临时多放一份,让每张卡收到的 token 数恒定。

**共享专家不过网。** 每张卡对自己的 token 算。DeepEP 的三种和 NIXL EP 是异步后端,收尾被拆成「发起合并」和「等合并到」两步,共享专家插在中间算,正好填上 SM 空等的那段;别的后端在这一步 token 数不多时,把共享专家放到另一条 CUDA 流上,和路由、专家计算并行。

**双批重叠把通信藏进另一半批的计算。** 一步的批按 token 数从中间切成两个微批,两个 CPU 线程各跑一个微批的整个前向,每到分发或合并就让给对方,GPU 上计算和通信各走一条流。

![双批重叠下一层 MoE 的稳定排程:一步的批按 token 数对半切成微批 0 和微批 1,请求 r3 被劈到两边;CPU 线程 0 只跑微批 0、线程 1 只跑微批 1,轮流发 kernel,每次发起分发或合并后就让给对方;GPU 计算流上依次是微批 0 的投影与注意力、微批 1 的专家、微批 1 的共享专家与微批 0 的专家、微批 0 的共享专家与微批 1 的投影与注意力,同一时刻通信流上是另一个微批的分发或合并,DeepEP 高吞吐时通信只给 20 个 SM;底部列出整步不切的条件:纯 decode 批不到 32 个 token、带 prefill 的批不到 512 个 token、补齐到最忙副本后 token 最少的副本不超过一半、批里有请求读另一条请求这一步才写的前缀块](/opensource/vllm-lite/11b-dbo-two-microbatches.svg)

DeepEP 高吞吐的通信只占一部分 SM,其余留给另一个微批的计算;DeepEP 低延迟和 NIXL EP 本来就不占 SM。双批只录整图 CUDA graph,录好以后回放不再需要两个线程来回切。

**专家负载均衡(EPLB)复制热专家、按统计重排。** 每一层记录每个物理专家这一步收了多少 token,只记重排前最近 1000 步;每 3000 步用改编自 DeepSeek 开源 EPLB 的算法重算一张「物理槽 → 逻辑专家」的表:冗余槽一个一个分给「负载 ÷ 副本数」最大的专家,再把专家装箱到各卡,让每张卡的负载尽量平;机器数能整除专家组数时,先按机器分组,少跨机。新表里还留在原卡的专家尽量不挪槽位,只搬必须搬的权重。默认异步:后台线程在单独一条流上逐层把新权重收进缓冲,全组都收好了,才在两步之间一起换上。一个逻辑专家有多个副本时,token 按自己的序号哈希到其中一份。

**弹性专家并行让副本数在服务中改。** 扩容时,新副本的非专家权重不读盘,由老副本点对点送过去,新副本在老副本之间均分;每个副本的专家槽数固定不变,扩容多出来的槽全当冗余副本,靠一次 EPLB 重排把热专家摊上去;缩容最少只能缩到剩下的槽还装得下全部逻辑专家。通信组改用不挂在 PyTorch 全局进程组上的独立组,先在旁边建好备用组,切换那一刻阻塞前向。缩容反过来,先重排把要撤的卡上的专家挪走,再撤卡。

为什么实际不吃亏:两次通信被关在准备和收尾里,换一种 all-to-all 不用动专家 kernel,双批重叠也只需在这两件里插让出点;双批切不切由全体副本每步一起定,纯 decode 批不到 32 个 token、带 prefill 的批不到 512 个 token 时不切,免得两半都太小、打不满 GPU;EPLB 只改「逻辑专家落在哪个槽」,模型和路由结果都不动。

## 三、代价

**默认后端最省事,也最费网。** all-gather 让每张卡收下全组的 token,卡越多,白收的比例越高。

**开了专家并行不等于走了 all-to-all。** 只有数据并行或预填充上下文并行大于 1 时才走;单副本的 TP 8 加专家并行,每张卡放整份专家、对全部 token 算本卡那几个,最后照样一次 all-reduce,点名的 all-to-all 实现被忽略,不报错。

**高吞吐和低延迟整个实例只能二选一。** 高吞吐配数据并行时 CUDA graph 被关(08 章),decode 吃亏。DeepEP 低延迟和 NIXL EP 按每副本每步的 token 上限开缓冲,没显式设时这个上限从常规默认值降到 256,长 prompt 要切成很多步做 prefill,TTFT 变长。官方文档也说 DeepEP 两种模式是为 PD 分离调的,混合负载可能表现差;两头都要,就起 prefill、decode 两套实例(12 章)。

**双批重叠的前置条件多。** 只收 DeepEP 高吞吐、DeepEP 低延迟和 NIXL EP;只录整图;默认会让 runner 退回 V1(07 章);级联注意力被关掉。各副本里 token 最少的那个不超过补齐后最大值的一半时,整步都不切;批里有请求要读另一条请求这一步才写的前缀块时,也不切。

**EPLB 吃显存,也吃搬运。** 冗余专家的个数是全局总数,每张卡多放「冗余数 ÷ 卡数」个专家的权重,官方文档给的量级是 DeepSeek-V3 每卡多 1 个专家约 2.4 GB;逻辑专家数加冗余数必须能被卡数整除。改成同步搬以后,重排那一步不出 token。除默认后端、DeepEP v2 和 FlashInfer 单边外,其余后端配 EPLB 时共享专家不再和通信重叠,源码注释说是有正确性问题。MoonEP 不能开 EPLB。均衡度只进日志,没有 Prometheus 指标。

**弹性专家并行的限制最多。** 副本必须用 Ray 拉起;必须开 EPLB;不能配 PP,不能配外部或混合负载均衡;异步 EPLB 时必须装 NIXL;runner 退回 V1(07 章);切组时阻塞前向;默认不等在途请求排空就开始切。

## 四、和 SGLang、TensorRT-LLM 比:通信实现谁来挑、热专家怎么摊

对照对象是 SGLang(基准 826d5170ae)与 TensorRT-LLM(基准 59f5c47f2e)。三家都要把专家分到卡上、都要把 token 送到专家再收回、都要对付热专家,分歧在专家份数是不是独立的数、通信实现由谁挑、以及热专家在什么时候被摊开。

| 必须有人做的事 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 专家分几份 | 一个布尔开关;份数由数据并行 × 预填充上下文并行 × TP 推出来 | 专家份数单独给一个数,默认 1;点名 DeepEP 等全交换后端后被改成等于 TP | 专家并行与专家 TP 各有一个数,可以混切 |
| 通信实现谁来挑 | 启动时用户点名一种,默认 all-gather 加 reduce-scatter | 用户点名,默认不走全交换,每卡跑全部 token 再 all-reduce | 注意力数据并行时按优先级自动试:NVLink 单边、双边、NCCL EP、DeepEP、DeepEP 低延迟,都不行才退 all-gather;环境变量可强制 |
| DeepEP 两种模式怎么用 | 整个实例二选一 | 默认按这一批有没有 prefill 自动在两种之间切 | 在自动候选链里按顺序取第一个能起来的 |
| 把通信藏进计算 | 双批重叠,默认关,decode 不到 32、prefill 不到 512 个 token 不切 | 双批重叠,默认关 | 基准的配置接口里没有把一批切成两个微批的开关 |
| 热专家怎么摊 | 冗余槽加每 3000 步按最近 1000 步重排,默认异步搬;另有 MoonEP 每步在线复制 | 冗余槽加每 1000 步重排 | 槽位与专家脱钩;每步更新层数默认 0,即启动摆好不再动,大于 0 才在线搬 |
| 服务中改规模 | HTTP 接口改数据并行副本数,新卡权重由老卡点对点送 | 弹性专家并行,通信库选 Mooncake 或 NIXL | 基准的配置接口里没有运行中改专家并行度的开关 |

vLLM 的取舍是把「怎么通信」做成一个用户显式挑的零件,靠模块化 kernel 让任意一种都能和专家 kernel 拼上,于是能一口气接 10 种实现,代价是选错了不报错,而且一个实例只有一种。TensorRT-LLM 替用户按硬件自动挑;SGLang 让 DeepEP 在一个实例里按批切换模式,用户少踩一个坑。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。唯一的运行时接口是弹性专家并行的 `/scale_elastic_ep`(见下)。`--eplb-config` 收 JSON,也可以拆成 `--eplb-config.window_size 1000` 这样逐项给。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--enable-expert-parallel` / `-ep` · 专家整份分卡 | 启动 | 关 | 开:专家层组为 DP × PCP × TP 张卡,每卡放 1/N 个专家;DP 与 PCP 都为 1 时不走 all-to-all,只是 all-reduce | DP 或 PCP 大于 1 时有 `Using … all2all manager.`;不开且 DP 大于 1 时打 `Detected DP deployment with no --enable-expert-parallel. Falling back to AllGather+ReduceScatter dispatch/combine.` |
| `--all2all-backend` · 分发与合并用哪种实现 | 启动 | `allgather_reducescatter` | 取值见下表;`pplx`、`naive` 已删除,告警后回落默认;`flashinfer_all2allv` 是 `flashinfer_nvlink_two_sided` 的旧名 | 日志 `Using DeepEPLLAll2AllManager all2all manager.` 这类 |
| `--moe-backend` · 专家计算用哪族 kernel | 启动 | `auto` | 点名后只试这一族,FP8 遇到批格式时自动换成对应的批版本;不支持当前量化或格式直接报错 | 日志 `Using … Fp8 MoE backend out of potential backends: [...]` |
| `--enable-dbo` · 双批重叠 | 启动 | 关 | 开:all-to-all 须是 `deepep_low_latency`、`deepep_high_throughput`、`nixl_ep` 之一,否则启动断言;关级联注意力;未设 `VLLM_USE_V2_MODEL_RUNNER` 时退回 V1 | 日志 `Disabling cascade attention when DBO is enabled.`;debug 日志 `Aborting ubatching` |
| `--dbo-decode-token-threshold` · 纯 decode 批至少多少 token 才切 | 启动 | 32 | 调低:小批也切,两半都打不满 GPU;调高:只有大批重叠;不能小于微批数 | 同上 |
| `--dbo-prefill-token-threshold` · 带 prefill 的批至少多少 token 才切 | 启动 | 512 | 同上 | 同上 |
| `--ubatch-size` · 不开双批重叠时切几个微批 | 启动 | 0 | 大于 1 就按这个数切;开 `--enable-dbo` 时固定 2 | — |
| `VLLM_DBO_COMM_SMS` · 双批重叠时划给通信的 SM | 环境变量 | CUDA 20、ROCm 64 | 调高:DeepEP 高吞吐的通信快、DeepGEMM 能用的 SM 少;低延迟与 NIXL EP 不占 SM,不受影响;ROCm 上 DeepEP 高吞吐配双批重叠时强制 0 | — |
| `--enable-eplb` · 专家负载均衡 | 启动 | 关 | 开:要求 `-ep` 且 TP × PCP × DP 大于 1;只 CUDA、ROCm、XPU | 日志 `Rearranging experts …` |
| `--eplb-config.window_size` · 用最近多少步的负载 | 启动 | 1000 | 调小:只看最近的流量,对突变反应快,统计噪声大 | — |
| `--eplb-config.step_interval` · 每多少步重排一次 | 启动 | 3000;计数从 3/4 处起,首次重排在第 750 步 | 调小:跟得紧,搬权重更频繁;空转批也计步 | 日志 `Rearranging experts` 的间隔 |
| `--eplb-config.num_redundant_experts` · 全局多放几个专家副本 | 启动 | 0 | 调大:热专家能多复制,每卡多占「它 ÷ 卡数」个专家的显存,KV 容量降;逻辑数加它须被卡数整除;不开 EPLB 时非 0 直接报错 | 启动日志 KV 容量 |
| `--eplb-config.use_async` · 后台逐层搬权重 | 启动 | true | false:同步搬,重排那一步停住,rank 0 打耗时 | 日志 `Rearranged experts in … s.` |
| `--eplb-config.communicator` · 搬专家权重用什么 | 启动 | 不设:XPU 用 `torch_xccl`;装了 NIXL 用 `nixl`;弹性专家并行用 `pynccl`;否则 `torch_gloo` | `torch_nccl`、`pynccl` 与异步 EPLB 互斥 | 报错 `communicator is incompatible with async EPLB` |
| `--eplb-config.log_balancedness` 与 `.log_balancedness_interval` · 打均衡度 | 启动 | false、1 | 开:rank 0 按间隔打一行,每次多一次跨卡同步 | 日志 `EPLB step: … avg_tokens=…, max_tokens=…, balancedness=…` |
| `--expert-placement-strategy` · 初始摆位 | 启动 | `linear` | `round_robin`:隔卡轮流放;只对有专家分组、无冗余、不开 EPLB 的模型,且只配 `deepep_low_latency` 或 `nixl_ep`,否则告警回落 `linear` | 告警 `Falling back to linear expert placement` |
| `--enable-elastic-ep` · 弹性专家并行 | 启动 | 关 | 开:须 `--enable-eplb`;扩缩时须 `-dpb ray`,否则断言 `Only ray DP backend supports scaling elastic EP`;不能配 PP、外部或混合负载均衡;异步 EPLB 时要装 NIXL;runner 退回 V1 | `/is_scaling_elastic_ep` |
| `--elastic-ep-max-dp-size` · 弹性扩容的副本数上限 | 启动 | 不设:等于初始 DP | 调大:EPLB 的映射表与 NIXL EP 的缓冲按上限预留;不能小于初始 DP;扩到超过它时报错 | 报错 `Cannot scale to data_parallel_size` |
| `VLLM_ELASTIC_EP_DRAIN_REQUESTS` · 扩缩前等在途请求排空 | 环境变量 | 0 | 1:先等排空,超时用请求体里的 `drain_timeout`(默认 120 秒),超时回 408 | 日志 `waiting for requests to drain before scaling` |
| `--max-num-batched-tokens` · 每副本每步 token 上限(02 章) | 启动 | 常规默认;`deepep_low_latency` 或 `nixl_ep` 且开 `-ep`、DP 大于 1 时改为 256 | 调大:长 prompt 少切几步,低延迟缓冲按它开,显存涨 | 启动日志里的批上限 |
| `VLLM_DEEPEP_BUFFER_SIZE_MB` · DeepEP 的 NVLink 与 RDMA 缓冲 | 环境变量 | 1024 | 调小省显存;太小时大批放不下 | — |
| `VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE` · 跨机也走机内 kernel | 环境变量 | 0 | 1:GB200 这类多机 NVLink 上提高 prefill 吞吐 | — |
| `VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL` · 低延迟模式走多机 NVLink | 环境变量 | 0 | 1:GB200 这类系统上降低 decode 延迟 | — |
| `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD` · 多少 token 以内共享专家走副流 | 环境变量 | 256 | 只对非异步后端生效;调高:大批也并行,多一份输入拷贝 | — |
| `VLLM_DISABLE_SHARED_EXPERTS_STREAM` · 关掉共享专家副流 | 环境变量 | 0 | 1:排障用,共享专家串行 | debug 日志 `Disabling MoE shared_experts cuda stream` |
| `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` 与 `VLLM_RANDOMIZE_DP_DUMMY_INPUTS` · 压测时让路由均匀 | 环境变量 | 空、0 | `uniform_random` 与 1:专家负载被人为摊平,输出不再正确,只用于量通信上限 | — |

**all-to-all 各取值。** 都要求 `-ep` 且 DP 或 PCP 大于 1 才生效。

| 取值 | 是什么 | 适合 | 限制 |
|---|---|---|---|
| `allgather_reducescatter` | 分发 all-gather,合并 reduce-scatter | 默认;不装额外库 | 每卡收全部 token;同步 |
| `deepep_high_throughput` | DeepEP 高吞吐,连续布局,异步 | 多机 prefill、PD 的 prefill 实例 | 配 DP 时关 CUDA graph(08 章) |
| `deepep_low_latency` | DeepEP 低延迟,按专家分桶的定长批,异步 | 多机 decode、PD 的 decode 实例 | 每副本每步默认 256 token |
| `deepep_v2` | DeepEP 2.0 的统一接口,走 NCCL | 装了 DeepEP 2.0 的集群 | 要 NCCL 2.30.4 以上 |
| `mori_high_throughput` / `mori_low_latency` | AMD 的 MoRI | ROCm 多机 | 同步,不能配双批重叠 |
| `nixl_ep` | NIXL EP,按专家分桶的定长批,异步 | 弹性专家并行(扩缩时能复用 kernel 与图) | 每副本每步默认 256 token |
| `moonep` | MoonEP,每步在线复制专家,形状恒定 | 单机 NVSwitch | 只收不量化的 BF16 模型;不能 EPLB;只 `linear` 摆位 |
| `flashinfer_nvlink_one_sided` / `flashinfer_nvlink_two_sided` | FlashInfer 的单边、双边 all-to-all | 跨机 NVLink(MNNVL)系统 | 同步;双边时共享专家不重叠 |

**怎么看。** 一是启动日志:DP 或 PCP 大于 1 时 EP 组打一行 `Using DeepEPHTAll2AllManager all2all manager.` 这类,没有这一行就说明点名的后端没生效,有这一行还要确认开了 `-ep`;专家 kernel 打 `Using TRITON Fp8 MoE backend out of potential backends: [...]` 这类,方括号是按优先级排好的候选;DeepEP 高吞吐关图时打 `DeepEP: Disabling CUDA Graphs since DeepEP high-throughput kernels are optimized for prefill …`。二是 EPLB:`Rearranging experts (async mode) ...` 标出每次重排,同步模式另有 `Rearranged experts in … s.` 给耗时;打开均衡度后 rank 0 按间隔打 `EPLB step: … balancedness=…`,值是各卡平均 token 数除以最多那张卡,越接近 1 越均。三是接口:`POST /scale_elastic_ep` 带 `{"new_data_parallel_size": N}` 触发扩缩,`/is_scaling_elastic_ep` 查是否在切。多机 InfiniBand 集群初始化卡住,官方文档让先设 `GLOO_SOCKET_IFNAME=eth0`。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单机 8 张 H200,DeepSeek-V3,在线混合负载,不想装 DeepEP | `vllm serve deepseek-ai/DeepSeek-V3-0324 -tp 1 -dp 8 -ep` | 注意力 8 份各存各的 KV、专家每卡 1/8;默认后端每卡收下 8 个副本的全部 token,单机 NVLink 上扛得住 |
| 两台 8 张 H200,DeepSeek-V3,PD 分离里的 decode 实例 | 主节点 `vllm serve deepseek-ai/DeepSeek-V3-0324 -dp 16 -dpl 8 -dpa <主节点IP> -dpp 13345 -ep --all2all-backend deepep_low_latency --enable-eplb --eplb-config.num_redundant_experts 32 --enable-dbo`;第二台用同一条命令,末尾追加 `--headless -dpr 8` | 低延迟 all-to-all 可进图;32 个冗余专家让每卡多放 2 个专家的权重,换热专家不再拖住全组;双批重叠让 runner 退回 V1;每副本每步只收 256 token,长 prompt 交给 prefill 实例 |
| 同一集群里 PD 分离的 prefill 实例 | 同上,只把后端换成 `--all2all-backend deepep_high_throughput` | 用连续布局的大块传输换 prefill 吞吐;CUDA graph 被关(08 章),不适合再兼 decode;批上限回到常规默认 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 点名了 DeepEP,日志里却没有 `Using … all2all manager` | DP 与 PCP 是不是都为 1 | 专家并行要配 DP 大于 1 才走 all-to-all;单副本 TP 下它只是 all-reduce |
| 启动断言 `Microbatching currently only supports the deepep_low_latency, deepep_high_throughput, and nixl_ep all2all backends` | 开了双批重叠却用默认后端 | 换成这三种之一 |
| 开了双批重叠,吞吐没变 | 有没有退回 V1 的告警(07 章);每步 token 数是否低于阈值;debug 日志里 `Aborting ubatching` 多不多 | 调低阈值;各副本负载差得多时先解决分配(10 章) |
| 用 DeepEP 高吞吐后 decode 很慢 | 有没有 `DeepEP: Disabling CUDA Graphs` | decode 改用 `deepep_low_latency`,或拆 PD 两套实例 |
| 低延迟后端下长 prompt 的 TTFT 很长 | 批上限是不是被降成了 256 | 显式调大 `--max-num-batched-tokens`,缓冲显存跟着涨;或交给 prefill 实例 |
| 某张卡总是最慢 | 打开 `log_balancedness` 看 `balancedness` | 开 `--enable-eplb` 并给冗余专家 |
| 每隔一段时间整组卡一下 | 有没有 `Rearranged experts in … s.` | 改回异步 EPLB,或调大 `step_interval` |
| 报 `num_redundant_experts is set to … but EPLB is not enabled` | 只给了冗余数 | 加 `--enable-eplb` |
| 开 EPLB 后 MoE 层变慢 | 后端是不是 `allgather_reducescatter`、`deepep_v2`、`flashinfer_nvlink_one_sided` 以外的那几种 | 共享专家不再重叠是预期;可换这三种之一 |
| 报 `enable_expert_parallel must be True to use EPLB.` | 没开 `-ep` | 加上 |
| `moonep` 启动报错 | 模型是不是量化的,是否开了 EPLB | 换别的后端 |
| InfiniBand 下报 `cannot register cq buf` | 宿主机与容器的 `ulimit -l` | 设成 unlimited |
| 报 `init failed for transport: IBGDA` | 节点缺 IBGDA 内核模块 | 按官方文档在每台 GPU 节点跑驱动配置脚本后重启 |

## 六、常见误区

- **以为开了专家并行就走 all-to-all,DeepEP 也就用上了。** 名字叫专家并行,官方 MoE kernel 文档还写着后端能配「EP+DP 或 EP+TP」。代码里只有 DP 或 PCP 大于 1 才建 all-to-all;单机 TP 8 加 `-ep` 时每卡放整份专家、最后一次 all-reduce,`--all2all-backend` 填什么都被静默忽略。先看日志里有没有 `Using … all2all manager`,再确认没有 `Detected DP deployment with no --enable-expert-parallel`:DP 大于 1 却没开 `-ep` 时,前一行照样会打,token 走的却是把专家按 TP 切开的那条路。
- **以为默认后端是真正的 all-to-all。** 参数名叫 all-to-all 后端,默认值又是第一个列出来的,很容易当成标准实现。它的分发是 all-gather,每张卡收下全组所有 token 再只算自己的;16 卡、256 选 8、路由均匀时约 59% 是白收。多机大规模部署不换后端,网先满。
- **以为冗余专家数是「每卡多放几个」。** 官方文档的参数表写的就是「per EP rank」。代码里它是全局总数,加在逻辑专家数上再均分到各卡;DeepSeek 系 256 个专家配 32 个冗余、16 张卡,每卡多放 2 个,不是 32 个。按每卡去算显存,会把 KV 预算算少一大截。
- **以为双批重叠开了就一直在重叠。** 它每步由所有副本一起决定:任何一个副本 token 太少、切完第二半会空,或者批里有前缀读写交叉,整步都不切,只打 debug 日志;默认还会让 runner 退回 V1。官方双批重叠文档说只支持 DeepEP,代码还收 NIXL EP。看效果要先确认它真切了。
- **以为 EPLB 是按负载变化触发的。** 叫负载均衡,自然以为哪张卡热了才动。实际按步数定时:默认每 3000 步一次,首次在第 750 步,只看重排前最近 1000 步;空转批也推进计数但不计负载。服务刚起不久就会重排一次,那时的负载未必代表稳态流量。
- **以为 DeepEP 两种模式能按批自动切。** 用过 SGLang 的人会这么以为,那边默认按批里有没有 prefill 切换。vLLM 一个实例只有一种;混合负载选高吞吐,decode 丢了 CUDA graph,选低延迟,prefill 被 256 的批上限切碎。官方的答案是 PD 分离两套实例。
- **以为 `/scale_elastic_ep` 只在开了弹性专家并行时才有,而且受 API key 保护。** 只要是生成模型,这个路由就挂在 HTTP 上;官方安全文档把它和 `/pause` 一起列为不带 token 就能调、能造成拒绝服务的接口。对外暴露端口前要在网关挡掉。

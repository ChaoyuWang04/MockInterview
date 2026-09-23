# vLLM 10|并行:TP、PP、DP 与上下文并行

一张卡放不下权重、放不下长上下文的 KV,或者一份模型扛不住流量时,vLLM 沿哪几维把活切开,每切一维又要在哪里多付一次同步。

说法都在源码基准 `94f4170df3`(tag `v0.30.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

第一处坏在**能不能起来**。一个 80 层、隐藏维 8192 的 70B 级稠密模型,BF16 权重约 140 GB,一张 80 GB 的卡连权重都放不下。切开之后又多出通信:按 TP 切,每层要把激活拼回来 2 次,80 层一步就是 160 次 all-reduce;decode 一步 64 个 token 时每次只有 1 MiB,贵的不是带宽,是每次都要走一遍的固定延迟,TPOT 被它按住。

第二处坏在**并发**。长上下文时 KV 比权重还大,而 TP 切 KV 只能按头切。KV 头数少于 TP 时,多出来的卡只能复制:MLA 模型的 KV 只有 1 份共享潜向量,TP 8 就是 8 张卡各存一份一模一样的 KV,加卡不加 KV 容量,并发上不去。

第三处坏在**数据并行的副本之间**。稠密模型的副本互不相干,不均只是分配问题。MoE 模型不一样:专家层的通信组横跨所有副本,每一步前向全体都得到场做一次集合通信。某个副本这一步没有请求,它也不能歇着,否则其余副本卡死在通信里;而一步的时长由最慢的副本定,一个副本在做长 prefill,其余副本的 decode 都陪着等,吞吐被最慢的那个拖住。

## 二、解法:哪一维放不下就切哪一维,切到哪一维就在哪一维上对齐

最天然的直觉是按维度切,vLLM 也是这么做的。卡按一张网格编号,最内层是 TP,往外依次是预填充上下文并行、PP、DP;相邻编号的卡先组成 TP 组,所以同一台机器上的卡应该先给 TP。每一维一个进程组,各走各的通信。

**TP:每次拼回来都先问专用实现,NCCL 兜底。** 权重按注意力头和 MLP 的列行切开,每层在注意力输出和 MLP 输出处各做 1 次 all-reduce。这一步的消息很小,所以 TP 组的 all-reduce 不直接交给 NCCL,而是按固定顺序问一串实现,第一个肯接的做:FlashInfer 的 all-reduce、vLLM 自写的 all-reduce、PyTorch 对称内存,最后才是 NCCL。前三个各有一个按卡型和卡数查表的消息上限,H100 8 卡时 FlashInfer 只接 0.5 MB 以内的,对称内存接到 64 MB。对称内存要 NVSwitch 的多播,自写的那一个超过 2 张卡就要求 NVLink 全连,PCIe 机器上直接关掉;给 PCIe 单机另有一条 FlashInfer 实现,要手动打开。启动日志会按问的顺序列出这一组里真正可用的实现。别的进程组只用 NCCL。KV 头按 TP 切,头数不够分时每张卡复制一份。

**PP:层分段接力,末段采样再广播回去。** 层按段均分,除不尽时多出的层从倒数第二段往前各加 1 层,末段不加,因为它还扛着输出层。激活点对点传给下一段。只有末段做采样,采出的 token 要广播回前面各段,这次广播用单独一个通信器、放在副流上发,不跟激活的点对点抢线。Model Runner V2 下,同一条请求相邻两次 decode 至少隔段数那么多步,于是在跑的请求自然分成段数份轮流上,并发够分成这么多份流水线才满;批队列深度见 07 章。

**DP:稠密模型各跑各的,MoE 模型步调一致。** 每个副本一个 EngineCore(01 章)。稠密模型的副本在启动时就把自己当成单副本,副本之间不建任何通信;数据并行时另有一个协调进程,对稠密模型只做一件事,把各副本的排队数与 KV 占用汇总给 API 进程挑副本,打分方法在 01 章。MoE 模型的专家层按「数据并行度 × TP」张卡组成一个通信组(切法见 11 章),副本就必须一起走:

![MoE 模型开 4 个数据并行副本时一个请求波次的时间线:引擎全体暂停时,新请求发给 DP1,API 进程同时通知协调进程,协调进程向 DP0、DP2、DP3 广播开新波次;此后 4 个副本每一步都一起前向,有请求的副本跑真批,没请求的跑空转批;第 1 步和每第 16 步全组做一次「谁还有活」的同步,第 16 步时 DP1、DP2 还有活就继续,第 28 步起所有请求都已结束,但要空转到第 32 步的同步步才一起暂停,由 DP0 向协调进程报波次结束、波次号加 1;稠密模型不走这一套](/opensource/vllm-lite/10a-dp-wave-lockstep.svg)

这张图里有三件事。一是**空转批**:没请求的副本跑一个只有 1 个 token 的假 decode 陪着;每一步前向之前,各副本先在 CPU 上交换一次本步 token 数,按最大值补齐 CUDA graph 的形状,保证大家走同一张图、进同一次集合通信。二是**波次**:全体在「运行」和「暂停」两个状态之间整体切换;运行时每隔固定步数做一次 all-reduce,问「谁还有活」,全员都没活才一起停下,这一次从运行到暂停算一个波次。三是**协调进程叫醒**:暂停时新请求只会落到一个副本,API 进程发请求的同时通知协调进程,由它广播「开新波次」叫醒其余副本。另有一个可选的节拍:所有副本只在同一批步上收新的 prefill,不让某一个副本单独做 prefill 把全体拖慢。

**上下文并行:两种,各治一段。** 解码上下文并行(DCP)不加卡,复用 TP 组里的卡:原本复制的那几份 KV 改成按 token 位置轮流存,第 i 个 token 放在组内第「i 除以组大小的余数」号卡上,每张卡只存 1/c。decode 时各卡对自己那份 KV 算部分注意力,再按 log-sum-exp 合并。对 GQA 模型,组大小最多到「TP ÷ KV 头数」,正好把复制消掉;MLA 模型可以一直开到 TP。预填充上下文并行(PCP)要加卡:一条长 prompt 切成 2 倍组大小那么多块,第 i 张卡拿第 i 块和倒数第 i 块,因果注意力下前面的块算得少、后面的块算得多,这样配对各卡算量相当;decode 不切,各卡复制。PCP 算完的 KV 会在组内 all-gather,完整写进每张卡,KV 一个字节不省,省的是长 prompt 的 prefill 时间。

**多机怎么起,负载怎么分。** 起法有两种。多进程:每台机器跑同一条命令,告诉它一共几台、自己第几台、主节点地址,非主节点只起 worker、不起 HTTP。Ray:在一台机器上一条命令拉起全部。数据并行跨机时负载均衡有三种模式:内部模式只有主节点有 API 进程,一个入口,按协调进程推来的负载挑副本;混合模式每台机器各有 API 进程,只往本机副本发,机器之间交给上游负载均衡器;外部模式每个副本一个独立端点,全交外部路由,只给 MoE 用,稠密模型直接起互不相干的独立实例就行。另有一种变体,由每台机器上一个监督进程替本机每个副本各起一个外部模式的端点。

为什么实际不吃亏:TP 的消息大小跟这一步的 token 数走,不跟模型大小走,小消息交给专用实现省掉固定延迟;稠密模型的 DP 完全没有同步;MoE 的同步大部分是 CPU 上交换几个整数,「谁还有活」默认 16 步才问一次;DCP 每层多出的通信和上下文长度无关,省下的 KV 却随上下文线性增长。

## 三、代价

**TP 每层都要等。** 每层 2 次 all-reduce 是硬成本,TP 跨机后走网卡,TPOT 明显变长。PCIe 机器超过 2 张卡时,自写和对称内存都用不上,只剩 NCCL 和一条要手动打开的 FlashInfer PCIe 实现。自写那一个跨机只在 Blackwell 的多节点 NVLink 域里能开。注意力头数必须能被 TP 整除;KV 头数少于 TP 时 KV 复制,显存不按 TP 线性降。

**PP 有气泡,单条请求更慢。** 在跑的请求不够分成段数份时,总有段在空等。同一条请求相邻两次 decode 隔着段数步,单流的 TPOT 跟着变长;PP 换来的是能把更大的模型装下、或者避开机间的 TP,不是更低的延迟。模型要实现流水线接口才能开;层数除不尽时最重的那一段决定一步。

**MoE 的 DP 在陪跑上烧卡。** 没请求的副本也在跑空转批,走图时还被补齐到最忙那个副本的形状。所有请求结束后,还要空转到下一个同步步才能停,默认最多多跑 15 步。一个副本在做 prefill,其余副本整步陪等。所有副本的并行配置必须一模一样,启动时逐个核对。

**DCP 每层多 1 到 3 次组内通信。** MLA 默认是先把 query 全组 all-gather,算完再 all-gather 合并用的 log-sum-exp、reduce-scatter 输出,一共 3 次;换成 all-to-all 合并少 1 次,把 query 投影复制到每张卡再省 1 次。调度器看到的块大小乘上组大小,前缀缓存的命中粒度随之变粗。滑动窗口、分块局部注意力和 Mamba 层不支持;PD 分离下只有 MLA 能开(12 章)。

**PCP 的限制最多。** 只支持 MLA 模型,只在 Model Runner V2 上有,不能和 PP 同开,CUDA 上不能和 DP 同开;要加卡,KV 一点不省。

**多机要各起一条命令。** 多进程方式下每台机器都要起、端口和地址都要对上;官方文档里「多机默认 Ray」的说法和代码不符,见第六节。

## 四、和 SGLang、TensorRT-LLM 比:没活的副本怎么陪跑

对照对象是 SGLang(基准 826d5170ae)与 TensorRT-LLM(基准 59f5c47f2e)。三家在 MoE 部署里都要让没请求的副本陪着进集合通信,也都要把 TP 的小 all-reduce 做快,分歧在副本之间靠什么对齐、多久对一次,以及请求由谁分给副本。

| 必须有人做的事 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| MoE 部署里的一个 DP 副本是什么 | 一个独立的 EngineCore 进程,专家层的通信组横跨所有副本 | 注意力数据并行:一个 TP 组切出几个副本,各 rank 的调度器在组内对齐 | 注意力数据并行:同一执行器里的一组卡改成各接各的请求 |
| 这一步某个副本没请求 | 跑 1 个 token 的空转批,走图时补齐到最忙副本的形状 | 跑空转批 | 塞一条假请求 |
| 多久对一次齐 | 每步交换 token 数补齐形状;「全体还有没有活」默认 16 步问一次,全员没活才整体暂停 | 每一轮调度都在组内 all-gather 各副本的 token 数,全组都是 0 才不跑空转批 | 每次取新请求都在组内 all-gather 各 rank 的状态 |
| 全体暂停后谁叫醒别人 | 单独的协调进程广播「开新波次」 | 没有单独的协调进程 | 没有单独的协调进程 |
| 请求怎么分到副本 | API 进程按协调进程推来的负载挑,或交外部路由 | 单独的数据并行控制进程按策略分发 | 执行器里的路由器在各 rank 状态汇齐后分配,有按 KV 命中分配的版本 |
| TP 的 all-reduce 用谁 | FlashInfer、自写、对称内存按上限依次试,NCCL 兜底 | 多条加速路按顺序查,自写那条默认开 | 一个策略字段选实现,默认自动 |
| 多机怎么起 | 每台同一条命令,非主节点不起 HTTP;或 Ray 一条命令 | 每台同一条命令,只改节点序号 | 靠 MPI 或 Slurm 拉起 |

vLLM 的取舍是把副本做成互相独立的引擎进程,稠密模型因此零同步、可以各自扩缩;代价是 MoE 模型要额外补一个协调进程和波次协议,才能让一群独立进程在需要时步调一致,并且只能按固定间隔发现「大家都闲了」。SGLang 和 TensorRT-LLM 把副本放在同一个通信组里、每轮都对齐一次,用不着另起一个进程去叫醒谁。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。并行度、负载均衡模式、通信实现都没有运行时接口可改;多机部署时每台机器的这些参数必须一致。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--tensor-parallel-size` / `-tp` · TP 度 | 启动 | 1 | 调高:每卡权重与 KV 变少、并发升;每层 2 次 all-reduce,跨机后 TPOT 明显升;注意力头数须能整除;KV 头数少于它时 KV 复制 | 日志 `rank N in world size M is assigned as DP rank …, PP rank …, PCP rank …, TP rank …` |
| `--pipeline-parallel-size` / `-pp` · 流水线段数 | 启动 | 1 | 大于 1:能装下更大的模型、能避开机间 TP;单条请求的 TPOT 变长,在跑请求不够分成段数份时有气泡;模型须实现流水线接口 | 日志 `Hidden layers were unevenly partitioned: [...]` |
| `VLLM_PP_LAYER_PARTITION` · 每段几层,逗号分隔 | 环境变量 | 不设:均分,余数从倒数第二段往前各加 1 | 手写:个数不等于段数或总和不等于层数直接报错;把重的段调轻 | 报错 `does not match pp_size` 或 `does not match num_hidden_layers` |
| `--data-parallel-size` / `-dp` · 数据并行副本数 | 启动 | 1 | 调高:吞吐按副本加;稠密模型副本互不通信;MoE 模型专家层组变成 DP × TP 张卡、副本步调一致;`--max-num-seqs` 按每个副本算 | `ps` 里 `VLLM::EngineCore_DP0`…与 `VLLM::DPCoordinator`;日志 `Started DP Coordinator process` |
| `--data-parallel-size-local` / `-dpl` · 本机起几个副本 | 启动 | 不设:单机等于副本数;多进程跨机时按节点数推 | 设 0:本机只起 API 进程;跨机时每台按实际卡数填 | 各节点 `EngineCore_DP` 进程数 |
| `--data-parallel-start-rank` / `-dpr` · 本机第一个副本的序号 | 启动 | 不设 | 非主节点设它;不带 `--headless` 时自动进混合模式 | 日志 `Inferred data_parallel_rank` |
| `--data-parallel-hybrid-lb` / `-dph` · 混合负载均衡 | 启动 | 关 | 开:每台机器自己的 API 进程只发本机副本;必须给本机副本数,非主节点再给起始序号;不能配 `--headless`;本机只有 1 个副本时自动改成外部模式 | 每台机器都有 HTTP 端口 |
| `--data-parallel-external-lb` / `-dpe`,或直接给 `--data-parallel-rank` / `-dpn` · 外部负载均衡 | 启动 | 关 | 开:每个副本一个端点、本机副本数固定 1;只 MoE 可用,稠密模型报错;协调进程仍在 0 号副本旁边跑波次 | 报错 `Non-MoE models do not support external data parallel mode` |
| `--data-parallel-multi-port-external-lb` / `-dpm` · 每台机器一个监督进程,替本机每个副本各起一个外部模式端点 | 启动 | 关 | 开:端点端口为 `--port` 加本地序号,聚合健康检查在 `--data-parallel-supervisor-port`(默认 9256);本机副本数至少 2;和上两种模式互斥 | 报错 `Cannot use more than one data parallel load balancing mode` |
| `--data-parallel-address` / `-dpa`、`--data-parallel-rpc-port` / `-dpp` · 副本握手的主节点地址与端口 | 启动 | 不设:多进程用 `--master-addr`(127.0.0.1),Ray 用本机 IP;端口 29550 | 跨机时所有节点填同一个 | 启动卡在等远端引擎 |
| `--data-parallel-backend` / `-dpb` · 副本怎么拉起 | 启动 | `mp` | `ray`:一条命令起全部副本,不用填地址与端口;和 `--nnodes` 大于 1 不能同用 | 报错里提示 `--nnodes` 需要 `--data-parallel-backend mp` |
| `--dp-sync-interval` · MoE 副本每几步问一次「谁还有活」 | 启动 | 16 | 调大:同步次数少;请求全部结束后最多多空转「它减 1」步;所有副本必须一致 | debug 日志 `Wave N finished, pausing engine loop.` |
| `--prefill-schedule-interval` · MoE 副本每几步才一起收新 prefill | 启动 | 1 | 设 4–8:各副本在同一批步上一起做 prefill,不再一个副本拖全体,步长抖动小;新请求 TTFT 最多多等「它减 1」步;排队积压时自动不限;只对 MoE 的 DP 生效 | 各副本 `vllm:inter_token_latency_seconds` 的尾部 |
| `--disable-nccl-for-dp-synchronization` · DP 补齐形状的交换走不走 NCCL | 启动 | 不设:开异步调度时为真(走 CPU) | 只影响 V1 ModelRunner;V2 始终走 CPU 组 | 日志 `Disabling NCCL for DP synchronization when using async scheduling.` |
| `--nnodes` / `-n`、`--node-rank` / `-r`、`--master-addr`、`--master-port` · 多进程跨机 | 启动 | 1、0、127.0.0.1、29501 | 每台跑同一条命令只改 `--node-rank`,非 0 号机加 `--headless`(01 章);总卡数须能被节点数整除;只能配 `mp` 执行器 | 非主节点日志 `Launching vLLM (…) headless multiproc executor` |
| `--distributed-executor-backend` · worker 怎么拉起(默认值见 01 章) | 启动 | CUDA 上设了 `--nnodes` 就是 `mp`;卡不够又没设节点数直接报错 | `ray`:单命令跨机,要先起好 Ray 集群;只有显式指定、`-dpb ray` 或已在 Ray 放置组里才会用 | 报错 `World size (…) is larger than the number of available GPUs` |
| `--distributed-timeout-seconds` · 建组与集合通信的超时 | 启动 | 不设:PyTorch 默认,NCCL 600 秒 | 多机下载或加载慢时调大 | 超时报错 |
| `--disable-custom-all-reduce` · 关掉 vLLM 自写的 all-reduce | 启动 | 关;平台不支持或批不变模式下强制开 | 开:只去掉自写那一个,FlashInfer 与对称内存照旧 | 日志 `Using [...] all-reduce backends (in dispatch order) for group 'tp…'` |
| `VLLM_ALLREDUCE_USE_FLASHINFER` · TP 组试不试 FlashInfer 的 all-reduce | 环境变量 | 1 | 0:不试;批不变模式下本来就不用 | 同上 |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` · TP 组试不试 PyTorch 对称内存 | 环境变量 | 1 | 0:不试;没有 NVSwitch 多播时它自己告警后退出 | 告警以 `SymmMemCommunicator:` 开头 |
| `VLLM_USE_NCCL_SYMM_MEM` · NCCL 对称内存 | 环境变量 | 0 | 1:4 卡以上时 16 KB 以内和 128 KB(8 卡)或 512 KB(4 卡)以上的消息改走它,排在 FlashInfer 之后、自写之前 | 同上,列表里多出 `NCCL_SYMM_MEM` |
| `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC` · PCIe 单机的 FlashInfer all-reduce | 环境变量 | 0 | 1:单机 2、4、8 张 PCIe 卡的小 all-reduce 改走它;源码注释说还在验证,所以默认关 | 同上,列表里多出 `FLASHINFER_PCIE_IPC` |
| `VLLM_SKIP_P2P_CHECK` · 信不信驱动报的 P2P | 环境变量 | 1(信) | 0:启动时实测一遍 P2P;自写 all-reduce 卡住时用,源码注释点名 535 系列驱动 | 告警 `Custom allreduce is disabled because your platform lacks GPU P2P capability` |
| `--decode-context-parallel-size` / `-dcp` · DCP 组大小 | 启动 | 1 | 大于 1:每卡 KV 变 1/c、并发升;每层多 1–3 次组内通信;须整除 TP;GQA 模型要 TP 大于 KV 头数、它不超过「TP ÷ KV 头数」、每个 KV 头对应的 query 头数能被它整除 | 报错 `Decode context parallelism for GQA/MQA requires` 或 `exceeds the maximum supported value`;启动日志 KV 容量 |
| `--dcp-comm-backend` · DCP 合并输出的通信方式 | 启动 | 不设:`ag_rs`;GLM-MoE-DSA 默认 `a2a` | `a2a`:MLA 每层集合通信从 3 次减到 2 次;PCP 同开时只能 `ag_rs` | TPOT;有对称内存时另打 `Using direct symmetric-memory DCP A2A for MLA.` |
| `--dcp-q-replicate` · 每卡复制 MLA 的 query 投影 | 启动 | 不设:关(GLM-MoE-DSA 开) | 开:省掉每层 query 的 all-gather,多算一份投影 | TPOT |
| `--cp-kv-cache-interleave-size` · DCP 下每张卡连续放几个 token | 启动 | 1;不写且接 NIXL 做 PD 时自动改成块大小 | 调大到块大小:按整块轮流放;须不大于并能整除块大小 | 启动断言 `should be greater than or equal to and divisible by` |
| `--prefill-context-parallel-size` / `-pcp` · PCP 组大小 | 启动 | 1 | 大于 1:总卡数乘它,长 prompt 的 TTFT 降、KV 不省;只 MLA、只 V2、不能配 PP、CUDA 上不能配 DP;与 DCP 同开时 DCP 只能取 1、它本身或 TP × 它 | 报错 `MRV2 PCP currently supports MLA models only.`、`PCP does not support data parallelism on CUDA yet.` |

**怎么看。** 一是启动日志:每个 rank 打一行 `rank N in world size M is assigned as DP rank …, PP rank …, PCP rank …, TP rank …`,先确认网格和你想的一样;TP 组打一行 `Using [...] all-reduce backends (in dispatch order) for group 'tp:…'`,方括号里就是这一组真正可用的实现,只剩 `'PYNCCL'` 说明专用实现一个都没起来,往上找原因(`Custom allreduce is disabled because …`、以 `SymmMemCommunicator:` 开头的告警、`Failed to initialize FlashInfer All Reduce workspace`);PP 层数不均时打 `Hidden layers were unevenly partitioned`。二是进程:数据并行时 `ps` 里有 `VLLM::DPCoordinator` 和一串 `VLLM::EngineCore_DPn`,MoE 模型把日志级别调到 debug 能看到 `Wave N finished, pausing engine loop.`。三是 `/metrics`:按副本看 `vllm:num_requests_running` 与 `vllm:num_requests_waiting`,MoE 部署下一个副本长期为 0 而 GPU 利用率不为 0 是空转批,不是泄漏。多机建组卡住时,官方排障文档建议先 `NCCL_DEBUG=TRACE` 看 NCCL 在连哪个网卡,必要时指定 `NCCL_SOCKET_IFNAME` 与 `GLOO_SOCKET_IFNAME`。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单机 8 张 L40S(48 GB,PCIe、没有 NVLink),70B 级稠密模型 BF16 在线服务 | `vllm serve <模型> -tp 2 -pp 4` | 2 张 PCIe 卡的 TP 仍能用自写 all-reduce,4 段流水线避开 8 卡 PCIe 上只剩 NCCL 的 TP;单条请求 TPOT 变长,需要至少 4 份在跑请求才填满流水线 |
| 单机 8 张 H200,DeepSeek-R1(MLA、FP8),长上下文多并发 | `vllm serve deepseek-ai/DeepSeek-R1 -tp 8 -dcp 8` | KV 从 8 份复制变成每卡 1/8,同样显存能多放约 8 倍的上下文;每层多 3 次组内通信,前缀缓存按 8 倍的块命中 |
| 两台 8 张 H200,DeepSeek-V3 这类大 MoE,专家并行加数据并行,单一入口 | 主节点 `vllm serve deepseek-ai/DeepSeek-V3 -dp 16 -dpl 8 -dpa <主节点IP> -dpp 13345 -ep --prefill-schedule-interval 4`;另一台同样参数加 `--headless -dpr 8` | 注意力按副本切、专家按 16 张卡切(11 章);没请求的副本陪跑空转批,prefill 对齐到每 4 步一次,新请求 TTFT 最多多等 3 步 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 启动报 `World size (…) is larger than the number of available GPUs` | 是不是想跨机却没设节点数 | 多进程:每台加 `--nnodes`、`--node-rank`、`--master-addr`,非 0 号机 `--headless`;或显式 `--distributed-executor-backend ray` |
| 启动报 `Total number of attention heads (…) must be divisible by tensor parallel size` | 头数与 TP | 换能整除的 TP,剩下的卡给 PP |
| TP 8 卡 TPOT 比预期差一大截 | `all-reduce backends` 那行是不是只剩 `'PYNCCL'`;有没有 `more than two PCIe-only GPUs` 告警 | PCIe 机器改成小 TP 加 PP,或试 `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`;NVLink 机器设 `VLLM_SKIP_P2P_CHECK=0` 实测 P2P |
| 以为关了自写 all-reduce 就是纯 NCCL,结果和别的环境对不上 | `all-reduce backends` 列表 | 再设 `VLLM_ALLREDUCE_USE_FLASHINFER=0` 与 `VLLM_ALLREDUCE_USE_SYMM_MEM=0` |
| 多机启动卡在建组 | NCCL 走的网卡、各节点地址端口是否一致 | `NCCL_DEBUG=TRACE` 看连接;指定网卡;调大 `--distributed-timeout-seconds` |
| 启动报 `Configuration mismatch detected for engine` | 各节点命令行是否完全一致 | MoE 数据并行下所有副本的并行相关参数必须相同 |
| MoE 部署里没请求的副本 GPU 也不闲 | 是不是空转批 | 设计如此;副本间长期不均先看 01 章的挑选,或减副本数 |
| MoE 部署 TPOT 周期性跳高 | 跳高时刻是否有副本在做长 prefill | 设 `--prefill-schedule-interval` 4–8;再配 02 章的单条 prefill 上限 |
| 稠密模型给了 `--data-parallel-rank` 报错 | 报错 `Non-MoE models do not support external data parallel mode` | 稠密模型起独立实例,外面自己挂负载均衡 |
| 开了 DCP 启动报 `exceeds the maximum supported value` | KV 头数 × DCP 是否超过 TP | GQA 模型按「TP ÷ KV 头数」设 |
| 开 DCP 报 `Decode Context Parallelism (DCP) requires attention implementations to return the softmax LSE` | 当前注意力后端 decode 时交不出 log-sum-exp | 换一个支持 DCP 的后端(06 章),或关掉 DCP |
| 开了 DCP 后前缀命中率掉了 | 命中粒度变成块大小 × DCP | 预期代价;前缀短的负载别开太大 |
| 开 PCP 起不来 | 是不是非 MLA 模型、是否同开了 DP 或 PP | 非 MLA 用 DCP 省 KV;TTFT 靠 02 章的分块 prefill |

## 六、常见误区

- **以为多机默认走 Ray。** 官方并行文档写着「多机默认 Ray、单机默认多进程」,于是在两台机器上直接给一个超过单机卡数的 TP,等它自己去连 Ray。代码里 CUDA 上卡不够又没给节点数会直接报错,不会自动转 Ray;Ray 只在显式指定、数据并行后端选了 Ray、或进程本来就在 Ray 放置组里时才用。反过来,设了 `--nnodes` 就只能用多进程执行器。
- **以为稠密模型开 DP 也要像 MoE 那样同步。** 数据并行文档把 MoE 的空转批和协调进程写在同一页,开头又说对稠密与 MoE 都适用,读起来像一套机制。稠密模型的副本在启动时就被改成单副本,彼此不建通信组、不跑空转批,协调进程只收负载;也正因为这样,稠密模型给外部模式的参数直接报错,官方让你起独立实例。
- **看到 MoE 部署里空闲副本 GPU 满载,以为请求泄漏或卡死。** 大家习惯「没请求就空闲」。MoE 的数据并行下,只要全组有一个副本有活,其余副本每步都跑空转批,走图时还被补齐到最忙那个副本的形状;最后一条请求结束后还要空转到下一个同步步,默认最多 15 步。真要确认,看 debug 日志里有没有 `Wave N finished`。
- **以为 `--disable-custom-all-reduce` 就等于回到 NCCL。** 名字和旧版行为都这么暗示,对照实验时常靠它排除自写通信。当前 TP 组的 all-reduce 前面还排着 FlashInfer 和 PyTorch 对称内存,两个都默认开,这个参数只去掉自写那一个;要纯 NCCL 得把两个环境变量也设 0,以 `all-reduce backends` 那行日志为准。
- **在 PCIe 多卡机器上照样开大 TP。** 单机多卡的通用建议是「单机用 TP」。自写 all-reduce 在超过 2 张没有 NVLink 全连的卡上直接关掉,只打一行告警,对称内存又要 NVSwitch 多播,8 卡 PCIe 的 TP 默认每层都走 NCCL,PCIe 专用的那条 FlashInfer 实现要手动打开;官方文档自己也建议这种机器用 PP 代替 TP。
- **以为 DCP 能随便开到 TP。** 官方上下文并行文档举的都是 MLA 模型开到等于 TP 的例子。GQA 模型的上限是「TP ÷ KV 头数」,KV 头数不少于 TP 的模型一开就报错,每个 KV 头对应的 query 头数还得能被它整除;只有 MLA 能一路开到 TP。
- **以为 PCP 省 KV。** 名字里有「上下文」,和 DCP 并列,容易当成一对省显存的开关。PCP 算完的 KV 在组内 all-gather 回每张卡,一个字节不省,省的只是长 prompt 的 prefill 时间,还要多占卡;省 KV 的是 DCP。官方文档把 PCP 的两种做法都写成「开发中」,代码里 V2 的 PCP 已经能用,但只收 MLA 模型、不能配 PP,CUDA 上不能配 DP。
- **以为 `--max-num-seqs` 是整个服务的并发上限。** 单实例时它确实是。开了数据并行,它按每个副本算,4 个副本就是 4 倍;而 02 章的 `--max-num-queued-reqs` 是整个服务一个计数。两个按单实例的习惯设成一样大,满载时会在副本还没跑满前就开始回 503。

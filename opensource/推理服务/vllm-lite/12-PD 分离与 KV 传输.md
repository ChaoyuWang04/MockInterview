# vLLM 12|PD 分离与 KV 传输

prefill 和 decode 拆到两组实例之后,一条请求怎么在两边之间交接、KV 由谁发起搬运、两边形状不同怎么对齐、出了事谁收拾,以及多模态编码器怎么再拆出一段。

说法都在源码基准 `94f4170df3`(tag `v0.30.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

混部时 prefill 和 decode 挤在同一次前向里,一条长 prompt 的续算块会把这一步撑到满额(机制见 02 章)。拿 Qwen3-8B 在 H100 上算:纯 decode 的一步主要是把 16 GB 权重读一遍,约 5 毫秒;塞进 8096 个 prefill token 后,光矩阵乘就要约 1.3e14 次浮点运算,跑满 BF16 峰值也要约 134 毫秒,在跑的每条流这一个 token 间隔涨了 27 倍;一条 32768 token 的 prompt 至少要连着 4 步这样的胖步,每条流连停 0.5 秒以上。坏的是 **TPOT 的尾部**,均值看不出来。

调小总额能把尖刺压低,但 prompt 切得越碎,TTFT 越长;官方文档的原话是合适的切块大小在实践中很难定。更根本的是两段要的东西不一样:prefill 吃算力,想要更大的 TP 把一条 prompt 算快;decode 吃带宽,想要更大的批把权重读一遍摊给更多请求。同一组卡只能选一种并行方式、一个批大小。

多模态模型再多一段。图像、音频要先过编码器,算出的嵌入进了序列 prefill 才能开始;编码器同样吃算力,和 prefill 在同一张卡上排队,图多的负载下 **TTFT** 跟着编码器一起涨;每个只做 decode 的实例也得把编码器权重装进显存,哪怕 decode 阶段用不上它。

## 二、解法:两组实例各算各的,KV 由连接器交接

最天然的直觉:prefill 和 decode 各起一组实例,前面放一个代理,prefill 算完把 KV 交给 decode。vLLM 就是这么做的,而且引擎本身不知道另一半在哪。两组实例之间不互相登记,配对全靠代理:它把 P 算完返回的一小段传输参数原样塞进发给 D 的请求,这段参数里有 P 的实例编号、块号、握手地址和 TP 大小。

![一条请求在 P、代理、D 三条泳道上的时间线。上半是默认的拉:代理先把请求发给 P 并把最多生成数改成 1,P 算完整条 prompt 后块不还,灰条表示这些块扣着等 D 来读,租约 30 秒、D 每 5 秒续一次;代理拿到 P 的块号再转发给 D,D 分好块停在门外(红条),然后发起 NIXL 读,读完通知 P 才还块,D 重算最后 1 个 prompt token 并采出首 token,P 采的那个被代理丢掉。下半是推:代理同时发给两边,D 进门就分好块并把块号登记给 P,红条一直延续到 P 算完用 NIXL 写进来为止,P 写完即还块。红条与灰条的长度就是两边空占块的时间](/opensource/vllm-lite/12a-pd-pull-vs-push-timeline.svg)

**D 这边不另开一条路。** 远端 KV 在 D 的调度器眼里就是一种外部前缀命中:先查本地前缀缓存(04 章),再问连接器「外面还有多少」,得到 prompt 长度减去本地命中;按这个数分好块,把请求停在「等远端 KV」的状态,块占着、不进批(准入与延迟队列见 02 章)。读完的消息回来,请求回到等待队列,已算长度等于整条 prompt,调度器按「整条命中」的老规矩退 1 个 token,于是 D 重算最后一个 prompt token、自己采出首 token。P 这边也一样平常:请求照常 prefill,结束时调度器在还块之前问一句连接器,连接器答「这些块我先扣着」,顺手把传输参数塞进响应。

**连接器是一个两半的接口。** 每个实例里连接器被构造两次,调度器进程一份,每个 worker 一份;两半只靠每步随调度结果下发的元数据、和随前向结果回传的完成清单说话。

- 调度器那一半有 6 个时刻:新请求进门时登记(D 从这时起就给 P 发心跳);查外部命中时回答「多少 token、是不是异步」;分好块后记下要收哪些块;每步打包下发的元数据;收到 worker 的完成清单时更新状态;请求结束时决定块先不还,并给出返回客户端的传输参数。
- worker 那一半:启动时把 KV 显存交给传输库登记;每步前向前绑定元数据、开始加载;要逐层流水的实现在每层注意力里等这一层到、发这一层走;前向后等保存;每步回报哪些请求收完、发完、收失败。Model Runner V2 下,异步加载在本步前向发出之后才提交,不挡本步的计算。
- KV 下放(05 章)也走这套接口,是它的另一种用法。几个实现能叠成一个组合连接器:加载只从第一个说「有」的取,保存则每个都写。LMCache 有两种接法:一种把 LMCache 的适配层装进每个 worker,另一种连一个独立的 LMCache 服务进程,多个 vLLM 实例共用它的 KV。

**拉还是推。** 默认的 NIXL 连接器是 D 读 P。另有一个推的 NIXL 实现:代理同时发两边,D 一进门就分好块,经 NIXL 的通知通道把块号登记给 P,P 算完直接写进 D 的显存;P 那边一个专门的写线程负责配对「登记到了」和「块算完了」,哪边先到都行。Mooncake 连接器是同一种形状:代理给两边同一个传输号,D 先经 P 的引导服务查到 P 各个 rank 的地址,再把自己的块地址连同传输号报过去,P 算完用 Mooncake 的传输引擎写过来。两种 NIXL 协议握手时互相拒绝。

**两边形状不同时怎么对齐。**

![一个 8 个 KV 头的示意模型在两边 TP 差 2 倍时的读法:D 的 TP 大时,P 的 2 张卡各存 4 个头,D 的 4 张卡两两去读同一张 P 卡,各取其中 2 个头,每层每块 1 段,P 卡收齐 2 张 D 卡的通知才还块;P 的 TP 大时,P 的 4 张卡各存 2 个头,D 的每张卡把自己每块的头切成 2 份,各从一张 P 卡读 1 段;右侧对比一块里的两种排法,token 在外时同样 2 个头在每个 token 位里各占一小截、一块要拆成 4 段读,头在外时这 2 个头是块的前一半、1 段读完;MLA 模型各卡的 KV 一样,D 的每张卡只读一张 P 卡的整块](/opensource/vllm-lite/12b-hetero-tp-head-mapping.svg)

- TP 不同:两边 TP 必须成整数倍。几张卡读一张、一张读几张都是按头切片去读,为了让每次读的那几个头在显存里是连续的一段,NIXL 与 Mooncake 连接器把非 MLA 模型的块内排法改成头在外、token 在内。
- 块大小不同:只支持 P 的块比 D 小且能整除,D 的 1 块收 P 的几块,收完在本地重排成自己的布局;分组记账(03 章)开着时两边块大小必须相同。
- 其余的配置在第一次握手时核对:两边各算一个兼容哈希,覆盖版本、模型名、精度、KV 头数与层数、注意力后端、KV 精度、分组记账开没开、投机解码配置和推拉方式,不同就拒绝。TP 与块大小不进哈希,握手后单独校验。

**失败与超时。** P 扣着的块有一份 30 秒的租约。D 从请求进门那一刻就开始每 5 秒给 P 发一次心跳,每次把租约往后推 20 秒;心跳搭在 NIXL 的通知通道上,一条心跳续同一个 P 上的所有请求。D 读完的通知一到,块立刻还。D 挂了,心跳一停,P 最多 30 秒收回;D 只是排队排得久,心跳让块一直留着。D 这边读失败时,默认直接把这条请求判错返回;也可以改成在 D 上把没读到的部分重算。

**编码器再拆一段。** 多模态模型可以另起一个只跑编码器的实例,它不加载语言模型、不建 KV 缓存。代理先把请求里的每个媒体项发给编码器实例,拿回编码结果的句柄,再把请求交给 P;P 的调度器发现编码结果还没到,这一步先不排它,到了再排。编码结果的搬运走另一套同样分两半的接口,内置 3 个实现:经共享文件系统、经 Mooncake,以及一个带 CPU 缓存层、可经 NIXL 点对点读的实现。最后这个读不到时,请求里还留着原始媒体就退回本地编码,否则判错。编码缓存本身见 15 章。

为什么实际不吃亏:按 03 章那个模型(32 层、8 个 KV 头、头维 128、BF16),8192 token 的 KV 是 1 GiB(每 token 128 KiB,见 03 章),400 Gb/s 的网卡约 21 毫秒;参数量按 82 亿算,这段 prompt 的 prefill 在 H100 上至少约 136 毫秒。D 等 KV 时只有这一条在门外,别的请求每步照算;P 交出之后也不再为这条请求做任何计算。

## 三、代价

**不提吞吐,只换尾延迟。** 官方文档明说拆开不提升吞吐。同样多的卡分成两组,每组的批都变小;P 组与 D 组的负载配比一旦对不上,一边闲着一边排队。换来的是 TPOT 不再被 prefill 打断、两段能各选并行方式。

**TTFT 多出几段。** 默认的拉要串行走两次 HTTP,代理等 P 回完才发 D;D 收进门后要等读完的消息回到调度器,再花 1 步重算最后一个 token。第一次碰到某个 P 时还要先握手。推省掉了中间那次往返,代价见下一条。

**块在两边都有空占。** 拉的时候,P 算完到 D 读完之间块扣在 P 上;D 慢或排队长,P 的池子被这些等交出的块吃掉,新的 prefill 进不来。D 这边,请求一进门就按整条 prompt 占块,调度器还只在空块够所有在途读取跑完时才收新的远端请求。推的时候,D 的块在 P 算的整个过程里空占。

**前缀缓存各管各的。** P 和 D 各有一份块哈希缓存,D 本地已命中的块不再读。多轮对话上一轮的回答只在 D 上,P 默认看不到,下一轮整段重算;要复用得打开双向传输,让 P 先从 D 把旧块读回来,还要一个按会话记住上一轮 D 块号的有状态代理。

**能配的组合打了折。** 编码器-解码器模型挂连接器起不来;混合 Mamba 模型两边 TP 必须相同;动态量化的 KV 不支持,每块的缩放因子不随块传;DCP 在 PD 下只有 MLA 能开(10 章);不支持分组记账的连接器会让它自动关掉(03 章);推的实现不支持 DCP,也不支持双向传输。

**失败默认直接报错。** 默认策略下,D 读失败的请求以错误结束,客户端自己重试。改成在 D 上重算,等于在为 decode 调好的实例上跑一整段 prefill,又回到了拆开要避免的那种打断。

## 四、和 SGLang、TensorRT-LLM 比:谁先动手,首 token 谁出

对照对象是 SGLang(基准 826d5170ae)与 TensorRT-LLM(基准 59f5c47f2e)。三家都把 prefill 和 decode 拆成两组实例、KV 经 RDMA 在显存之间直传,分歧在代理怎么发、谁发起搬运、首 token 由谁产出:

| 必须有人做的事 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 代理怎么发 | 默认先发 P,拿到块号再发 D;推的实现与 Mooncake 两边同时发 | 网关同时发两边,靠一个随机房间号配对 | 编排服务先发只做 context 的请求,拿到结果再发只做 generation 的请求 |
| 谁发起搬运 | 默认 D 读 P;推的实现里 D 先登记块号,P 写 D | D 先分好页,把页号发给 P,P 写 D | D 侧发起请求,P 侧应答并发送 |
| P 什么时候发 | 整条 prompt 算完 | 每算完一块发一块 | context 阶段算完 |
| P 只算 prefill 靠谁 | 靠代理把最多生成数改成 1 | 引擎在 P 模式下自己强制成 1 | 请求类型标成只做 context |
| 首 token 谁出 | D 重算最后一个 prompt token 自己采,P 采的被代理丢掉 | P 采的首 token 随元数据写进 D,D 直接用 | P 采出,随 context 参数交给 D |
| 传输后端怎么挂 | 注册表按名字惰性加载,也能从外部模块加载 | 内置后端继承同一个基类,默认 Mooncake | 默认不装传输器,要显式指定后端 |

vLLM 的取舍是让引擎只认一个通用接口:P 和 D 之间只传 KV,不传采样结果,D 把远端 KV 当成普通的前缀命中处理,代价是 D 多花 1 步重算一个 token、默认还多一次代理往返。SGLang 与 TensorRT-LLM 把首 token 一起带过去,D 收完就能直接 decode,代价是采样结果也成了交接的一部分,SGLang 连首 token 的 logprobs 都要随 KV 写过去。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。请求级只有 `kv_transfer_params` 这一个字段,由代理填:P 腿写 `{"do_remote_decode": true}`;拉的 D 腿原样带上 P 响应里的那一份;推的 D 腿由代理拼出 P 的 `remote_engine_id`、`remote_host`、`remote_port`、`tp_size` 和共用的 `remote_request_id`;Mooncake 两腿共用一个 `transfer_id`,D 腿再带 `remote_bootstrap_addr`。聊天接口的 D 腿还可以在里面放 `prompt_token_ids`,省掉 D 上的套模板与切词。编码器分离用同样方式的 `ec_transfer_params`。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--kv-transfer-config` · 挂哪个连接器、当什么角色 | 启动,JSON;也能逐键写成 `--kv-transfer-config.kv_connector NixlConnector` | 不设:不分离 | 设了之后 P、D 各起各的,中间的代理要自己放 | 日志 `Creating v1 connector with name: … and engine_id: …` |
| `kv_connector` · 用哪个实现 | 上面的 JSON | 无 | `NixlConnector`(拉,等于 `NixlPullConnector`)、`NixlPushConnector`、`MooncakeConnector`、`MoRIIOConnector`(ROCm)、`LMCacheConnectorV1`、`LMCacheMPConnector`、`MultiConnector` 等 17 个注册名;`kv_connector_module_path` 指向外部模块时优先用外部的 | 同上 |
| `kv_role` · 交出方还是接收方 | 同上;设了连接器就必填 | 无 | `kv_producer` 给 P,`kv_consumer` 给 D;`kv_both` 对 NIXL 已标弃用 | 警告 `Using kv_role='kv_both' with NixlConnector is deprecated` |
| `kv_load_failure_policy` · D 读失败怎么办 | 同上 | `fail` | `recompute`:在 D 上重算没读到的部分,请求不报错,D 上的 TPOT 跟着抖 | 失败请求的 `finish_reason` 为 error |
| `kv_buffer_device` · 传输缓冲放哪 | 同上 | 平台设备类型,CUDA 上是 `cuda` | `cpu`:先拷进主机内存再传,给 NIXL 不能直接读写显存的加速器用;不支持两边块大小不同 | — |
| `enable_permute_local_kv` · 允许两边 KV 布局不同 | 同上 | `false` | `true`:D 收到后在头在外与 token 在外两种布局间转换;实验性,不支持分组记账 | — |
| `backends` · NIXL 用哪个传输插件 | `kv_connector_extra_config` | `["UCX"]` | 可换 `LIBFABRIC`、`GDS` 等,看 NIXL 编译时带了哪些插件 | — |
| `num_threads` · NIXL 的 UCX 线程数 | 同上 | 4 | 调高:并发传输更多;每个线程占网卡的门铃页,过多会让同机的 DeepEP 初始化失败;换了非 UCX 插件时不生效 | 报错里有 `mlx5dv_devx_alloc_uar` |
| `kv_lease_duration` · P 扣块的租约秒数 | 同上,P 侧 | 30 | 调高:D 排队久时不容易过期,D 挂掉后 P 收块更慢;心跳间隔 = 租约 ÷ 6,每次续 = 租约 × 2 ÷ 3 | `Num KV expired reqs`;`vllm:nixl_num_kv_expired_reqs` |
| `enforce_handshake_compat` · 握手时比兼容哈希 | 同上 | `true` | `false`:不比了,配置真不一致时 KV 会被错读而不报错 | 日志 `NIXL compatibility check passed (hash: …)` |
| `bidirectional_kv_xfer` · 多轮时 P 从 D 读上一轮的 KV | 同上,P 与 D 都设 | `false` | `true`:要有状态代理;D 的块在请求结束后按下面的 TTL 留着;推的实现不支持 | — |
| `kv_recompute_threshold` · 双向时少于多少 token 就不读、直接重算 | 同上 | 64 | 调高:短前缀直接在 P 重算,省一次往返 | — |
| `decoder_kv_blocks_ttl` · 双向时 D 留块多久 | 同上 | 480 秒 | 调低:D 的池子早腾出来,会话隔久了回来要重算;不靠心跳续 | — |
| `push_registration_timeout` · 推模式下 D 登记后等多久 | 同上,D 侧 | 同 `decoder_kv_blocks_ttl` | 到期只丢登记、打警告,请求本身不失败,要靠代理超时或中止 | 警告 `NixlPushConnector: registration for request … timed out` |
| `VLLM_NIXL_SIDE_CHANNEL_HOST` / `VLLM_NIXL_SIDE_CHANNEL_PORT` · NIXL 握手地址 | 环境变量 | `localhost` / 5600 | 跨机必须设成对端能连上的 IP;同机多个实例端口要错开;实际端口 = 基准 + DP 序号 | 握手失败计入 `Num failed transfers` |
| `UCX_TLS` / `UCX_NET_DEVICES` · UCX 走哪些通道、哪些网卡 | 环境变量,UCX 自己读 | 不设由 UCX 决定 | NCCL 的网卡变量对 NIXL 不起作用 | `KV Transfer metrics` 里的吞吐 |
| `VLLM_KV_CACHE_LAYOUT` · 手动指定 KV 布局 | 环境变量 | 不设:NIXL 与 Mooncake 对非 MLA 模型用 `LBHNC` | `BLHNC`:同一块的各层连成一段,传输描述符更少 | 日志 `NixlConnector setting KV cache layout to LBHNC for better xfer performance.` |
| `VLLM_MOONCAKE_BOOTSTRAP_PORT` · P 侧 Mooncake 引导服务端口 | 环境变量 | 8998 | 同机多个 P 要错开 | — |
| `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT` · Mooncake 下 P 最多扣块多少秒 | 环境变量 | 480 | 调低:D 出事后 P 早收块 | — |
| `num_workers` / `mooncake_protocol` / `device_name` · Mooncake 发送线程、协议、网卡白名单 | `kv_connector_extra_config` | 10 / `rdma` / 空(全部探测) | 网卡里 IB 与 RoCE 混插时用白名单让两边落在同一种链路上 | — |
| `--ec-transfer-config` · 编码器分离的连接器与角色 | 启动,JSON | 不设 | `ec_connector` 填 `ECExampleConnector`、`ECMooncakeConnector` 或 `ECCPUConnector`;`ec_role` 为 `ec_producer` 时自动只跑编码器、不建 KV 缓存 | 日志 `Creating connector with name: … and engine_id: …` |
| `ec_cpu_bytes` / `ec_enable_nixl` / `consumer_ack_timeout_s` · CPU 编码缓存层大小、点对点读、等应答多久 | `ECCPUConnector` 的 `ec_connector_extra_config` | 必填 / `false` / 30 秒 | 开点对点读要装 NIXL;编码端一步很长时调大应答超时;要求 Model Runner V2 | — |
| `VLLM_EC_SIDE_CHANNEL_HOST` / `VLLM_EC_SIDE_CHANNEL_PORT` · 编码结果点对点读的控制地址 | 环境变量,生产方 | `localhost` / 5601 | 跨机必须改成可达地址 | — |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` · PyTorch 可扩展段分配器 | 环境变量 | 不设 | 与任何 KV 连接器冲突,除非同时开 `--enable-cumem-allocator`;显存页被重映射后已登记的 RDMA 区域失效 | 启动报错 `is incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**怎么看。** 一是启动日志:每个实例都会先警告 `Initializing KVConnectorBase_V1. This API is experimental`,再打 `Creating v1 connector with name: …`;第一次与某个对端握手成功时打 `NIXL compatibility check passed`。二是稳态:每 10 秒一行 `KV Transfer metrics: Num successful transfers=…, Avg xfer time (ms)=…, P90 xfer time (ms)=…, Avg MB per transfer=…, Throughput (MB/s)=…, Num failed transfers=…, Num KV expired reqs=…`;P 上的警告 `Releasing expired KV blocks for request … before lease expired.` 说明 D 没在租约内来读。D 上等远端 KV 的请求计在 `Waiting` 里的 `Deferred`(02 章)。三是 Prometheus:`vllm:nixl_xfer_time_seconds`、`vllm:nixl_bytes_transferred`、`vllm:nixl_num_failed_transfers`、`vllm:nixl_num_kv_expired_reqs`,前两个相除是单次传输的实际带宽。四是响应:D 返回的用量里,缓存命中数报的是 P 那边的命中,不是 D 自己那份接近 100% 的「命中」。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表;代理用仓库里的示例,生产环境换成自己的路由层。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单机 2 张卡先跑通,8B 模型,1 个 P、1 个 D | P:`CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve Qwen/Qwen3-8B --port 8100 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`;D:同样的命令换成卡 1、`VLLM_NIXL_SIDE_CHANNEL_PORT=5601`、`--port 8200`、`"kv_role":"kv_consumer"`;代理:`python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192 --prefiller-hosts localhost --prefiller-ports 8100 --decoder-hosts localhost --decoder-ports 8200` | 用 1 张卡的 prefill 容量换另一张卡上不被打断的 decode;用来验证连通与正确性,吞吐不代表线上 |
| 两台 8 卡 H100,70B 模型,长 prompt 短输出:A 机 1 个 P(TP 8),B 机 2 个 D(各 TP 4) | A:`VLLM_NIXL_SIDE_CHANNEL_HOST=<A_IP> UCX_NET_DEVICES=all vllm serve meta-llama/Llama-3.3-70B-Instruct -tp 8 --port 8100 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_connector_extra_config":{"kv_lease_duration":60}}'`;B 上第 1 个 D:`CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NIXL_SIDE_CHANNEL_HOST=<B_IP> VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve <同一个模型名> -tp 4 --port 8200 -cc.cudagraph_mode=FULL_DECODE_ONLY --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'`,第 2 个换卡 4–7、端口 5601 与 8201;代理的 `--decoder-hosts` 写两遍 `<B_IP>`,`--decoder-ports 8200 8201` | P 用 8 卡把 prefill 算快,D 用 2 份 4 卡扩批;每张 D 卡要读 2 张 P 卡;租约放宽到 60 秒防跨机排队时过期,代价是 D 挂掉后 P 最多扣块 60 秒;D 只录 decode 的整图(08 章) |
| 多轮 agent,想让下一轮复用上一轮留在 D 上的 KV | 在上一行两边的 `kv_connector_extra_config` 里都加 `"bidirectional_kv_xfer":true`;代理换成 `python examples/disaggregated/disaggregated_serving/disagg_proxy_multiturn.py --prefiller-host <P_IP> --prefiller-port 8100 --decoder-host <D_IP> --decoder-port 8200`,客户端每轮带同一个 `conversation_id` | 用 D 上每个会话最多 480 秒的块占用,换多轮时 P 不再重算历史;客户端要是删掉了推理模型的思考段,P 的 prompt 不再是 D 序列的前缀,读回来的 KV 位置是错的 |

组合与推荐值是按语义推的起点,不是实测最优。

编码器分离照 `examples/disaggregated/disaggregated_encoder/disagg_1e1p1d_example.sh` 起:编码器实例 1 张卡,`--ec-transfer-config` 的 `ec_role` 设 `ec_producer`,同时关前缀缓存、把 `--max-num-batched-tokens` 调得很大;P 同时挂 `ec_consumer` 与 `kv_producer`,D 只挂 `kv_consumer`;代理用同目录的 `disagg_epd_proxy.py`,分别给 `--encode-servers-urls`、`--prefill-servers-urls`、`--decode-servers-urls`。

| 症状 | 先查 | 然后 |
|---|---|---|
| D 返回的只有 1 个 token,或 D 从头 prefill | 代理有没有把 P 响应里的 `kv_transfer_params` 转给 D;P 腿是不是流式 | 流式响应不带这个字段,P 腿必须非流式;D 日志有 `Got invalid KVTransferParams` 或 `Got kv_transfer_params, but no KVConnector found` 时对照检查 |
| 第一条请求就失败,D 日志 `NIXL compatibility hash mismatch` | 两边的模型名字符串、vLLM 版本、精度、KV 精度、注意力后端、分组记账、投机解码配置、推还是拉 | 改成一致;模型名要逐字相同,一边 HF 名一边本地路径也算不同 |
| P 上反复出现 `Releasing expired KV blocks`,D 的请求报错 | D 的 `Deferred` 是否堆积;跨机网络 | 加 D 或调大 `kv_lease_duration`;D 读失败默认判错,想不报错改 `kv_load_failure_policy` 为 `recompute` |
| 跨机时 `Num failed transfers` 一直涨,同机正常 | `VLLM_NIXL_SIDE_CHANNEL_HOST` 是否还是 `localhost`;`UCX_NET_DEVICES` | 设成本机可达 IP;UCX 变量按网卡配,不要配 NCCL 的 |
| 启动报 `is incompatible with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 环境里的 PyTorch 分配器设置 | 去掉这项,或加 `--enable-cumem-allocator` |
| 启动报 `does not support HMA but HMA is enabled` | 是否显式开了分组记账 | 去掉显式开关让它自动关,或换支持分组记账的连接器(03 章) |
| 同机的 DeepEP 初始化报 `mlx5dv_devx_alloc_uar` | NIXL 的 `num_threads` | 调低线程数 |
| P 的 `KV cache usage` 高、新请求进不去,算力却闲 | 等交出的块占了多少;D 读得慢还是来得晚 | 加 D、缩短租约;或换推的实现让 P 算完就写走 |
| EPD 下多模态请求等了约 60 秒后报错 | 编码器实例的 CPU 缓存层是否把结果挤掉了;请求里还有没有原始媒体 | 加大 `ec_cpu_bytes`;代理别把媒体改写掉,读不到时才能退回本地编码 |

## 六、常见误区

- **以为 P 实例会自动只做 prefill。** 用过 SGLang 的人会这么想,它的 P 模式在引擎里把生成长度强制成 1。vLLM 的 P 只认请求里的传输参数,生成长度靠代理改成 1;直接对 P 发一条没改过的请求,它会照常生成到底再交块,前面的 decode 全白算,D 只读 prompt 那一段。
- **以为首 token 是 P 算的,D 从第 2 个接着吐。** 另外两家都是这样,P 响应里也确实带着 1 个 token。实际上那个 token 被代理丢掉,D 重算最后一个 prompt token 后自己采首 token;D 的 TTFT 里因此总有一步前向,开采样时 D 吐的首 token 也未必和 P 那个相同。
- **以为 D 挂了,P 要 480 秒后才收块。** 旧的设计文档里写着一个 480 秒的 NIXL 超时环境变量,Mooncake 至今也有一个 480 秒的同类变量。NIXL 那个变量在基准上已经不存在,换成了 30 秒租约加心跳,D 一停心跳,P 最多 30 秒就收;只有 Mooncake 还是固定的 480 秒。
- **以为 `kv_role` 写 `kv_both` 最省事。** Mooncake 的文档确实把它当成灵活选项。NIXL 已把它标成弃用;而且只要角色里带交出方,连接器就按「交出必须可靠」处理,D 上被抢占的请求在路上那一步的输出也被丢掉(07 章),重算时再采一遍。P 写 `kv_producer`、D 写 `kv_consumer`。
- **以为模型一样就能握手。** 兼容哈希里放的是模型名字符串而不是权重指纹,一边写 HF 仓库名、一边写本地路径,权重再相同也被拒;vLLM 版本也在里面,滚动升级时一新一旧的 P 与 D 配不上。
- **以为两边 TP 随便配。** 异构 TP 只支持成整数倍;非 MLA 模型在 P 的 TP 大于 KV 头数(KV 在 P 上被复制)时,不能再让 D 的 TP 比 P 小;混合 Mamba 模型两边必须一样。70B 这类 8 个 KV 头的模型,P 最多开到 TP 8 才能配一个 TP 更小的 D。
- **以为设了 `recompute` 更稳。** 字面上是「失败了自己补」。实际是把一段 prefill 塞回为 decode 调好的实例上,那一步所有 decode 一起卡住;官方文档也提醒这会让 decode 实例的尾延迟变差。线上宁可让代理重试整条请求。
- **以为编码器点对点读的应答超时是 2 秒。** CPU 编码缓存连接器的使用文档在配置表和超时说明里都写 2 秒,代码里的缺省其实是 30 秒,和生产方扣住编码结果的租约同一个常量。以代码为准:应答慢的时候先别急着调它。

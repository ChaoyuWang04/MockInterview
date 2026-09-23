# vLLM 05|KV offload 与 HiSparse

显存装不下的 KV 往哪放、什么时候写出去、要用时怎么拉回来而不拖慢别人;以及稀疏注意力模型怎么连一条在跑的请求都只在 GPU 留热的那部分。

说法都在源码基准 `94f4170df3`(tag `v0.30.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

04 章的前缀缓存只活在空闲显存里,容量就是那块 KV 池。拿 03 章那个 32 层、8 个 KV 头、头维 128 的 BF16 模型算:一个 token 的 KV 是 128 KiB,40 GiB 的池装 327680 个 token。agent 和多轮对话一个会话动辄 32000 token,池里同时只放得下 10 个会话。第 11 个会话进来,最久没用的那个就被挤掉;它调完一次工具、隔两分钟再回来,整段 32000 token 从头 prefill。坏的是 **TTFT 的长尾和 prefill 吞吐**:内容明明重复,命中率却被显存容量卡死,而同一台机器上 512 GB 以上的内存大多闲着。

算一笔账就知道留着值:把这 32000 token 的 KV(约 4.2 GB)从内存拷回来,按 50 GB/s 的有效带宽约 84 毫秒;重算一遍 prefill 要约 7.8e14 次浮点运算,H100 就算跑满 BF16 峰值也要 0.8 秒。

稀疏注意力模型坏在另一处。DeepSeek 稀疏注意力这类模型,decode 每一步只读 top-k 个位置(常见是 2048),可一条 128K 上下文的请求有 131072 行 KV 全堆在 GPU 上,每一步用到的只有 1/64。并发被这些冷 KV 卡住,**长上下文 decode 的并发**上不去。

## 二、解法:把前缀缓存改成直写的多级缓存

最天然的直觉:CPU 内存当前缀缓存的下一层,再往下接磁盘、对象存储或别的实例;每算完一块就往下抄一份,GPU 丢块时不用再拷;新请求在 GPU 上没查到的部分,用同一套块键(04 章的链式哈希)往下层查,查到就拉回来。这些都通过 KV connector 接口挂进引擎,接口本身和它在调度器、worker 两侧的钩子见 12 章。

![写出与拉回的时间轴:上半是写出,请求 A 的 prompt 分两步算完,每步算完的 128 块到下一步前向发出之前才由拷贝流写进 CPU,拷贝和那一步的前向并行,写入期间 CPU 里这些块查不到,写完后 EngineCore 里的写线程再把它们抄成磁盘文件,decode 产出的块默认不写,A 结束时块照常还池不等写完;下半是拉回,请求 B 带着同一段 4096 token 前缀进来,CPU 命中时 B 先在门外等 256 块拷回 GPU,两步后进批只算剩下的部分,只在磁盘命中时要先后台查文件在不在、再读回 CPU、再拷回 GPU,四步后才进批](/opensource/vllm-lite/05a-offload-write-and-load-timeline.svg)

**写出:算完就抄,晚一步开拷。** 每一步调度时,把这一步会算完的满块登记成写出任务,worker 等到下一步前向发出之前才提交拷贝。晚这一步是故意的:采样结果要先回传,写出不和它抢。拷贝在单独的流上,流上先等计算流把这些块写完,再和下一步的前向并行;走 GPU 的拷贝引擎,不占算力。默认只写 prompt 的块,decode 生成的块不写。CPU 层满了就按 LRU 挤掉没人在读的块;连能挤的都没有,这次写出直接放弃、打一条警告。它是尽力而为的缓存,少写一次只是以后少一次命中。

**请求结束不等写完。** 块照常还池,进 04 章那条空闲队列。写出任务盯着这些块:要是池子在拷完之前就把其中一块分给了别人,下一步前向开始前先等这次拷贝结束。只有这种抢块的情况会让计算停下来等。

**拉回:命中的请求在门外等,别人照算。** 新请求先查 GPU 的前缀缓存,再从 GPU 命中的位置往后逐块查 CPU,第一次查不到就停。命中了就先给它分好 GPU 块,但不放进这一步的批。这一步的前向发出之后才提交 CPU 到 GPU 的拷贝,拷贝流等这一步算完再开拷,和下一步别的请求的前向并行。拷完的消息回到调度器,它回到等待队列,再被调度时这一段已经算作命中,只 prefill 剩下的,拉回来的块同时登记进 GPU 前缀缓存。命中的块还在写入途中时,请求推迟一步再查,而不是当场重算。

**多层:只有 CPU 能碰 GPU。** 多层时 CPU 是唯一的主层,磁盘、对象存储、别的实例的 CPU 都是下层,只和 CPU 交换。块写进 CPU 并确认之后,EngineCore 进程里的 I/O 线程再把它抄进每个下层。查找时 CPU 没有才问下层;下层命中先读回 CPU,请求这几步留在队里,读回后再走 CPU 到 GPU 那一段。块键默认只取决于内容,所以下层可以挂在多个实例之间共享。

**两套 CPU 下放实现。** 默认的下放连接器自己管一个 CPU 块池,一个存取单位可以是几个 GPU 块拼成的一组,淘汰策略可换,能接上面那条多层链。另有一个简版 CPU 下放连接器,靠一个环境变量切过去,思路不同:CPU 侧直接复用 GPU 前缀缓存的那套块池与 LRU;写出在每步前向发出之后提交,等这一步算完才开拷,放在最低优先级的流上,由后台线程下发;拷贝期间把源 GPU 块的引用计数加 1,让它根本不会被分走,而不是事后去等。它还有一个懒写模式,只在 GPU 空闲队列里快被重新分配的块上动手,每步保持 2 步满额 token 那么多块已经写好,属于写回而不是直写。它必须开着前缀缓存,否则自己关掉;能直接写本地磁盘,但没有多层链。

为什么实际不吃亏:写出和拉回都在单独的流上,和前向重叠;请求级的等待只让命中的那一条晚几步进批,不让整批停;一次 CPU 命中省下的是开头那笔 0.8 秒对 84 毫秒的账。

**HiSparse:在跑的请求也只在 GPU 留热块。** 前面讲的都是请求之间的缓存;稀疏注意力模型换一个层次,在一条请求 decode 的每一步里分冷热。

- **主机上永远有一份全量。** prefill 写出的 KV 在前向里同时镜像进主机的钉住内存;decode 写出的行默认也同步镜像。GPU 上常驻的页因此只是主机全量的写回缓存,放掉它不需要再搬数据。
- **池子不紧时什么都不放。** 常驻页照常留在 GPU,注意力直接读。GPU 空闲块跌到池子的 10%(或一条请求热区的块数,取大的)以下,请求才去领一块固定大小的热区,默认是「每步查询数加 1」乘以 top-k 行,不开推测解码、top-k 2048 时是 4096 行;领到之后,它已经镜像完的页就还给池子,内容在被复用之前仍然可读。
- **每一步在 kernel 里决定去哪取。** 注意力选出 top-k 个位置后,由一个融合的解析 kernel 逐个找,顺序见下图。全程在 GPU 上,不回 CPU 做决定,能被 CUDA graph 抓住。
- **索引器的 KV 不动。** 算 top-k 要用的索引器 KV 仍是普通的 GPU 缓存组,全量留在显存;想把它也下放,就同时挂上前面那个下放连接器。
- **主机层兼做前缀缓存。** 主机页写完才按块哈希公布,后来的请求可以命中;PD 分离时 decode 实例看整段前缀放不放得进显存,放得下直接落 GPU,放不下就落主机层。

![上半是一步 decode 里的查找:索引器挑出的每个 top-k 位置先问在不在 GPU 常驻页上,在就直接读;不在再问在不在这条请求的 GPU 热区里,在就读并更新 GPU 上的 LRU;都不在就让出热区里最久没用的一行,从主机钉住内存经 PCIe 拷进来;三条路最后都给稀疏注意力一个 GPU 物理行号。下半按行数成比例画一条 128K 请求:主机上全量 131072 行,GPU 热区 4096 行是它的 1/32,这一步要读的 2048 行是它的 1/64](/opensource/vllm-lite/05b-hisparse-lookup-path.svg)

top-k 位置在相邻两步之间变化不大,热区里 LRU 命中的占大头,只有没命中的行走一次 PCIe;128K 上下文的请求,GPU 上的热区只有它全量行数的 1/32。

## 三、代价

**CPU 层要比 GPU 的 KV 大才有用。** 因为是直写,GPU 上的块早就抄进了 CPU;CPU 层要是不比全部卡的 KV 加起来大,它只是 GPU 的镜像,多不出命中。CUDA 上这块内存是 /dev/shm 里的共享映射,启动时整块注册成钉住内存;容器的共享内存默认很小,不改大直接起不来。容量是所有卡加起来的总数,每张卡分到总数除以卡数;只有单机 TP 的 MLA 模型各卡 KV 一样,存一份就够。

**每个 prompt 块都写一遍。** 用不用得上都写,PCIe 写带宽成了常态开销,和拉回、PD 传输共用一条总线。单层时可以设一个门槛,同一块被第 2 次提交写出时才真写,只留热数据;多层时不支持这个门槛。

**命中按存取单位对齐,还得比 GPU 多出一整个单位。** 单位默认等于 GPU 块,调大后 I/O 更大块、记账更少,命中也更粗;下层比 GPU 前缀缓存多命中不到一个单位时,当作没命中。decode 的块默认不写,多轮对话上一轮的回答要等下一轮把它当 prompt 算过,才会进 CPU。

**拉回不是免费的等待。** CPU 命中的请求当步不算,要等拷贝完成的消息回到调度器才进批;只在磁盘或对象存储命中时,先要后台查一遍在不在,再读回 CPU,再拷回 GPU,每一环都要等下一次调度来问,比 CPU 命中多等 2 轮。连接器不比较拉回和重算哪个更快,只要多命中一个单位就拉。每层一块小于 28 KiB 的模型(MLA 常见),一次拉回 16 块以上时,CPU 到 GPU 的拷贝改走一个最多占 12 个 SM 的 Triton kernel,拉回会和前向抢算力。

**抢块会让整批等。** 被抢占的请求、或者还没写完就被重新分配的块,都让下一步前向开始前同步等写出结束。

**磁盘层只增不减。** 文件系统下层没有容量上限,也不淘汰;换模型、块大小、并行方式或精度,会在同一根目录下另起一个子目录,旧的留在原地。清前缀缓存时下层也不清。

**HiSparse 挑模型、挑平台。** 只支持配置里带 top-k 字段的 DSA 模型,不支持 DeepSeek V4;只在 NVIDIA CUDA 上跑,不支持流水线并行和 decode 上下文并行;要求 Model Runner V2 和分组记账(03 章);设计文档标着实验性。热区没命中的行每一步都走一次 PCIe,缺得多时直接进 TPOT。主机池按每个数据并行副本计,钉住内存,超过可用内存的 95% 就拒绝启动。

## 四、和 SGLang、TensorRT-LLM 比:写出早晚、拉回时谁等

对照对象是 SGLang(基准 826d5170ae)与 TensorRT-LLM(基准 59f5c47f2e)。三家都要回答同样几件事:什么时候往内存写、拉回来时谁等、内存以下接什么、稀疏注意力的冷 KV 放哪。

| 必须有人做的事 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 什么时候往 CPU 写 | 每步算完的满块在下一步开头写,默认只写 prompt 的块 | 插进前缀树那一刻写(默认);可改成同一段第 2 次插入才写,或驱逐时才写 | 块被踢出 GPU 的那一刻才写;主机层默认不开 |
| 拉回时谁等 | 命中的请求在门外等整段拷完,别的请求照算 | 按层重叠:第 N 层在算时后面的层在传,注意力取某层前等这一层到 | 分配块时发起拷贝,本步的拷贝在主流上同步之后才开算 |
| 内存以下 | 磁盘、对象存储、别的实例,全部经 CPU 中转,由 EngineCore 里的线程读写 | 可换的存储后端,按页哈希存取 | 磁盘层只在第二代管理器上有 |
| 稀疏注意力的冷 KV | 主机全量加 GPU 热区;GPU 常驻页是写回缓存,池子紧时才切到热区;主机层兼做前缀缓存 | 主机全量加 GPU 热区,要求关掉前缀缓存;DSA 与 DeepSeek V4 都支持,ROCm 也能跑 | DSA 配置里没有把 KV 放主机的选项 |

vLLM 的取舍是写得最早、等得最局部:每个 prompt 块都付一次 PCIe 写,换来驱逐时不用再拷,拉回时也只让命中的那一条等。SGLang 同样早写,但把拉回藏进请求自己的前向里逐层重叠,命中当步就能算;TensorRT-LLM 拖到驱逐才写,省了写带宽,代价是拉回的那一步整批一起等。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。请求里能改的是 `kv_transfer_params` 的三个字段:`max_load_tokens`(这条最多拉回多少 token,0 表示不拉)、`max_offload_tokens`(只写前多少 token,0 表示不写)、`kv_load_tiers`(只查哪几个下层);三者都标着实验性。运行时只有 `POST /reset_prefix_cache?reset_external=true`(要开开发接口,见 04 章)能清 CPU 层。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--kv-offloading-size` · CPU 层多少 GiB,打开下放的快捷方式 | 启动 | 不设:不下放 | 设了自动挂 `OffloadingConnector`、`kv_role` 设 `kv_both`、`cpu_bytes_to_use` 设成这个值;是所有卡的总和 | 日志 `Creating offloading spec with name: CPUOffloadingSpec` |
| `--kv-offloading-backend` · 快捷方式挂哪一家 | 启动 | `native` | `lmcache`:改挂 `LMCacheMPConnector`,容量由独立的 LMCache 服务进程管,上一行的大小不传过去 | 启动日志里的 connector 名 |
| `VLLM_USE_SIMPLE_KV_OFFLOAD` · 快捷方式改用简版实现 | 环境变量 | 0 | 设 1:挂 `SimpleCPUOffloadConnector`;要求开着前缀缓存 | 日志 `SimpleCPUOffloadScheduler: Allocating … offload blocks` |
| `--kv-transfer-config` · 完整写法 | 启动,JSON | 不设 | `kv_connector` 填 `OffloadingConnector`、`SimpleCPUOffloadConnector` 或 `HiSparseConnector`;`kv_role` 必填;要和 PD 或外部缓存同时用,填 `MultiConnector` 再列子项(12 章) | — |
| `cpu_bytes_to_use` · CPU 层总字节数 | `kv_connector_extra_config` | 下放连接器必填;简版默认 8 GiB | 调大:能留的前缀变多,/dev/shm 与钉住内存同比变大;要大于全部卡的 KV 之和才有意义 | 日志 `Created mmap file /dev/shm/vllm_offload_… (… GB)`;`vllm:kv_offload_cpu_cache_usage_perc` |
| `block_size` / `blocks_per_chunk` · 一个存取单位几个 token / 几个 GPU 块 | 同上 | GPU 块大小 / 1;两者只能设一个 | 调大:每次 I/O 更大、记账更少,命中更粗;`block_size` 必须是 GPU 块大小的倍数,各组块大小不同的模型只能用 `blocks_per_chunk` | 启动报错会给出可用的倍数 |
| `offload_prompt_only` · 只写 prompt 的块 | 同上 | `true` | 设 `false`:decode 生成的块也写,多轮对话上一轮的回答能直接命中;写带宽随输出长度涨 | `vllm:kv_offload_store_bytes` |
| `eviction_policy` · CPU 层淘汰策略 | 同上 | `lru` | `arc`:兼顾最近与频次,扫一遍的长 prompt 不容易冲掉热前缀;也可填自写类加 `cache_policy_module_path` | — |
| `store_threshold` · 被提交几次才真写 | 同上,只对单层 | 0(不过滤) | 设 2:只写第 2 次出现的块,写带宽降,第一次复用必然不命中;多层时设 2 以上启动报错 | `vllm:kv_offload_stores_skipped` |
| `max_tracker_size` · 上一行计数表的上限 | 同上 | 64000 | 调大:门槛记得住更久以前的块,多占 CPU 内存 | — |
| `spec_name` · 单层还是多层 | 同上 | `CPUOffloadingSpec` | `TieringOffloadingSpec`:可接 `secondary_tiers` | 日志 `Created secondary tier #0 (fs)` |
| `secondary_tiers` · 下层列表,按顺序查 | 同上 | 空 | `fs` 写本地或共享文件系统(`root_dir` 必填);`obj` 走 NIXL 写 S3 兼容存储;`p2p` 从别的实例的 CPU 层拉,也能当 PD 传输(12 章) | 同上 |
| `n_read_threads` / `n_write_threads` · 文件系统下层的读写线程数 | `secondary_tiers` 里的一项 | 16 / 16 | 按存储的并发能力调;prefill 命中多时多给读线程 | `vllm:kv_offload_tiering_read_time` 与 `_write_time` |
| `backpressure` · 下层写不动时丢写 | `kv_connector_extra_config` 或单个下层里 | 不设:不丢 | 设了(必须给 `backpressure_cls`,如 `EMABackpressureDetector`):按每 MiB 写延迟的滑动平均判断过载,默认策略从高水位起按比例丢写,到 3 倍高水位全丢;`fs` 默认按本地 NVMe 定水位,`obj` 与标了 `REMOTE` 的按网络存储定 | 日志 `Tier #… back-pressure activated`;`vllm:kv_offload_tiering_backpressure_stores_dropped` |
| `lazy_offload` · 简版改成快被挤掉时才写 | 简版的 `kv_connector_extra_config` | `false` | `true`:只写 GPU 空闲队列前端的块,写带宽最省,GPU 上还热的块不备份 | — |
| `kv_offload_backend` / `disk_path` / `disk_capacity_bytes` · 简版写本地磁盘 | 同上 | `cpu` / 无 / 100 GiB | `disk`:容量按磁盘算,按 LRU 淘汰;`disk_path` 必填 | 日志 `backend=disk` |
| `host_pool_gib` · HiSparse 的主机池 | `HiSparseConnector` 的 `kv_connector_extra_config` | 必填,须大于 0 | 每个数据并行副本一份;调大:能放下的冷 KV 更多;超过可用内存 95% 拒绝启动 | 启动报错 `HiSparse pinned host pool needs` |
| `device_buffer_size` · 每条请求的 GPU 热区行数 | `--attention-config` 的 `hisparse_config` | (每步查询数 + 1) × `index_topk` | 调大:热区命中率涨,每条请求多占显存;不能小于每步查询数 × top-k,不超过 32768;不按块大小对齐会打警告 | `vllm:hisparse_cache_misses`、`vllm:hisparse_host_to_device_bytes` |
| `eager_host_mirror` · decode 的行在前向里就写一份到主机 | 同上 | `true` | `false`:decode 行只在 GPU,页被放掉时才拷去主机,放页那一刻多一次搬运 | — |
| `--prefix-caching-hash-algo`、`PYTHONHASHSEED` · 块键怎么算 | 启动 / 环境变量 | 见 04 章 | 多实例共享磁盘、对象存储或 p2p 下层时必须一致;快速哈希必须设同一个种子 | p2p 握手时拒绝种子不同的对端 |

**怎么看。** 一是启动日志:`Creating offloading spec with name: …`、`Created mmap file /dev/shm/vllm_offload_<engine_id>.mmap (X GB)`、多层时每个下层一行 `Created secondary tier #i (…)`;简版是 `SimpleCPUOffloadConnector: role=…, per_rank=… GB, … mode=eager|lazy, backend=cpu|disk`。二是稳态日志:每 10 秒那行多一项 `External prefix cache hit rate`,是下层命中占「GPU 没命中的那部分」的比例;下面另起一行 `KV Transfer metrics: …`,列出这段时间的拉回与写出字节数和耗时;CPU 层挤不出空位时是 warning `Request … cannot store chunks`。三是 Prometheus:`vllm:kv_offload_load_bytes`、`vllm:kv_offload_store_bytes` 与对应的 `_time`,相除就是实际带宽;`vllm:kv_offload_cpu_cache_usage_perc` 是 CPU 层被在途拷贝钉住的比例,长期接近 1 说明写出会被丢;`vllm:kv_offload_lookup_async_delay_seconds` 是请求因为等下层查询而推迟了多久;`vllm:kv_offload_allocation_failure` 是写出放弃的次数;多层另有 `vllm:kv_offload_tiering_*` 一组;外部命中的计数器是 04 章那对 `vllm:external_prefix_cache_queries` 与 `_hits`。HiSparse 看 `vllm:hisparse_cache_hits`、`_cache_misses`、`_host_to_device_bytes` 三个计数器,未命中率 = misses ÷ (hits + misses)。

**手算一遍。** CPU 层按「存取单位」切槽:一个单位的字节数 = 每卡一个 GPU 块的字节数 × 卡数 × 每单位的块数,再向上对齐到内存页(x86 上 4 KiB);槽数 = `cpu_bytes_to_use` ÷ 这个数。03 章那个模型一块 2 MiB,单卡、单位 = 1 块,给 200 GiB 就是 102400 槽、1638400 个 token,是 40 GiB GPU 池的 5 倍。同一个模型换成 TP 4,每卡一块 512 KiB,4 卡合计还是 2 MiB 一个单位,槽数不变。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单卡 H100、8B 模型跑多轮 agent,机器有 512 GB 内存 | `vllm serve Qwen/Qwen3-8B --kv-offloading-size 200`;容器加 `--shm-size 210g` 或 `--ipc=host` | 用 200 GiB 钉住内存和每个 prompt 块一次 PCIe 写,换约 5 倍于显存的前缀容量;隔几分钟回来的会话不再从头 prefill |
| 4 个实例共享一块 NVMe 或共享卷上的前缀,要跨实例命中 | `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","cpu_bytes_to_use":107374182400,"secondary_tiers":[{"type":"fs","root_dir":"/mnt/kv","n_read_threads":32}]}}'`,四个实例的模型名、块大小、并行方式、精度、哈希算法保持一致 | 用磁盘空间和 3 步以上的拉回延迟换跨实例、跨重启的命中;目录不会自己清,要另配定期清理 |
| PD 分离的 decode 实例,8 卡跑 DSA 模型(如 DeepSeek V3.2),长上下文并发被显存卡住 | `vllm serve <模型> -tp 8 --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"HiSparseConnector","kv_role":"kv_both","kv_connector_extra_config":{"host_pool_gib":512}},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]}}'` | 用 512 GiB 钉住内存和热区未命中时每步的 PCIe 读,换长请求的 GPU 占用降到热区加索引器;放不进显存的导入直接落主机层;TPOT 随未命中率上升。单独挂 HiSparse 也能起,但启动警告会把它标成调试用的兜底路径 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 启动报 `Insufficient space in /dev/shm for CPU KV offload shared region in /dev/shm` | 容器的 /dev/shm 多大 | 加 `--shm-size` 或 `--ipc=host`;或调小 `cpu_bytes_to_use` |
| 启动报 `cpu_bytes_to_use must be specified in kv_connector_extra_config` | 是不是手写了 `--kv-transfer-config` 却没给容量 | 补上;或改用 `--kv-offloading-size` |
| 开了下放,`External prefix cache hit rate` 一直是 0 | CPU 层是否比全部卡的 KV 之和小;请求是否只差不到一个存取单位 | 加大 CPU 层;多轮场景把 `offload_prompt_only` 设 `false` |
| 日志反复出现 `Request … cannot store chunks` | `vllm:kv_offload_cpu_cache_usage_perc` 是否贴着 1 | CPU 层被在途拉回钉满,加大容量或降低并发 |
| 命中了下层,TTFT 反而更高 | `vllm:kv_offload_lookup_async_delay_seconds`、下层的读耗时 | 加读线程;冷数据多时用 `kv_load_tiers` 让短请求只查 CPU,或用 `max_load_tokens` 限量 |
| 磁盘被写满 | `root_dir` 下有几个 `<模型>_<摘要>` 子目录 | 文件系统下层不淘汰,删掉旧子目录并配定期清理 |
| 多实例共享同一个 `root_dir`,互相不命中 | 各实例的哈希算法、`PYTHONHASHSEED`、块大小、并行方式、精度 | 统一这几项;任何一项不同都会落到不同子目录 |
| RL 换完权重、清过前缀缓存,输出仍像旧权重 | 是否挂了磁盘、对象存储或 p2p 下层 | 清缓存不清下层,子目录也不含权重版本;换 `root_dir` 或清空目录 |
| 设了 `VLLM_USE_SIMPLE_KV_OFFLOAD=1`,日志出现 `Detected prefix caching disabled, disabling CPU offload` | 是否关了前缀缓存 | 打开前缀缓存,或改用默认的下放连接器 |
| HiSparse 启动报 `HiSparse is only supported for DSA models with index_topk` 或 `requires NVIDIA CUDA` | 模型与平台 | 只有 CUDA 上的 DSA 模型能用;流水线并行和 decode 上下文并行也要去掉 |
| 启动警告 `HiSparse host-resident KV is configured with connector … debug/fallback paths` | `kv_connector` 是不是单独的 `HiSparseConnector` 或别的 connector | 官方验证过的是经 `MultiConnector` 和 `NixlConnector` 或 `MooncakeStoreConnector` 组合的用法 |
| HiSparse 开了之后 TPOT 变高 | `vllm:hisparse_cache_misses` 占比、`_host_to_device_bytes` 速率 | 调大 `device_buffer_size`;或降并发,让池子不那么早跌到 10% 以下 |

## 六、常见误区

- **以为 CPU 层给个 16 GB 试试就能看到命中率涨。** 把它当成「GPU 放不下的溢出区」,觉得多少都有帮助。实际它是直写:GPU 上的块一算完就抄下去了,CPU 层小于全部卡的 KV 之和时,里面基本就是 GPU 已有内容的副本,下层命中率接近 0。官方调参建议也写着要大于 GPU 的 KV 总量。
- **以为多轮对话里上一轮的回答已经被下放了。** 名字叫「KV 下放」,自然以为算过的都下去了。默认只写 prompt 的块,上一轮的回答要等下一轮作为 prompt 被算过才进 CPU;两轮之间它要是先被挤出 GPU,就得重算。输出长、复用多的场景把 `offload_prompt_only` 关掉。
- **以为 CPU 命中的请求当步就能开算。** SGLang 里内存命中的那段和本步前向逐层重叠,从那边过来的人会这么想。vLLM 让命中的请求在门外等整段拷完、完成的消息回到调度器才收;下层命中还要多两轮查询和读回。短 prompt 的负载拉回未必比重算划算,可以用请求里的 `max_load_tokens` 关掉。
- **以为 `--kv-offloading-size` 是每张卡的。** 看到「每卡一份内存池」的写法太多,就按单卡填。它是所有卡加起来的总数,TP 8 时每卡只分到 1/8;同一个值,换 TP 之后每张卡能留的前缀就变了。
- **以为磁盘层会按容量自己淘汰。** 简版的磁盘后端确实有容量上限、按 LRU 淘汰,两套实现被混为一谈。多层链里的文件系统下层没有容量参数,只写不删,盘满之后新块再也写不进去。
- **以为清了前缀缓存,所有层都干净了。** `/reset_prefix_cache?reset_external=true` 只清 CPU 主层,代码里特意不清下层,好让持久存储跨重启复用。RL 换权重、模型名不变时,磁盘和对象存储里的旧 KV 仍会被当成命中拉回来,而且不报错。
- **以为 HiSparse 一开显存占用就降。** 看着文档说「冷 KV 放主机」,开了之后显存曲线却和原来一样,就以为没生效。它在池子不紧时什么都不放,GPU 空闲块跌到 10% 以下才让请求切到热区、把页还给池子;省下的显存体现在满载时能多收的请求上。索引器的 KV 也一直全量留在 GPU。

# SGLang 11|PD 分离与 EPD

prefill 为什么会打断 decode、拆开之后多出来的传输这件事、推还是拉是两套实现、分层传是能力上限不是默认行为、首字 token 在哪一侧产出、为什么瓶颈总是 P,vLLM 解读的「PD 分离与 KV 传输」一章已经讲透。这一页只说 SGLang 的握手与传输本身、代价、以及和 vLLM 不一样的地方。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

SGLang 的调度器一步只做一件事,有 prefill 批就先跑 prefill(04 章)。一条 8000 token 的 prompt 在 70B 模型上 prefill 要 200–800 毫秒,这一步里所有在跑的 decode 请求的下一个 token 都在等;并发 100–500、prompt 又长的负载下,TPOT 的 P99 从 30 毫秒跳到 500 毫秒以上,均值看着还行。DP attention 再放大一次:8 个 DP rank 每步要同步,只要 1 个 rank 在做 prefill,其他 7 个 rank 的 decode 都陪着等。多模态模型再多一层:ViT 编码 1–10 张图要 50–500 毫秒算力,和 prefill 在同一张卡上抢,图多的负载下所有请求的 TTFT 一起变差,而且 encoder 权重占着每张 decode 卡的显存。同一张卡上按 SM 分区的 PD 多路复用(`--enable-pdmux`,07 章)只能缓解,根治要把它们拆到不同的卡上。

## 二、解法:D 先腾地方,P 算完就推

把 prefill 和 decode 拆成两组服务器,KV 算完由 RDMA 网卡直接从 P 的显存写进 D 的显存。最天然的直觉是「P 算完把结果寄给 D」,SGLang 实际的顺序反过来:**D 先腾出地方,再告诉 P 往哪写。** 一条请求的路径:

- 网关(18 章)给每条请求生成一个 63 位随机数当房间号 `bootstrap_room`,同一份请求同时发给 P 和 D,两边都带着 P 的地址和房间号。P 只算这一条 prompt 的 prefill,`max_new_tokens` 被强制成 1。
- P 启动时起一个握手服务(默认 8998 端口)当花名册,每个 rank 把自己的 IP、ZMQ 端口、TP/DP/PP 坐标注册进去。D 收到请求先去查一次「这个 P 有几个 rank、页大小、KV 精度、各在哪」,页大小或精度对不上直接报错;查到就缓存,以后只走 ZMQ。
- D 的预分配队列按先来先服务分格子:prompt 长度加 512 个 token 的 decode 余量,还要保证「预分配的加上在跑的能踢出去的」够任何一条请求跑到头。分好后把页号、元数据槽号、`decode_prefix_len` 打包发到 P 对应 rank 的 ZMQ 端口。`decode_prefix_len` 是 D 侧开了前缀树时它已经有的前缀长度,P 从这个位置以后才发。
- P 的 bootstrap 队列里,请求收到这条消息才从 Bootstrapping 变成 WaitingForInput,进等待队列开始算;没收到就一直等,上限 300 秒。
- P 每算完一块就发一块(chunked prefill 分块,不满一页的尾巴留给下一块;命中前缀树的那段在 forward 之前就先发)。发送是把自己的页号和 D 的页号配成连续区间,交给后台传输线程,**所有层合成一个批一次 RDMA 写过去。** 最后一块之后再写一份 64 字节对齐的元数据:首 token、cached_tokens、logprobs、推测解码的隐状态;然后给 D 发一条状态消息。
- D 的传输队列轮询到 Success,请求进等待队列,以「已建好」的批合入 running batch,首 token 直接用 P 传来的,不重算。P 给网关回一条长度 0 的响应,客户端看到的流全部来自 D。

为什么不吃亏:8000 token 在 70B 模型(GQA,8 个 KV 头)上的 KV 约 2.6 GB,400 Gb/s 网卡 50–100 毫秒,MLA 模型只有它的 1/5;prefill 本身要 200–800 毫秒。传输在 P 的后台线程做,D 预分配和收数据时 GPU 在给别的请求 decode,两边的 GPU 都不停。

## 三、代价

- **D 的显存有一部分被在途请求占着。** 预分配是 prompt 加 512 个 token 的余量,KV 到之前这些格子空转;`pre-allocated usage` 是单独的一个数。每 worker 并发不超过 32 时清单还多留 2 倍的行。
- **元数据缓冲两侧都按倍数买断。** P 是最大并发的 2 倍,D 是清单行数的 2 倍(MiniMax 是 8 倍),每行 10–15 个 64 字节对齐的小张量,推测解码时还带一份隐状态。
- **TTFT 多出三段。** 握手(D 查表、发元数据、P 换状态)、传输、D 轮询的最多 1 步。P 侧常驻 4–12 个传输线程和 4 条队列,按目标 D 分片。
- **前缀缓存断成两半。** P 有自己的树,D 默认关树用块缓存;多轮对话上一轮回答的 KV 在 D,P 看不到,下一轮全量重算。要么 D 开树(和推测解码互斥),要么走 HiCache 的 L3 加 D 侧 offload(05 章)。
- **失败面变大。** 两侧各一个 300 秒超时;P 挂了 D 靠心跳(5 秒一次,连错 2 次)发现,受影响请求全部 500;D 挂了 P 靠 Mooncake 会话失败发现,同一个 D 的后续块直接跳过。
- **功能表打折。** D 侧树与推测解码、HiSparse、fake 互斥;角色切换与 DP、EP、PP、DCP、推测解码互斥;P 关 decode 的 CUDA graph、D 关 prefill 的;unified memory 要求两侧 TP 相同;两侧 TP 不同时默认按 token 切片传,开中转缓冲才有 2–5 倍;Responses API 的有状态功能不支持;fake 后端只能给 D。

## 四、和 vLLM 不一样的八处

1. **方向只有推。** D 先分配再把页号发给 P,P 用 RDMA 写进 D;全树没有 D 去读 P 的实现。vLLM 的 NIXL 默认是拉,推是另一套代码。
2. **握手有一个服务。** P 起一个 HTTP 花名册,rank 启动时注册、D 查表缓存,请求级的元数据走 ZMQ;vLLM 两端没有一句 RPC,坐标随传输参数由网关带。
3. **网关同时发。** SGLang 自家网关生成房间号后同时把请求发给 P 和 D,P 失败网关直接回 502,不等 D 超时;vLLM 是先 P 后 D 两次 HTTP。
4. **首 token 随元数据走。** P 采样的首 token、logprobs、推测解码的隐状态都走 RDMA 到 D 的元数据槽,D 不重算;vLLM 只有一个实现有 opt-in 的首字回传。
5. **分块发、整层批。** chunked prefill 每块算完就发,命中的前缀在 forward 前先发;但每次发送把所有层合成一个批,没有逐层和计算重叠,和 vLLM 那章的结论一样。NVLink 自定义显存池下改成按层并行提交,仍不是流水。
6. **后端是继承不是注册表。** 5 个内置(mooncake 默认、nixl、ascend、mori、fake)加一个 mooncake_tcp 别名,全部继承同一个基类,握手、心跳、状态机共用;后端不反过来规定 KV 布局。vLLM 是 16 个注册名惰性加载,connector 能规定布局。
7. **失败处理是双向的。** D 用心跳探 P,P 用会话失败探 D;D 放弃一条请求时给 P 发 ABORT、P 回 ACK;权重更新后的真抢占由 D 反向调 P 的 `/generate` 让它按新权重重算(04 章)。
8. **角色能运行时切换,encoder 能再拆一段。** `/pd_role_switch` 在空闲时把一台 P 变成 D 或反过来,KV 池不重分;EPD 把 ViT 编码拆成 `--encoder-only` 服务器,语言端 `--language-only` 不加载 encoder 权重,embedding 默认 ZMQ 直送调度器,也可以走 Mooncake RDMA;vLLM 那章只有 P 和 D 两段。

## 五、调参与观测

**参数在哪调。** 启动参数 `--disaggregation-*` P 和 D 各自给,改了要重启;`SGLANG_DISAGGREGATION_*` 环境变量是超时、线程、队列,P 侧的和 D 侧的分开生效;请求参数 `bootstrap_host`、`bootstrap_port`、`bootstrap_room` 由网关填,绕过网关直发又缺房间号就是 400;运行时接口只有 `/pd_role_switch`(要 `--enable-pd-role-switch` 且完全空闲),另外 `/release_memory_occupation` 会顺手清空 PD 队列(17 章)。EPD 的 `--encoder-*` 一组参数在 encoder 端和语言端各给一半。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--disaggregation-mode` · 这台是 P 还是 D | 启动 | null | prefill:起握手服务、关 decode 图、`max_new_tokens` 强制 1;decode:关 prefill 图、树默认关成块缓存 | 就绪前日志 `CommonKVBootstrapServer started successfully`;D 侧警告 `KV cache is forced as chunk cache` |
| `--disaggregation-transfer-backend` · 用哪条传输 | 启动,两侧一致 | mooncake | nixl 换 UCX 或 LIBFABRIC 插件;ascend 昇腾;mori AMD;mooncake_tcp 没 RDMA 网卡时用,慢 10 倍以上;fake 只给 D 压测,不传 KV | 启动日志 `NIXL KVManager initialized with backend`;`kv_transfer_speed_gb_s` |
| `--disaggregation-bootstrap-port` · P 的握手服务端口 | P 启动;D 侧是「去找 P 的哪个端口」 | 8998 | 同机多个 P 必须错开,否则 bind 冲突;D 侧默认也按 8998 找 | P 日志 `started successfully on host:port`;D 报 `Failed to get prefill server info` |
| `--disaggregation-ib-device` · RDMA 网卡 | 启动,mooncake | 无,自动探测 | 按 GPU 给 JSON 让每张卡走最近的网卡;写错了传输直接失败 | 启动警告 `falling back to auto discovery`;`kv_transfer_speed_gb_s` |
| `--disaggregation-decode-enable-radix-cache` · D 侧开前缀树 | D 启动 | 关 | 开:D 已有的前缀不再传,多轮的 TTFT 降;与推测解码、HiSparse、fake 互斥;DP attention 下标实验 | 警告 `Radix cache is enabled for decode server`;`kv_transfer_total_mb` 变小 |
| `--disaggregation-decode-retraction-backup` · D 抢占时 KV 存哪 | D 启动 | 按池推断 | cpu_tensor 每请求一份 CPU 张量;host_pool 预留 0.2 倍的 HiCache 池,满了不回退(03 章) | Decode batch 行的 `#retracted-req` |
| `--num-reserved-decode-tokens` · 预分配时每条多留几步 | D 启动 | 512 | 调高:在途请求少被抢占,但并发上限降;调低反过来 | `pre-allocated usage`;警告 `Retract requests` |
| `--disaggregation-decode-extra-slots` · 给在途请求多留几行清单 | D 启动 | 0;每 worker 并发不超过 32 时是 2 倍 | 小并发下让传输和计算重叠;大并发默认不留 | `#transfer-req` 贴着并发上限就该加 |
| `--disaggregation-decode-polling-interval` · D 每几轮轮询一次队列 | D 启动 | 1 | 调高:CPU 开销降,KV 到了要晚 N 步才进批,TTFT 加 N 步 | TTFT 比 P 侧传输完成时间多出的部分 |
| `--optimistic-prefill-attempts` · P 不等握手先算 | P 启动 | 0 | 调 1–2:TTFT 减掉握手时间;握手迟迟不来会让位给已握手的请求,重算 | Prefill batch 行的 `#optimistic-req`;`num_prefill_retries_total` |
| `--disaggregation-enable-kv-checksum` · 传完校验 | 两侧 | 关 | 每请求算一遍 Adler-32,对不上就 abort;排查错字时开,常态关 | `num_transfer_failed_reqs_total` |
| `--enable-pd-role-switch` · 允许运行时换角色 | 两侧 | 关 | 每台都起握手服务;与 DP、EP、PP、DCP、推测解码互斥;切换时清空前缀树 | `/pd_role_switch` 的返回;失败后实例标 unhealthy 要重启 |
| `--load-balance-method` · P 的 DP rank 怎么选 | P 启动 | PD prefill 下 auto 解析为 follow_bootstrap_room | 房间号取模选 rank,D 不用再查;改 round_robin 每条请求多一次注册和查询 | D 日志 `follow_bootstrap_room conflict` |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` · P 等 D 元数据的上限 | P 环境变量 | 300 秒 | 调高:D 排队久也不失败,但 D 挂了 P 的树锁晚释放 | 警告 `timed out when bootstrapping`;`num_bootstrap_failed_reqs_total` |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` · D 发完元数据后等 KV 的上限 | D 环境变量 | 300 秒 | 调高:P 排队长也不失败,但 D 的预分配显存被占更久 | 警告 `fail to receive KV Cache transfer done signal` |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` 与 `_MAX_FAILURE` · D 探 P 的频率和判死次数 | D 环境变量 | 5 秒(最低 2)/ 2 次 | 调低发现得快但误判多;判死后该 P 的所有在途请求 500 | 错误行 `Lost connection with prefill instance` |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` 与 `_QUEUE_SIZE` · P 的传输线程和队列 | P 环境变量 | 线程 = CPU 数 × 0.5 / 8,夹在 4–12;队列 4 | 线程必须不少于队列;队列按目标 D 分片,设 1 就是先来先发 | `kv_transfer_speed_gb_s`;`#inflight-req` 堆积 |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` · 走 NVLink | 两侧环境变量 | 无 | NVLINK 或 BAREX 用自定义显存池(03 章),按层并行发;INTRA_NODE_NVLINK 同机;元数据仍走 TCP | 启动日志 `Initialized custom memory pool` |
| `SGLANG_DISAGG_STAGING_BUFFER` 与 `_POOL_SIZE_MB` · 两侧 TP 不同时的中转缓冲 | 两侧环境变量 | 关 / 4096 MB | 异构 TP 吞吐 2–5 倍;MLA 模型不能开;`chunked_prefill_size` 要是页的倍数;TP 相同自动跳过 | 启动 RuntimeError |
| `SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX` · 命中的前缀先发 | P 环境变量 | 开 | 关掉后前缀和新算的一起在最后发,TTFT 长 | `kv_transfer_latency_ms` |
| `--encoder-only` / `--language-only` / `--encoder-urls` · EPD 三件套 | encoder 端 / 语言端 / 语言端 | 关 / 关 / 空 | 语言端不给 urls 就等 encoder 用 `--encoder-register-urls` 来注册(端口 8997);只支持 Qwen-VL、Kimi、GLM 等 16 种架构 | 语言端日志 `Encoders are expected to register dynamically` |
| `--encoder-transfer-backend` · embedding 怎么送 | 两端一致 | auto,解析为 zmq_to_scheduler(Kimi K3 且 TP 大于 1 是 zmq_to_tokenizer) | mooncake:RDMA 直写,要配 ib_device;`SGLANG_ENCODER_MM_RECEIVER_MODE=grpc` 时走 gRPC | 启动日志 `Encoder transfer backend auto-resolved to` |
| `--enable-adaptive-dispatch-to-encoder` · 图少的本地算 | 语言端 | 关 | 开:少于 2 个媒体项(`SGLANG_ENCODER_DISPATCH_MIN_ITEMS`)本地编码,多的才发 encoder;批请求一律不发 | 警告 `not supported in EPD disaggregation mode` |
| `--enable-mm-global-cache` · 跨实例复用 ViT 输出 | encoder 端 | 关 | 同一张图别的 encoder 算过就从 Mooncake 拿;和传输后端无关 | encoder 端命中日志 |

**怎么看。** P 的 `Prefill batch` 行多三个数:`#bootstrap-req`(在等 D 的元数据)、`#inflight-req`(算完在传)、`#optimistic-req`;D 的 `Decode batch` 行多四个:`pre-allocated usage`、`#prealloc-req`、`#transfer-req`、`#retracted-req`。`/metrics` 里同一组数是 `num_prefill_bootstrap_queue_reqs`、`num_prefill_inflight_queue_reqs`、`num_decode_prealloc_queue_reqs`、`num_decode_transfer_queue_reqs`、`pending_prealloc_token_usage`,传输本身看 `kv_transfer_speed_gb_s`、`kv_transfer_latency_ms`、`kv_transfer_total_mb`、`kv_transfer_bootstrap_ms`、`kv_transfer_alloc_ms`(P 侧只按最后一块计时),出事看 `num_bootstrap_failed_reqs_total`、`num_transfer_failed_reqs_total`、`num_prefill_retries_total`;标签 `engine_type` 分 prefill 和 decode。请求级三行:P 侧 `Prefill bootstrap failed for request` 与 `Prefill transfer failed for request`,D 侧 `Decode transfer failed for request`,都带 rid 和房间号。P 的传输线程一旦抛异常整个实例报 `Transfer thread failed ... is dead`。

| 症状 | 先查 | 然后 |
|---|---|---|
| D 报 `Failed to get prefill server info` 且 503 `not fully registered` | P 的 rank 没注册齐;多机时握手服务在 `--dist-init-addr` 那台 | 等 P 全部就绪;核对 D 请求里的 bootstrap_port 和 P 的端口 |
| 启动就 `Page size mismatch` 或 `KV cache dtype mismatch` | 两侧 `--page-size`、`--kv-cache-dtype` | 改成一样 |
| 请求 400 `without bootstrap room id` | 是不是绕过网关直发了 P 或 D | 走网关;或请求里自己填 `bootstrap_host`、`bootstrap_room` |
| P 大量 `timed out when bootstrapping` | D 的 `pre-allocated usage` 和 `#prealloc-req`:D 分不出格子 | 降 `--num-reserved-decode-tokens`;加 D;D 挂了就重启 |
| D 大量 `fail to receive KV Cache transfer done signal` | P 的 `#bootstrap-req`、`#inflight-req`:P 排队还是传输堵 | 加 P;`kv_transfer_speed_gb_s` 低就查网卡 |
| `Lost connection with prefill instance` | P 进程和网络 | 重启 P;网关摘除该 P |
| `kv_transfer_speed_gb_s` 只有 1–5 | 是不是 mooncake_tcp 或 ib_device 没配对 | 按 GPU 指定网卡;两侧 TP 不同就开中转缓冲 |
| D `Retract requests` 频繁且 `#transfer-req` 高 | 在途请求把余量吃光 | 调高 `--num-reserved-decode-tokens`;调低 `--max-running-requests` |
| 多轮对话每轮都全量 prefill | D 侧树是不是关着;有没有 L3 | D 开树;或 HiCache L3 加 `--disaggregation-decode-enable-offload-kvcache`(05 章) |
| P 报 `Session ... failed` | 对应 D 的 Mooncake 会话死了 | 查 D;开 `SGLANG_ENABLE_FAILED_SESSION_PROBE` 让它 30 秒一探自动恢复 |
| `/pd_role_switch` 返回 `not idle` | `#running-req`、四个 PD 队列是否归零 | 断流量等空;换 decode 还要给 `decode_cuda_graph_memory_gb` |
| EPD 单图请求没走 encoder | `--enable-adaptive-dispatch-to-encoder` 开着,阈值 2 | 关掉它,或 `SGLANG_ENCODER_DISPATCH_MIN_ITEMS=1` |

## 六、常见误区

- **「D 收到 KV 才分配显存」。** 因为单机是算完才写。实际反过来,D 先分配再把页号发给 P,KV 到之前格子就占着,所以 D 的 `token usage` 之外还有一个 `pre-allocated usage`;D 显存紧的第一反应该看后者。
- **「P 卡在 bootstrap 队列是 P 的问题」。** 因为队列在 P 上。实际 P 只知道房间号,等的是 D 那条元数据;P 的 `#bootstrap-req` 堆高,先去看 D 的 `#prealloc-req` 和预分配使用率。
- **「300 秒超时说明传输慢」。** 因为名字里有 timeout。实际两个 300 秒是等对端的上限,不是传输时长;8000 token 传 50–100 毫秒,超时几乎都是排队或对端挂了。
- **「两侧的 `--disaggregation-bootstrap-port` 意思一样」。** 因为参数名一样。P 侧是自己监听的端口,D 侧是「去找 P 的哪个端口」,默认都是 8998 所以平时看不出;同机起两个 P 就撞上了,Rust 服务器模式下 P 侧还会被改成 API 端口。
- **「文档写线程池默认是 CPU 数的 0.75 倍」。** 文档确实这么写。代码是 0.5 倍再除以 8、夹在 4–12;32 核机器是 4 个线程不是 12 个,想提带宽要显式设。
- **「fake 后端可以用来压 P」。** 因为它叫 fake。P 侧直接 assert 拒绝,fake 只在 D 侧假装收到了 KV,用来单测 decode 的吞吐。
- **「D 侧开了树就和单机一样命中」。** 因为树是同一棵。开了只是 D 已有的前缀不再传,P 该算的还是算;而且它和推测解码互斥,DP attention 下还要网关按前缀路由才有命中。
- **「KV 是逐层流水传的」。** 因为博客里画过。默认所有层合成一个批一次写;NVLink 自定义池下按层并行提交,也只是并行,不和 forward 重叠;开了 chunked prefill 才有块级别的边算边传。
- **「D 抢占后会自己重算」。** 因为单机是这样。D 不会算 prefill:普通抢占把 KV 备份到内存再恢复(03 章);权重更新后的真抢占是 D 反过来调 P 的 `/generate` 让 P 按新权重重算,P 的 HTTP 端口是它注册花名册时自己报的。
- **「首 token 由 D 算」。** 因为流从 D 出来。首 token 是 P 采的,和 logprobs、隐状态一起随元数据传过去,D 拿来直接吐;所以 TTFT 里含传输时间,`top_logprobs` 超过 128 个在 P 侧直接报错。
- **「`SGLANG_DISAGGREGATION_QUEUE_SIZE` 越大越快」。** 因为叫队列。队列按目标 D 分片,多队列只是让不同 D 之间不互相堵;线程数必须不少于队列数,否则启动 assert。
- **「EPD 的 embedding 必须走 Mooncake」。** 因为 PD 的 KV 走 Mooncake。默认是 ZMQ 把张量直接送到语言端调度器,Mooncake 是可选;`--enable-mm-global-cache` 是另一件事,管的是 ViT 输出跨实例复用。

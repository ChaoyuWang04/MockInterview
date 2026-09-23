# TensorRT-LLM 09|PD 分离与 KV 传输

prefill 和 decode 若一直挤在同一张卡的同一批里，首 token 和后续间隔会互相拖，两段也不能各自选并行。

说法都在源码基准 `59f5c47f`(tag `v1.2.1`)上核过，「当前」与「默认」都指这组基准。

## 一、核心问题

默认同卡 IFB 把新请求的 prefill 和别人的 decode 塞进同一步。一条 2000–8000 token 的 prompt 要占 200–800 毫秒算力时，正在吐字的请求这一步只能等，**TPOT 的尾部被 prefill 打断**；反过来，decode 先占满一步预算，新 prompt 进不来，**TTFT 被拖长**。两段还被迫共用一套并行：prefill 想多切权重、decode 想多切请求或上下文，同卡做不到。长输入、中等输出、并发 10–500 时，这两个指标一起坏。

同卡切块只能缓解，根治是把两段拆到不同 GPU 池。拆开之后多出来一件混部没有的事：整份 KV 必须搬走，而且要在 decode 开始前搬完。

## 二、解法:分池，NIXL 搬块，传输藏进别人的计算

最天然的直觉是「prefill 算完把 KV 寄给 decode」。本项目默认并不拆：传输器后端没有默认值，不配就不装传输器，分阶段请求会被拒。配上之后，prefill 实例和 decode 实例各占一组卡，KV 走 NIXL 在设备内存之间直传；两侧并行可以不同，传输时做布局变换。

![默认同卡 IFB 用虚线画在同一张卡上、不分池；拆开后 prefill 池算完由 NIXL 把块搬到 decode 池，本请求在传时别人仍在算，编排进程先发 context 再发 generation](/opensource/TensorRT-LLM/09a-pd-pool-kv-transfer.svg)

一条请求的路径：客户端打到编排服务；编排先把请求标成只做 context 发给 prefill 实例，拿到首 token 和一组上下文参数；再把请求标成只做 generation 发给 decode 实例。decode 侧用这组参数连上 prefill 侧把块拉过来。连接器是块池接到外部存储的另一条插口，见 03 章，不是这一页的传输器。Helix 只出现在 decode 池上，见 07 章。多实例要对上同一条全局请求号，协调细节在 15 章。

为什么不吃亏：传输和计算按**不同请求**重叠——这条在搬块时，GPU 正在给别的请求做 prefill 或 decode。多卡实例之间也可以并行传。搬的量相对一次超长 prefill 往往更短；P 侧块传到确认就还，周转比同卡占到生成结束快一个数量级。

## 三、代价

**默认仍是同卡。** 不写传输后端，工人照常起，但不装传输器。Dynamo 是仓外编排卫星，不是本引擎默认 PD 路径。

**多一跳 HTTP 加一次设备直传。** TTFT 里多出编排往返、建连和搬块。头几条请求要把通信建起来，带宽看起来偏低，压测要先热身。

**两侧配置必须手对齐。** 后端、超时、弹跳缓冲没有跨工人协商；对不上不会当成配置错误报出来，只是传失败或退回逐块写。

**功能打折。** 文档只保证解码器模型；每层 KV 的类型和头数要齐。同一实例兼做两种分阶段请求没有最优调度。Python 侧弹跳缓冲要有限超时；流水线按块边算边传还有一串静态约束，开错启动就失败。

**失败面变大。** 默认 60 秒等不到块就取消。请求号撞车会写坏传输。关掉传输重叠后，decode 工人会干等搬完再算。

## 四、prefill 和 decode 怎么分池、KV 怎么搬

这一页比的是 prefill 和 decode 怎么分池、KV 怎么搬、传输能不能藏进别人的计算。

| 维度 | TensorRT-LLM | vLLM | SGLang |
|---|---|---|---|
| 默认是否分池 | 否，同卡 IFB；要显式给传输后端 | 实例配成生产或消费才分 | 实例用分离模式才分 |
| 谁编排 | 本仓另起编排服务，先 context 再 generation | 引擎外网关发两次 HTTP | 网关同时发给两侧 |
| 传输库算不算对等引擎 | 不算；NIXL 只搬块 | 连接器注册表里多种实现 | 内置多种传输后端 |
| 重叠画在哪 | 这条在传、别人在算；不是默认逐层流水 | 默认拉，重叠在消费侧载入 | 后台线程推，D 预分配后收 |
| 两侧并行 | 允许不同张量并行与流水线并行，传时改布局 | 由连接器约束布局 | 同 TP 走快路径，不同要中转 |
| 首 token | prefill 侧产出，随上下文参数交给 decode | 多数实现不回传，要显式开 | 随元数据槽回传 |

重心在本项目：分池是开关不是默认；搬块是传输器不是 03 章的连接器；重叠是跨请求而不是同请求逐层。

## 五、调参与观测

这一页的参数全是启动配置，改了要重新拉起整个作业。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `cache_transceiver_config.backend` · 用哪套库搬 KV | 工人 YAML | 无，不装传输器 | 填 `NIXL`：RDMA 或 NVLink 直传；两侧必须相同 | 日志 `Using KvCacheTransceiverV2` 或 `cache_transceiver is disabled` |
| `cache_transceiver_config.transceiver_runtime` · Python 还是 C++ 传输器 | 工人 YAML | `auto` | `PYTHON`：弹跳与流水线；`CPP`：无弹跳 | 同上；混用 Mamba 池会启动报错 |
| `cache_transceiver_config.kv_transfer_timeout_ms` · 等多久算传失败 | 工人 YAML | `60000` | 调高：对端排队久也不取消，块占更久 | 请求被取消、传输超时日志 |
| `cache_transceiver_config.kv_cache_bounce_size_mb` · 弹跳缓冲多大 | 工人 YAML | `0`，逐块写 | 大于 0：散块合成一次写；C++ 传输器忽略 | 散块布局下带宽是否上去 |
| `cache_transceiver_config.agent_bounce_buffer_enable` · 弹跳是否走 C++ 代理 | 工人 YAML | 关 | 开：两侧共享一块缓冲，头数不对才默认走 | 对端加载后警告并退回标准 NIXL |
| `cache_transceiver_config.enable_pipelined_transfer` · 切块边算边传 | 工人 YAML | 关 | 开：后一块还在算时先传已完成块；约束很多 | 启动失败或请求被拒 |
| `cache_transceiver_config.max_tokens_in_buffer` · decode 同时收几条 | 工人 YAML | 空，按模型最长序列推 | 手写太小会卡住传输入口 | decode 侧排队、TTFT 变长 |
| `TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP` · 禁止传算重叠 | 环境变量 | `0`，重叠 | `1`：decode 干等搬完 | TPOT 变差、GPU 在收块时空 |
| `TRTLLM_NIXL_KVCACHE_BACKEND` · NIXL 底下走 UCX 还是 libfabric | 环境变量 | `UCX` | `LIBFABRIC`：要带插件的构建，非法值警告后回 UCX | 启动警告 |
| 编排配置里的 context / generation 地址 · 两池有几台 | `trtllm-serve disaggregated -c` | 无，必填 | 多写几台提高该阶段并发 | 编排口 8000 能通、工人端口探活 |

**怎么看。** 工人日志先确认传输器起来了，再确认 prefill 与 decode 两侧都起来。编排口打补全；失败先看「Disaggregated serving is not enabled」。带宽看热身之后，不要拿头几条当稳态。

**典型配置。** 三套能直接抄走的起法，参数名和默认值都来自上表。prefill 工人关 overlap 调度（官方示例），decode 工人保持默认重叠。

| 什么场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 两卡同机、小模型、要拆开 TTFT 与 TPOT | 工人：`CUDA_VISIBLE_DEVICES=0 trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --host localhost --port 8001 --config pd-ctx-nixl.yaml`（yaml 写 `disable_overlap_scheduler: true` 与 `cache_transceiver_config: {backend: NIXL}`）；另一张卡端口 8003 用 `pd-gen-nixl.yaml`（只写同一套 `cache_transceiver_config`）；编排：`trtllm-serve disaggregated -c pd-orch-1p1d.yaml`（一台上文、一台生成） | 用一块卡专做 prefill、一块专做 decode，换掉同卡互相拖；多一次搬块 |
| 两台上文加一台生成、长 prompt 中等输出 | 两张卡起上文（8001、8002，同一份 `pd-ctx-nixl.yaml`），一张卡起生成（8003，`pd-gen-nixl.yaml`），编排 `trtllm-serve disaggregated -c pd-orch-2p1d.yaml` | 用更多 prefill 卡换 TTFT；decode 卡仍可能先满 |
| 两侧张量并行不同、块在显存里很散 | 两侧 yaml 把 `kv_cache_bounce_size_mb` 写成正数（例如 256），其余仍是 NIXL；编排不变 | 用一块弹跳缓冲换一次大写；两侧必须相同，否则退回逐块 |

组合与推荐值是按语义推的起点，不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| `Disaggregated serving is not enabled` | 该工人 yaml 有没有传输后端 | 两侧都写成 NIXL，不要只配编排 |
| 日志 `cache_transceiver is disabled` | 后端仍是空 | 改 yaml 重启工人 |
| 头几条特别慢、带宽很低 | 是不是刚建连 | 先热身再测 |
| decode 侧 GPU 空、传输却在跑 | 是否关掉了传输重叠 | 去掉该环境变量 |
| 传失败但启动没报配置错 | 两侧超时、弹跳、运行时是否一致 | 手对齐，没有自动协商 |
| 请求写坏或串台 | 全局请求号是否碰撞 | 不要自己填重复号；多实例走 15 章协调 |
| NVLink 域不同时挂住 | 域 UUID 是否跨域 | 按文档收窄 UCX 设备与方案 |

## 六、常见误区

- **以为默认已经 PD。** 因为文档大谈分池。实际后端无默认，不写就不装传输器，请求仍走同卡 IFB。
- **以为传输器就是 03 章的连接器。** 因为都碰 KV 块。连接器把块接到外部缓存；传输器只在 prefill 池和 decode 池之间搬。
- **以为 NIXL 是对等引擎。** 因为它常和别的框架并列出现。它是搬块的库，对照表里的对等列仍只该是推理引擎。
- **以为重叠是同一条请求逐层流水。** 因为图上画过边算边传。默认是这条在传、别人在算；按块流水要另开开关，且约束很严。
- **以为同一实例兼做两种分阶段请求更省卡。** 因为能跑通。调度没有为混部优化，文档明确不推荐。
- **以为编排进程会帮两侧对配置。** 因为请求从它转。配置是各工人自己的，对不上只在传输时爆。

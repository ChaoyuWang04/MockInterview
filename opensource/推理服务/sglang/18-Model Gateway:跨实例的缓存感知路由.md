# SGLang 18|Model Gateway:跨实例的缓存感知路由

vLLM 解读没有对应章;单实例内的数据并行分发见本项目 01 章。这一页只说网关(SGLang Model Gateway,简称 SMG,一个独立的 Rust 进程)在多台实例之间怎么选实例、怎么给 PD 配对、代价是什么、和实例内的分发有什么不一样。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

一台实例的前缀树(04 章)只认自己算过的东西。8 台实例前面放一个普通的轮询负载均衡,同一个用户的第 2 轮对话有 7/8 的概率落到没算过第 1 轮的那台,树的命中率从单机的 60%–75% 掉到 20% 上下,prefill 算力翻倍,TTFT 跟着翻倍,GPU 显存里还存着 8 份互相看不见的重复前缀。多轮对话、agent 循环、共享 system prompt 的负载最惨,单条请求越长、轮数越多,浪费越大。

PD 分离(11 章)再加一层:一条请求要同时送到一台 P 和一台 D,两边还得拿着同一个房间号,普通负载均衡器做不了这件事。实例挂了、实例在加载模型、实例的显存被打满,这些也都要有人看着,DP 控制器(01 章)只在一个进程里,看不到别的机器。

## 二、解法:网关自己记一棵近似树,按前缀猜实例

最天然的直觉是「谁算过就发给谁」。网关照着做,只是它不去问实例「你缓存了什么」,而是**自己记下每条请求发去了哪台** :把请求的原文当成一条字符串插进一棵树,树上每个节点标着「这段字符哪几台实例见过」。新请求来了从根往下走,走到分叉为止,匹配到的字符数除以请求总字符数就是命中率;命中率超过 `cache_threshold` 就发给匹配最长的那台,否则发给当前在飞请求最少的那台。发完再把这条请求插回树里。

它是猜,不是查:按字符不按 token,不知道实例有没有把那段前缀驱逐掉。猜错的代价只是那台实例多算一次 prefill,和轮询一样,不会出错。猜对的收益是 prefill 少算一段,所以实际不吃亏。官方在 8 卡 A100、多组长前缀的负载上测出命中率 20% 到 75%,吞吐 82665 token/s 到 158596 token/s。

猜有一个副作用:同一批前缀会被持续送到同一台,那台越来越忙。所以每次选之前先看负载:所有实例里在飞请求最多的减去最少的超过 `balance_abs_threshold`,并且最多的超过最少的 `balance_rel_threshold` 倍,两条同时成立才算失衡,失衡时放弃命中、直接发给最闲的那台。两条要同时成立,是为了让「1 台在跑 3 条、1 台空着」这种小数字不触发,也让「100 条对 90 条」这种比例接近的不触发。

PD 分离时网关给每条请求生成一个 63 位随机房间号,把 P 的地址、握手端口、房间号写进请求体,同一份请求同时发给一台 P 和一台 D;P 和 D 各用一套策略独立选,cache_aware 下 P 池和 D 池各有自己的树,互不驱逐。P 回了非 2xx 或连接失败,网关立刻断掉 D 那条连接并回 502,不让 D 干等 300 秒。

## 三、代价

- **多一跳。** 每条请求多一次 HTTP 转发;流式响应经网关逐块转。gRPC 模式下切词、推理解析、工具调用解析都搬进网关做,网关 CPU 变成要盯的东西。
- **树占网关内存。** 每台实例最多记 `max_tree_size` 个字符,默认 67108864,8 台就是 512 MB 量级的字符加节点开销;每 `eviction_interval` 秒扫一次全树按 LRU 删叶子,扫的时候要遍历整棵树。
- **命中是猜的。** 实例内部驱逐了网关不知道;权重更新、`/flush_cache` 之后网关的树还在,继续把请求送去那台,只是命中数变 0。
- **多个网关不共享树。** 网关横向扩成 3 个副本,每个副本各记各的,官方估命中率掉 10%–20%;mesh 同步只把插入操作发出去,接收端没有接线。
- **负载看的是网关自己数的在飞请求数** ,不是实例的 token 数、队列长度。一条 100 token 的请求和一条 100000 token 的请求在网关眼里一样重。

## 四、和实例内分发不一样的七处

1. **在哪。** DP 控制器(01 章)是主进程里的一个环节,只管本进程起的 dp_size 个 rank;网关是独立进程,管的是任意多台实例,实例可以在运行时用 `/workers` 加进来或摘掉,也可以让它去 Kubernetes 按标签自己找。官方文档已经把实例内的 DP 标成不推荐,让所有 DP 场景都走网关,`sglang_router.launch_server --dp-size 4` 起的就是 4 台单 DP 实例加 1 个网关。
2. **树记什么。** 04 章的真树按 token 存,节点值是 KV 槽号,命中直接省 prefill;网关的树按字符存,节点值是「哪几台见过」,一个模型一棵多租户树,命中只是一个路由决定。字符和 token 的分界不一样,所以「命中率 0.3」和实例日志里的 `#cached-token` 不是一回事。
3. **命中错了会怎样。** 真树错不了;网关猜错只多算一次。反过来网关也不知道真树里的前缀什么时候被踢,两边各驱逐各的。
4. **驱逐怎么做。** 真树显存不够才踢;网关按每台实例的字符总数上限踢,定时线程扫,时间戳用一个自增计数器而不是时钟,并且只有 1/8 的匹配会刷新时间戳,LRU 是近似的。
5. **负载是什么。** DP 控制器的 `total_tokens` 和 `total_requests` 读的是调度器发来的真实快照;网关的 cache_aware 读自己数的在飞请求数。`power_of_two` 本来每 30 秒去拉实例的 `/v1/loads` 想拿 token 数,但它读的是响应里的 `aggregate.total_tokens`,而实例返回的 JSON 里没有这个键,所以拉回来永远是 -1,退回在飞请求数比较;HTTP 非 PD 模式下只有 cache_aware 和 manual 两种策略会给实例计在飞请求,`power_of_two` 拿到的两台都是 0,选出来的就是随机抽到的第 1 台(PD 模式下非流式请求两边都计数,这一条不成立)。
6. **失败处理。** DP 控制器没有:rank 挂了整个实例挂。网关每 60 秒探一次 `/health`,连错 3 次下线、连对 2 次回来;每台一个熔断器,窗口 120 秒内错 10 次就打开 60 秒;失败按退避重试最多 5 次,每次重试重新选实例;4xx 算客户端的错,不计入熔断。
7. **PD 配对。** 实例内没有这件事;网关同时选 P 和 D,两套策略,两棵树,房间号由网关生成。

## 五、调参与观测

**参数在哪调。** 网关是独立进程,参数全是它自己的启动参数(Rust 二进制 `sgl-model-gateway` 或 Python 启动器 `python -m sglang_router.launch_router`;和实例一起起用 `sglang_router.launch_server`,网关参数加 `--router-` 前缀),改了要重启。运行时能动的只有实例列表:`POST /workers` 加一台(body 给 `url`,可选 `model_id`、`worker_type`、`bootstrap_port`、`labels`),`GET /workers` 看全部,`PUT` 和 `DELETE /workers/{id}` 改和摘;`/add_worker`、`/remove_worker`、`/list_workers` 是旧名字,基准上已经不在路由表里。策略本身运行时改不了。指标在另一个端口(默认 29000)的 `/metrics`。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后(哪个指标往哪变) | 怎么看 |
|---|---|---|---|---|
| `--policy` · 选实例的策略 | 启动 | cache_aware | random、round_robin 无状态;power_of_two 抽 2 台比在飞数;prefix_hash 按前 256 个 token 的哈希上一致性环,只在 gRPC 模式下有 token;manual 按 `X-SMG-Routing-Key` 粘住;bucket 和 consistent_hashing 只有 Python 启动器接受 | `smg_worker_selection_total` 的 policy 标签 |
| `--cache-threshold` · 命中率低于它就不按命中走 | 启动 | 0.3 | 调高(0.5):短前缀不再粘同一台,分布更匀,命中降;调低(0.1):更粘,单台容易过热 | 实例侧 `Prefill batch` 行的 `#cached-token` 与 `#new-token` 之比;各实例 `smg_worker_requests_active` 是否偏斜 |
| `--balance-abs-threshold` · 最忙减最闲超过多少才算失衡 | 启动 | 64 | 调低(16):更早放弃命中去均衡,命中降、尾延迟稳;调高:反过来 | `smg_worker_requests_active` 各实例的差 |
| `--balance-rel-threshold` · 最忙是最闲的几倍才算失衡 | 启动 | 1.5 | 和上一条同时成立才触发;某台完全空着时最闲是 0,倍数条件自动成立,只剩绝对差 | 同上 |
| `--eviction-interval` · 多久扫一次树做 LRU 驱逐 | 启动 | 120 秒(Python 启动器 60 秒) | 调短:树更新鲜、扫描 CPU 更多;调长:树里留着实例早已踢掉的前缀,命中猜错变多 | debug 日志 `Cache eviction completed for` |
| `--max-tree-size` · 每台实例在树里最多记多少字符 | 启动 | 67108864 | 调小:网关内存降,长会话早被踢出树;调大:反过来。它是字符数不是节点数 | 网关进程 RSS |
| `--prefill-policy` / `--decode-policy` · PD 模式下 P 和 D 各用什么策略 | 启动,要 `--pd-disaggregation` | 不给就都用 `--policy` | P 用 cache_aware 让前缀粘 P,D 用 power_of_two 或 round_robin 摊开;D 的树没什么用 | `smg_worker_selection_total` 的 worker_type 标签 |
| `--prefill URL [PORT]` / `--decode URL` · P 和 D 的地址 | 启动 | 空 | P 后面跟握手端口(11 章的 8998),不给就发 null 让实例用默认;运行时用 `/workers` 加 | `GET /workers` 的 `stats` |
| `--dp-aware` · 一台多 DP 实例按 rank 拆成多台看 | 启动 | 关 | 开:去实例 `/server_info` 读 `dp_size`,注册成 `url@0`…`url@N-1`,请求体带 `data_parallel_rank`;实例侧会打一行「已弃用,用 routed_dp_rank」的警告 | `GET /workers` 里 URL 带 `@` |
| `--worker-urls` / `--service-discovery` + `--selector` · 实例从哪来 | 启动 | 空 / 关 | 静态列表或 Kubernetes 按标签发现(PD 用 `--prefill-selector`、`--decode-selector`,P 的握手端口读 pod 注解 `sglang.ai/bootstrap-port`),每 60 秒同步一次 | `smg_discovery_workers_discovered` |
| `--health-check-interval-secs` 与 `--health-failure-threshold` / `--health-success-threshold` · 探活频率与判定 | 启动 | 60 秒 / 3 / 2 | 每台 GET `/health`,5 秒超时;调短发现得快、实例多时探活流量多 | `smg_worker_health`;`smg_worker_health_checks_total` |
| `--cb-failure-threshold` 等 4 个 · 熔断 | 启动 | 10 次 / 3 次 / 60 秒 / 120 秒 | 调低:一台实例抖几下就被摘 60 秒;`--disable-circuit-breaker` 关掉 | `smg_worker_cb_state`;`smg_worker_cb_transitions_total` |
| `--retry-max-retries` 等 5 个 · 重试 | 启动 | 5 次,50 毫秒起,×1.5,上限 30000 毫秒,抖动 0.2 | 只对 408、429、500、502、503、504 重试,每次重新选实例;非幂等的流式请求也会重发,`--disable-retries` 关掉 | `smg_worker_retries_total`;`smg_worker_retries_exhausted_total` |
| `--max-concurrent-requests` / `--queue-size` / `--queue-timeout-secs` · 限流与排队 | 启动 | -1(不限)/ 100 / 60 秒 | 开了之后超过并发的进队列,队列满回 429,排队超时回 408 | `smg_http_rate_limit_total` |
| `--request-timeout-secs` · 一条请求最长多久 | 启动 | 1800 秒 | 按最长生成时间给;太短长生成被网关掐断 | `smg_router_request_errors_total` |
| `--model-path` / `--tokenizer-path` · 网关自己的分词器 | 启动 | 无 | gRPC 模式必给,网关在自己进程里切词、解析推理块和工具调用;HTTP 模式不用 | 启动日志;`/v1/tokenizers` |
| `--prometheus-port` · 指标端口 | 启动 | 29000(Python 启动器要显式给) | 关不掉,只能换端口 | `curl :29000/metrics` |
| `--enable-igw` · 一个网关服务多个模型 | 启动 | 关 | 开:按请求里的 `model` 字段找实例,每个模型第 1 台实例注册时 `labels.policy` 决定该模型的策略;这样建出来的 cache_aware 用的是代码里的另一组默认值(0.5 / 32 / 1.1 / 30 秒 / 10000 字符),不是命令行的 | `GET /workers` 的 `model_id` |
| `--enable-mesh` / `--mesh-peer-urls` · 多网关互联 | 启动 | 关 / 空 | 开:网关之间同步实例状态与限流配置,树的插入只发不收 | `/ha/status`、`/ha/workers`、`/ha/policies` |

**怎么看。** 网关自己的三处:`smg_worker_requests_active` 按实例看在飞数是否偏斜(cache_aware 生效的直接证据是「偏斜但没超过失衡阈值」);`smg_worker_selection_total` 按 policy 和 worker_type 看每次选择走了哪个策略;`smg_worker_health`、`smg_worker_cb_state` 看有没有实例被摘。`GET /workers` 是运行时最有用的接口,一行一台,带 `is_healthy` 和 `load`。网关没有命中率指标,命中要去实例日志 `Prefill batch` 行看 `#cached-token`,或者响应 `meta_info` 里的 `cached_tokens`。debug 级日志里 `Load balancing triggered | max: … | min: …` 是失衡分支被触发的痕迹,`Removed stale worker … from cache tree` 是树里的租户已经不在了。`/engine_metrics` 把所有实例的 `/metrics` 合并成一份,`/v1/loads` 把所有实例的负载拉一遍,`/flush_cache` 给所有 HTTP 实例各发一次。gRPC 模式多出 `smg_router_ttft_seconds` 和 `smg_router_tpot_seconds`,HTTP 转发模式没有。

| 症状 | 先查 | 然后 |
|---|---|---|
| 加了网关命中率没变 | `--policy` 是不是 cache_aware;实例是否都注册在同一个 `model_id` 下 | 换策略;IGW 模式下核对 `labels.policy` |
| 1 台实例一直满、其他空着 | `smg_worker_requests_active` 的最大减最小是否一直没到 64 | 调低 `--balance-abs-threshold`;或调高 `--cache-threshold` |
| 命中率高但尾延迟差 | 同上,是命中把负载堆到 1 台了 | 先动绝对阈值,别关 cache_aware |
| 多轮对话第 2 轮还是全量 prefill | 请求原文的开头是否每轮都变(时间戳、随机 id 放在 system prompt 最前面) | 把变的部分挪到后面;树按字符从头匹配 |
| 网关内存一直涨 | `--max-tree-size` × 实例数 | 调小;确认驱逐线程在跑(debug 日志) |
| `power_of_two` 看着像随机 | 是否 HTTP 非 PD 模式;`/v1/loads` 拉回来是不是 -1 | 换 cache_aware,或换 gRPC 模式 |
| PD 模式请求 502 `prefill_server_error` | P 的状态;P 的握手端口对不对 | 看 P 日志;`--prefill URL PORT` 补端口 |
| PD 模式 400 `without bootstrap room id` | 是不是绕过网关直发了 | 走网关 |
| 实例明明活着却被摘 | `smg_worker_cb_state` 是 open 还是 `smg_worker_health` 是 0 | 熔断等 60 秒自动半开;探活看 `/health` 5 秒内回没回 |
| 实例加载模型时网关报没可用实例 | 注册流程等 `/health` 200,最长 1800 秒 | 等;或调 `--worker-startup-timeout-secs` |
| 429 | `--max-concurrent-requests` 和 `--queue-size` | 加大或设 -1 |
| 长请求被掐断 | `--request-timeout-secs` | 按最长生成时间加大 |
| `--dp-aware` 下实例日志刷警告 | 网关发的是 `data_parallel_rank`,实例要的是 `routed_dp_rank` | 基准上只是警告,功能正常;别关 |

## 六、常见误区

- **「网关知道每台实例缓存了什么」。** 因为名字叫缓存感知。实际它只记自己发过什么,按字符猜;实例内部踢了什么、`/flush_cache` 清了什么、换权重之后旧前缀还在不在,它一概不知道。命中率要去实例日志看,网关没有这个指标。
- **「命中率阈值 0.3 太低,调到 0.8 命中更高」。** 因为「阈值高等于要求高」。0.8 意味着请求前 80% 的字符都得见过才粘,多轮对话里每一轮新增的内容都会让比例掉下去,大部分请求走最闲实例,反而不粘了。要粘就调低,分布不均再用两个失衡阈值兜底。
- **「`--max-tree-size` 是节点数」。** 文档这么写。代码比的是每台实例在树里的字符总数,67108864 就是 64 M 字符每台,8 台实例是 512 M 字符的上限。
- **「两个失衡阈值任一成立就均衡」。** 因为文档写的是「before rebalancing」。代码是两条同时成立才触发,只调一条经常没反应;某台空着时倍数条件自动成立,这时只有绝对差在起作用。
- **「`power_of_two` 比的是实例的 token 数」。** 文档说 Load Monitor 喂它实例负载。基准上它读的 `aggregate.total_tokens` 在实例的 `/v1/loads` 响应里不存在(那份 JSON 只有 timestamp、version、accelerator、num_accelerators、loads 5 个键),永远拿到 -1;HTTP 非 PD 模式下它连在飞请求数也没人给它数,两边都是 0,选出来的是随机抽到的第 1 台。PD 模式下非流式请求两边都计数,这时它比的是在飞请求数。
- **「PD 模式下 D 也该用 cache_aware」。** 因为「都开总没错」。D 不做 prefill,D 侧树默认关着(11 章),网关给 D 建的那棵树只会把同前缀的请求往同一台 D 堆;D 用 power_of_two 或 round_robin。
- **「代码里的默认值就是命令行默认值」。** 结构体默认是 0.5 / 32 / 1.1 / 30 秒 / 10000,命令行默认是 0.3 / 64 / 1.5 / 120 秒 / 67108864,Python 启动器的驱逐间隔又是 60 秒。IGW 模式按 `labels.policy` 建出来的策略走结构体默认。查生效值看启动日志,不看文档表。
- **「多起几个网关副本树会同步」。** 因为有 `--enable-mesh`。mesh 同步的是实例状态、限流配置,树的插入只有发送端,接收端在基准上没有接线;每个副本各猜各的,官方自己估命中率掉 10%–20%。要粘就在网关前面按用户 id 做会话亲和。
- **「重试是安全的」。** 因为默认开着。408、429、5xx 都重试、每次换一台,流式请求发到一半断了也会重发,客户端可能收到两份开头。对不能重复的请求用 `--disable-retries`。
- **「网关的 `/health` 200 说明后面有实例」。** `/health` 和 `/liveness` 一样永远 200;要看有没有健康实例用 `/readiness`。
- **「Rust 前端就是网关」。** 01 章说过一次:`SGLANG_RUST_SERVER` 是嵌在单实例里的 HTTP 层,网关是实例之间的独立进程,两者叠着用。`experimental/sgl-router` 是第 3 样东西:一个只服务 1 个模型的精简路由器,靠订阅实例发布的 KV 事件建索引而不是靠猜,和 llm-d 的路由器走同一条路;llm-d 是 Kubernetes 上的集群层,SGLang 给它发 KV 事件。跨实例这一层在基石解读里没有对应章,所以本页只和实例内的分发比,不和 vLLM 比。

# SGLang 10|大规模 MoE:EP、DeepEP、DP attention、TBO 与 EPLB

专家并行怎么切、路由为什么天生不均衡、all-to-all 为什么让最慢的卡决定整轮、微批重叠的原理、超大 MoE 为什么要配数据并行,vLLM 解读的「分布式并行」一章已经讲透(「EP:切专家,负载天生不均衡」「EP 的重叠:微批与辅助流」「DP:单个副本打满之后」三部分)。这一页只说 SGLang 遇到的问题、它的解法、代价、以及和 vLLM 不一样的地方。02 章讲过 DP attention 下调度保守度自动乘 0.3 与 prefill delayer,06 章讲了注意力后端,07 章讲过 DP attention 下 decode 图各卡补到全局最大批,本章只回指不重讲。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

DeepSeek V3 每层 256 个路由专家,专家权重是模型的大头,一张卡放不下,只能按专家整个分卡。分完之后一个 token 要用的 8 个专家散在别的卡上,通信是形状每步都变的 all-to-all,不是 TP 那种形状固定的 all-reduce;路由是动态的,某个专家热门,持有它的卡算得慢,其他卡在集合通信的同步点上等它,整轮时间由最慢的卡决定;dispatch 和 combine 在网络上跑的时候,SM 空转,每层 2 次。

MLA 把 KV 压成 1 个头,TP 切不开它:TP=8 时 8 张卡各存一份完整 KV,显存被重复占 8 次,并发上不去。高并发 decode 的 TPOT 和 `gen throughput` 先坏,长 prompt 的 TTFT 其次,路由分布漂移时会突然冒出一张慢卡把整机拖住。

## 二、解法:注意力分数据、专家分卡、通信藏进计算、热门专家复制

**EP:整个专家分卡。** 最天然的直觉是 8 张卡各拿 1/8 的专家。SGLang 的 `--ep-size` 是独立的数,可以小于 tp:tp=8、ep=2 时每张卡拿全部专家的 1/2,再把每个专家的中间维切 4 份(MoE 的 TP 等于 tp 除以 ep 再除以 moe_dp)。默认不走 all-to-all:每张卡对同一批 token 跑自己那份专家,不在本卡的专家把 id 置 -1 跳过,算完 all-reduce 求和,通信形状和纯 TP 一样,所以小规模下不吃亏。一旦选了任何一个 all-to-all 后端,ep 被强制改成等于 tp,日志打一行。

**DeepEP 与 all-to-all 后端。** 命令行有 11 个名字:`deepep`、`deepep_v2`、`mooncake`、`nixl`、`mori`、`pplx`、`flashinfer`、`flashinfer_megamoe`、`megamoe`、`ascend_fuseep`、`ascend_tp`(NPU 上被折回 none),默认 `none`。DeepEP 有两种模式:normal 给 prefill,吞吐高、SM 数可配、不能录 CUDA graph;low_latency 给 decode,每张卡每层派出的 token 有固定上限、可以录图。`--deepep-mode auto` 按这一步有没有 extend 自动切;pplx 只有 low_latency,mori 默认 normal,deepep_v2 是另一套缓冲(direct 单机、hybrid 跨机),和 normal/low_latency 无关。专家计算在 low_latency 下走 DeepGEMM 的 masked 分组 GEMM,normal 下走 contiguous 分组 GEMM。

**DP attention:注意力分数据,FFN 照旧。** 8 张卡各自接一批请求、各存各的 KV,MLA 的 KV 不再重复;到了 MoE 层再把各卡的 token 拼到一起走 TP 或 EP。为什么不吃亏:每一步调度器只 all-gather 8 个整数(token 数、算 logprob 的 token 数、能否走 decode 图、有没有 extend、能否 TBO、前向模式、能否走 prefill 图、prefix 上限),没有请求的卡造一个 IDLE 空批陪跑;MoE 层的输入按两种方式拼:补到最大再 all-gather,或补到总和再 all-reduce,prefill 用后者,decode 按通信量小的选。`--dp-size` 要显式给,tp 要能被 dp 整除,dp 小于 tp 时剩下的 tp/dp 是注意力内部的 TP;dp=1 时开关静默失效。

**TBO:一批切两半,通信和计算错开。** 半批 A 算注意力时半批 B 在 dispatch。decode 按请求数对半切,prefill 按 token 和找平衡点,平衡度差过 0.48 就把一条序列劈成两块。生效要同时满足:开关开;decoder 层是 DeepSeek V2/V3、Qwen3 MoE、MiMo V2、DeepSeek V4 这 4 种之一;DP attention 下各卡前向模式相同且都投票能切;prefill 不在 low_latency 模式。decode 图只录 TBO 版本,桶对齐到 2 的倍数;prefill 下 DeepGEMM 只用总 SM 减去 DeepEP 占的那些。没有最小 token 阈值,小批也切。SBO 是单批内把共享专家和 combine 错开,SM90 上不可用。

**EPLB:记录、重排、搬权重。** 记录器数每层每个逻辑专家被选了多少次,每 1000 步按 DeepSeek 的 EPLB 算法算一张新表:物理槽 = 逻辑专家数 + 冗余数,热门专家复制到多张卡;然后按新旧两张表用 P2P 把权重从旧位置搬到新位置。搬运在前向循环里做,一次搬全部层,或者按 `--eplb-rebalance-layers-per-chunk` 分几步搬;搬的时候不出 token。token 落到哪个副本:static 每 rank 一张固定表,dynamic 按行号轮转,lp 用求解器给概率。弹性 EP 是它的延伸:mooncake 或 nixl 后端下某张卡挂了,余下的卡按记录重排,缺的专家从磁盘或 DRAM 备份补回来继续服务;`/scale_elastic_ep` 运行时加卡。

**DeepGEMM:两套 kernel,启动时全部预编译。** contiguous 给 prefill,masked 给 decode;JIT 按 M 维预编译,默认把 1 到 16384(块大于 8192 时到 2 倍块,封顶 131072)每个 M 都编一遍,只有每台机器的第一个 rank 编,产物在 `~/.cache/sglang/deep_gemm`。

## 三、代价

- **ep 被强制等于 tp。** 选了 all-to-all 后端就没有混合切,想要 ep 小于 tp 只能用 none 后端的 all-reduce 路径。
- **normal 模式没有图。** `--deepep-mode normal` 把 decode 和 prefill 两个阶段的 CUDA graph 都关掉;low_latency 每 rank 每层派 128 个 token 的缓冲上限,调高显存跟着涨,硬上限 1024。
- **DP attention 每步多一次同步。** 8 个整数的 all-gather 走 CPU 组或 NCCL,IDLE 批照样跑 MoE;chunked prefill 被除以 dp,保守度乘 0.3;lm head 要么每卡全量权重,要么全词表 all-gather,decode 节点默认改用 all-to-all。
- **TBO 的 kernel 变小。** 小批下每个 kernel 打不满 GPU;decode 图的桶翻倍;和可断 prefill 图、统一内存、DSA 的 index 共享互斥。
- **EPLB 重排时停服。** 一次全层重排的时间在日志 `rebalance end time=` 里;冗余专家吃显存;记录器的统计常驻。迭代数必须不小于记录缓冲。
- **DeepGEMM 冷启动 10–20 分钟。** 没有跑过 `sglang.compile_deep_gemm` 的机器第一次起服务要等。

## 四、和 vLLM 不一样的七处

1. **EP 是独立的数,不是压平的 TP。** vLLM 一个布尔量,专家份数等于 TP 乘 DP。SGLang `--ep-size` 自己给,可以小于 tp 做混合切;只有选了 all-to-all 后端才被改成等于 tp。
2. **走不走 all-to-all 由后端名决定,不由并行度推。** vLLM 开了 EP 但 DP、CP、SP 都是 1 时仍走 all-gather 加 reduce-scatter。SGLang 默认 `none` 是每卡跑全部 token 再 all-reduce,`--moe-a2a-backend` 指定了才走 dispatch 与 combine。
3. **后端是 11 个名字,DeepEP 两种模式按批阶段自动切。** vLLM 是几个 all-to-all kernel 加变长后端。SGLang 一个 `--deepep-mode auto` 就在 prefill 的 normal 和 decode 的 low_latency 之间换,用户不用管。
4. **TBO 没有最小 token 阈值,但有一堆生效条件。** vLLM 的双批重叠 decode 低于 32、prefill 低于 512 个 token 不切,重叠方式是四值枚举。SGLang 小批也切,只有 0.48 的两块切分阈值;但只对 4 种 decoder 层生效,DP attention 下还要各卡投票一致。
5. **DP 协调每步同步 8 个整数,不是每 16 步 1 个布尔。** vLLM 副本之间只同步「还有没有请求」。SGLang 每步都 all-gather 各卡的 token 数和前向模式,空卡造 IDLE 批,因为 MoE 层的输入要拼起来。
6. **DP attention 不要求 MoE,也不要求 EP。** vLLM 的数据并行断言模型必须是 MoE。SGLang 的 DP attention 是给 MLA 省 KV 的,Qwen 这类标准注意力也能开,不开 EP 也能用。
7. **EPLB 是运行时的,还有弹性 EP。** vLLM 的 EPLB 只要求先开 EP。SGLang 除了要求 ep 大于 1,还有记录、导出、`--init-expert-location` 离线加载三个入口,重排在前向循环里分块做,卡挂了能重排接着服务。

## 五、调参与观测

**参数在哪调。** 并行形状(`--tp-size`、`--ep-size`、`--dp-size`、`--enable-dp-attention`)、后端(`--moe-a2a-backend`、`--moe-runner-backend`、`--deepep-mode`)、TBO 和 EPLB 的开关与周期全是启动参数,改了要重启。DeepEP 的缓冲大小与 DeepGEMM 的编译策略是环境变量。运行时只有五个接口:`/start_expert_distribution_record`、`/stop_expert_distribution_record`、`/dump_expert_distribution_record` 管专家分布记录;`/scale_elastic_ep` 与 `/is_scaling_elastic_ep` 管弹性 EP 加卡。`/set_internal_state` 改不了这一页的任何参数。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--ep-size` · 专家切几份 | 启动 | 1 | 大于 1 且后端 none:每卡少放专家,通信仍是 all-reduce;选了 all-to-all 后端被改成 tp | 日志 `The expert parallel size is adjusted from … to the tensor parallel size` |
| `--moe-a2a-backend` · 用哪种 all-to-all | 启动 | none | deepep:走 dispatch 与 combine,ep=tp;flashinfer、flashinfer_megamoe、pplx 要求开 DP attention 且 dp=tp;deepep_v2 强制 deep_gemm 且不能开 TBO | 启动报错信息;`/get_server_info` |
| `--moe-runner-backend` · 专家 GEMM 用哪个 kernel | 启动 | auto | 19 个选项;deepep 低时延模式下必须有 DeepGEMM;cutlass 的 FP8 只能 ep=1 | 启动断言 |
| `--deepep-mode` · normal / low_latency / auto | 启动 | auto | normal:两个阶段的 CUDA graph 都关,TPOT 明显变长;low_latency:prefill 也走 masked 路径,受每 rank token 上限约束 | 日志 `Cuda graph is disabled because deepep_mode=` |
| `--deepep-config` · DeepEP normal 模式的调优 JSON | 启动 | 无 | 给 normal_dispatch 与 normal_combine 各一组配置,两组 num_sms 必须相同;SM 数少于总 SM 数的 50% 时打警告 | 日志 `Only use … SMs for DeepEP communication` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` · low_latency 每 rank 每层缓冲多少 token | 环境变量 | 128 | 每 rank 一步派出的 token 数(decode 批乘草稿数)必须在它之内;调高缓冲显存涨,硬上限 1024 | 就绪前 `available_gpu_mem` |
| `--enable-dp-attention` 加 `--dp-size` · 注意力按数据并行 | 启动 | 关;1 | 开:KV 不再重复,`max_total_num_tokens` 每卡翻到 tp/dp 倍量级;每步多一次 8 整数同步;块大小除以 dp、保守度乘 0.3(02 章);dp=1 时静默不生效 | 日志 `DP attention is enabled. chunked prefill size is adjusted`;每个调度进程的日志前缀 `DP0 TP0` |
| `--enable-dp-lm-head` · lm head 在注意力 TP 组内切词表 | 启动 | 关 | 开:省掉跨 DP 的 all-gather;每卡批小时 GEMM 效率低 | `/get_server_info` |
| `--enable-tp-lm-head-all-to-all` · lm head 用 all-to-all 代替 all-gather | 启动 | decode 节点且 dp=tp 时开 | 关:退回全词表 all-gather;和 dp-lm-head 互斥 | 同上 |
| `--moe-dense-tp-size` · 稠密 MLP 层的 TP | 启动 | 无 | 大 TP 下稠密层维度太小报 GEMM 错时设 1;它改变 MoE 输入是 all-gather 还是 all-reduce | 启动报错 |
| `--enable-two-batch-overlap` · 双批重叠 | 启动 | 关 | 开:decode 与 prefill 各按半批错开,吞吐上去;decode 图只录 TBO 版、桶对齐到 2;可断 prefill 图被关;不满足 4 种层报未实现 | `Prefill batch` 与 `Decode batch` 行的吞吐;`SGLANG_TBO_DEBUG` |
| `--tbo-token-distribution-threshold` · 两半 token 差多少改两块切 | 启动 | 0.48 | 调到 0 关掉两块切,只按序列边界切;必须不大于 0.5 | 无 |
| `--enable-single-batch-overlap` · 单批内共享专家与 combine 错开 | 启动 | 关 | 开:decode 每层省一段;SM90 上直接报错 | 启动报错 |
| `--enable-eplb` · 专家负载均衡 | 启动 | 关 | 开:自动打开 stat 记录器,每 1000 步重排;要求 ep 大于 1;分配算法自动选 static(有 a2a)或 dynamic(none) | 日志 `[EPLBManager] rebalance start` 与 `end time=` |
| `--eplb-rebalance-num-iterations` · 多少步重排一次 | 启动 | 1000 | 调小:跟负载变化更紧,停服更频繁;必须不小于记录缓冲 | 同上出现的频率 |
| `--eplb-rebalance-layers-per-chunk` · 一次搬几层 | 启动 | 无(全搬) | 设 4 或 8:每次前向只搬几层,单次停顿短,整轮重排拖长;设了就不打 `time=` | `time=` 是否出现 |
| `--eplb-min-rebalancing-utilization-threshold` · 均衡度高于多少就不重排 | 启动 | 1.0 | 设 0.8:窗口均衡度已高于 0.8 时跳过;1.0 等于每次都重排且不记历史 | 日志 `Skipped ep rebalancing` |
| `--eplb-algorithm` · 重排算法 | 启动 | auto | auto:专家分组数能被节点数整除选 deepseek_hierarchical,否则 deepseek;弹性 EP 强制 elasticity_aware | `/get_server_info` |
| `--ep-num-redundant-experts` · 多放几个冗余专家槽 | 启动 | 0 | 设 32:热门专家有地方复制,每卡多放 32/ep 个专家的权重;逻辑加冗余必须被 ep 整除 | 就绪前 `available_gpu_mem` |
| `--ep-dispatch-algorithm` · token 落到哪个副本 | 启动 | 无 | static 固定表、dynamic 按行号轮转、lp 求解器;none 后端下 static 与 lp 报错 | 启动报错 |
| `--init-expert-location` · 启动就按记录摆专家 | 启动 | trivial | 给 .pt 或 .json:含 physical_to_logical_map 直接用,含 logical_count 先算一次 EPLB | 日志 `init_expert_location from init_by_…` |
| `--expert-distribution-recorder-mode` · 记录粒度 | 启动 | 无;EPLB 下 stat | stat 累计次数;stat_approx 只在 normal 模式下用 dispatch 计数近似;per_pass、per_token 留原始明细,内存大 | dump 出的 .pt 大小 |
| `--expert-distribution-recorder-buffer-size` · 记录环形缓冲多少步 | 启动 | 等于重排迭代数;无 EPLB 时 1000 | -1 无限;调大内存涨 | 无 |
| `--expert-balancedness-report-mode` · 均衡度往哪报 | 启动 | off | server_log 打日志;prometheus 出 `sglang:eplb_balancedness`;both 都要 | `/metrics` |
| `--elastic-ep-backend` · 弹性 EP 用哪个通信库 | 启动 | 无 | mooncake 或 nixl:卡挂了能重排接着服务;运行时加卡还要 `--max-ep-size`、关两个阶段的图、dp=ep=tp、nixl、round_robin | `/is_scaling_elastic_ep` |
| `--enable-elastic-expert-backup` · 专家权重在 DRAM 留一份 | 启动 | 关 | 开:恢复时从内存补专家,不读盘;和运行时加卡互斥 | 日志 `[Elastic EP]` |
| `--enable-waterfill` · 共享专家当第 9 个路由专家派到最闲的卡 | 启动 | 关 | 开:强制 deepep 后端和共享专家融合;只有 DeepSeek V3/R1 且 ep 不小于 2 | 日志 `Waterfill is enabled` |
| `SGLANG_ENABLE_JIT_DEEPGEMM` · 用不用 DeepGEMM | 环境变量 | 开 | 关:退回 triton 等 runner;SM 低于 90 自动关 | 启动日志 `DeepGEMM` 字样 |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` 与 `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` · 预编译全部 M 还是抽样 | 环境变量 | 开;关 | 快速预热:1 到 1024 全编,再往上按 2、4、8、16 步长抽样,约 3072 个 kernel;关预编译:第一次遇到某个 M 才编 | 启动日志 `DeepGEMM warmup` 进度 |
| `SGLANG_DG_CACHE_DIR` · 编译产物放哪 | 环境变量 | `~/.cache/sglang/deep_gemm` | 指到持久盘,多次启动不重编 | 第二次启动的耗时 |
| `SGLANG_ENABLE_METRICS_DP_ATTENTION` · 各卡协作指标 | 环境变量 | 关 | 开:`/metrics` 多出按「几张卡在 prefill」分标签的 token 数与前向秒数 | `sglang:dp_cooperation_realtime_tokens_total` |
| `SGLANG_LOG_EXPERT_LOCATION_METADATA` · 重排前后打专家表 | 环境变量 | 关 | 开:rank 0 打 before、target、diff、after 四段 | 日志 `[EPLBManager] rebalance layout` |

**怎么看。** DP attention 下每个调度进程各打自己的日志,前缀是 `DP0 TP0 EP0` 这种,对着比各卡 `Decode batch` 行的 `#running-req` 和 `token usage`,差得多就是路由网关或前置 DP 控制器分得不均。专家分布:`/start_expert_distribution_record` 开始记,`/dump_expert_distribution_record` 把 rank 0 累计的 logical_count 写成 `/tmp/expert_distribution_recorder_<时间戳>.pt`(目录由 `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` 定),这份文件可以直接喂给 `--init-expert-location`。均衡度是每层「各卡平均 token 数除以最多那张卡」,1.0 是完美,`--expert-balancedness-report-mode` 决定它进日志还是 `/metrics`。DeepEP 的 SM 数和 DeepGEMM 的编译进度都在启动日志里。

| 症状 | 先查 | 然后 |
|---|---|---|
| 启动说 ep 被改成 tp | 是否选了 all-to-all 后端 | 想混合切就换回 none |
| 开了 deepep 后 decode 明显变慢 | `--deepep-mode` 是否 normal | 改 auto,让 decode 走 low_latency 和 CUDA graph |
| DeepEP 报 token 数超限 | 每 rank decode 批乘草稿数是否超过 128 | 调 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`,不超 1024 |
| 开了 `--enable-dp-attention` 却没效果 | `--dp-size` 是否还是 1 | 显式给 dp,通常等于 tp |
| DP attention 下某张卡一直空转 | 各卡 `Decode batch` 行的 `#running-req` | 上游分发不均;prefill 不同步开 delayer(02 章) |
| 长 prompt 的 TTFT 突然变长 | chunked prefill 是否被除以 dp | 显式指定块大小(02 章) |
| 开 TBO 报未实现 | 模型是不是 4 种 decoder 层之一 | 不是就关 |
| 开 TBO 后小批更慢 | 每步 token 数 | 没有阈值,低并发场景关掉 |
| 一张卡持续比别的卡慢 | `sglang:eplb_balancedness` 或 dump 出的分布 | 开 EPLB 加冗余专家;或用 dump 文件 `--init-expert-location` |
| EPLB 每次重排卡几秒 | `rebalance end time=` | 设 `--eplb-rebalance-layers-per-chunk` 分步搬 |
| EPLB 一直 `Skipped ep rebalancing` | 阈值是否设得太低 | 均衡度已经高于阈值,这是正常 |
| 启动等了 20 分钟没就绪 | 日志里 `DeepGEMM JIT Pre-Compile` | 先跑 `sglang.compile_deep_gemm`,或把缓存目录指到持久盘 |
| 弹性 EP 加卡失败 | `/is_scaling_elastic_ep` 与启动断言 | 两个阶段的图都关、dp=ep=tp、nixl、round_robin 缺一不可 |

## 六、常见误区

- **以为 `--ep-size 8 --tp-size 8` 就走了 all-to-all。** 文档和博客都把 EP 和 DeepEP 放一起讲。这个基准上后端默认 `none`,专家分了卡,通信仍是每卡跑全部 token 再 all-reduce。要走 dispatch 与 combine 必须显式给 `--moe-a2a-backend`。
- **以为 `--deepep-mode normal` 是「稳妥模式」。** 名字像默认档。它把两个阶段的 CUDA graph 都关掉,decode 每步多 1000 到 2000 次下发。生产用 auto,normal 只给 prefill 节点或排障。
- **以为 `--enable-dp-attention` 一个开关就够。** dp_size 默认 1,开关在 dp=1 时被静默改回关,没有任何报错。一定要同时给 `--dp-size`。
- **以为 DP attention 只有 DeepSeek 能用。** 它是给 MLA 省 KV 的,收益最大在那里,但 Qwen 这类标准注意力也能开;而且它不要求 EP。
- **看到某张卡 `#running-req` 很低就去调 EPLB。** DP attention 下各卡的请求是上游分的,EPLB 平衡的是专家,不是请求。先看分发。
- **以为 TBO 像 vLLM 那样有小批保护。** 没有阈值,决定切不切的只有前向模式一致和层类型。低并发下开了反而慢,自己关。
- **开了 TBO 之后 prefill 图没了,以为是 bug。** 可断 prefill 图的自动关闭规则里写着 TBO。decode 图还在,只是只录 TBO 版本。
- **以为 EPLB 是异步的。** 重排在前向循环里做,搬权重那段不出 token。一次全搬的时间在 `time=` 里,长了就分块。
- **以为均衡度阈值默认 1.0 是「只在完美均衡时才不重排」。** 1.0 的效果是每次都重排,而且不记历史。想跳过要显式设一个小于 1 的数。
- **把 `--ep-num-redundant-experts` 当成 EPLB 的开关。** 它只多放几个槽。不开 EPLB 也不给 `--init-expert-location`,冗余槽就是空转的显存。
- **以为 dump 出来的分布文件只能看。** 它带 logical_count,直接喂给 `--init-expert-location`,下次启动就按上次的负载摆专家,不用等 1000 步。
- **以为 DeepGEMM 编译只发生一次。** 缓存目录默认在用户目录,换机器、换容器就重来 10 到 20 分钟。持久化目录或提前跑 `sglang.compile_deep_gemm`。
- **以为弹性 EP 就是加卡。** 它先是容错:卡挂了重排接着服务;运行时加卡是另一套前置条件,少一条启动就断言。

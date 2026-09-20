# SGLang 10|大规模 MoE:EP、DeepEP、DP attention、TBO 与 EPLB

专家并行怎么切、路由为什么天生不均衡、all-to-all 为什么让最慢的卡决定整轮、微批重叠的原理、超大 MoE 为什么要配数据并行,vLLM 解读的「分布式并行」一章已经讲透,知识库另有 MoE 与 DeepEP 的专篇。这一页只说 SGLang 把这几件事做成了什么形状、代价落在哪、和 vLLM 差在哪。张量并行、流水线并行与上下文并行是 09 章的主场,零开销重叠调度是 02 章,CUDA graph 是 07 章,本章只回指不重讲。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

超大 MoE 上有三件事同时在等,各自先坏掉的指标不一样。

**等网络。** 专家权重是模型的大头,一张卡放不下,只能按专家整份分卡;分完之后一个 token 要用的那几个专家散在别的卡上,通信是形状每步都变的 all-to-all,不是张量并行那种形状固定的 all-reduce。派出去和收回来各一次,每层两段,这两段跑在网上的时候 SM 空转。

**等最慢的卡。** 路由是动态的,某个专家突然变热,持有它的那张卡算得久,别的卡卡在集合通信的同步点上陪等,整轮时间由最慢的一张决定。这件事随负载漂移——昨天均衡的部署,今天换一批流量就能冒出一张慢卡把整机按住。

**等显存。** MLA 把 KV 压成 1 个头,张量并行切不动它,8 张卡各存一份完整的 KV,同一份数据被重复占 8 次,能跑的并发被这份冗余卡死。

高并发 decode 的 TPOT 和生成吞吐先坏,长 prompt 的 TTFT 其次。

## 二、解法:五个各管一件事的开关,默认全关

SGLang 里没有一个叫「大规模 MoE」的模式。上面三种等各有对策,每个对策是一个独立开关,默认全部关着,互相之间还有一串强制改写和前置条件。**这是这一章最要紧的一句话**,第六节一半的误区都来自把其中两个开关当成了一个。

**专家整份分卡,但切几份是独立的数。** 最天然的直觉是 8 张卡各拿 1/8 的专家。SGLang 的专家份数不跟着张量并行度走,可以更小:张量并行 8、专家份数 2 时,每张卡拿全部专家的一半,再把每个专家的中间维切 4 份。

**默认不走 all-to-all。** 只分卡、不点名全交换后端时,每张卡对同一批 token 跑自己那份专家,不在本卡的专家编号置 -1 跳过,算完 all-reduce 求和——通信形状和纯张量并行一样,所以小规模下不吃亏。一旦点名了全交换后端,专家份数被强制改成等于张量并行度,日志打一行说明,混合切就没有了。

![同一层 MoE 的三种摆法,每一种都画成 4 张卡、8 个逻辑专家:第一种是默认路径,四张卡各自拿到同一批完整 token(四条紫条一样长),每张卡的 8 个槽里只有 2 个是绿的、其余 6 个是被置 -1 跳过的灰槽,末尾一条横贯全宽的 all-reduce 把四份部分结果加起来;第二种是点名全交换后端之后,每张卡的紫条只剩四分之一长,一条 dispatch 带和一组交叉的连线把 token 送到持有对应专家的卡上,灰槽消失、每张卡只剩 2 个绿槽,算完再由一条 combine 带送回原来那张卡;第三种是再加冗余槽与负载均衡,每张卡的槽从 2 个变成 3 个,热门的那几个专家用红色标出、在多张卡上各放一份](/opensource/sglang/10a-expert-placement.svg)

**全交换后端是一张 11 个名字的清单,两种批阶段各一套实现。** DeepEP 是其中的主线,它有两种模式:大吞吐的那种给 prefill,通信占几个 SM 可以配,但录不了 CUDA graph(07 章);低时延的那种给 decode,每张卡每层派出的 token 有固定上限,可以录图。默认档按这一步里有没有 prefill 自动在两者之间切,用户不用管。专家的分组 GEMM 跟着分两套:低时延那边是带掩码的布局,大吞吐那边是连续布局。

**注意力分数据,FFN 照旧。** 8 张卡各接各的请求、各存各的 KV,MLA 那份冗余就没了;到了 MoE 层再把各卡的 token 拼到一起过专家。为什么不吃亏:每步各卡只 all-gather 8 个整数——这一步的 token 数、要算对数概率的 token 数、能不能走 decode 图、这批里有没有 prefill、能不能双批重叠、前向模式是哪个、能不能走 prefill 图、前缀长度上限——没有请求的卡造一个空批陪跑。MoE 层的输入有两种拼法:补到最长的那份再 all-gather,或补到总和再 all-reduce;prefill 用后者,decode 按哪种通信量小选哪种。副本数必须显式给,张量并行度要能被它整除;只开开关不给副本数,开关被静默改回关。

**一批切两半,通信和计算错开。** 半批 A 算注意力的时候,半批 B 正在网上派 token。decode 按请求数对半切,prefill 按 token 数找平衡点,两半差得过大就把一条序列劈成两块。它没有最小 token 阈值,小批照切;但生效条件是一串:模型的 decoder 层必须是 4 种之一,开了注意力数据并行还要求各卡这一步的前向模式一致、且都投票说能切。另有一种单批内的重叠,把共享专家的计算和收回 token 的那段通信错开,SM90 上不可用。

**热门专家复制一份。** 记录器数每层每个逻辑专家被选了多少次,每 1000 步按 DeepSeek 的 EPLB 算法算一张新表:物理槽数等于逻辑专家数加冗余槽数,热门的逻辑专家在多张卡上各放一份;再按新旧两张表用点对点通信把权重从旧位置搬到新位置。搬运在前向循环里做,可以一次搬全部层,也可以每次前向只搬几层,搬的那段不出 token。token 落到哪个副本有三种挑法:每卡一张固定表、按行号轮转、或用求解器给概率。它的延伸是弹性专家并行——某张卡挂了,余下的卡按记录重排,缺的专家从内存或磁盘的备份补回来接着服务,也能在运行时加卡。

## 三、代价

- **点名全交换后端就没有混合切。** 专家份数被改成等于张量并行度;想要更小的份数,只能留在默认那条 all-reduce 路径上。
- **大吞吐模式把图全关掉。** 它同时关掉 decode 和 prefill 两个阶段的 CUDA graph(07 章),decode 每步多出 1000–2000 次 kernel 下发。低时延模式则给每张卡每层的派发缓冲一个固定上限,默认 128 个 token,硬上限 1024,调高之后显存跟着涨。
- **注意力分数据每步多一次同步。** 8 个整数的 all-gather 走 CPU 组或设备组;没请求的卡照样造空批跑一遍 MoE;分块 prefill 的块大小被除以副本数、调度保守度乘 0.3(02 章);输出词表那一层要么每卡全量权重,要么全词表 all-gather。
- **切两半会把 kernel 切小。** 小批下每半批都打不满 GPU;decode 图只录切两半的版本、桶对齐到 2 的倍数;和可断 prefill 图、统一内存、稀疏注意力的索引共享互斥。
- **重排时停服。** 一次全层重排要停多久,日志里有一行时间;冗余专家吃显存;记录器的统计常驻内存;重排周期不能小于记录缓冲。
- **DeepGEMM 冷启动 10–20 分钟。** 没预编译过的机器第一次起服务,要把每个可能的 M 都编一遍,只有每台机器上的第一个 rank 编。

## 四、和 vLLM 不一样的七处

![两边都必须做的 5 件事各占一行,左栏 vLLM、右栏 SGLang:第一行「决定每张卡放哪些专家」,两边都是 8 个格子,左边全灰表示份数由张量并行度乘副本数推出来、选不了,右边全绿并套着两个虚线组框,表示份数自己给、这里切成了 2 份;第二行「决定 token 怎么到专家那里」,左边只有 all-gather 加 reduce-scatter 一条路,右边有两条——上面一条是默认的「每卡跑全批 token 再 all-reduce」,下面一条是点名后端后的 dispatch 与 combine;第三行「把通信藏进计算」画成从左往右的时间条,紫段是算、黄段是通信,左边 token 少时是一整条算完再通信、token 多时才切成上下错开的两半批,右边两档都是错开的两半批;第四行「让各副本步调一致」,左边 16 个步的格子里只有最后一个是通信色,右边每一步都有 8 个通信色格子;第五行「把热门专家摊开」,左边是一个迁移入口,右边是记录、导出、启动摆位、故障重排四个入口。两边的参数名都在第五节](/opensource/sglang/10b-vllm-diff-five-rows.svg)

1. **专家份数是独立的数,不是压平的张量并行。** vLLM 那边是一个布尔开关,份数由张量并行度乘副本数推出来;SGLang 自己给一个数,可以小于张量并行度做混合切,只有点名了全交换后端才被改成相等。
2. **走不走 all-to-all 由后端名字决定,不由并行度推。** vLLM 开了专家并行、而副本数与上下文并行、序列并行都是 1 时,仍走 all-gather 加 reduce-scatter;SGLang 的默认是每卡跑全部 token 再 all-reduce,必须显式点名一个后端才走派出去与收回来。两边的坑同形,踩法不同。
3. **后端是一张 11 个名字的清单,模式还按批阶段自动切。** vLLM 是几个 all-to-all kernel 加一个变长后端;SGLang 的默认档在 prefill 的大吞吐模式和 decode 的低时延模式之间自己换,不用用户挑。
4. **双批重叠没有小批保护。** vLLM 的 decode 低于 32、prefill 低于 512 个 token 就不切,重叠方式是一个四值枚举;SGLang 小批照切,唯一的阈值是「要不要把一条序列劈成两块」,却另有一串模型类型与各卡投票的前置条件。
5. **副本之间每步同步 8 个整数,不是每 16 步一个布尔。** vLLM 的副本只需要知道「大家是不是都空了」;SGLang 每步都要 all-gather 各卡的 token 数和前向模式,因为 MoE 层的输入要拼起来,空卡也得造批陪跑。
6. **注意力的数据并行不要求 MoE,也不要求专家并行。** vLLM 的数据并行引擎直接断言模型必须是 MoE;SGLang 这一路是为 MLA 省 KV 做的,收益最大也在那里,但标准注意力的模型照样能开,不开专家并行也能用。
7. **专家负载均衡是运行时的,还带容错。** vLLM 只要求先开专家并行;SGLang 除了要求份数大于 1,还有记录、导出、启动时按上次的记录摆专家三个入口,重排在前向循环里可以分块做,卡挂了能重排接着服务。

## 五、调参与观测

**参数在哪调。** 并行形状、全交换与 GEMM 后端、双批重叠与负载均衡的开关和周期全是启动参数,改了要重启;DeepEP 的缓冲大小与 DeepGEMM 的编译策略是环境变量。运行时只有五个接口:`/start_expert_distribution_record`、`/stop_expert_distribution_record`、`/dump_expert_distribution_record` 管专家分布记录,`/scale_elastic_ep` 与 `/is_scaling_elastic_ep` 管弹性专家并行加卡。`/set_internal_state` 改不了这一页的任何参数。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--ep-size` · 专家切几份 | 启动 | 1 | 大于 1 且后端是 `none`:每卡少放专家,通信仍是 all-reduce;点名 all-to-all 后端后被改成等于 tp | 日志 `The expert parallel size is adjusted from … to the tensor parallel size` |
| `--moe-a2a-backend` · 用哪种 all-to-all | 启动 | `none` | `deepep`:走 dispatch 与 combine,ep 改成 tp;`flashinfer`、`flashinfer_megamoe`、`pplx` 要求开 DP attention 且 dp=tp;`deepep_v2` 强制 `deep_gemm` 且不能开 TBO;`ascend_tp` 会被折回 `none` | 启动日志与报错;`/server_info` |
| `--moe-runner-backend` · 专家 GEMM 用哪个 kernel | 启动 | `auto` | 19 个选项;DeepEP 低时延模式下必须有 DeepGEMM;`cutlass` 的 FP8 只能 ep=1 | 启动断言 |
| `--deepep-mode` · `normal` / `low_latency` / `auto` | 启动 | `auto` | `normal`:两个阶段的 CUDA graph 都关,TPOT 明显变长;`low_latency`:prefill 也走掩码路径,受每 rank token 上限约束 | 日志 `Cuda graph is disabled because deepep_mode=` |
| `--deepep-config` · DeepEP `normal` 模式的调优 JSON | 启动 | 无 | 给 dispatch 与 combine 各一组配置,两组的 SM 数必须相同;SM 数少于总数的 50% 且没开 TBO 时打警告 | 日志 `Only use … SMs for DeepEP communication` |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` · 低时延模式每 rank 每层缓冲多少 token | 环境变量 | 128 | 每 rank 一步派出的 token 数(decode 批乘草稿数)必须不超过它;调高缓冲显存涨,硬上限 1024 | 就绪前 `available_gpu_mem` |
| `--enable-dp-attention` 加 `--dp-size` · 注意力按数据并行 | 启动 | 关;1 | 开:KV 不再重复,`max_total_num_tokens` 每卡涨到 tp/dp 倍量级;每步多一次 8 整数同步;块大小除以 dp、保守度乘 0.3(02 章);dp=1 时静默不生效 | 日志 `DP attention is enabled. chunked prefill size is adjusted`;调度进程日志前缀 `DP0 TP0` |
| `--enable-dp-lm-head` · 输出词表层在注意力 TP 组内切 | 启动 | 关 | 开:省掉跨 DP 的 all-gather;每卡批小时 GEMM 效率低 | `/server_info` |
| `--enable-tp-lm-head-all-to-all` · 输出词表层用 all-to-all 代替 all-gather | 启动 | 纯 decode 节点且 dp=tp>1、注意力 CP 为 1 时开,其余关 | 关:退回全词表 all-gather;和 `--enable-dp-lm-head` 互斥;补齐行数不等的批自动落回 all-gather | 同上 |
| `--moe-dense-tp-size` · 稠密 MLP 层的 TP | 启动 | 无 | 大 TP 下稠密层维度太小报 GEMM 错时设 1;它同时改变 MoE 输入走 all-gather 还是 all-reduce | 启动报错 |
| `--enable-two-batch-overlap` · 双批重叠 | 启动 | 关 | 开:decode 与 prefill 各按半批错开,吞吐上去;decode 图只录 TBO 版、桶对齐到 2;可断 prefill 图被关;层类型不在 4 种之内报未实现 | `Prefill batch` 与 `Decode batch` 行的吞吐;`SGLANG_TBO_DEBUG` |
| `--tbo-token-distribution-threshold` · 两半 token 差多少就改两块切 | 启动 | 0.48 | 设 0 关掉两块切,只按序列边界切;必须不大于 0.5 | 无 |
| `--enable-single-batch-overlap` · 单批内共享专家与收回 token 错开 | 启动 | 关 | 开:decode 每层省一段;SM90 上直接报错 | 启动报错 |
| `--enable-eplb` · 专家负载均衡 | 启动 | 关 | 开:自动打开 `stat` 记录器,每 1000 步重排;要求 ep 大于 1;落位算法自动选 `static`(有 all-to-all 后端)或 `dynamic`(`none`) | 日志 `[EPLBManager] rebalance start` 与 `end time=` |
| `--eplb-rebalance-num-iterations` · 多少步重排一次 | 启动 | 1000 | 调小:跟负载变化更紧,停服更频繁;必须不小于记录缓冲,否则启动断言 | 上面那两行出现的频率 |
| `--eplb-rebalance-layers-per-chunk` · 一次搬几层 | 启动 | 无(一次全搬) | 设 4 或 8:每次前向只搬几层,单次停顿短,整轮重排拖长;设了就不打 `time=` | `time=` 是否出现 |
| `--eplb-min-rebalancing-utilization-threshold` · 均衡度高于多少就跳过重排 | 启动 | 1.0 | 设 0.8:窗口均衡度已高于 0.8 时跳过;1.0 的实际效果是每次都重排且不记历史 | 日志 `Skipped ep rebalancing` |
| `--eplb-algorithm` · 重排算法 | 启动 | `auto` | `auto`:专家分组数能被节点数整除选层级版,否则选普通版;弹性专家并行下强制用容错版 | `/server_info` |
| `--ep-num-redundant-experts` · 多放几个冗余专家槽 | 启动 | 0 | 设 32:热门专家有地方复制,每卡多放 32/ep 个专家的权重;逻辑数加冗余数必须被 ep 整除 | 就绪前 `available_gpu_mem` |
| `--ep-dispatch-algorithm` · token 落到哪个副本 | 启动 | 无(开 EPLB 时自动选) | `static` 每 rank 一张固定表、`dynamic` 按行号轮转、`lp` 用求解器;`none` 后端下 `static` 与 `lp` 直接报错 | 启动报错 |
| `--init-expert-location` · 启动就按记录摆专家 | 启动 | `trivial` | 给 `.pt` 或 `.json`:含物理到逻辑的映射就直接用,只含计数就先算一次 EPLB | 日志 `init_expert_location from init_by_…` |
| `--expert-distribution-recorder-mode` · 记录粒度 | 启动 | 无;开 EPLB 或开均衡度上报时自动 `stat` | `stat` 累计次数;`stat_approx` 只在 `normal` 模式下用 dispatch 计数近似;`per_pass`、`per_token` 留原始明细,内存大 | dump 出的 `.pt` 大小 |
| `--expert-distribution-recorder-buffer-size` · 记录环形缓冲多少步 | 启动 | 等于重排迭代数;没开 EPLB 时 1000 | 设 -1 为无限;调大内存涨 | 无 |
| `--expert-balancedness-report-mode` · 均衡度往哪报 | 启动 | `off` | `server_log` 打日志;`prometheus` 出 `sglang:eplb_balancedness`;`both` 都要 | `/metrics` |
| `--elastic-ep-backend` · 弹性专家并行用哪个通信库 | 启动 | 无 | `mooncake` 或 `nixl`:卡挂了能重排接着服务;运行时加卡还要 `--max-ep-size`、关掉两个阶段的图、dp=ep=tp、`nixl`、轮询分发 | `/is_scaling_elastic_ep` |
| `--enable-elastic-expert-backup` · 专家权重在内存留一份 | 启动 | 关 | 开:恢复时从内存补专家,不读盘;和运行时加卡互斥 | 日志 `[Elastic EP]` |
| `--enable-waterfill` · 共享专家当成额外一个路由专家派到最闲的卡 | 启动 | 关 | 开:后端不是 `deepep` 或 `megamoe` 时被改成 `deepep`,并强制打开共享专家融合;只在 DeepSeek V3/R1 且 ep 不小于 2 上支持 | 日志 `Waterfill is enabled with moe_a2a_backend=` |
| `SGLANG_ENABLE_JIT_DEEPGEMM` · 用不用 DeepGEMM | 环境变量 | 开 | 关:退回 triton 等 runner;SM 低于 90 或没装库时自动关 | 启动日志里的 `DeepGEMM` 字样 |
| `SGLANG_JIT_DEEPGEMM_PRECOMPILE` 与 `SGLANG_JIT_DEEPGEMM_FAST_WARMUP` · 预编译全部 M 还是抽样 | 环境变量 | 开;关 | 开快速预热:1–1024 全编,再往上按 2、4、8、16 步长抽样,约 3072 个 kernel;关预编译:第一次遇到某个 M 才编 | 启动日志的预编译进度 |
| `SGLANG_DG_CACHE_DIR` · 编译产物放哪 | 环境变量 | `~/.cache/sglang/deep_gemm` | 指到持久盘,换容器重启不重编 | 第二次启动的耗时 |
| `SGLANG_ENABLE_METRICS_DP_ATTENTION` · 各卡协作指标 | 环境变量 | 关 | 开:`/metrics` 多出按「几张卡在 prefill」分标签的 token 数与前向秒数 | `sglang:dp_cooperation_realtime_tokens_total` |
| `SGLANG_LOG_EXPERT_LOCATION_METADATA` · 重排前后打专家表 | 环境变量 | 关 | 开:rank 0 打 before、target、diff、after 四段 | 日志 `[EPLBManager] rebalance layout` |

**怎么看。** 开了注意力数据并行之后,每个调度进程各打自己的日志,前缀是 `DP0 TP0 EP0` 这种;把各卡 `Decode batch` 行的 `#running-req` 和 `token usage` 对着比,差得多就是上游路由网关或 DP 控制器分得不均,不是专家不均。专家分布用三个接口取:`/start_expert_distribution_record` 开始记,`/dump_expert_distribution_record` 由 rank 0 把累计计数写成 `/tmp/expert_distribution_recorder_<时间戳>.pt`(目录由 `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` 定),这份文件可以直接喂给 `--init-expert-location`。均衡度是每层「各卡平均 token 数除以最多那张卡」,1.0 是完美,进日志还是进 `/metrics` 由 `--expert-balancedness-report-mode` 决定,指标只在专家并行的 rank 0 上注册。DeepEP 占了几个 SM、DeepGEMM 编到第几个 kernel,都在启动日志里。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。组合与推荐值是按语义推的起点,不是实测最优。

| 什么场景 | 在默认起法上加什么 | 拿什么换什么 |
|---|---|---|
| 单机 8 卡跑中等规模 MoE,并发 50–500 | `--tp-size 8 --ep-size 2` | 专家分卡省显存,通信仍是 all-reduce,不引入 all-to-all 的抖动和 DeepEP 的部署负担;代价是每卡仍要算全部 token 的路由 |
| 多机大规模 MoE 在线服务,decode 吞吐优先 | `--tp-size 16 --enable-dp-attention --dp-size 16 --moe-a2a-backend deepep --deepep-mode auto --enable-two-batch-overlap --enable-eplb --ep-num-redundant-experts 32` | 用一整套开关换 decode 吞吐:KV 不再重复、通信藏进计算、热门专家摊开;代价是冗余专家的显存、每 1000 步一次停服重排、启动多一段 DeepGEMM 编译,而且 ep 被钉死等于 tp |
| 纯 prefill 节点或排障 | `--moe-a2a-backend deepep --deepep-mode normal --deepep-config <调优 JSON>` | 用大吞吐模式换 prefill 的 all-to-all 带宽;代价是两个阶段的 CUDA graph 全关,这台机器不适合再承担 decode |

| 症状 | 先查 | 然后 |
|---|---|---|
| 启动说 ep 被改成了 tp | 是不是点名了 all-to-all 后端 | 想混合切就换回 `none` |
| 开了 DeepEP 之后 decode 明显变慢 | `--deepep-mode` 是不是 `normal` | 改 `auto`,让 decode 走低时延模式和 CUDA graph |
| DeepEP 报 token 数超限 | 每 rank 的 decode 批乘草稿数是否超过 128 | 调 `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`,不超过 1024 |
| 开了 `--enable-dp-attention` 却没效果 | `--dp-size` 是不是还是 1 | 显式给 dp,通常等于 tp |
| 某张卡一直空转 | 各卡 `Decode batch` 行的 `#running-req` | 上游分发不均;prefill 不同步的话开 delayer(02 章) |
| 长 prompt 的 TTFT 突然变长 | 分块 prefill 的块大小是不是被除以了 dp | 显式指定块大小(02 章) |
| 开 TBO 报未实现 | 模型的 decoder 层是不是那 4 种之一 | 不是就关掉 |
| 开 TBO 之后小批更慢 | 每步的 token 数 | 没有小批保护,低并发场景自己关 |
| 一张卡持续比别的卡慢 | `sglang:eplb_balancedness` 或 dump 出的分布 | 开 EPLB 加冗余专家;或拿 dump 文件喂 `--init-expert-location` |
| 每次重排卡几秒 | 日志 `rebalance end time=` | 设 `--eplb-rebalance-layers-per-chunk` 分步搬 |
| 一直 `Skipped ep rebalancing` | 阈值是不是设低了 | 均衡度已经高于阈值,这是正常的 |
| 启动等了 20 分钟没就绪 | 启动日志里的 DeepGEMM 预编译进度 | 先跑 `sglang.compile_deep_gemm`,或把 `SGLANG_DG_CACHE_DIR` 指到持久盘 |
| 弹性专家并行加卡失败 | `/is_scaling_elastic_ep` 与启动断言 | 两个阶段的图都关、dp=ep=tp、`nixl`、轮询分发,缺一不可 |

## 六、常见误区

- **以为 `--ep-size 8 --tp-size 8` 就走了 all-to-all。** 文档和博客总把专家并行和 DeepEP 放在一起讲,看上去像一回事。这个基准上后端默认 `none`:专家确实分了卡,通信仍是每卡跑全部 token 再 all-reduce。要走 dispatch 与 combine,必须显式给 `--moe-a2a-backend`。
- **以为 `--deepep-mode normal` 是「稳妥模式」。** 名字像个默认档,实际上它把 decode 和 prefill 两个阶段的 CUDA graph 都关掉,decode 每步多 1000–2000 次下发。生产用 `auto`,`normal` 只留给纯 prefill 节点或排障。
- **以为 `--enable-dp-attention` 一个开关就够。** `--dp-size` 默认 1,而开关在 dp=1 时被静默改回关,不报错也不警告。一定要同时给 `--dp-size`。
- **以为注意力数据并行只有 DeepSeek 能用。** 它是为 MLA 省 KV 的,收益最大在那里,但 Qwen 这类标准注意力也能开;而且它不要求专家并行。
- **看到某张卡 `#running-req` 很低就去调 EPLB。** 开了注意力数据并行之后,各卡的请求是上游分的,EPLB 平衡的是专家不是请求。先看分发。
- **以为 TBO 像 vLLM 那样有小批保护。** 没有阈值,决定切不切的只有前向模式一致和层类型。低并发下开了反而慢,自己关。
- **开了 TBO 之后 prefill 图没了,以为是 bug。** 可断 prefill 图的自动关闭规则里就写着 TBO。decode 图还在,只是只录切两半的那一版。
- **以为 EPLB 是异步的。** 重排在前向循环里做,搬权重那段不出 token。一次全搬要多久看 `time=`,长了就分块搬。
- **以为均衡度阈值默认 1.0 是「只在完美均衡时才不重排」。** 1.0 的实际效果是每次都重排,而且不记历史。想跳过就得显式设一个小于 1 的数。
- **把 `--ep-num-redundant-experts` 当成 EPLB 的开关。** 它只是多开几个槽。不开 EPLB 也不给 `--init-expert-location`,这些槽就是空转的显存。
- **以为 dump 出来的分布文件只能拿来看。** 它带着逻辑专家计数,直接喂给 `--init-expert-location`,下次启动就按上次的负载摆专家,不用再等 1000 步。
- **以为 DeepGEMM 只编译一次。** 缓存目录默认在用户目录下,换机器、换容器就重来 10–20 分钟。把 `SGLANG_DG_CACHE_DIR` 指到持久盘,或提前跑 `sglang.compile_deep_gemm`。
- **以为弹性专家并行就是加卡。** 它首先是容错:卡挂了重排接着服务。运行时加卡是另一套前置条件,少一条启动就断言。
- **以为 `ascend_tp` 是一个 all-to-all 后端。** 它在解析时被直接折回 `none`,选不选一个样;NPU 上的 `none` 同样保持 `none`。

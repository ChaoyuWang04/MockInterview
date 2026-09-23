# vLLM 08|torch.compile 与 CUDA graph

decode 一步前向在 GPU 上只算一小会儿,CPU 逐个下发 kernel 的时间却不跟着缩;怎么在启动时把前向编译、融合、按形状录成图,让每一步只剩一次重放,又不让变长的注意力把图卡死。

说法都在源码基准 `94f4170df3`(tag `v0.30.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

PyTorch 按层执行时,前向里的每个算子都是一次独立的 kernel 下发:归一化、QKV 投影、位置编码、写 KV、注意力、输出投影、残差加、MLP 的投影与激活,每一个都要先过一遍 Python 调用和算子分发,再交给驱动。prefill 的批动辄上千个 token,每个 kernel 在 GPU 上跑得够久,下发藏在计算后面;decode 每条请求只算 1 个 token,批一小,kernel 转眼就算完,GPU 算完一个、CPU 还没发出下一个,两个 kernel 之间全是空档。源码注释里留了一个量级:DeepSeek R1 一次前向里 Inductor 插入的形状与对齐检查约 400 次 Python 调用,光这一项就让每次前向多出约 2 毫秒。

坏的是**小批 decode 的 TPOT**。模型越小、批越小、张量并行把每张卡上的矩阵切得越薄,每个 kernel 越短,下发占的比例就越大,换一张更快的卡反而更明显。07 章的异步调度把「调度、准备输入」挪进了 GPU 忙的时间,挪不走的是前向内部这一长串下发。

把前向录成 CUDA graph 是现成的办法,难处在形状。vLLM 的一步可以同时装着 prefill 分块、短追加和 decode,token 总数每步都变;注意力的元数据(每条请求 query 多长、历史多长、KV 在哪几块)也每步都变,多数注意力 kernel 一旦录进图就换不了长度。图录少了覆盖不到,录多了启动慢、显存被吃掉。所以这一章还有第二个会坏的指标:**启动时间**,编译和录图都得在接第一条请求之前做完。

## 二、解法:编译一次,在注意力处切段,按 token 数分档录两种图

最天然的直觉:把一步前向整个录成一张图,以后每步只发一次重放。vLLM 把这件事拆成编译、录图、派发三步,让同一份编译产物服务形状不同的三种批:

![一次前向在三种批形状下怎么跑:最上面是启动时的编译,Dynamo 抓出整张前向后在注意力与写 KV 处切成段 0 到段 N,每段交给 Inductor 按 1 到 8192 个 token 编一份;下面三行与它上下对齐,13 条请求的纯 decode 批补到 16 档后整个前向连注意力是 1 张整图,300 个 token 的混合批补到 304 档后每段各是 1 张分段图而注意力留在图外现场下发,8192 个 token 的 prefill 块超过最大档 512 不用图、逐个下发编好的 kernel;最底下是 H100 在线服务默认的 51 个档位刻度,512 以下先密后疏,512 以上是不用图的区域](/opensource/vllm-lite/08a-three-batch-shapes.svg)

**编译:只让 token 数这一维是变量。** 启动时第一次空跑前向,Dynamo 把模型的整个前向抓成一张计算图,输入里只有 token 数这一维是符号。注意力、写 KV、Mamba、线性注意力这类层事先注册成不透明的自定义算子,Dynamo 不往里看,只记下它们输出的形状。抓完之后丢掉全部守卫,服务期间不会因为形状变化重新编译。然后在这些不透明算子处把图切开,相邻两次注意力之间是一段;中间各层的段结构相同,编一次就到处复用。每段交给 Inductor,按「1 到一步 token 上限」这整个区间编一份,运行时任何批大小都用它。

**融合发生在编译里,模型代码不用改。** 开着编译时,vLLM 自己手写的归一化、激活等 kernel 默认全部让位给 PyTorch 原生写法,好让 Inductor 把归一化、残差加、激活和相邻的逐元素运算合成少数几个 Triton kernel。在 Inductor 之前,vLLM 还有一组自己的图改写 pass 按固定模式匹配:张量并行时把 all-reduce 和其后的残差加、RMSNorm 合成一个 FlashInfer kernel,Hopper 与数据中心 Blackwell 上张量并行大于 1 就默认打开;量化模型把 norm 或激活与量化合成一步(13 章);序列并行这类改写通信的 pass 也挂在这里(10 章)。

**录图:每一档录两种。** 启动时按 token 数铺一排档位:1、2、4,之后每 8 个一档到 248,再每 16 个一档到上限。上限默认取「最大并发的 2 倍」与 512 中的小者,数据中心 Blackwell 是 1024,且不超过一步的 token 预算;H100 在线服务落在 512,共 51 档。每档录两遍。**整图**把整次前向连注意力一起录成一张,只给纯 decode 批用;**分段图**只录注意力之间的那些段,注意力连同写 KV 留在图外,按真实长度现场下发,给混合批用。两种图共用一个显存池,先录分段图、再录整图,都从大档往小档录,整图的中间激活正好落进分段图已经占下的块里。

**派发:每步按批的形状查表。** ModelRunner 数出本步的 token 数,向上补到最近一档,补出来的行标成空行。整批都是 decode、每条的 token 数相同(推测解码时是 1 加草稿数,09 章),走整图;否则走分段图;token 数超过最大档,这一步不用图,逐个下发编好的 kernel,因为批大到这个份上,每个 kernel 都算得够久。数据并行时各副本先对齐到同一档,只要有一个副本用不了图,这一步大家都不用(10 章)。V1 与 V2 两套 ModelRunner 各有一份查表逻辑,规则相同(07 章)。

**图的模式由配置与后端的声明合出来。** 默认的第 2 级优化要的是「decode 整图 + 其余分段图」。配置期先按特性改写:池化模型去掉整图;编码器-解码器模型只留 decode 整图;LMCache 开逐层传输时只留分段图;DeepEP 高吞吐模式配数据并行时整个关掉;序列并行、注意力加量化融合要看到整张图,手动打开它们就不切段、只留整图(这两项在各级优化下默认都关)。KV cache 建好之后,ModelRunner 挨个问注意力后端(06 章)能进哪种图,答案有四档:任何批、只收同长批、只收单 token 的 decode、不能进,取其中最弱的一个。后端撑不起混合批进整图,就退成 decode 整图加分段图;后端一种整图都不收,就退成只剩分段图;开着推测解码而后端只收单 token 的 decode,也退成只剩分段图。每退一次打一行告警;退到最后仍然要整图、后端却一种都不收,启动报错。

**少数新架构绕开编译器。** NVIDIA 卡上,DeepSeek V3.2 与 V4、Kimi K3、MiniMax M3 稀疏版等 24 个架构名默认打开「可断图」:不追踪、不编译,直接对整个前向录图,录到被标记的注意力类算子就结束当前一段,现场执行它,再开下一段。源码注释写明这个思路来自 SGLang。

为什么这样实际不吃亏:编译只编一个形状区间,多花的是一次性的启动时间,产物还能落盘复用;分段图每层只有注意力这一处留在图外,其余算子的下发全都省掉;补齐的浪费有上限,256 以下最多多算 7 行,256 以上最多 15 行;超过最大档的大批本来就不怕下发开销。

## 三、代价

**启动多出两段时间。** 首次启动要跑 Dynamo 抓图和 Inductor 编译,再把全部档位的图录一遍;源码注释说录图通常 5–20 秒。编译产物按哈希落盘,哈希里有相关环境变量、全部配置、Dynamo 追踪时经过的每个源文件内容和编译器版本,任何一项变了都要冷编一遍。

**有的融合让编译翻倍。** all-reduce 融合只在通信量小时划算,于是在阈值处把 token 区间切成两截,每一段图两截各编一份。以 H100、张量并行 8 路、隐藏维 8192 的 BF16 模型为例,阈值 0.5 MB 折合 32 个 token,每段都要按 1–32 与 33–8192 编两遍。

**显存先被图扣走。** 图的显存池在分配 KV 之前就估好,从 KV 预算里扣掉(03 章):分段图全部实测一遍,整图只录最大的 2 张,再按单张的增量外推其余。档位越多、上限越大,KV 分到的越少。

**覆盖有上限。** H100 在线服务默认一步最多排 1024 条请求,图却只录到 512 个 token;decode 批在 513–1024 条之间的每一步都没有图可用。

**录进去就改不了。** 档位、模式、融合都在启动时定死,运行期没有接口能改。重放时出错看不到 Python 调用栈,排障只能关图复现。

**多了一层要维护的缓存。** 官方调试文档自己承认:torch.compile 本身的缓存很稳,vLLM 叠在上面的这层缓存并不总是对的。出错时表现为加载失败或行为古怪,只能关掉缓存或删目录重编。

## 四、和 SGLang、TensorRT-LLM 比:编译器是不是默认必经的一站

对照对象是 SGLang(基准 826d5170ae)与 TensorRT-LLM(基准 59f5c47f2e)。三家都给纯 decode 批录整图,也都得给形状多变的混合批另找出路;分歧在 torch.compile 默认在不在执行链上,以及算子融合写在哪一层。

| 必须有人做的事 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 默认过不过 torch.compile | 过:整个前向先编译再录图;24 个新架构默认改走不编译的可断图 | 不过:整模型编译要手动打开,标为实验 | 不过:编译配置默认为空 |
| 纯 decode 批 | 按 token 数分档录整图,向上补到最近一档 | 按批大小分档录整图 | 按批大小分档录整图;补齐默认关,不落在档上整步不用图 |
| 混合批与 prefill | 编译后在注意力处切出的分段图 | CUDA 上默认可断图,不追踪,录到标记处断开 | 默认不录图,分段与可断都要显式打开 |
| all-reduce 与 RMSNorm 这类融合写在哪 | 编译器里的图改写 pass 按模式匹配,对所有模型生效 | 层代码里按条件调融合 kernel,按一张架构名单自动打开 | 模型文件里逐层声明要哪种 all-reduce 融合 |

vLLM 的取舍是把融合与切段都交给编译器,模型代码只写最朴素的 PyTorch,加一个融合 pass 就对所有能编译的模型生效;代价是冷启动多一段编译、多一层缓存,编译器追踪不了的新架构还得另走可断图。另外两家把融合写进模型和层,启动轻,新模型要一个个接。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。`-cc.xxx` 是 `--compilation-config` 的点号写法,也可以整份写成 JSON 传给 `-cc`;用户显式设过的字段压过优化级别给的默认。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `-O0` … `-O3`(即 `--optimization-level`)· 优化级别 | 启动 | `2` | `0`:不编译、不录图、融合全关、FlashInfer 自动调优关,启动最快;`1`:编译加只录分段图,只开 norm 与激活的量化融合(ROCm 另有两项);`2`:再加 decode 整图与 all-reduce 融合;`3`:当前与 `2` 完全相同 | 日志 `Enabled custom fusions:` 列出开着的融合 |
| `--enforce-eager` · 全关 | 启动 | 关 | 开:等于 `-cc.mode=none -cc.cudagraph_mode=none`,并关掉 JIT kernel 预热;启动最快,小批 decode 最慢 | 告警 `Enforce eager set, disabling torch.compile, CUDAGraphs, and JIT kernel warmup` |
| `-cc.cudagraph_mode` · 图的模式 | 启动 | `FULL_AND_PIECEWISE`(`-O1` 下 `PIECEWISE`) | `PIECEWISE`:decode 也只用分段图,省整图显存和录图时间;`FULL_DECODE_ONLY`:混合批不用图,省分段图显存;`FULL`:混合批也进整图,只有「任何批都能进」的后端撑得住,否则自动退,后端完全不能进图则启动报错;`NONE`:只关图不关编译 | 进度条 `Capturing CUDA graphs (PIECEWISE)`、`(FULL)`;退化告警 `CUDAGraphMode.… is not supported with … backend … setting cudagraph_mode=…` |
| `-cc.mode` · 编译方式 | 启动 | `-O1` 起为 `3`(`VLLM_COMPILE`),`-O0` 为 `0` | `0` 或 `none`:不编译,分段图跟着退,`FULL_AND_PIECEWISE` 变 `FULL_DECODE_ONLY`、`PIECEWISE` 变 `NONE`,手写 kernel 全部打开;`1`:原生 torch.compile,不切段、没有 vLLM 的 pass,且 V2 不支持,退回 V1(07 章) | 日志 `Cudagraph mode … is not compatible with compilation mode … Overriding to …` |
| `--max-cudagraph-capture-size` · 最大档 | 启动 | min(最大并发 × decode 长度 × 2, 512),SM10x 为 1024,再不超过 `--max-num-batched-tokens` | 调高到最大并发:大 decode 批也有图;档位变多,录图更久、图显存更大 | `--cudagraph-metrics` 表里 `Runtime Mode` 为 `NONE` 的行 |
| `--cudagraph-capture-sizes` · 直接给档位列表 | 启动 | 自动生成 | 只录列出的档,最大值即上限;与 `-cc.cudagraph_capture_sizes` 互斥 | 同上 |
| `--performance-mode` · 性能倾向 | 启动 | `balanced` | `interactivity`:1–32 每个整数一档,小批几乎不补齐,32 以内的档位从 7 个变成 32 个;`throughput`:最大并发与 token 预算翻倍(02 章),但默认最大档仍卡在 512 | `--cudagraph-metrics` 表的 `Num Paddings` |
| `-cc.compile_sizes` · 额外按固定形状编译 | 启动 | 空 | 列出的 token 数各再编一份静态形状,并打开 Inductor 的 max-autotune 与坐标下降调优;首次编译慢得多;V1 下必须取档位里的数,否则启动报错 | 日志 `Compile and warming up model for size N` |
| `-cc.pass_config.fuse_allreduce_rms` · all-reduce 与 RMSNorm 融合 | 启动 | `-O2` 起在 SM90 与 SM10x、TP > 1、装了 FlashInfer、非批不变模式时开 | 关:每层多一次独立的 all-reduce 与 norm;开:按通信量阈值多切一个编译区间,编译时间翻倍 | `Enabled custom fusions:` 里有 `allreduce_rms` |
| `-cc.custom_ops` · 手写 kernel 开关 | 启动 | 编译时为 `none`(交给 Inductor 生成),不编译时为 `all` | `+rms_norm` 这类:该算子改用 vLLM 手写 kernel,Inductor 不再把它与邻居融合,norm 加量化的融合 pass 则会打开;名字写错只打告警 | DEBUG 日志 `enabled custom ops:`;告警 `Op '…' … has no effect` |
| `-cc.splitting_ops` · 在哪些算子处切段 | 启动 | 不设:16 个注意力类算子加 2 个写 KV 算子 | 设 `[]`:不切段,分段模式被改成 `FULL` 或 `NONE`;依赖整图的融合可以打开 | 告警 `Piecewise compilation with empty splitting_ops` |
| `-cc.use_inductor_graph_partition` · 让 Inductor 编完整图再分区 | 启动 | 关(各级优化都显式设关) | 开:融合在整图上做完再在注意力处分区,整图融合与分段图可以并存;编译更久,要 torch ≥ 2.9 | — |
| `VLLM_USE_BREAKABLE_CUDAGRAPH` · 可断图 | 环境变量 | 不设;命中默认架构名单时自动设 `1` | `1`:编译方式强制为 `none`,分段图改用可断图;对名单里的架构设 `0`:退回编译路径 | 日志 `Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1` |
| `VLLM_CACHE_ROOT` · 编译缓存的根 | 环境变量 | `~/.cache/vllm` | 缓存在其下 `torch_compile_cache/`;容器里挂成持久卷,重启才能命中 | 日志 `Using cache directory: … for vLLM's torch.compile` |
| `VLLM_DISABLE_COMPILE_CACHE` · 关编译缓存 | 环境变量 | `0` | `1`:每次都冷编,同时关掉 AOT 编译;排查缓存问题用 | 日志 `vLLM's torch.compile cache is disabled.` |
| `VLLM_USE_AOT_COMPILE` · Dynamo 产物也落盘 | 环境变量 | torch ≥ 2.10 且缓存没关时为 `1` | `0`:热启动也要重新跑 Dynamo 追踪 | 日志 `Directly load AOT compilation from path …` |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` · 缓存格式 | 环境变量(或 `-cc.compile_cache_save_format`) | `binary` | `unpacked`:存成目录,可以直接改 Inductor 生成的代码调试;多进程不安全 | 缓存目录结构 |
| `-cc.debug_dump_path` · 导出编译中间产物 | 启动(或 `VLLM_DEBUG_DUMP_PATH`) | 不设 | 按 rank 导出 Dynamo 改写后的代码与各阶段计算图 | 目录下 `rank_*_dp_*` |
| `--cudagraph-metrics` · 统计每步走了哪种图 | 启动 | 关 | 开:每个日志周期打一张表,列未补齐与补齐后的 token 数、补了几行、走哪种模式、出现次数 | 日志 `**CUDAGraph Stats:**` |
| `VLLM_LOGGING_LEVEL=DEBUG` · 调试日志 | 环境变量 | `INFO` | 分段图录图时记下输入地址、重放时逐次校验;torch 低于 2.12 时还打开 Inductor 的运行时断言,变慢 | 断言 `Input addresses for cudagraphs are different during replay` |

**怎么看。** 一是启动日志,按时间顺序:`Dynamo bytecode transform time: … s` 是抓图耗时;`Compiling a graph for compile range (1, 8192) takes … s` 是 Inductor 冷编,命中缓存时换成 `Directly load the compiled graph(s) for compile range (1, 8192) from the cache, took … s`;`torch.compile took … s in total` 是编译总账;录图结束打 `Graph capturing finished in N secs, took X GiB`,紧跟一行 `CUDA graph pool memory: … GiB (actual), … GiB (estimated)` 对照预扣是否准。二是模式有没有被退:搜 `setting cudagraph_mode=`、`Overriding cudagraph_mode`、`Overriding to`,句中写着原因和退到了哪一档。三是运行期:开 `--cudagraph-metrics` 后,那张表里 `Runtime Mode` 列就是每步实际走的 `FULL`、`PIECEWISE` 或 `NONE`,`Num Paddings` 是补了多少空行。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单卡 H100 在线服务,Qwen3 这类稠密模型,并发 256 以内 | `vllm serve Qwen/Qwen3-8B`,把 `VLLM_CACHE_ROOT` 指到持久卷 | 什么都不改:首次启动付编译和录图的时间,换小批 decode 的下发开销;之后重启命中缓存;确认日志里没有模式退化 |
| H100 高并发,decode 批经常超过 512 条 | `vllm serve <模型> --max-cudagraph-capture-size 1024` | 档位从 51 个涨到 83 个,多花录图时间和图显存,换 513–1024 条的 decode 也有整图 |
| PD 分离里的 decode 实例 | `vllm serve <模型> -cc.cudagraph_mode=FULL_DECODE_ONLY`,再配 12 章的 KV connector | 这台几乎只跑 decode,省掉分段图的显存和录图时间;偶尔的混合批没有图;挂 LMCache 且开逐层传输时会被改回分段图 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 每次重启都要等很久编译 | 有没有 `Directly load` 那两行;缓存目录路径每次是否一样 | 容器里把 `VLLM_CACHE_ROOT` 挂成持久卷;配置、代码、torch 版本一变哈希就变,属正常;只求起得快用 `-O1` 或 `-cc.mode=none` |
| 启动报 `CUDAGraphMode.… is not supported with … backend`,原因里写着 `NEVER` | 显式设了 `FULL` 类模式,而后端声明不能进图 | 改回默认或 `PIECEWISE`,并确认编译开着;或换后端(06 章) |
| 启动日志有 `setting cudagraph_mode=PIECEWISE`,小批 decode 变慢 | 句中点名的最弱后端是谁,是不是开了推测解码 | 换一个能进同长批的后端(06 章);混合模型里 Mamba 这类层常是最弱的那个 |
| 并发过某个值后 TPOT 突然变差 | `--cudagraph-metrics` 表里 `NONE` 行是否变多;批大小是否超过最大档 | 调大 `--max-cudagraph-capture-size`,同时留意 KV 容量被图显存吃掉多少 |
| 小批时补齐浪费大 | 表里 `Num Paddings` | `--performance-mode interactivity`,或自己给 `--cudagraph-capture-sizes` |
| 告警 `Cudagraph mode … requires piecewise capture, but the loaded model provides neither a compiled submodule nor breakable CUDA graphs`,混合批从此没有图 | 模型这一路没被编译(关了编译,或模型类声明不编译) | 设 `VLLM_USE_BREAKABLE_CUDAGRAPH=1` 用可断图补上分段图;或接受只有 decode 整图 |
| 开图与关图输出不一致,或重放时崩 | 用 `-cc.cudagraph_mode=NONE` 对照;`VLLM_LOGGING_LEVEL=DEBUG` 查输入地址断言 | 确认是图的问题后只关图,不必全关;报给上游 |
| 升级或改代码后加载缓存报错、行为古怪 | `VLLM_DISABLE_COMPILE_CACHE=1` 能否复现 | 删 `VLLM_CACHE_ROOT` 下的 `torch_compile_cache` 重编 |
| KV 容量比预期小 | `Graph capturing finished … took X GiB` 与估计值那一行 | 降最大档,或改用 `PIECEWISE`、`FULL_DECODE_ONLY` 少录一种图 |

## 六、常见误区

- **以为 `--enforce-eager` 只是关 CUDA graph。** 名字读起来像「立即执行,不录图」,排查图的问题时大家第一反应就加它。它同时关掉了 torch.compile 和挂在编译里的全部融合 pass,还关了 JIT kernel 预热,拿它测出来的性能和线上差的不只是图。只想关图用 `-cc.cudagraph_mode=NONE`,只想关编译用 `-cc.mode=0`,官方调试文档也是这么分三档的。
- **以为后端声明「任何批都能进整图」,混合批默认就在整图里。** FlashAttention 第 3 代、Triton 注意力确实这么声明。可默认模式是「decode 整图 + 其余分段图」,混合批一律走分段图;只有显式设 `FULL` 才让混合批也进整图。代码注释说大多数模型上默认这种组合更快。
- **以为图能覆盖到最大并发。** 两个默认值看起来是配套的,其实各算各的:H100 在线服务的最大并发是 1024,最大档却是 512,decode 批在 513–1024 条之间每步都没有图。`throughput` 模式把最大并发翻到 2048,最大档仍是 512。
- **以为 `-O3` 比 `-O2` 更激进。** 名字暗示多一档优化。当前代码里两者的默认值表逐项相同,官方文档也写着「目前等于 `-O2`」。
- **以为 `-O2` 会按模型自动打开序列并行和注意力加量化融合。** 级别表里这两项写的是「是否稠密模型」「是否量化模型」,读起来像条件判断;实际上那两个判断是写死的 `False`,注释说依赖的属性暂时都当假处理。想要只能手动在 `-cc.pass_config` 里打开,而且一开就不再切段(10 章、13 章)。
- **以为 Inductor 的缓存在 `/tmp/torchinductor_用户名`。** 官方调试文档讲删 torch 自带的缓存时指的就是这个目录,排障时常只删它。本基准钉的 torch 是 2.13,AOT 编译默认开着,它把 Inductor 的缓存目录改到了 `VLLM_CACHE_ROOT` 下的 `torch_compile_cache/torch_aot_compile/<哈希>/inductor_cache`,Dynamo 的产物也在这棵树里,只删 `/tmp` 那个等于没删。要清就清 `VLLM_CACHE_ROOT` 下的 `torch_compile_cache`。
- **以为 TTFT 偶尔冒尖是遇到新形状在现场编译。** 用过原生 torch.compile 的人会这么想,它见到新形状就可能重编。vLLM 丢掉了全部守卫,所有 token 区间在启动时第一次空跑就编完,服务期间形状怎么变都不会重编;运行期冒尖要去查调度与 prefill(02 章),不是编译。
- **以为 `-cc.custom_ops` 开得越多越快。** 手写 kernel 听起来总比生成的强。开着编译时它们默认全关,正是为了让 Inductor 把归一化、激活和相邻运算融成一个 kernel;打开某一个,它就成了融合链上的断点。只有 norm 加量化那条融合需要手写的量化 kernel 在场;权重按块量化的 FP8 模型,代码会自动把它打开。

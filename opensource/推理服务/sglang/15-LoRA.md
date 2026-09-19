# SGLang 15|LoRA

LoRA 为什么能算对、为什么必须拆成 shrink 和 expand 两个 kernel、槽位这层间接是怎么来的、adapter 在磁盘、内存、显存三级之间怎么进出,vLLM 解读的「LoRA」一章已经讲透。这一页只说 SGLang 遇到的问题、它自己的解法、以及它和 vLLM 不一样的地方。全页按基准 commit 逐条核对过源码;每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

一份 base 权重同时服务多个 adapter(挂在 base 上的一对瘦矩阵,每个业务方一套),GPU 上一次只装得下固定的套数,SGLang 默认 8 套,而且 base 自己占其中 1 套。三种负载把三个指标弄坏:

- 同时在跑的 adapter 种类超过槽数:后面的请求只能在队列里等,KV 和算力都空着,TTFT 的尾部以秒计。负载偏斜时更糟,3–4 个热 adapter 的长请求一直霸着槽,冷 adapter 的请求可以一直等下去。
- adapter 换得勤:每换一个要从内存往显存搬一次权重,默认搬的时候前向停着,批里所有请求的 TPOT 一起抖一下。
- 一个批里 adapter 很杂:每段序列各起一次 kernel,kernel 数随请求数涨,每步时延跟着涨。

## 二、解法:所有 adapter 叠在一块显存里,按槽号取

最天然的直觉:把可能用到的 adapter 权重预先叠在一块显存里,一个 adapter 一个槽,每个请求带一个槽号,kernel 按槽号取权重算。为什么实际不吃亏:LoRA 只多算 rank 那么窄的一段;SGLang 默认后端 csgmv 又把同一 adapter 连着的请求合成一段、再按 16 个 token 切成块,kernel 的数量只跟块数走,不跟请求数走,adapter 分布再偏斜也不会多起 kernel。

SGLang 自己加的东西:

- **base 也占一个槽。** 不用 adapter 的请求在池里是一个空名字,和真 adapter 平等排队。槽满了先踢真 adapter,只有整批都是 LoRA 请求时才踢 base;base 被踢或重新装回时,那个槽整体清零,CUDA graph 回放读到的是 0 而不是上一个 adapter 的残留。
- **踢谁可选。** 默认最久没用的先走,可以改成先进先出。本批正在用的和钉住的不能踢。
- **排空器。** 某个 adapter 的请求等超过阈值秒数,调度器挑一个在跑的、剩余 token 最少的 adapter,不再给它放新请求(只放能在它剩余长度 1.2 倍内跑完的),让它跑完腾槽。默认关。
- **搬运与前向重叠。** 开了开关,新 adapter 从内存到显存的搬运走另一条 stream,调度器只把已经搬到位的 adapter 放进批。它要求权重在内存里锁页,所以强制内存里最多只能留 2 倍槽数的 adapter。
- **按引用计数卸载。** 每个在飞的请求给它用的 adapter 计数加 1。卸载先把名字从注册表摘掉,新请求进不来;等计数归零才真正在 GPU 侧删。
- **直接吃张量。** 除了按路径装,还有一个接口直接接序列化好的张量,不落盘,RL 训练侧刚训完的 adapter 可以直接推过来。

## 三、代价

- 显存按「槽数 × rank 上限 × 目标模块数」买断,精度跟 base,启动就占。目标模块设成 all 会把所有支持的线性层加 embedding 和 lm_head 全包进去,每层多两次小矩阵乘,哪怕 adapter 只训了注意力。
- LoRA kernel 每步都发,整批都是 base 也发,进去发现 rank 是 0 立刻退。CUDA graph 里捕的就是这条路径。
- 调度多一道准入:新请求要么它的 adapter 已在跑,要么槽还有空,否则跳过留在队列。排空器让一部分请求为了公平多等。
- 槽位分配用排序不用哈希,因为各 TP 进程的 Python 哈希种子不同,而槽号必须逐进程一致。
- 重叠搬运的反面:adapter 分先后到位,同一步能凑到一起 prefill 的 adapter 变少,搬运短、prefill 长的时候比不开还慢。

## 四、和 vLLM 不一样的十处

1. **槽包含 base。** vLLM 槽数默认 1,base 走 0 号不占槽;SGLang 默认 8,base 占 1 个,同时在跑的真 adapter 通常是 7 个。
2. **默认后端是分块 SGMV。** vLLM 按序列分组;SGLang 先合并连续同 adapter 的序列,再按 16 切块,块大小 16、32、64、128 可调。老的 triton 后端保留,高并发下慢 20%–80%。
3. **内存这一级默认不限,满了也不用客户端重给路径。** vLLM 内存侧容量默认等于槽数,被挤出去的 adapter 下次请求时靠请求里带的路径重读磁盘;SGLang 的 `--max-loaded-loras` 默认不限,设了以后满了在 tokenizer 侧把最久没用的非钉住 adapter 整个卸掉(先装新的再卸旧的),请求里只有名字,服务自己记着路径,下次点名时自动重装,只有从没装过的名字才报错。
4. **踢法可选,钉住有上限。** vLLM 只有 LRU,钉住是「不在就先搬上去再钉」的强语义;SGLang 可选 lru 或 fifo,钉住最多槽数减 1 个,防止把非钉住的和 base 饿死。
5. **有排空器。** vLLM 没有对应机制。
6. **搬运与前向重叠是显式开关。** vLLM 默认在同一条流上异步拷贝;SGLang 默认同步搬,开关后独立流加 Event。
7. **热加载默认开,还能直接吃张量。** vLLM 要开环境变量才有热加载端点;SGLang 的 3 个接口默认就在,但要求 dp_size 为 1 或开 DP attention。
8. **捕图不造假 adapter。** vLLM 捕 CUDA graph 时要造一批假 adapter 才能捕到真路径;SGLang 的 kernel 总是发射、槽号为空就早退,捕图用「全是 base」的批就够。而且 csgmv 和 triton 后端下 prefill 也能进 CUDA graph,DP attention 下除外。
9. **推测解码有白名单。** 只允许 NGRAM、EAGLE、EAGLE3、DFLASH、DSPARK,再加 4 条禁用组合:DSPARK 配非 static 的 ragged verify、`--speculative-adaptive`、experimental_sgl_trtllm 的 MoE runner、`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`,启动即报错。
10. **少 3 个参数,多 1 条拒绝。** 没有 LoRA 精度参数(跟 base)、没有内存侧容量默认值(不限)、没有全分片开关;带新词表的 adapter 两边都拒绝,SGLang 另外拒绝 DoRA。

## 五、调参与观测

**参数在哪调。** 启动参数(`--enable-lora`、`--max-loras-per-batch` 这些)决定池的形状和后端,改了要重启;`--lora-paths` 只是初始集合。请求参数只有一个:`/generate` 的 `lora_path`(字符串或与输入等长的列表,空即 base),OpenAI 接口写在 `model` 字段里,格式是 `base:adapter`。运行时 3 个接口:`/load_lora_adapter` 按名字和路径装(可带 `pinned`),`/unload_lora_adapter` 按名字卸,`/load_lora_adapter_from_tensors` 直接接张量;装进来的 adapter 的 rank 和目标模块必须在启动时开的池子里,超了要重启。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--enable-lora` · 打开 LoRA 路径 | 启动 | 关;给了 `--lora-paths` 自动开 | 开了每步多发 LoRA kernel,纯 base 也慢一点 | 启动日志 `Using csgmv as backend of LoRA kernels` |
| `--lora-paths` · 启动就装的 adapter,名字=路径,JSON 可带 pinned | 启动 | 空 | 没给时必须给 rank 上限和目标模块,否则起不来 | `/v1/models` 列出全部 adapter |
| `--max-loras-per-batch` · 一个批里 adapter 种类的上限,含 base | 启动 | 8 | 调高:能同时跑的种类多,排队少,显存按比例涨。调低:反过来 | 装 adapter 前后日志里的 `avail mem`;`#queue-req` 高但 `#running-req` 低 |
| `--max-lora-rank` · 池按多大的 rank 开 | 启动 | 从初始 adapter 取最大 | 调高:能热装大 rank,显存与算量线性涨;比它大的装不进 | 装载报错 `incompatible with the current LoRA memory pool configuration` |
| `--lora-target-modules` · 哪些层留 LoRA 位 | 启动 | 从初始 adapter 取并集 | 设 all:什么 adapter 都装得进,每层多两次小乘,显存多占 | 日志 `resolved to [...] by inspecting the base model` |
| `--lora-backend` · kernel 后端 | 启动 | csgmv | triton:老路径,高并发慢 20%–80%。torch_native:只用来对数 | 启动日志那一行 |
| `--max-lora-chunk-size` · 每块的 token 上限 | 启动 | 16 | 调大:块变少,长 prompt 多 adapter 时更快;短请求多时块填不满 | 只能压测对比 |
| `--lora-eviction-policy` · 槽满了踢谁 | 启动 | lru | fifo:轮换型负载下踢得更均匀 | 只有 debug 级日志 |
| `--max-loaded-loras` · 内存里最多留几份 | 启动 | 不限 | 调低:省内存;超了彻底卸载,下次点名重装,那次请求 TTFT 多一次磁盘读 | 日志 `Unloading least recently used LoRA adapter`、`Reloading evicted adapter` |
| `--enable-lora-overlap-loading` · 搬运与前向重叠 | 启动 | 关 | 开:换手高、adapter 大时 TTFT 中位数降 35%;搬运短、prefill 长时反而变慢;要求内存侧上限不超过 2 倍槽数 | 压测 TTFT 中位与 p99 |
| `--lora-drain-wait-threshold` · 等多久算饿,秒 | 启动 | 0,关 | 设 1–5:冷 adapter 的尾延迟压下来,热 adapter 吞吐略降 | 只有 debug 级日志;看排队时间分布 |
| `--lora-strict-loading` · 权重键对不上就报错 | 启动 | 关 | 开:少装了模块直接失败,不再静默跳过 | 警告 `weight(s) skipped because they did not match any target module` |
| `pinned` · 钉住不踢 | `--lora-paths` 的 JSON 或 `/load_lora_adapter` | false | 钉住的免搬运;最多槽数减 1;钉住的越多,给其他 adapter 剩的槽越少 | 报错 `not allowed to pin all slots` |
| `lora_path` · 这条请求用哪个 adapter | 请求体;OpenAI 用 `model` 的 `base:adapter` | 空,走 base | 没装过的名字直接报错 | 响应错误 `has never been loaded` |

**怎么看。** LoRA 没有任何 Prometheus 指标,只能靠日志和接口。info 级只有这些:启动时的后端行;装和卸各一对 `LoRA adapter loading starts / completes`,带 `avail mem`,前后相减就是这个 adapter 在显存里的大小;装完 `loaded weights for target modules [...]` 列出真正接上的模块;`max-loaded-loras` 触发的 `Unloading least recently used` 和 `Reloading evicted adapter`。踢槽、排空、重叠搬运全是 debug 级,排查槽位抖动要临时开 `--log-level debug`。`/v1/models` 列出当前注册的 adapter,`/get_server_info` 读回启动参数。稳态看 Decode batch 那一行:`#queue-req` 高、`#running-req` 低、`token usage` 也低,三者同时出现就是卡在槽,不是卡在 KV。

| 症状 | 先查 | 然后 |
|---|---|---|
| 队列长、`token usage` 低、`#running-req` 少 | 在跑的 adapter 种类是否贴着槽数 | 加 `--max-loras-per-batch`;显存不够就开排空器 |
| 冷 adapter 的请求等待以秒计而 KV 空着 | 热 adapter 的长请求是否霸槽 | `--lora-drain-wait-threshold` 设 2 |
| 换 adapter 时全体 TPOT 抖一下 | debug 日志里是否每批都在装新 adapter | 热的钉住;或开重叠搬运并设 `--max-loaded-loras` |
| 开了重叠搬运 TTFT 反而长 | 单个 adapter 搬运耗时和 prefill 耗时哪个大 | 搬运短就关掉 |
| 热装报 rank 或目标模块不兼容 | 启动时的 `--max-lora-rank`、`--lora-target-modules` | 重启,显式给两个参数 |
| 热装报 add tokens 或 DoRA | adapter 目录里的 added_tokens.json 和 adapter_config.json | 离线合并进 base 单独部署 |
| 请求报 `never been loaded` | 这个名字是否装过 | 先 `/load_lora_adapter` |
| 装过的 adapter 请求变慢一次 | 是否被 `--max-loaded-loras` 卸了又自动重装 | 提高上限或钉住 |
| 卸载接口一直不返回 | 该 adapter 是否还有请求在飞 | 等它们跑完;卸载前停发 |
| 启动报 `LoRA with EAGLE ... does not support` | 4 条禁用组合 | 关掉对应那项 |
| 装载警告 `weight(s) skipped` | adapter 的目标模块和 base 架构是否对得上 | 开 `--lora-strict-loading` 复现,换 adapter |
| 换了 adapter 答案没变 | `lora_path` 传没传;OpenAI 用的是不是 `base:adapter` | 改请求 |

## 六、常见误区

- **槽数 8 就是能同时跑 8 个 adapter。** 文档写的是「每个批用到的 adapter 数上限」,人自然只数真 adapter。实际上 base 也占一个槽,常态是 7 个真 adapter 加 base;整批都是 LoRA 请求时能到 8,但 base 被踢出去,下一个不带 adapter 的请求进来又要占槽清零。
- **开了重叠搬运总不会更慢。** 名字听着就是白拿的优化。实际上它让 adapter 分先后到位,本来能凑成一个 prefill 批的请求被拆开跑。文档自己的例子:4 个 adapter 各搬 2 毫秒、prefill 20 毫秒,不开 28 毫秒,开了最坏 82 毫秒。只在换手高、adapter 大、PCIe 紧的时候开。
- **钉住越多越稳。** 钉住等于免搬运,人就想把常用的全钉上。实际上钉住的槽对其他人是死的:能进批的非钉住 adapter 数是槽数减钉住数,钉到只剩 1 个空槽时所有非钉住请求串行跑。系统最多让钉槽数减 1 个。
- **目标模块设 all 省心。** 热加载什么 adapter 都不用重启。实际上池子按 all 开,每层每个线性层加 embedding 和 lm_head 都留了位,显存多占,每层多两次小乘,哪怕装进来的 adapter 只训了注意力。
- **adapter 目录里有 added_tokens.json 就装不上。** 有人看到拒绝新词表的规则就先删文件。实际上 id 小于 base 词表大小的条目会被当成从 base 拷来的自动过滤,只有真新增的 token 才拒绝;真新增的没有绕法,离线合并。
- **推测解码和 LoRA 不能一起用。** draft 模型没有 adapter,人就推断两者互斥。实际上 EAGLE 一族和 NGRAM 都行,adapter 只作用在 target 模型上,draft 照常裸跑;卡住的是那 4 条组合,启动就报,报错信息里写了原因。
- **内存侧上限满了请求会失败。** 名字是 max,人以为超了就拒。实际上超了是把最久没用的整个卸掉,下次点名时按记住的路径自动重装,代价是那条请求多等一次磁盘读;只有从没装过的名字才报错。
- **看 info 日志就能判断槽在不在抖。** 装卸有 info 行,人以为踢槽也有。实际上踢槽、排空、重叠搬运全是 debug 级,info 里一片安静不代表没抖。
- **rank 上限会自动推断,热加载随便装。** 推断只看启动时那批,后来装 rank 更大的直接报不兼容。要热加载就显式给 `--max-lora-rank` 和 `--lora-target-modules`。
- **多 DP 副本上也能热加载。** 接口在,人就调。实际上 dp_size 大于 1 且没开 DP attention 时接口直接断言失败。
- **同一个 adapter 换个名字再装是免费的。** 路径一样,人以为会复用。实际上只打一行警告,照样再占一份内存和一个槽。

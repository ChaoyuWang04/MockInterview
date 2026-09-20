# SGLang 15|LoRA

LoRA 凭什么能算对、为什么必须拆成收缩和扩张两个 kernel、槽位这层间接是怎么来的、adapter 在磁盘、内存、显存三级之间怎么进出,vLLM 解读的「LoRA」一章已经讲透;低秩分解本身在知识库里有一篇。这一页只说 SGLang 遇到的问题、它的解法、以及它和 vLLM 不一样的地方。全页按基准 commit 在源码上逐条重核过;每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

一份 base 权重同时服务多个 adapter(挂在 base 上的一对瘦矩阵,每个业务方一套)。把「同一批里同时活着的 adapter 种类」这个数推上去,三处先坏:

**kernel 的工作量按最长的那条请求开。** 老路径给一条请求开一组线程,而组的高度是按这一批里最长的那条序列定的。32 条请求里混进一条 2000 token 的 prefill、其余 31 条各出 1 个 token,那 31 组也照 2000 的高度开满,进去发现自己只有 1 行就退掉。种类越杂、长短越不齐,空转的比例越高,每步时延跟着涨。

**GPU 上能同时活着的种类是个小数,base 自己还占一个。** 默认 8,其中一个名额留给「不用 adapter」这一类。顶到上限之后,后面的请求只能在队列里等,而这时 KV 和算力都空着。负载偏斜时更糟:3–4 个热门 adapter 的长请求一直霸着名额,冷门 adapter 的请求可以一直等下去,TTFT 的尾部以秒计。

**换手勤,搬运挡在前向前面。** 每换一种就要把权重从内存搬进显存,默认搬的时候前向停着,批里所有请求的 TPOT 一起抖一下。

## 二、解法:先按身份把 token 排一遍队,再按固定长度切块

![一批 token 怎么变成线程组:上半是按请求分段的老路径,4 条请求各占一列、每列 3 层,层高由这一批最长的那条 48 个 token 定,12 个线程组里只有 5 个真在算;下半是默认路径,整批 51 个 token 先按身份排成一排、同身份的连成一片再按 16 个切,只得到 6 个线程组,其中只有装 base 的那一个空转](/opensource/sglang/15a-sort-and-chunk.svg)

最天然的直觉:既然慢是因为「一段里混着不同身份的 token,而段又长短不一」,那就先把整批 token 按身份排成一排,同身份的连成一片,再把每一片切成等长的小块,一块交给一组线程。块长固定,于是线程组的总数正比于这一批真实的 token 数——和请求数无关,和最长那条有多长也无关,身份分布再偏斜也一样。

为什么实际不吃亏:排队用的是一次稳定排序,在 CPU 上做,token 在显存里从没真的搬过家,kernel 拿着那张下标表间接取行;而这一步和 GPU 的前向是错开的(错开怎么做到的见 02 章)。

槽位那一侧,SGLang 自己加了这么几样:

- **base 也占一格。** 不用 adapter 的请求在池子里是一个空名字,和真 adapter 平等排队。要腾位置时先踢真 adapter,只有整批都带 adapter、实在没别的可踢时才踢 base;base 被踢或重新装回,那一格整体清零,图回放时读到的是 0 而不是上一个 adapter 的残留。
- **踢谁可换。** 默认最久没用的先走,也可以换成先进先出。本批正在用的和钉住的不进候选,候选空了直接报错,不会挑一个在用的下手。
- **排空器。** 某个 adapter 的请求等待超过阈值、而且正在跑的种类已经顶满,调度器就在跑着的里面挑一个「剩下要出的 token 最少」的,不再给它放新请求(只放能在它剩余长度 1.2 倍以内跑完的),让它自己跑完腾位置。默认关。
- **搬运与前向重叠。** 打开之后,新 adapter 的搬运走另一条流、记一个事件;调度器每轮先把完成的事件收掉,只把已经到位的 adapter 放进批。它要求权重在内存里锁页,所以同时强制内存里最多只留 2 倍格数的 adapter。
- **按引用计数卸载。** 每条在飞的请求给它用的那一种计数加 1。卸载先把名字从注册表摘掉,新请求进不来;等计数归零才真的发到后端去删。
- **直接吃张量。** 除了按路径装,还有一条口子直接接序列化好的张量、不落盘,训练侧刚训完的 adapter 可以直接推过来。

## 三、代价

- **显存按「格数 × rank 上限 × 所有目标模块的进出维之和」买断**,精度跟 base,启动就占满,实际装了几个 adapter 都一样。这块地和 KV 池分的是同一份静态预算,划走多少 KV 就少多少(池的账见 03 章)。目标模块开成全集会把自动探测出来的每个线性层加上词嵌入和输出头全包进去,每层多两次小矩阵乘,哪怕装进来的 adapter 只训了注意力。
- **每一步都要在 CPU 上给这批 token 排一次序、再分一次段。** token 越多这一步越贵,而且它在前向的关键路径上。
- **LoRA 那一层每步都发**,整批都是 base 也发,进了 kernel 读到这一格的 rank 是 0 才退。图里捕下来的正是这条路径。
- **调度多一道准入。** 新请求要么它那一种已经在跑,要么池子还容得下,否则跳过、留在队列里。排空器再让一部分请求为了公平多等一会儿。
- **格号的分配用排序不用哈希**,因为各 TP 进程的 Python 哈希种子不同,而同一个 adapter 在每个进程里必须落到同一格。
- **重叠搬运的反面。** adapter 分先后到位,同一步能凑到一起 prefill 的种类变少;搬运短、prefill 长的时候比不开还慢。

## 四、和 vLLM 不一样的十一处

![一个 adapter 怎么进出,两边都必须做的三件事各占一行:第一行从磁盘读进内存,vLLM 那一份的容量默认等于格数、SGLang 默认不限;第二行从内存搬进 GPU 的一格,vLLM 画出一个格子、不用 adapter 的 token 是池外的一个负数哨兵,SGLang 画出八个格子、其中一格常态被 base 占着;第三行是被挤出内存之后怎么回来,vLLM 那条箭头从服务进程虚线框外的客户端请求发出,SGLang 那条从虚线框里服务自己记着的名字到路径表发出](/opensource/sglang/15b-vllm-diff-adapter-life.svg)

1. **base 占不占格。** vLLM 的池子里只有真 adapter,不用 adapter 的 token 在送进 kernel 的索引里是一个负数哨兵,在池外;SGLang 让 base 占一格,默认 8 格里常态是 7 个真 adapter 加一个 base。
2. **一批 token 怎么分段。** vLLM 一条请求(也就是一段)一组线程;SGLang 默认先把整批 token 按身份排序、再切成定长块,组数跟总 token 数走。按请求分段那条老路保留着,官方自己测下来高并发时比新路慢 20%–80%。块长可调。
3. **缩放系数在哪乘。** vLLM 在装载时就把它折进升维那半边权重,运行时 kernel 完全不知道它存在;SGLang 每一格存一个系数,扩张 kernel 每步按格号读一次再乘。好处是换 adapter 只动权重,代价是多一次读——以及它只实现了「除以 rank」这一种算法。
4. **rank 上限能填什么。** vLLM 只接受一串固定档位(它要按 rank 去查分块配置表);SGLang 接受任意正整数,不给就从启动时那批 adapter 里取最大值。
5. **内存这一级默认不限,被挤掉了也不用客户端重给路径。** vLLM 内存侧的容量默认等于格数,被挤出去的 adapter 下次请求时靠请求里带着的路径重读磁盘;SGLang 默认不限,设了上限才在收请求那一侧把最久没用的非钉住 adapter 整个卸掉(先装新的、再卸旧的),而请求里只有名字,服务自己记着路径,下次点名自动重装,只有从没装过的名字才报错。
6. **踢法可换,钉住有上限。** vLLM 只有最久未用一种,而且钉住是「不在就先搬上去再钉」的强语义;SGLang 两种踢法可选,钉住最多到格数减 1,防止把非钉住的和 base 一起饿死。
7. **有排空器。** 这是一道专门对付「热门 adapter 霸位、冷门 adapter 饿死」的公平性闸门,vLLM 解读的 LoRA 一章里没有对应物。
8. **搬运与前向重叠是显式开关,而且默认关。** vLLM 默认就在同一条流上异步拷;SGLang 默认同步搬,打开开关才走独立流加事件,同时强制内存侧最多留 2 倍格数。官方在对抗性负载下测到 TTFT 中位数降约 35%,也给了反例:4 个 adapter 各搬 2 毫秒、每个 prefill 20 毫秒,不开一共 28 毫秒,开了最坏 82 毫秒。
9. **热加载默认开,还能直接吃张量。** vLLM 要先打开一个环境变量才有这几个端点(理由是端点接受任意路径);SGLang 的三个端点默认就在,但要求数据并行只有一个副本,或者开了 DP attention。
10. **和图捕获、和推测解码谈判的结果不同。** 捕图这一侧:vLLM 捕图时得先造一批假 adapter,否则捕到的是早退那条路;SGLang 的 kernel 总是发射、格号为空就早退,拿「整批都是 base」的批去捕就够。两边各自还有一串「哪一档图能带 LoRA」的条件,主场在 07 章。推测解码那一侧,SGLang 给 LoRA 划了一张算法白名单,外加 4 条启动即报错的禁用组合,主场在 08 章。
11. **少几个旋钮,多一条拒绝。** 没有 LoRA 精度参数(跟 base 走)、没有内存侧容量的默认值(不限)、没有全分片开关;带新词表的 adapter 两边都拒绝,SGLang 另外拒绝 DoRA。

## 五、调参与观测

**参数在哪调。** 这一章的参数全是启动参数,改了要重启;`--lora-paths` 只是初始集合。请求参数只有一个:`/generate` 的 `lora_path`(字符串或与输入等长的列表,空即 base),OpenAI 接口写在 `model` 字段里,格式是 `基座名:adapter名`,它的优先级高于显式的 `lora_path`。运行时 3 个接口:`/load_lora_adapter` 按名字和路径装(可带 `pinned`)、`/unload_lora_adapter` 按名字卸、`/load_lora_adapter_from_tensors` 直接接张量;热装进来的 adapter,rank 和目标模块必须落在启动时开好的池子里,超了只能重启。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--enable-lora` · 打开 LoRA 路径 | 启动 | 关;给了 `--lora-paths` 自动打开并打一行警告 | 开了每步都发 LoRA kernel,整批都是 base 也发 | 启动日志 `Using csgmv as backend of LoRA kernels.` |
| `--lora-paths` · 启动就装的 adapter | 启动 | 空 | 三种写法:裸路径、`名字=路径`、JSON(可带 `pinned`);不给就必须同时给 `--max-lora-rank` 和 `--lora-target-modules`,否则起不来 | `/v1/models` 列出全部 adapter |
| `--max-loras-per-batch` · 一个批里 adapter 种类的上限,含 base | 启动 | 8 | 调高:能同时跑的种类多、排队少,显存按比例涨;调低反过来 | 指标 `sglang:lora_pool_slots_total`;装卸日志里的 `avail mem` |
| `--max-lora-rank` · 池子按多大的 rank 开 | 启动 | 取初始 adapter 里最大的那个 | 任意正整数;调高能热装大 rank,显存与算量线性涨,比它大的装不进 | 装载报错 `incompatible with the current LoRA memory pool configuration` |
| `--lora-target-modules` · 哪些层留 LoRA 位 | 启动 | 取初始 adapter 的并集 | 20 个模块名可选,或 `all`(自动探测全部线性层再加词嵌入与输出头);设 `all` 什么 adapter 都装得进,但每层多两次小乘、显存多占 | 日志 `resolved to [...] by inspecting the base model` |
| `--lora-backend` · kernel 后端 | 启动 | `csgmv` | 命令行放出 4 个:`csgmv`;`triton` 是按请求分段的老路,高并发慢 20%–80%;`ascend`;`torch_native` 只用来对数。注册表里另有一个内部后端,`flashinfer` 已废弃、选中即报错 | 启动日志那一行 |
| `--max-lora-chunk-size` · 每块的 token 上限 | 启动 | 16 | 只对 `csgmv` 生效;取 16、32、64、128;调大:块变少、每块更满,长 prompt 配多 adapter 时更快;短请求多时块填不满反而浪费 | 只能压测对比 |
| `--lora-eviction-policy` · 格子满了踢谁 | 启动 | `lru` | `fifo`:轮换型负载下踢得更均匀 | 只有 debug 级日志 |
| `--max-loaded-loras` · 内存里最多留几份 | 启动 | 不限 | 必须不小于 `--max-loras-per-batch`,也不小于初始 adapter 数;设了以后超限就把最久没用的非钉住 adapter 整个卸掉,下次点名自动重装,那一次请求多一次磁盘读 | 日志 `Unloading least recently used LoRA adapter`、`Reloading evicted adapter` |
| `--enable-lora-overlap-loading` · 搬运与前向重叠 | 启动 | 关 | 开:换手高、adapter 大时 TTFT 中位数降约 35%;搬运短、prefill 长时反而变慢;必须同时设 `--max-loaded-loras` 且不超过 2 倍槽数,否则起不来 | 压测 TTFT 的中位与 p99 |
| `--lora-drain-wait-threshold` · 等多久算饿,秒 | 启动 | 0,关 | 设 1–5:冷门 adapter 的尾延迟压下来,热门 adapter 吞吐略降;只在正在跑的种类已顶满时才动作 | 只有 debug 级日志;看排队时间分布 |
| `--lora-strict-loading` · 权重键对不上就报错 | 启动 | 关 | 开:少装了模块直接失败,不再只打一行警告 | 警告 `weight(s) skipped because they did not match any target module` |
| `--experts-shared-outer-loras` · MoE 的外层权重在专家之间共享 | 启动 | 不给就从 adapter 权重自动探测 | 显式设了就不再探测;一批里混着两种格式的 adapter 会抛错 | 装载报错 |
| `pinned` · 钉住不踢 | `--lora-paths` 的 JSON 或 `/load_lora_adapter` | false | 钉住的免搬运;最多钉到槽数减 1;钉住的越多,留给其他 adapter 轮转的格子越少 | 报错 `not allowed to pin all slots` |
| `lora_path` · 这条请求用哪个 adapter | 请求体;OpenAI 用 `model` 的 `基座名:adapter名` | 空,走 base | 从没装过的名字直接报错;一条批请求里不同 adapter 的个数不能超过内存侧上限 | 响应错误 `has never been loaded` |

**怎么看。** 三处。

一是指标。开了 LoRA 之后多三个 gauge:`sglang:lora_pool_slots_total`(池子一共几格)、`sglang:lora_pool_slots_used`、`sglang:lora_pool_utilization`(前两者之比);`/v1/loads` 默认也带一个 `lora` 段,字段同名,可以用 `include=lora` 单独取。**口径要当心**:分母是格数、含 base 那一格,而分子只数正在跑的那一批里非空的 adapter 种类,base 不算。所以整批都不带 adapter 时它是 0,而稳态满负荷通常也只到 7/8——指标自己的说明里「1.0 表示池子满了」这句话要照这个口径读。

二是日志。info 级只有这些:启动时的后端行;装和卸各一对 `LoRA adapter loading starts / completes`,带 `avail mem`,前后相减就是这个 adapter 在显存里的大小;装完 `loaded weights for target modules [...]` 列出真正接上的模块;内存侧上限触发的 `Unloading least recently used LoRA adapter` 与 `Reloading evicted adapter`。踢格、排空、重叠搬运全是 debug 级,排查格位抖动要临时把日志级别降到 debug。

三是接口与那一行统计。`/v1/models` 列出当前注册的全部 adapter,`/server_info` 读回启动参数。稳态看 Decode batch 那一行:`#queue-req` 高、`#running-req` 低、`token usage` 也低,三者同时出现就是卡在格子,不是卡在 KV。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。组合与推荐值是按语义推的起点,不是实测最优。

| 什么场景 | 在默认起法上加什么 | 拿什么换什么 |
|---|---|---|
| 就一个 adapter、永不切换 | 别走 LoRA 路径:离线把它合进 base,当普通模型起 | 用「多存一份权重」换掉整条 LoRA 链路的每步开销 |
| 十几个 adapter、流量集中在少数几个 | `--enable-lora --max-loras-per-batch 8 --max-lora-rank <实际最大 rank> --lora-target-modules <实际并集>`,最热的两三个在 `--lora-paths` 的 JSON 里带 `pinned` | 用钉住换热门 adapter 免搬运;代价是留给其他 adapter 轮转的格子只剩 5 个 |
| 上百个 adapter、长尾且换手频繁 | `--max-loras-per-batch 16 --max-loaded-loras 32 --enable-lora-overlap-loading --lora-drain-wait-threshold 2` | 用显存(格数翻倍)和锁页内存换换手时的 TTFT;排空器再用热门 adapter 的一点吞吐换冷门 adapter 的尾延迟。搬运本来就不是瓶颈时,重叠反而会拖长 TTFT |

| 症状 | 先查 | 然后 |
|---|---|---|
| 队列长、`token usage` 低、`#running-req` 少 | 正在跑的 adapter 种类是不是贴着格数 | 加 `--max-loras-per-batch`;显存不够就开排空器 |
| 冷门 adapter 的请求等待以秒计而 KV 空着 | 热门 adapter 的长请求是不是霸着格子 | `--lora-drain-wait-threshold` 设 2 |
| 换 adapter 时全体 TPOT 抖一下 | debug 日志里是不是每批都在装新 adapter | 把热的钉住;或者开重叠搬运并设 `--max-loaded-loras` |
| 开了重叠搬运 TTFT 反而长 | 单个 adapter 的搬运耗时和 prefill 耗时哪个大 | 搬运短就关掉 |
| 长短混合的负载下每步都慢,清一色短请求又测不出来 | `--lora-backend` 是不是老的按请求分段那条 | 换回默认;压测要带上真实的长短混合 |
| 热装报 rank 或目标模块不兼容 | 启动时的 `--max-lora-rank`、`--lora-target-modules` | 重启,把两个参数显式给足 |
| 热装报 add tokens 或 DoRA | adapter 目录里的 `added_tokens.json` 和 `adapter_config.json` | 离线合并进 base 单独部署 |
| 请求报 `has never been loaded` | 这个名字在这个进程里装过没有 | 先调 `/load_lora_adapter` |
| 装过的 adapter 某一次请求特别慢 | 是不是被 `--max-loaded-loras` 卸了又自动重装 | 提高上限或者钉住 |
| 卸载接口一直不返回 | 这个 adapter 还有没有请求在飞 | 等它们跑完;卸载前先停发 |
| 启动报 `LoRA with EAGLE ... does not support` | 8 章那 4 条禁用组合 | 关掉对应那一项 |
| 装载警告 `weight(s) skipped` | adapter 的目标模块和 base 架构对不对得上 | 开 `--lora-strict-loading` 复现,换 adapter |
| 换了 adapter 答案没变 | `lora_path` 传没传;OpenAI 用的是不是 `基座名:adapter名` | 改请求 |

## 六、常见误区

- **以为格数 8 就是能同时跑 8 个 adapter。** 帮助文本写的是「一个批里 adapter 的数量上限」,人自然只数真 adapter。实际上 base 也占一格,常态是 7 个真 adapter 加 base;整批都带 adapter 时确实能到 8,但 base 被挤出去了,下一个不带 adapter 的请求进来又得占一格,而且那一格还要先整体清零。
- **以为 LoRA 这块只能看日志。** 因为格位抖动确实只有 debug 日志,人就以为整块都没有指标。实际上开了 LoRA 就多三个 gauge,`/v1/loads` 里也多一段;只是它们只回答「这一步用了几种」,不回答「这一步踢了谁」。而且利用率的分子不含 base,除非整批都带 adapter,否则它永远到不了 1.0。
- **以为用 PEFT 训出来的 adapter 装进来就一定等价。** 装载只认 `peft_type` 是不是 LoRA,不看缩放公式;而这里写死了「乘 alpha 除以 rank」。改用「除以 rank 的平方根」训出来的那一类 adapter 会被照前一种算,不报错、不警告,只是效果比训练时弱一截——rank 16 就弱到四分之一。这是本章唯一一条会静默给出错误答案的坑,`adapter_config.json` 里那个开关要在上线前看一眼。
- **以为开了重叠搬运总不会更慢。** 名字听着就是白拿的优化。实际上它让 adapter 分先后到位,本来能凑成一个 prefill 批的请求被拆开跑。文档自己给了反例:4 个 adapter 各搬 2 毫秒、每个 prefill 20 毫秒,不开 28 毫秒,开了最坏 82 毫秒。只在换手高、adapter 大、PCIe 紧的时候开,而且不设内存侧上限根本起不来。
- **以为钉住越多越稳。** 钉住等于免搬运,人就想把常用的全钉上。实际上钉住的格子对其他人是死的:能进批的非钉住 adapter 数是格数减钉住数,钉到只剩一格时所有非钉住请求串行跑。系统最多让你钉到格数减 1,再多在装载时就拒。
- **以为目标模块设成全集省心。** 热加载什么 adapter 都不用重启。实际上池子按全集开,每层每个线性层加上词嵌入和输出头都留了位,显存多占、每层多两次小乘,哪怕装进来的 adapter 只训了注意力。
- **以为 adapter 目录里有 `added_tokens.json` 就装不上。** 有人看到「拒绝新词表」的规则就先去删文件。实际上 id 小于 base 词表大小的条目会被当成从 base 拷过来的自动滤掉,只有真新增的 token 才拒绝;真新增的没有绕法,只能离线合并。
- **以为内存侧上限满了请求会失败。** 名字是 max,人以为超了就拒。实际上超了是把最久没用的那个整个卸掉,下次点名时按记住的路径自动重装,代价只是那条请求多等一次磁盘读;只有从没装过的名字才报错。用张量直接推进来的 adapter 例外——它没有真实路径,被卸掉就回不来了。
- **以为 rank 上限会自动推断,热加载随便装。** 推断只看启动时那批 adapter,后来装 rank 更大的直接报不兼容。要热加载就在启动时把 rank 上限和目标模块显式给足。
- **以为多个数据并行副本上也能热加载。** 三个端点都在,人就调。实际上副本数大于 1 且没开 DP attention 时,断言直接失败。
- **以为同一个 adapter 换个名字再装是免费的。** 路径一样,人以为会复用。实际上只打一行警告,照样再占一份内存和一个格子。
- **以为推测解码和 LoRA 不能一起用。** 草稿模型没有 adapter,人就推断两者互斥。实际上 adapter 只作用在目标模型上,草稿照常裸跑,好几种算法都能配;卡住的是那几条组合,启动就报,报错信息里写了原因(见 8 章)。

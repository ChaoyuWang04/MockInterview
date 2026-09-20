# SGLang 05|HiCache:三层缓存与 HiSparse

KV 为什么要在被丢之前留一份副本、多级缓存的三条收益各有什么前提,vLLM 解读的「PagedAttention 与 KV Cache 管理」一章的多级缓存两节已经讲透。这一页只说 SGLang 遇到的问题、它的解法、代价、以及和 vLLM 不一样的地方。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

前缀树只用空闲显存(04 章)。多轮对话的上一轮回答是叶子,先被丢;agent 一次跑 10–100 步,每步的工具输出都是叶子;并发拉满时树被驱逐空。这些负载的命中率不是被算法卡死的,是被显存容量卡死的:一张 80 GB 的卡,权重占掉一半,剩下的槽位在 4096 并发下 2–5 分钟就轮换一遍,上一轮对话隔了 5 分钟再来,KV 早就没了,从头算 prefill。坏的指标是多轮和长上下文负载的 TTFT,而且是长尾:命中的那次 20–50 毫秒,没命中的那次 200–5000 毫秒。

同一集群的 8 个实例各自算同一份系统提示词,互相看不见,这是第二层浪费。

## 二、解法:树不变,节点多记两个地址

把显存当 L1,本机内存当 L2,外部存储当 L3,像 CPU 的三级缓存。树还是那棵树,每个节点除了显存格子号,再记一个内存格子号和一个「已备份」标记;L3 不记在树里,要用时按页哈希去问后端有没有。

- **先备份,后驱逐。** 一段 KV 插进树的时候(请求结束、或者一次 prefill 结束),默认就异步拷一份到内存;拷完再拷一份到 L3。等显存不够要丢叶子时,有备份的节点只释放显存格子,树上的节点还在,指着内存;没备份的节点才真删。
- **L3 只收已经在内存里的。** 显存到内存的那份拷完并确认之后,才把同一段按页哈希交给写出线程发往 L3;发之前先查一遍存在性缓存,L3 已经有的页不再写;MLA 模型各 rank 的 KV 完全一样,只有 rank 0 往外写。
- **匹配一次拿到三层的命中。** 新请求走树,得到「显存里命中多少、内存里还有多少」两个数;准入按两个数的和算预算。内存里那段在准入时整段拉回显存,要么全拉要么不拉。L3 那段在排队时就开始预取到内存,预取到哪算哪。
- **拉回和算重叠。** 从内存拉回显存按层做:第 N 层在算,第 N+1 层在传,注意力后端取第 N 层 KV 时才等第 N 层传完。
- **为什么实际不吃亏。** 一个 token 的 KV 是 256 字节到 2 KB,PCIe 拉 5000 个 token 是 1–10 毫秒,重算是 50–500 毫秒;而且拉回是异步的,GPU 在算别的批。写出去那份也是后台线程做的,不占 forward 时间。

## 三、代价

- **内存按倍数买断,而且是钉住的。** 默认内存池是显存池的 2 倍,启动就分配、注册成不可换页内存,每张卡各买一份;8 卡机器 30 GB 一份就是 240 GB。机器空闲内存不够直接启动失败。
- **默认策略下每段 KV 都写一遍。** 默认那条是插进树就拷,不管以后用不用;写内存的带宽和写 L3 的 I/O 是常态开销,不是偶发。
- **L3 的延迟进 TTFT。** 预取等不等、等多久是策略;等完最保险,但 L3 后端慢的时候 TTFT 跟着慢。
- **页变大,命中退到页边界。** L3 按页存取,文档所有示例都是 64 个 token 一页;树在任意 token 劈开的好处在 L3 这一层没有了。
- **预取占的是内存池的配额。** 排队请求预取进来的页在准入前一直锁在内存里,占用超过内存池的一半就不再发新预取;长队列下 L3 命中再高也拉不动。
- **多了 4 个后台线程和一串 TP 同步。** 预取命中数、预取完成数、写回完成数都要 TP 取最小值再决定,每次都是一次集合通信。

## 四、和 vLLM 不一样的七处

1. **什么时候写副本。** vLLM 的 connector 每个调度步末尾扫一遍新算完的块,主动拷到 CPU。SGLang 在 KV 插进树的那一刻拷,而且有三种策略(名字见第五节):默认是第 1 次插就拷;也可以等同一段被第 2 次插入才拷,只备份热数据;还可以拖到驱逐时才拷,内存满了就直接丢并打警告。
2. **谁知道副本在哪。** vLLM 的 block 池不知道 offload 存在,靠 connector 解耦。SGLang 反过来:树节点自己记内存地址和备份标记,驱逐逻辑按写策略分支,控制器归树管;能换的只有 L3 后端,实现 get、exists、set 三个方法就能接一个新的。
3. **粒度。** vLLM 把若干 block 合成一个 chunk 存 CPU。SGLang 内存池和显存池一样按 token 编号,L3 按页哈希,哈希链从请求的命名空间(LoRA、cache_salt)起算,页越大 I/O 越省、命中越粗。
4. **第三层。** vLLM 的 CPU 之下是本机文件系统,满了才换下去。SGLang 的 L3 是 12 种可选后端,内置的有本地文件、模拟器、Mooncake、HF3FS、NIXL、AIBrix、EIC、SiMM、UMBP、共享内存、昇腾 MemCache,加一个 dynamic 装自己写的类;分布式后端在实例之间共享,L2 永远是实例私有的。
5. **跨实例。** vLLM 的 CPU 层跟着进程走。SGLang 同一份系统提示词 A 实例算过,B 实例从 L3 拿;不同 TP 的部署也能共用,写 KV 时按最小公倍数切头;MLA 模型 8 个 rank 只有 rank 0 往 L3 写。
6. **拉回来的规矩。** vLLM 逐 chunk 查、能拿多少拿多少。SGLang 内存那段要么整段拉回要么放弃改重算;L3 那段分三种停法:best_effort 能算就停、wait_complete 等完、timeout 等 1 秒加每 1024 个 token 0.25 秒。
7. **step 内换页。** vLLM 那章说 offload 不是无限上下文的银弹,因为 full attention 每步要读全部 KV。SGLang 也一样,HiCache 只在请求之间换。真正在 step 内按需拉的是 HiSparse:DeepSeek 稀疏注意力每步只看 top-k 个 token,所以 GPU 只留一个每请求 4096–6144 个槽的小缓冲,全量 KV 放内存,每步按注意力分数把 top-k 那些换进来,上一个 token 的 KV 异步备份出去。它是另一套池,不建树,没有 L3,只对 DSA 和 DeepSeek V4 模型,文档说只在 PD 分离的 decode 实例上用。

## 五、调参与观测

**参数在哪调。** 三层的开关、内存池大小、布局、写策略、预取策略都是启动参数,改了要重启。L3 后端例外:`PUT /hicache/storage-backend` 挂、`DELETE` 卸、`GET` 查、`POST /hicache/storage-backend/clear` 清空,要求服务完全空闲(有请求在跑或在排队返回 400);DP 多副本要全部成功才算成功,部分成功不回滚,失败先卸再挂。后端自己的键(Mooncake 地址、预取阈值、超时系数)走 `--hicache-storage-backend-extra-config`,JSON 串或 `@文件`。HiSparse 是另一组参数,只在 decode 实例上给。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--enable-hierarchical-cache` · 打开三层缓存 | 启动 | 关 | 开:多轮、长上下文的 TTFT 长尾消失;内存按倍数被占 | 启动日志 `Allocating kv hierarchical KV host pool` |
| `--hicache-ratio` · 内存池是显存池的几倍 | 启动 | 2.0 | 调高:L2 命中率涨,但热数据存满后再涨没用;小于 1 只警告不报错 | 日志 `host pool ... smaller than the device pool`;`/metrics` 的 `hicache_host_used_tokens` 贴着 total 就该加 |
| `--hicache-size` · 内存池多少 GB,每张卡一份 | 启动 | 0(用 ratio) | 覆盖 ratio;8 卡写 30 就是 240 GB | 启动失败信息 `Not enough host memory available` 里有要多少、有多少 |
| `--hicache-write-policy` · 什么时候往内存写副本 | 启动 | write_through | selective:写带宽降,冷数据不备份,首次命中后才有;write_back:驱逐时才写,内存紧时会丢 | `hicache_dropped_tokens_total` 的 reason 标签;警告行 `write_back: KV subtree dropped` |
| `--hicache-mem-layout` · 内存池怎么排 | 启动 | page_first | page_first 一页连续,零拷贝进 L3;page_first_direct 给 direct 后端;layer_first 和显存一样,Mooncake 不接受;page_head 给异构 TP 切头 | 启动警告 `switching to ... layout` |
| `--hicache-io-backend` · 显存和内存之间用什么搬 | 启动 | kernel | kernel 是 GPU 辅助的搬运核,文档说快到 3 倍;direct 是逐页 memcpy,配 page_first_direct;两者和布局不配会被自动改 | 启动警告 `switching to direct io backend` |
| `--page-size` · 一页几个 token | 启动 | 1 | 开 L3 必须调,示例都用 64;越大 I/O 越省、命中退到页边界 | `cached_tokens` 是页的倍数 |
| `--hicache-storage-backend` · L3 用哪个后端 | 启动或运行时挂载 | 无 | file 默认写 `/tmp/hicache`,只在本机;mooncake、hf3fs、nixl、aibrix、eic、simm 跨实例 | `GET /hicache/storage-backend` |
| `--hicache-storage-prefetch-policy` · 预取什么时候停 | 启动或挂载时 | timeout | best_effort:TTFT 最低,命中最少;wait_complete:命中最高,L3 慢时 TTFT 跟着慢;timeout:折中 | 日志 `HiCache prefetch success/dropped ... completed= matched= loaded=` |
| `prefetch_threshold` · L3 命中多少 token 才值得预取 | extra_config | 256,不低于一页 | 调低:短命中也去拉,L3 压力大;调高:只拉长前缀 | 预取日志里 revoked 的次数 |
| `prefetch_timeout_base` 与 `prefetch_timeout_per_ki_token` · 超时公式 | extra_config | 1 秒、0.25 秒每 1024 token,无上限 | 调高:多等一会换命中;文档写的 2 秒、0.1 秒、上限 30 秒是老实现的值 | timeout 策略下 completed 与 matched 的差 |
| `--hicache-storage-prefetch-retry-*` · 预取没命中时再查几次 | 启动 | 每 8 个调度轮再查、最多 8 次 | 备份还在写时第 1 次查会漏;设 0 关重试 | 警告行 `storage prefetch reissue cap reached` |
| `--hicache-host-memory-mode` · 内存是缓存层还是中转站 | 启动 | cache | buffer_only:内存只做 L3 的中转,ratio 默认 1.2,要有 L3,不能 write_back,不能在 decode 实例 | 启动校验 |
| `--disaggregation-decode-enable-offload-kvcache` · PD 的 decode 端把生成的 KV 也写 L3 | 启动 | 关 | 开了多轮对话下 prefill 端能从 L3 拿到上一轮的回答;要有 L3;自己一套内存池 | decode 端日志 `Enable offload kv cache for decode side` |
| `--enable-hisparse` 与 `--hisparse-config` · 稀疏注意力的 GPU 小缓冲 | decode 启动 | 关;top_k 2048、缓冲 2 倍 top_k、内存 2 倍、核块 960 | 内存倍数按机器给:1 TB 给 5,2 TB 给 10;必须同时 `--disable-radix-cache` | 错误行 `HiSparse: host mem pool alloc failed` |

**怎么看。** 响应 `meta_info` 里的 `cached_tokens_details` 分 `device`、`host`、`storage` 三个数,开了 L3 还带 `storage_backend`;OpenAI 接口要 `--enable-cache-report` 才在 `usage.prompt_tokens_details` 里给。日志 `Prefill batch` 行的 `#cached-token` 是三层合计,分层只在 `/metrics`:`hicache_host_used_tokens` 对 `hicache_host_total_tokens` 看 L2 满没满,`load_back_tokens_total` 和 `load_back_duration_seconds` 看拉回,`hicache_backup_tokens_total` 看写出,`hicache_dropped_tokens_total` 按 reason 看丢了多少没备份的,`storage_prefetch_hit_tokens_total` 对 `storage_prefetch_unfulfilled_tokens_total` 看 L3 预取兑现了多少。每次预取一行 `HiCache prefetch success` 或 `dropped`,里面 `occupied` 超过内存池一半就不再发新预取。

| 症状 | 先查 | 然后 |
|---|---|---|
| 启动报 `Not enough host memory available` | 报错里的要求量是每张卡一份,总量按卡数乘 | 降 ratio 或 size;空闲内存减 1 GB 保留再除以本机 rank 数才是预算 |
| `cached_tokens_details.host` 一直是 0 | 写策略是不是 selective(第 2 次才备份);内存池是否小于显存池 | 换 write_through;ratio 调到 2 以上 |
| 开了 L3 之后 TTFT 反而变长 | 预取策略是不是 wait_complete;L3 后端带宽 | 换 timeout;看 `prefetch_bandwidth` |
| 预取日志全是 dropped 或 revoked | 命中长度是否够 `prefetch_threshold`;`occupied` 是否到了池的一半 | 调低阈值;加内存 |
| 两个实例互相看不见对方的 KV | 后端是不是 file(默认 `/tmp/hicache` 只在本机);model_name、TP 是否一致 | 换分布式后端;异构 TP 配 `tp_lcm_size` |
| 挂载 L3 返回 400 | `#queue-req` 和 `#running-req` 是否归零 | 先断上游流量等空闲 |
| 启动日志里布局或 IO 后端被改了 | page_first 配了 direct,或 page_first_direct 配了 kernel | 按警告里的组合改配置,别再猜 |
| 警告 `write_back: KV subtree dropped` | 内存池太小,驱逐时来不及备份 | 加 ratio;或改回 write_through |
| MLA 模型 8 卡只有 1 张卡在写 L3 | 正常,KV 各 rank 相同,只有 rank 0 写 | 不用管 |
| PD 分离下多轮对话每轮都全量 prefill | decode 端有没有开 offload;有没有 L3 | 两个都要 |
| HiSparse 报 `host mem pool alloc failed` | `host_to_device_ratio` 按机器内存给了没 | 1 TB 给 5,2 TB 给 10 |

## 六、常见误区

- **ratio 调大一点,几台机器的内存就能凑成一个大 L2。** L2 是实例私有的,同一台机器上的两个实例也互不可见;ratio 只放大自己那份。跨实例只能靠 L3,而且 file 后端默认写在本机 `/tmp`,跨不了机器。
- **开了 HiCache 就是驱逐时才备份,像磁盘缓存那样。** 默认是插进树就拷,驱逐那一刻 GPU 什么都不做。只有 write_back 才是驱逐时拷,而这个模式下内存一满就直接丢子树,警告行很容易被当普通日志略过。
- **文档上的超时公式:2 秒加每 1024 token 0.1 秒,上限 30 秒。** 这是老树的实现,当前默认走统一树,是 1 秒加每 1024 token 0.25 秒,没有上限。extra_config 里的键名一样,填的值要按新公式想。
- **page_size 留 1 也能开 L3。** 阈值取 256 和一页的大者,页是 1 就是每个 token 一个哈希一个对象,I/O 慢到不可用;文档所有示例都写 64。开 L3 的第一件事是改页。
- **看 `#cached-token` 就知道 HiCache 有没有用。** 那是三层合计,树本来就有命中。要看 `cached_tokens_details.host` 和 `storage` 两个数,或 `/metrics` 里的 load_back 和 prefetch 计数。
- **配了 kernel 加 page_first_direct,以为两个都生效了。** 启动时会被改成 direct,只打一行警告。同理 direct 加 page_first 会被改成 page_first_direct,Mooncake 加 layer_first 会被改掉。看启动日志,别看启动命令。
- **wait_complete 最稳,生产就用它。** 它把 L3 的延迟全放进 TTFT,后端一抖动 TTFT 就抖。生产建议 timeout。
- **卸载 L3 后端会清掉里面的数据。** 只是停止使用和停后台线程,Mooncake、HF3FS 里的数据还在。清空是另一个端点,要 admin key。
- **有流量时也能挂卸 L3。** 必须完全空闲,否则 400;DP 下部分副本成功了也算失败,而且不回滚,要先卸再挂。
- **HiSparse 是 HiCache 的一种模式。** 不是。它自己一套显存池和内存池,不建树,没有 L3,反而要求关掉前缀树。HiCache 解决的是「请求之间」的复用,HiSparse 解决的是「一个请求内每步只需要 top-k」的 decode 显存;两个都开,各管各的。
- **开了 HiCache,100K 上下文的 decode 就能省显存。** 不能,full attention 每步读全部 KV,HiCache 只在请求之间换。能省的只有 DSA 模型上的 HiSparse。
- **换了权重之后 L3 里的旧 KV 会自己失效。** L3 的键是页哈希加模型名,同名模型换权重后键不变,`/flush_cache` 也不碰 L3。RL 训推一体要顺手调清空端点。
- **MLA 模型 8 卡只有 1 张卡在写 L3,以为坏了。** MLA 的 KV 各 rank 完全相同,写 8 份是浪费;rank 0 写是设计。

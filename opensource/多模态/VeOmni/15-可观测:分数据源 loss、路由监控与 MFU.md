# 可观测:分数据源 loss、路由监控与 MFU

这一页回答一件事:总 loss 看上去平稳时,怎么看出是哪个数据源在退化、哪一层的专家被挤爆、算力到底用上了几成,而且不改训练本身。说法都在源码基准 `019b1c0276`(tag `v0.1.12`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

**总 loss 是按 token 加权的平均,小数据源的退化被大数据源淹没。** 多源混训时,一个只占 5% token 的代码源,loss 涨了 0.3,总 loss 只动 0.015,落在每步的正常抖动里。要等下游评测才发现代码能力掉了。要早点看到它,得把 loss 按来源拆开记。

**拆开记恰好卡在三处。** 第一,打包之后一个微批只剩一行,几条来自不同来源的样本首尾相接(07 章),loss 是对整行一次算完的。第二,融合交叉熵那条路根本不物化 logits,从隐状态直接出一个标量,想要逐 token 的数都没有。第三,开了序列并行,这一行又被切到几张卡上,一条样本可能前半段在 rank 0、后半段在 rank 1。而模型的前向是生成的 HF 代码(02 章),不可能为了一个监控指标去改每个模型的前向与返回值。

**MoE 的负载失衡不报错。** 路由把 token 集中送给少数专家,热门专家所在的卡算得久、激活占得多,整个专家并行组等它(05 章)。loss 可能完全正常,只是每步慢、显存偏,没人数过每层每个专家分到多少 token,就没人知道。

**光看每秒 token 数判断不了卡用得好不好。** 同样的 token 数,换一个模型、换一种序列长度,计算量差得很远:注意力的计算量跟每条样本长度的平方走,而打包后那一行的长度不是样本长度。要按模型结构和真实样本长度估出这一步做了多少浮点运算,再除以硬件峰值,才能跨作业比。

## 二、解法

最天然的直觉:**不碰训练路径,在已有的口子上旁听,另记一本账。** 三件事都是这个做法:分数据源 loss 旁听 loss 的入口,路由监控旁听路由器的输出,MFU 旁听每一步的样本长度与耗时。

![按数据源拆 loss 的四步:collator 交出的一行里 3 条样本首尾相接,位置编号在每条样本开头归零,批里另有一列按拼接顺序排好的来源编号 A、B、A;开序列并行时这一行从中间切给两张卡,rank 1 那一段开头没有位置 0,是样本 2 的续段;每张卡在自己那一段上旁路算逐 token 交叉熵并按段求 loss 和与 token 数,在序列并行组内收集后按行、rank、段序排好,续段并进前一个起始段,拼回 3 个样本,与来源编号按顺序对上;最后按来源累加,跨微批累加后在数据并行组求和,每个来源的 loss 等于 loss 和除以 token 数](/opensource/VeOmni/15a-channel-loss-segments.svg)

**分数据源 loss 挂在 loss 的入口上,不挂在前向里。** 训练开始时,旁路记账器(ChannelLossComputer)剥开模型运行时、LoRA 包装与 Omni 的外壳,找到真正调用 loss 函数的那个模块,把它的 loss 函数换成一层包装:先原样调用、原样返回,再拿同一组输入另算一遍。走融合交叉熵的那条路,它把交叉熵空位(03 章)背后的实现换成一个分发器,同样是先调原实现、再通知当前正在旁听的记账器。旁听开关是一个上下文变量,只在采样步的模型前向期间打开,所以同一进程里的其他模型不会被记进来(DPO 的参考模型就是这样被排除的,见 14 章)。

**另算的那一遍在不求梯度的上下文里做。** 手上有 logits,就按每 512 个 token 一块、升到 fp32 算逐 token 交叉熵;融合那条路没有 logits,就拿隐状态和输出头权重,同样每 512 个 token 一块重做输出头投影再算。主 loss 与梯度一个字节都不变。

**切回样本靠位置编号,对上来源靠顺序。** 打包时每条样本的位置编号从 0 起(07 章),所以编号为 0 的地方就是样本起点。collator 同时在批里留了一列来源编号,每条样本一个,按拼接顺序排。记账器把一行切成段,第 i 段就归第 i 个来源编号。只统计标签不是忽略值的 token。补到固定长度时尾部 padding 也会形成一个位置从 0 起的段,只有能无歧义地认定它是纯 padding 时才丢掉。段数和来源数对不上,默认跳过这个微批并告警;打开严格模式则直接报错。

**序列并行下多一步拼接。** 每张卡只看得到自己那一段,开头不是 0 的就是上一张卡某条样本的续段。每张卡先对自己的段各求一个 loss 和与 token 数,然后在序列并行组内收集:段的元数据用对象收集,loss 和与 token 数分两个紧凑张量收集;按(行、rank、段内顺序)排好,续段加到前一个起始段上,拼回按样本的结果。

**口径是这一步内的 token 平均。** 每个来源的 loss 和与 token 数先在本卡跨微批累加,步末在数据并行组各求一次和,再相除。所以它是「这一步里、全体数据并行 rank 上、这个来源所有被监督 token 的平均交叉熵」,不跨步平滑。另外报两样:这个来源的 loss 和除以这一步所有被记到的 token 数(各来源加起来就是整体平均),以及这个来源的 token 数。

**存进 checkpoint 的只有来源编号到名字的登记表。** 指标名由来源编号的稳定编码加名字拼成,登记表保证续训前后同一个来源还是同一条曲线;它跟着作业侧的每 rank 状态文件走,目录见 13 章。

**路由监控在路由器上挂前向钩子,只数不算。** 训练开始时遍历模型,认得的路由器类各挂一个钩子,从路由器输出里取 top-k 专家编号,在卡上按专家计数,不同步。DeepSeek-V3 的 top-k 在路由器之后的补丁代码里算,补丁里多写了一行显式上报。层的顺序在挂钩时定死,热力图的行序续训前后不变。每隔 N 步,把各层计数叠成「层 × 专家」的矩阵,在 FSDP 组(数据并行乘序列并行,恰好是持有不同 token 的全部 rank)上求和,搬回 CPU,每行归一成比例,然后清零开始下一个窗口。

**失衡用「偏离均匀分布多少」来读。** 把每个专家的份额乘以专家数再减 1:均匀路由时全是 0;每层最大值是最热专家超出平均的倍数,1.0 就是它拿了平均的 2 倍;每层最小值落在 −1 到 0,−1 说明有专家一个 token 都没分到;再加一个每层的平均绝对偏离。各层再取最大与平均,配一张「层 × 专家」的热力图。

**MFU 是「估出来的 FLOPs ÷ 实际耗时 ÷ 峰值」。** 吞吐计量器(EnvironMeter)在每步开始时从每个微批取出真实样本长度(打包边界减去尾部 padding),多模态再从网格尺寸取出视觉 token 数。步末按模型类型查一张登记表,取出这个模型的公式。以 Llama、Qwen2 这类稠密模型为例:线性层部分是 6 × 参数量 × token 数,注意力部分按每条样本长度的平方累加;视觉语言模型的视觉塔单独算。除以本卡这一步的耗时得到每秒 FLOPs,在数据并行组求和;分母是按设备名查出的 BF16 稠密峰值乘以总卡数。开序列并行时,同组每张卡拿到的是完整的样本长度,只在数据并行组求和,每个 token 正好算一次。耗时从吞吐计量器的步开始算到它的步结束,取数据的等待不在里面。

**同一处还报两个长度。** 有效长度 = 这一步真实 token 总数 ÷ 全局批大小,动态批下每个批位是一份「微批 × 最大序列长度」的 token 预算,拿它比最大序列长度就是预算装满了几成;样本长度 = 真实 token 总数 ÷ 真实样本条数。

**模型也可以自己上报指标。** 前向可以在输出里带一个辅助指标字典,它和 loss 分两个口袋走:loss 已经按 token 占比缩放过,跨微批相加;辅助指标跨微批取平均,再在 FSDP 组上平均。两者最后都以同一个前缀发布,所以名字和本步的 loss 或回调自己发布的几个名字撞了,第一次前向就报错,而不是悄悄互相覆盖。当前树内只有 DeepSeek-V4 用它上报索引器的 KL。

**所有指标汇到训练器上的一个字典,由回调顺序决定谁先谁后**(回调顺序见 01 章)。分数据源 loss 排在吞吐计量器之后、wandb 之前,因为计量器每步会重建指标字典,wandb 要在它补完之后才上报。

## 三、代价

**采样步要多付一遍交叉熵。** 融合那条路等于在前向里把整行的输出头投影多做一遍;序列并行下每个采样步还要在组内做一次对象收集加两次张量收集,数据并行组再做一次对象收集和两次求和,对象收集要在 CPU 上序列化。所以默认每 10 步才采一次,中间的步没有数。

**一步一个点,小来源很吵。** 某个来源在这一步没被抽到就没有点;抽到的 token 少,这一步的平均就抖。口径不跨步平滑,要看趋势得自己在看板上平滑。

**来源编号会被吞吐计量器先拿走。** 多源 YAML 给每条样本打上的来源编号与名字(07 章),吞吐计量器在步开始时要用它们统计各源消耗,取的方式是从批里直接弹出;而它在回调列表里排在分数据源 loss 前面。结果是分数据源 loss 读元数据时,默认键表里能匹配上的那两个键已经不在了:默认模式下这一步静默不记,一行告警都没有;严格模式下第一步就报「找不到来源编号」。树里的多源测试专门把自己的检查回调挪到计量器前面,注释写的就是这个原因;分数据源 loss 自己的测试是直接构造批,没有经过计量器。

**只认因果语言模型。** 扩散训练器(09 章)、分类目标、RL 训练器在构造时就拒绝打开它。

**认不出来的一律报 0 或不报,而且多半不说。** 这是旁听做法的通病:它只认登记过的东西。FLOPs 公式只登记了 18 个模型类型,两代 Omni、Gemma3、GLM-MoE-DSA 与所有扩散模型不在表里,实际 FLOPs 静默记 0;峰值表只认名字里带特定型号的卡,表外的卡(包括 AMD 与寒武纪)峰值记成无穷大,MFU 同样是 0;表里 H20 与 910B 的峰值,源码注释明说不符合其余型号的那套换算。路由监控只登记了 4 个路由器类名,其中 Qwen3-VL MoE 与 Qwen3-Omni MoE 登记的类名在树里不存在(生成文件里的类名多了 Text、ThinkerText),挂钩数为 0 就关掉;它还要求模型配置顶层有专家数字段,按仓库钉死的 transformers 5.16.1,DeepSeek-V3 的配置里这个字段叫别的名字。当前真正能打开路由监控的只有 Qwen3-MoE 文本模型。

**路由监控不开 wandb 就是白算。** 每到间隔,所有 rank 都要参加一次求和,每个 rank 都会用 matplotlib 画一张热力图;但只有 rank 0 且开着 wandb 才上报,连那行汇总日志也在这个判断之后,关着 wandb 时结果算完即扔。

**评估回调是空的。** 按步、按轮的触发时机都在,真正做评估的函数体只有一行待实现。

**观测本身会改运行行为。** 定期清显存缓存与手动垃圾回收挂在吞吐计量器上:默认每 500 步清一次缓存;回收间隔大于 0 时,Python 的自动垃圾回收整个被关掉,只在每 500 步手动收一次。

## 四、和同类大厂框架不一样的地方

对照对象是 Megatron-LM(基准 fb6a123a09)。

两边都要回答「哪个数据、哪些专家、多少算力」,落点不同:VeOmni 在训练步里旁听,Megatron 把前两件分别交给评估流程和落盘文件,第三件只报绝对吞吐。

| 比较维度 | VeOmni | Megatron-LM |
|---|---|---|
| 按数据源看 loss | 训练步本身按来源拆,采样步上旁路重算交叉熵,跨卡拼回样本后按来源求 token 平均 | 训练 loss 不拆;验证时可以把多个验证集各自单独评一遍,每个集一条 loss,集合不能带权重 |
| 专家负载 | 前向钩子数每层每专家的 token,按间隔在持有不同 token 的全部 rank 上求和,归一后报偏离度与热力图 | 每层记辅助损失与 z-loss 的数值,可以按层上报;每专家 token 数按间隔由钩子写成每个 rank 一个 JSON Lines 文件,放在存档目录下,不做跨 rank 汇总 |
| 算力利用 | 按模型类型各登记一个 FLOPs 公式,除以内置的设备峰值表报 MFU | 一个由架构参数推出的解析公式覆盖所有结构,报每卡每秒 TFLOP,不除峰值 |

Megatron 的模型层是自己写的,一个解析公式读架构参数就能覆盖稠密、MoE 与多 token 预测,代价是它只给绝对吞吐,换卡型要自己换算。VeOmni 的模型来自 HF,结构千差万别,只能一个模型类型登记一个公式,未登记的就是 0。分数据源 loss 也是同一个取舍:Megatron 用独立的验证集回答「哪个数据退化了」,干净但要额外跑评估;VeOmni 直接在训练 batch 上看,零额外评估,换来的是采样稀疏、依赖批里的元数据契约。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `train.channel_loss.enable` · 分数据源 loss | YAML 或命令行 | 关 | 开:训练开始时包装 loss 入口;每个微批在送进模型前剥掉来源元数据;扩散、分类、RL 训练器直接报错 | `Channel loss: wrapped …loss_function` 或 `installed dispatcher for …veomni_causal_lm_loss` |
| `train.channel_loss.interval` · 采样间隔 | YAML 或命令行 | 10 | 调小:点更密,采样步多一遍交叉熵;1 是每步都算;小于 1 启动即报错 | 采样步才有 `channel_loss/*` |
| `train.channel_loss.source_id_keys` · 来源编号从哪个键读 | YAML | `channel_id`、`source_id`、`dataset_id`、`ds_idx` | 按顺序取第一个存在的键,值须每条样本一个 | 严格模式下找不到时的报错会列出查过的键 |
| `train.channel_loss.source_name_keys` · 显示名从哪个键读 | YAML | `channel_name`、`source_name`、`dataset_name`、`data_name` | 找不到名字时退回多源 YAML 里的名字,再退回 `source_<编号>` | 指标名 `<前缀>/source-i-<编号>__<名字>` |
| `train.channel_loss.extra_strip_keys` · 额外剥掉的键 | YAML | `cur_token_num` | 列进来的键在前向前从批里删掉,模型收不到它 | 前向报「意外的关键字参数」时加进来 |
| `train.channel_loss.strict` · 严格模式 | YAML 或命令行 | 关 | 开:任一步找不到来源编号、段数对不上、同一编号在各 rank 名字不一致,都直接报错;关:跳过并告警 | `Channel loss: source metadata count (…) does not match …` |
| `train.channel_loss.log_weighted_loss` / `log_token_count` | YAML | 都开 | 关掉就不报加权 loss 与 token 数 | `channel_loss_weighted/*`、`channel_tokens/*` |
| `train.channel_loss.loss_metric_prefix` 等三个前缀 | YAML | `channel_loss`、`channel_loss_weighted`、`channel_tokens` | 只改指标名 | 看板上的分组 |
| `train.moe_load_balance_monitor_interval` · 路由监控间隔 | YAML 或命令行 | 0(关) | 大于 0:每 N 步全局求和一次并清零;窗口越长越平滑 | `MoE router monitor: attached to N router module(s).` |
| `train.wandb.enable` / `project` / `name` / `id` | YAML 或命令行 | 关 / `VeOmni` / 不设 / 不设 | 开:rank 0 每步上报整张指标表,并把全部配置作为 run 配置;给了 `id` 就按续跑接上同一个 run | wandb 看板 |
| `train.profile.enable` · profiler | YAML 或命令行 | 关 | 开:训练开始即启动 profiler | `build profiler schedule - wait: …, warmup: …, active: …` |
| `train.profile.start_step` / `end_step` · 抓取区间 | YAML 或命令行 | 1 / 2 | 抓 `[start, end)` 这几步;起点大于 1 时前面留一步预热 | `Profiling result saved at …` |
| `train.profile.trace_dir` | YAML 或命令行 | `./trace` | 也可以是 `hdfs://`,先写本地缓存再拷上去 | 目录下的 `veomni_rank<R>_<时间戳>.pt.trace.json.gz` 与 `.pkl` |
| `train.profile.record_shapes` / `profile_memory` / `with_stack` / `with_modules` | YAML | 开 / 开 / 开 / 关 | 关掉前三个 trace 更小;`profile_memory` 同时决定记不记显存分配历史,显存快照就来自它 | trace 大小;`Profiling memory visualization saved at …` |
| `train.profile.rank0_only` | YAML 或命令行 | 开 | 关:每个 rank 都抓,文件数等于卡数,会打一行告警 | trace 目录 |
| `train.empty_cache_steps` | YAML 或命令行 | 500 | 每 N 步清一次显存缓存;非正数不清 | 每 N 步一次的耗时尖刺 |
| `train.gc_steps` | YAML 或命令行 | 500 | 大于 0:关掉 Python 自动垃圾回收,每 N 步手动收一次;非正数恢复自动回收 | 主机内存曲线 `cpu_used_memory(GB)` |
| `train.eval_steps` / `eval_epochs` | YAML 或命令行 | 0 / 1 | 当前不产生任何效果,评估函数是空的 | 无 |

**观测。** 本章看 4 处。一是指标表本身,下表是每个名字怎么来、跨卡怎么归约;它只在 wandb 里完整出现,文本日志里没有逐步的指标行。二是进度条:每个节点的 local rank 0 上,进度条后缀只列 `training/*` 与学习率,保留两位小数,所以学习率常显示成 `0.00`;分数据源 loss 在进度条之后才写进字典,进度条上看不到。三是路由监控:rank 0 且开着 wandb 时每个间隔打一行 `Step N: uploaded MoE load balance heatmap (steps a-b), max_vio max=… avg=…, …`,看 `max_vio max` 找最挤的那一层。四是 profiler 产物:chrome trace 用 Perfetto 打开,`.pkl` 是 PyTorch 的显存快照(读法见知识库「性能分析与Profiling」)。

| 指标 | 怎么来 | 跨卡怎么归约 |
|---|---|---|
| `training/total_loss`、`training/<各 loss>` | 本卡各微批之和,每个微批已按全局 token 占比缩放 | FSDP 组平均,结果是全局 token 平均 |
| `training/<辅助指标>` | 模型上报,各微批平均 | FSDP 组平均 |
| `training/grad_norm` | 裁剪时算出的范数 | FSDP 组平均 |
| `training/lr` | 各子调度器取最大 | 不归约 |
| `training/avg_effective_len`、`training/avg_sample_seq_len` | 真实 token 总数 ÷ 全局批大小、÷ 真实样本数 | token 与样本数在数据并行组求和 |
| `flops_achieved(T)`、`flops_promised(T)`、`mfu` | 本卡估算 FLOPs ÷ 本卡步耗时 | 实际值在数据并行组求和,峰值乘总卡数 |
| `tokens_per_second(M)`、`consume_tokens(M)`、`consume_tokens(B)`、`consumed_chunk_num` | 全作业,后三个是累计值并随 checkpoint 保存 | 数据并行组求和 |
| `max_memory_allocated(GB)`、`max_memory_reserved(GB)`、`num_alloc_retries` | 历史峰值,含义见 10 章 | 全体 rank 取最大 |
| `cpu_used_memory(GB)` 等三项 | rank 0 所在机器 | 不归约 |
| `multi_source/*` | 多源模式下按源名的消耗 token、样本数、本步占比 | 数据并行组收集后相加 |
| `channel_loss/*`、`channel_loss_weighted/*`、`channel_tokens/*` | 采样步上按来源的 loss 和 ÷ token 数 | 序列并行组内拼回样本,数据并行组对和与数各求和后再除 |
| `moe/max_vio/*`、`moe/min_vio/*`、`moe/avg_vio/*`、`moe/expert_load_heatmap` | 间隔窗口内的每层每专家计数,归一后算偏离 | FSDP 组求和 |

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| Qwen3-MoE 调负载均衡,想看每层专家分布 | `train.moe_load_balance_monitor_interval: 50`、`train.wandb.enable: true` | 用每 50 步一次全局求和与一张热力图,换到按层的最大偏离曲线;窗口 50 步抹掉单步噪声 |
| 单机 8 卡抓一段稳定步的性能 trace | `train.profile.enable: true`、`train.profile.start_step: 5`、`train.profile.end_step: 7`,其余保持默认 | 抓第 5、6 两步,前面留一步预热,避开首步的编译与分配;只在 rank 0 抓,文件小但看不到别的卡 |
| 验证分数据源 loss 的元数据能不能到达 | `train.channel_loss.enable: true`、`train.channel_loss.strict: true`、`train.channel_loss.interval: 1`,跑 20 步 | 用每步多一遍交叉熵与「一有问题就停」换掉默认的静默跳过;确认有 `channel_loss/*` 后再把间隔调回 10、关掉严格模式 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 开了分数据源 loss,看板上一条 `channel_loss/*` 都没有 | 数据是不是多源 YAML;打开严格模式看是否报找不到来源编号 | 来源编号被吞吐计量器先取走了,当前没有配置能绕开;要改代码,让分数据源 loss 在计量器弹出之前读到元数据 |
| 告警 `source metadata count (…) does not match packed segment count (…)` | 是否开了补齐到固定长度;数据是否每条样本一个编号 | 尾部 padding 认不出时会多一段;编号是每行一个而不是每条样本一个时也会对不上 |
| `mfu` 一直是 0 | 模型类型是否在 FLOPs 登记表里;设备名是否在峰值表里;是否开着 LoRA(11 章) | 只影响上报;看吞吐改看 `tokens_per_second(M)` 与每步耗时 |
| 日志 `model config has no 'num_experts'` 或 `no recognized router modules found` | 模型族与路由器类名 | 当前只有 Qwen3-MoE 文本模型能挂上;别的模型要在监控模块里登记提取函数 |
| 开了路由监控但 wandb 上没有热力图 | `train.wandb.enable`;是否 rank 0 | 打开 wandb;不开时结果被丢弃 |
| profiler 目录是空的 | 是否续训;`end_step` 是否小于续训起点的全局步 | 区间按全局步号比较,续训后要把两端设到续训之后 |
| 主机内存缓慢上涨后每 500 步回落 | `train.gc_steps` | 这是关掉自动回收后的正常形态;内存紧就调小或设成 0 |

## 六、常见误区

- **以为多源 YAML 加上 `train.channel_loss.enable` 就能看到各源 loss。** 会这么以为,是因为 07 章讲的来源编号就在批里,默认键表也列了 `ds_idx`。实际上吞吐计量器在同一个步开始回调里更早把 `ds_idx` 与 `source_name` 从批里弹走了,分数据源 loss 读不到;默认模式下连告警都没有,只是看板上永远空着。
- **以为 `channel_loss_weighted/*` 加起来应该等于 `training/total_loss`。** 会这么以为,是因为它的定义是来源 loss 和除以这一步全部被记到的 token 数,各来源加起来正好是一个整体平均。实际上它只覆盖采样步、只覆盖成功对上元数据的微批,交叉熵是另算的 fp32;主 loss 走的可能是融合算子、还叠着辅助损失。两者差一点是常态,差得多才要查段数告警。
- **以为 `mfu` 为 0 就是卡没在算。** 会这么以为,是因为其他框架的利用率掉到 0 通常意味着挂住了。实际上这里 0 最常见的原因是模型类型不在 FLOPs 登记表里(Omni、Gemma3、扩散模型都不在),或者卡的名字不在峰值表里,两种情况都不打任何日志。
- **以为 `tokens_per_second(M)` 是每卡吞吐。** 它是全作业的:分子在数据并行组上求了和,单位是百万。拿它和别的框架报的每卡数字比,要先除以数据并行度;而且它的耗时不含等数据,dataloader 卡住时它看上去仍然很好。
- **以为设了 `train.moe_load_balance_monitor_interval` 就有路由热力图。** 会这么以为,是因为参数说明写着「全局热力图」。实际上它要同时满足三件事:模型配置顶层有专家数字段、路由器类名在登记表里、开着 wandb。按当前代码,Qwen3-VL MoE 与 Qwen3-Omni MoE 登记的类名对不上生成文件,DeepSeek-V3 的配置字段不叫这个名字,都会打一行告警后关掉。
- **以为 `max_vio` 接近 0 就说明没有失衡。** 它是间隔窗口内的累计比例,窗口里多步的冷热互相抵消,单步的尖峰会被抹平;要看瞬时失衡,把间隔调小,或者直接看专家并行组里最慢的那张卡(05 章)。
- **以为 `train.eval_steps` 会触发评估。** 触发时机写好了,评估函数是空的,设多少都不会跑验证集,也不会有任何评估指标。
- **以为 profiler 默认抓的是有代表性的一步。** 默认区间是第 1 步,正好是首步的编译、分配与缓存预热;而且区间按全局步号比较,从 checkpoint 续训后起点早已越过,profiler 启动了却永远不会导出。
- **以为 `train.gc_steps` 只是多加一次定期回收。** 大于 0 时它先关掉 Python 的自动垃圾回收,默认 500 就是关着的;循环引用的对象要等到每 500 步才被收走,排查主机内存「泄漏」时先看这一项。

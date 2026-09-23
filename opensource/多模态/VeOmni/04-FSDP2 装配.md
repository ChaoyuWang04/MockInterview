# FSDP2 装配

这一页回答一件事:一个单卡放不下的模型,怎样做到从建出来、切开到装上权重,每张卡上始终只出现自己那一份。说法都在源码基准 `019b1c0276`(tag `v0.1.12`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

按最自然的写法,每个进程先照配置把整个模型建出来,再去调分片。模型一大,进程死在构造那一行:一个 200B 参数的 MoE,FP32 主权重就是 800 GB,一张 80 GB 的卡连它的 1/10 都放不下——分片还没开始。N 个进程各建一份,主机内存也一样顶不住。

就算躲过了建模这一关,**装权重是第二个峰值**。权重文件里存的是完整张量,读进来之后才能切;切之前,每张卡都得先拿着这个完整张量。一层融合后的专家权重本身就是一个大张量,读到它的那一刻,显存会在「本卡分片」之上再冲一截。让每个 rank 自己去读,共享存储又要被读 N 遍。

续训再叠一层。续训的参数最终来自分布式 checkpoint,可按正常流程,模型会先把 HF 初始权重完整装一遍,紧接着被 checkpoint 整个覆盖——白装一次,显存二次冲顶,大 MoE 恰好就 OOM 在这一步。

还有一个不报错的坑。先建空壳、再分配真实内存的做法,只拦截了参数的注册;缓冲区(旋转位置编码的频率表之类)在建模时已经是真值。分配内存那一步会把**所有**张量换成未初始化的内存,而分布式 checkpoint 不存非持久缓冲区,HF 的初始化函数也只重算一种形状的频率表。Gemma3 按层类型分开的频率表、它的嵌入缩放系数、Omni 音频塔的正弦位置编码,就这样变成了垃圾值:训练照跑,loss 就是对不上。

## 二、解法

最天然的直觉:**先定下每张卡拿哪一段,再让每张卡只为那一段分配、只往那一段装数。** VeOmni 把它固定成模型运行时里「切分并装载权重」那一步的构建顺序(01 章)。

![一次构建的六个步骤从左到右排开:元设备建空壳、先切额外并行、逐层分片、根模块分片、物化、装载权重;下面三行是同一张卡在每一步之后的显存。首次训练那一行前四步都是 0 字节,物化后出现 1/N 的绿色分片,装载时在分片之外多出一块红色的完整张量,那是峰值所在;续训那一行物化时不写随机初值,装载一步整格变灰,表示跳过 HF 装载、改由 DCP 在训练开始时按分片读回;对照那一行在第一步就是一整条红色的完整模型,长度是绿色分片的 N 倍](/opensource/VeOmni/04a-build-order-memory.svg)

**前四步只动形状。** 在 FSDP2 路径上,模型一律在元设备上建(为什么先切后填是成立的,见 FSDP 解读 05 章)。开了额外并行,先按模型自报的计划把专家这类参数切一刀,细节见 05 章。接着逐层调分片入口:**分组依据来自模型自己**——生成的 modeling 里声明的「不可拆分模块」类名(通常就是解码层),再并上用户追加的类名;凡是类名命中的模块实例各成一组,按嵌套关系排成子模块在前。根模块最后兜底,收下剩下的嵌入、输出头与末层 norm。

**根模块故意不指定前向后要不要释放。** 这样 FSDP2 对根的默认生效:根的参数在前向后留着,直到反向(见 FSDP 解读 06 章)。源码注释写明了理由:按块融合的 logprob 与蒸馏算子把输出头权重存进反向上下文,若前向后释放,反向会读到一块已经归还的空存储。解码层则按配置在前向后释放。

**混合精度在建模时就定了。** 开着混合精度(默认),模型直接以 FP32 建出来,这就是主权重;分片时挂上精度策略:计算用 BF16,梯度归约用 FP32(FSDP2 的精度策略本身见 FSDP 解读 08 章)。模型可以声明个别模块豁免精度策略,这些模块单独成组、保持全精度,且前向后不释放。重计算与按块编译都在分片之前挂上,开关与代价分别归 10 章和 03 章。

**物化只为本卡那一段分配。** 物化前先把每个非持久缓冲区抄一份,物化后原样写回,专治上一节那个坑;缓冲区若本身也在元设备上(由参数派生),没有东西可抄,只打一行警告。打开 FSDP 卸载时,物化落在主机内存,之后再把缓冲区单独搬回加速卡——卸载策略只管参数、梯度和优化器状态,不管缓冲区。

**装载有三条路**,差别在谁读盘、谁把数据送到各卡、每张卡临时多出什么。三条路都逐个张量处理,读进来先交给模型的权重转换器(02 章)换成训练布局,再切下本卡那一段写进去:

| 哪条路 | 谁读盘 | 怎么到各卡 | 每张卡临时多出什么 |
|---|---|---|---|
| 0 号读、广播(默认) | 只有 0 号 rank | 逐个张量广播完整值,各卡自己切 | 一个完整张量;属于额外并行组、又超过广播上限的大张量,0 号按专家维切块,点对点只发给持有那块的卡 |
| 每个 rank 各读 | 每个 rank 读全部权重文件 | 不通信,各自切 | 一个完整张量,读盘总量是 N 倍 |
| 按专家分片流式读 | 每个 rank 读稠密张量全文,专家张量只读自己那片 | 不通信 | 专家张量只有自己那片,完整的专家张量从不出现;要求权重文件已是融合后的专家布局 |

**续训时跳过初始权重。** 只要训练配置里给了续训的 checkpoint 路径、而且不是 LoRA,模型运行时就告诉装配这一步:物化后既不读 HF 权重、也不写随机初值,参数留给训练开始时的回调由分布式 checkpoint 按分片读回(那一侧见 13 章),读完再清一次显存缓存。LoRA 是例外:它的 checkpoint 只存可训练的那部分,冻结的底座还得从 HF 来。DPO 的参考模型同理,永远读 HF。

**梯度归约可以用低精度走线。** 默认梯度以 FP32 做 ReduceScatter,线上的字节是 BF16 的 2 倍。基准这一版新增了一条路:FP32 梯度桶在出线前转成参数精度,用一次全交换发出,接收端直接以 FP32 累加进 FP32 的输出分片。它借的是 FSDP2 的自定义 ReduceScatter 钩子(见 FSDP 解读 10 章),梯度缩放也挪进钩子里做。**只在分片组全在一台机器内时启用**,判断依据是每个成员读到的内核启动标识是否相同;跨机或判断不了的组退回原生归约;HSDP 下同一套副本对应的分片组必须一致同意,免得两边缩放口径不一致。

**几种形状对分片的影响。** 组间复制度大于 1 时,分片网格变成「复制 × 分片」二维,FSDP2 自己按 HSDP 处理(见 FSDP 解读 07 章);开了序列并行,分片维会与序列并行维融合成一维,序列并行的卡一起分担分片(06 章)。梯度累积时,每个微批的 ReduceScatter 照做;跨副本的 all-reduce 推迟到最后一个微批;若关掉了反向后释放,则第一个微批关、最后一个微批再打开(累积语义见 FSDP 解读 09 章)。

**梯度裁剪按参数的分法选路。** 纯 FSDP2 直接用 PyTorch 的裁剪,由分布式张量自己算全局范数;有额外并行参数组时,各组在自己的通信组里归约平方和,再合成一个全局系数(05 章);打开卸载时走自己的一版,只沿分片那一侧归约——HSDP 的副本梯度本来就相同。自己算范数的两条路都在归约核里用 FP32 累加,不另复制一份 FP32 梯度,注释说专家那种大张量复制一份就可能把剩余显存吃光。

**DDP 是第二条路。** 数据并行模式选 DDP 时整模型复制;参数还在元设备上时,物化与装载走同一个函数,续训跳过、0 号广播与各自读盘照样适用,只有按专家分片流式读不在这条路上;它不支持专家并行和编译,不广播缓冲区,开序列并行时在裁剪前补一次跨数据与序列并行组的梯度平均。模型若自带一个并行化方法,装配这一步会整个交给它,当前树内没有模型这么做。

**为什么实际不吃亏**:元设备上的切分是纯形状运算,不碰任何数据;物化之后每张卡只有 1/N;装载时临时多出的只是**一个**张量,而不是一个模型。

## 三、代价

**单卡不能用默认配置。** FSDP2 路径强制元设备建模,而只有 1 张卡时根本不包分片,元设备上的模型没人物化,启动就报错。单卡调通要换成 DDP 并指定加速卡建模。

**分组全靠模型声明。** 模型没有声明不可拆分模块、用户也没追加时,命中的类为空,整个模型只剩根这一组:每次前向都把全模型聚合一遍,显存与不分片几乎一样,且不报错。

**默认装载是串行的。** 所有张量都经 0 号读盘、再一个个广播,装载时间随参数量线性增长;每张卡都会短暂多出一个完整张量,**权重文件里最大的那个张量决定装载峰值**。广播上限只对属于额外并行组的张量起作用,一个超过上限的稠密张量照样整块广播。

**流式读只认融合布局。** HF 原版的 MoE 权重按专家分成一个个键,要靠转换器把它们拼成一个融合张量;逐片读做不到这件事,遇到就直接报错,不会悄悄退回整读。

**低精度走线不是白送。** 每个在途归约要多两块与输入等长的低精度缓冲,1 GiB 的梯度桶多 2 GiB;小桶反而更慢,官方实测每卡 4 MiB 时变慢、8 MiB 时基本持平。它先求和再缩放,极端的 BF16 值可能在 FP32 求和时溢出,而原生的求平均不会。分片组一旦跨机就退回原生,纯多机 FSDP 基本拿不到收益。

**卸载有两份账。** 打开卸载后参数、梯度与优化器状态常驻主机内存,每层前后都要搬运;默认还会锁页,超大 MoE 的每个 rank 各锁一份分片,同机 N 个 rank 锁 N 份,在容器里被记成不可回收的共享内存,装载阶段就可能把容器打爆。

**豁免精度的模块常驻显存。** 它们前向后不释放,源码注释自己也提醒这类模块参数不能多。

**关掉混合精度不等于回到 FP32。** 关掉后模型直接以 BF16 建出,没有 FP32 主权重,优化器也在 BF16 参数上更新。

## 四、和同类大厂框架不一样的地方

对照对象是 Megatron-LM(基准 fb6a123a09)。它的数据并行与分布式优化器、以及它自己的 FSDP 实现,见 Megatron 解读 05 到 07 章。

| 比较维度 | VeOmni | Megatron-LM |
|---|---|---|
| 第一个模型对象建在哪 | FSDP2 路径下强制在元设备上建整模型,分片之后才分配 | 默认每个 rank 直接在卡上建自己那一段:只建本流水级的层,张量并行层按切分后的尺寸分配;元设备建模是一个默认关闭的可选项,配置说明写明只在它自己的 FSDP 打开时可用 |
| 权重从哪来 | 启动时直接读 HF 权重文件,逐个张量转换布局、切片、写入 | 参考入口的存取模块只读写它自己的 checkpoint 格式,HF 权重要事先转换 |
| 续训时的初始权重 | 给了续训路径且不是 LoRA,就自动跳过随机初始化与 HF 读取 | 有一个「不做权重初始化」的开关,默认做初始化,跳不跳由用户决定 |
| 低精度梯度走线 | 梯度桶保持 FP32,只在出线前转成参数精度,本地 FP32 累加后写回 FP32 分片;只在分片组全在一台机器内时启用 | 同样是全交换加本地 FP32 累加,但只在梯度桶本身已是 BF16 时才有意义,累加结果写回 BF16 分片;不按分片组是否跨机来决定走不走 |

两边的差异来自模型从哪来。Megatron 自己写模型层,构造函数本来就知道自己是哪一段,「只建自己那段」不需要元设备;VeOmni 拿的是 HF 的 modeling,构造函数只会建整模型,只能先在元设备上建空壳再切。同样的原因,权重格式也是 HF 的,所以 VeOmni 必须在启动时自己完成转换与切片,三条装载路就是为这件事准备的。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。组间复制与分片度的形状在 01 章,专家并行度在 05 章,重计算在 10 章。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `model.accelerator.init_device` · 建模设备 | YAML 或命令行 | `meta` | FSDP2 下只能是 `meta`;只有 1 张卡时不包分片,必须写 `cuda` / `npu` / `mlu` 并换 DDP | 解析参数时的断言 |
| `model.accelerator.fsdp_config.fsdp_mode` · 数据并行模式 | YAML 或命令行 | `fsdp2` | `ddp`:整模型复制,不能开专家并行和编译;`eager` 保留未实现,直接报错 | 日志 `Apply data parallel to the model: fsdp2.` |
| `model.basic_modules` · 在 `_no_split_modules` 之外追加的分组类名 | YAML 或命令行 | 空 | 追加后每个命中的实例单独成组;分组越细,每次聚合越小 | 日志 `target classes to shard: {…}`,空集就是只有根一组 |
| `model.accelerator.fsdp_config.mixed_precision.enable` / `param_dtype` / `reduce_dtype` | YAML 或命令行 | `true` / `bfloat16` / `float32` | 关掉:模型以 BF16 建,无 FP32 主权重;`reduce_dtype` 与 `param_dtype` 相同则梯度以该精度归约 | 落盘的 `veomni_cli.yaml` |
| `model.accelerator.fsdp_config.mixed_precision.output_dtype` / `cast_forward_inputs` | YAML 或命令行 | 不设 / `true` | 透传给 FSDP2 的精度策略 | 同上 |
| `model.accelerator.fsdp_config.reshard_after_forward` · 解码层前向后释放 | YAML 或命令行 | `true` | `false`:层参数留到反向,省一次聚合,显存升;根不受它影响 | profiler 里反向的 all-gather |
| `model.accelerator.fsdp_config.reshard_after_backward` · 反向后释放 | YAML 或命令行 | `true` | `false` 且累积步数大于 1:第一个微批关、最后一个微批开,累积期间完整参数常驻 | 显存时间线;每微批的 all-gather 次数 |
| `model.accelerator.fsdp_config.forward_prefetch` · 手排预取 | YAML 或命令行 | `true` | 只在手排预取那条路上生效(开了额外并行或有精度豁免模块,见 05 章);关掉时前向与反向的手排预取一起关 | profiler 里专家聚合是否暴露在计算前 |
| `model.accelerator.fsdp_config.offload` · FSDP 卸载 | YAML 或命令行 | `false` | 参数、梯度、优化器状态放主机,物化也在主机上做 | 日志 `Enable FSDP2 CPU offload for parameters, gradients, and optimizer states.` |
| `model.accelerator.fsdp_config.offload_pin_memory` · 卸载缓冲锁页 | YAML 或命令行 | `true` | `false`:不锁页,超大 MoE 不被容器记成不可回收内存;每层搬运稍慢 | 容器内存里的共享内存占用 |
| `model.accelerator.fsdp_config.low_precision_reduce_scatter_comm` · 低精度 ReduceScatter | YAML 或命令行,必须是布尔值 | `false` | `true`:机内分片组以 `param_dtype` 走线、FP32 累加;只接受 `bfloat16`/`float16` 配 `float32`,两者相同则走原生;只支持 CUDA | 日志 `Registered bfloat16 ReduceScatter transport with FP32 output on N FSDP modules.` 与每个分片组的启用或回退行 |
| `model.accelerator.fsdp_config.max_load_broadcast_size` · 广播上限(GB) | YAML 或命令行 | `20.0` | 调小:更多属于额外并行组的大张量改为按专家维切块点对点发,装载峰值降 | 日志里每个张量的 `broadcast time (ms)` 与 `chunk and broadcast time (ms)` |
| `model.broadcast_model_weights_from_rank0` · 0 号读、广播 | YAML 或命令行 | `true` | `false`:每个 rank 自己读全部权重文件 | 日志 `Loading model weights from disk on rank0 then broadcasting to other ranks...` 或 `Every rank would read weights from disk and expect this to be slow!` |
| `model.ep_sharded_stream_load` · 按专家分片流式读 | YAML 或命令行 | `false` | `true`:专家张量每个 rank 只读自己那片;必须同时关掉上一项,否则解析即报错;模型没有额外并行计划时打一行忽略日志 | 日志 `Loading model weights via per-rank ExtraParallel-slice streaming (ep_sharded_stream_load)...` |
| `train.checkpoint.load_path` · 续训目录 | YAML 或命令行 | 不设 | 设了且不是 LoRA:跳过 HF 装载与随机初始化 | 日志 `Checkpoint resume enabled (load_path=…); skipping HF weight materialization before checkpoint restore.` |
| `model.optimizer.max_grad_norm` · 裁剪阈值 | YAML 或命令行 | `1.0` | 调小:更常被裁 | 每步日志的梯度范数 |

**观测。** 本章看 4 处。一是分组:`target classes to shard: {…}` 列出命中的类名,之后每个分组打一行 `layer_fqn=…, layer_mod._fsdp_modules=[…]`,数一数组数是否等于层数。二是装载走了哪条路:上表的三句装载日志只会出现其中一句;广播路径下每个张量都打一行广播耗时,OOM 时最后一行就是那个大张量。三是装载的副作用:`Find missing key(s) in state dict: …, initialize them.` 说明有参数不在权重文件里、被随机初始化了;`Detected tensor with all-zero values when reading safetensor: …` 说明权重文件里有全零张量;`Non-persistent buffer … is on meta …` 说明有缓冲区没法保住。四是显存:`VRAM usage after building model: cur …GB, max …GB.` 打在分片之前,那时模型还在元设备上,读数接近 0;装载峰值要看第一个 epoch 结束时那行的 max,或者自己用 profiler 抓。

**典型配置。** 四套能直接抄走的起法,参数名和默认值都来自上表。组合与推荐值是按语义推的起点,不是实测最优。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单机 8 卡,从 HF 权重起步做 SFT | 装载保持默认;加 `model.accelerator.fsdp_config.low_precision_reduce_scatter_comm: true` | 分片组全在机内,梯度以 BF16 走线;每个在途归约多 2 块低精度缓冲,换大桶上的 ReduceScatter 时延(官方在 2 机 16 张 H100 上测得每卡 64–1024 MiB 时降 11.9–14.8%) |
| 多机大 MoE,共享存储带宽足,权重已是融合专家布局 | `model.broadcast_model_weights_from_rank0: false`、`model.ep_sharded_stream_load: true`,专家并行度按 05 章配 | 用存储的并发读换掉 0 号串行广播,专家张量从不以完整形态出现在任何一张卡上 |
| 多机、机间带宽弱,每机 8 卡 | `model.accelerator.dp_shard_size: 8`,再加 `low_precision_reduce_scatter_comm: true` | HSDP 把参数聚合与 ReduceScatter 压在机内,并让后者以 BF16 走线;跨机只剩 FP32 的 all-reduce,且累积时只在最后一个微批做;代价是每卡常驻 1/8 而不是 1/全部卡数 |
| 分片加优化器状态仍放不下,主机内存充裕 | `model.accelerator.fsdp_config.offload: true`,超大 MoE 再加 `offload_pin_memory: false` | 主机内存与每层搬运时间换显存;不锁页再换掉容器上的不可回收内存,搬运更慢 |

| 症状 | 先查 | 然后 |
|---|---|---|
| 单卡启动报 `` Only FSDP training supports `init_device=meta`. `` | 是否只有 1 张卡却用着 FSDP2 | 换 `fsdp_mode: ddp` 且 `init_device: cuda` |
| 启动报 `Please use model.accelerator.init_device: meta for FSDP2 training` | 配置里是否把建模设备改成了加速卡 | FSDP2 下改回 `meta` |
| 显存占用像没分片,前向时陡升到整模型 | `target classes to shard` 是否为空集 | 用 `model.basic_modules` 追加解码层类名 |
| 装载阶段 OOM,最后一行是某个张量的广播日志 | 那个张量的形状与精度,是否属于额外并行组 | 属于就调小 `max_load_broadcast_size`;不属于就换流式读或每个 rank 各读 |
| 装载极慢,只有 0 号的磁盘与网络在忙 | 是否走着默认的广播路径 | 存储带宽够就换流式读或每个 rank 各读 |
| 流式读报 `NotImplementedError`,提到 per-expert -> fused | 权重文件是不是 HF 原版的逐专家布局 | 先转成融合布局,或关掉流式读 |
| 续训在装载阶段 OOM | 日志里有没有 `Checkpoint resume enabled`;是否开着 LoRA | LoRA 跳不过去,只能压装载峰值 |
| 某个模型 loss 从第一步就对不上基线 | 有没有 `Non-persistent buffer … is on meta` 警告 | 在模型的 `init_weights` 里重算那个缓冲区 |
| 开了低精度 ReduceScatter 却没变快 | 日志是否出现 `Using native ReduceScatter for shard group …: a shard group spans nodes` | 用 `dp_shard_size` 把分片组压回机内 |
| 开卸载后装载阶段容器被杀 | 容器内存里的共享内存是否暴涨 | `offload_pin_memory: false` |

## 六、常见误区

- **以为关掉 `forward_prefetch` 能省显存,或者对稠密模型有影响。** 会这么以为,是因为它名字像 FSDP2 的通用开关。实际上它只控制 VeOmni 自己手排的那套预取,而手排只在开了额外并行或有精度豁免模块时才会配置;稠密模型走 FSDP2 自己的默认预取,这个开关读都不会被读到。
- **以为 `VRAM usage after building model` 那行就是装载后的显存。** 会这么以为,是因为它的位置看上去在建模之后。实际上它打在冻结与 LoRA 之后、分片与装载之前,模型还在元设备上,cur 和 max 都接近 0;排查装载峰值时拿它当依据,会得出「装载不占显存」的错误结论。
- **以为 `max_load_broadcast_size` 是装载峰值的上限。** 会这么以为,是因为它的单位是 GB,又叫「上限」。实际上它只决定一个张量走不走「按专家维切块、点对点发」的路,而且只对属于额外并行组的张量生效;一个超过上限的稠密张量照样整块广播,每张卡照样要先拿着它。
- **以为关掉混合精度就是用 FP32 训练。** 会这么以为,是因为混合精度通常意味着「BF16 算、FP32 存」,关掉就该全是 FP32。实际上关掉后模型直接以 BF16 建出,没有 FP32 主权重;想要全 FP32,应当保持开启,并把 `param_dtype` 也设成 `float32`。
- **以为设了 `train.checkpoint.load_path`,所有模型都会跳过 HF 装载。** 会这么以为,是因为日志里确实出现了跳过的那一行。实际上 LoRA 作业在判断时就被排除(冻结的底座必须从 HF 来),DPO 的参考模型也永远读 HF;这两类续训的装载峰值和首次训练一样。
- **以为打开 `low_precision_reduce_scatter_comm` 在多机纯 FSDP 上也生效。** 会这么以为,是因为开关打开后没有报错,训练照跑。实际上分片组一旦跨机就退回原生归约,只在该组的 0 号打一行警告;要拿到收益,分片组必须压在机内,也就是配 HSDP。
- **以为单卡能用默认配置调通流程。** 会这么以为,是因为默认配置在 8 卡上开箱即用。实际上只有 1 张卡时不包分片,而 FSDP2 又强制元设备建模,启动就报错;单卡要显式换成 DDP 并在加速卡上建模。

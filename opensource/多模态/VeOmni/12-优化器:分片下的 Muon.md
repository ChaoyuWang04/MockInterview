# 优化器:分片下的 Muon

这一页回答一件事:选了 Muon 之后,已经被 FSDP 和专家并行切碎的每个矩阵,怎样既拿到整块去正交化,又不为此把全部参数每步聚合一遍。说法都在源码基准 `019b1c0276`(tag `v0.1.12`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

AdamW 是逐元素的:每张卡拿自己那一段梯度和状态就能算完,参数切成多少份都不影响结果,FSDP 能把优化器状态跟着参数一起分掉,靠的就是这一点。Muon 不是。它先攒动量,再把**整个动量矩阵**正交化,拿结果去更新(原理见知识库「优化器与学习率调度」第五节)。正交化是整块矩阵的函数:对行块各做一遍,每块各自被拉平,块与块之间不再正交,更新方向就错了。不报错,loss 只是悄悄走偏。

偏偏每个矩阵在 VeOmni 里都是切开的。FSDP2 把二维权重按行分给分片组里的每张卡;MoE 的专家是三维张量 [专家数, 行, 列],05 章先让专家并行沿第 0 维整块分专家,再由专家内 FSDP **默认沿第 1 维**切,每张卡手里是每个专家的半个矩阵。

最直接的补法是每一步把每个矩阵都聚合成整块再正交化,账有两笔。通信:每张卡每步把全模型的矩阵收一遍,等于在反向的 reduce-scatter 之外再多一轮全量 all-gather。计算:分片组里 N 张卡对同一个矩阵各算一遍 Newton-Schulz,N 张卡重复 N 遍。30B 级 MoE 的参数大头在专家上,这笔账也主要落在专家上。

还有一层分家。05 章说过专家参数与其他参数挂在不同网格上,一个优化器装不下两张网格的参数;Muon 又只管矩阵,词表嵌入、norm、偏置仍要 AdamW。一个模型因此要同时跑好几个子优化器,学习率调度、存取、日志都得跟着分,谁的超参从哪来也容易乱。

## 二、解法

最天然的直觉:**参数现在怎么切,就按怎么切去凑整块**。优化器每一步给每个参数看一眼它的分片布局,分进四条路,只在必须看到整块时才通信,而且只凑「这一个矩阵要的那一块」。

![Muon 在分片下的三条有通信差别的路:第一行是二维稠密参数,4 张卡的分片组里每卡持有 W0 到 W3 各四分之一行,一次 all-to-all 让卡 i 收齐 Wi 的全部行块并只对它做一次 Newton-Schulz,再一次反向 all-to-all 把结果行块送回;第二行是开了专家零通信的三维专家张量,卡 0 与卡 8 各持 8 个完整专家,本卡批量正交化、通信为 0;第三行是默认布局的三维专家张量,两张卡各持 16 个专家的一半行,在专家内 FSDP 组里 all-gather 成整块后各自把 16 个专家正交化一遍,只留自己那一半行的更新](/opensource/VeOmni/12a-muon-shard-paths.svg)

**四条路各管一类参数。** 没切开的参数(单卡、DDP,或网格上只是复制)在本卡整块算,零通信。二维参数只沿第 0 维按行切时走图里第一行:同一批参数按分片组大小 N 分组,每张卡认领其中一个当 owner,一次 all-to-all 让 owner 收齐自己那个矩阵的全部行块,算一次,再一次反向 all-to-all 送回。数学和全聚合完全一样,但每个矩阵只算一次;按手算,每张卡每个矩阵收发的量约是全聚合的 2/N。二维参数切在别的维、或布局里带着没归约完的部分和,就回退成全聚合、每卡各算一遍。三维专家张量只沿第 0 维切时,每张卡手里是若干完整专家,本卡批量正交化;切在其他维时,在专家内 FSDP 组里全聚合回本卡那批专家的整块,组里每张卡各算一遍,只留自己那部分。不开专家并行时,专家张量随普通 FSDP 沿第 0 维切,自然落在本地那一条。

**专家零通信改的是 FSDP,不是优化器。** 本地那一条要求专家内 FSDP 那一刀落在第 0 维。所以零通信的开关在切分阶段(04 章构建顺序里的逐层分片)就把专家的 FSDP 分片维从 1 改成 0,每张卡拿到完整专家;优化器那边一行不用改,分类时自然走本地。成立条件三条:计划里至少有一个三维专家张量;计划里的每个参数都沿第 0 维切;每个参数在专家并行切完之后的第 0 维都能被专家内 FSDP 度整除——128 个专家、专家并行度 8、专家内 FSDP 度 2,本卡 16 个专家,每卡 8 个完整专家。任一条不满足,打一行警告退回第 1 维。这个分片维也记进参数的分片说明,checkpoint 按它还原专家维(13 章)。

**哪些参数走 AdamW。** 二维和三维的可训参数默认都走 Muon。以下一律交给 AdamW:一维参数(norm、一维偏置);属于嵌入模块的参数;名字里带词表嵌入、输出头、一维卷积字样的参数;以及**不衰减名单**里点名的模块类和参数名——这份名单在 Muon 下同时就是「走 AdamW」的名单。一个参数都没分给 Muon 时直接报错,不静默退回 AdamW。

**子优化器按「优化器 × 网格」拆。** 不开额外并行,Muon 与 AdamW 各一个;开了,两者再各按网格拆,最多 4 个:专家的 Muon、其余的 Muon、专家的 AdamW、其余的 AdamW。外面套一个组合优化器,一步依次调每个子优化器,存取时把各自状态按参数名摊平合成一份(形态见 13 章)。选了 Muon,哪怕单卡也是组合优化器。梯度裁剪不按优化器分,而是按网格重新归拢参数,和 AdamW 时同一条路(05 章)。

**AdamW 这一侧的默认。** 模型运行时建优化器时总是打开 fused 实现,一个融合核更新一组参数。参数按不衰减名单分成衰减与不衰减两组,名单**默认是空的**,也就是 norm 与偏置默认照样衰减。另有一个低精度状态的 AdamW 变体:动量与方差存 BF16,用 Kahan 求和补回舍入误差。

**Muon 这一侧的学习率。** Muon 学习率不设时,默认按 AdamW 更新幅度对齐的缩放下直接沿用 AdamW 的学习率;换成原版缩放则取 25 倍。每个矩阵还按形状再乘一个步长系数,三维专家按单个专家的 [行, 列] 算,按头分块时按块的行数算。Muon 的权重衰减与 AdamW 的各设各的。

**调度器每个子优化器一个,共用一条曲线。** 曲线三种:常数、线性、余弦,都可以先线性预热,预热步数 = 总步数 × 预热比例,从起始学习率爬到主学习率。曲线是乘在每个参数组初始学习率上的**比例**,所以 Muon 组、视觉组都同比例缩放,最低学习率也按「最低 ÷ 主学习率」换成比例。余弦可以只在总步数的前一段里衰减完,之后停在最低值。

**为什么实际不吃亏**:二维参数的 owner 路径把计算砍成 1/N、通信砍到全聚合的 2/N 左右;专家开零通信后,参数大头那部分的正交化完全不通信,只是本卡几次批量矩阵乘。动量状态只有一份,比 AdamW 的两份省一半。

## 三、代价

**owner 路径靠所有卡步调一致。** all-to-all 按位置配对,要求每张卡以同样顺序、同样个数处理同一批参数。Muon 只跳过梯度为空的参数:某个参数在一张卡上有梯度、在另一张卡上没有,集合通信就配不上对,整组挂住。反过来,梯度是全零的参数照样做一次权重衰减——冻不住,只是慢慢被衰减掉。

**临时显存看路径。** owner 路径每个桶最多攒 N 个矩阵,每张卡同时只多一个整矩阵;回退的全聚合路径每个矩阵都在每张卡上短暂成整块;专家默认路径要在每张卡上放下本卡全部专家的整块,[16, 2048, 768] 这种张量整份出现在专家内 FSDP 组的每张卡上。

**零通信把整除约束请了回来。** 05 章让两刀切在不同维,就是为了躲开「专家数必须被两个并行度的乘积整除」;零通信要每张卡拿完整专家,这条约束在专家内那一刀上又回来了。不满足时它不报错,只是回到有全聚合和重复计算的默认路径。

**正交化本身要算。** 每个矩阵每步 5 轮 Newton-Schulz。默认实现用 Hopper 以上的专用矩阵乘核,需要额外装包;缺包或卡型不够时退回纯 PyTorch 的同算法实现,只警告一次,之后每步都慢。

**分组学习率不认。** Muon 会把全部参数按形状重新分一遍,调用方传进来的参数分组被整个忽略(视觉学习率那一侧见 08 章);想给某块参数单独的学习率,当前只能用 AdamW。

**日志里只有一个学习率。** 调度器按子优化器分了家,打日志时取的是第一个子调度器的学习率;Muon 下排第一的是 Muon 组,看到的是它的学习率,AdamW 那组要自己按比例推。

**MoE-LoRA 在 Muon 下裁剪会偏大。** AdamW 路径会单独记下「在专家并行组内复制的 LoRA 参数」,裁剪时不沿专家并行组重复累加;Muon 路径没建这份名单,共享 LoRA 的梯度平方和会按专家并行度被重复累加,范数偏大、裁剪偏狠(LoRA 本身见 11 章)。

## 四、和同类大厂框架不一样的地方

对照对象是 Megatron-LM(基准 fb6a123a09)。

它同样把矩阵交给 Muon、其余交给标量优化器,但「谁手里有整块」这件事是按它自己的并行方式解的,见 Megatron 解读 06 章。

| 比较维度 | VeOmni | Megatron-LM |
|---|---|---|
| 数据并行这一维怎么凑整块 | 参数本来就被 FSDP 按行切开;优化器每步用 all-to-all 把整块凑给 owner,算完再送回行块 | 梯度在每张卡上是完整的;按层分布式优化器把整个参数分给某个数据并行 rank,一个参数不跨分片边界,owner 更新后再把参数 all-gather 回所有卡 |
| 模型并行切开的矩阵 | 没有张量并行,只有 FSDP 与专家内 FSDP 两种切法,按布局自动选路 | 张量并行切开的矩阵有 4 种处理:聚合后每卡算整块、每步归约 Gram 矩阵的分布式算法、各分片各算(更新规则随张量并行度变)、按代价模型逐矩阵自动选 |
| 专家 | 专家内 FSDP 沿第 0 维切就零通信,否则组内全聚合 | 名字带 experts 且非共享的参数打上专家标记,不开按层分布式时单独成一组、用专家那套进程组 |
| 非矩阵参数 | 固定 AdamW | 标量优化器可选 Adam 或 Lion |
| 与 FSDP 的关系 | Muon 就是为 FSDP2 分片写的 | 自带 FSDP 的新版与这类优化器组合直接拒绝;也不支持 FP16 |

两条路的分歧来自底座。Megatron 的梯度在数据并行维上默认是完整的,「整块」不稀缺,难点在张量并行,所以它花力气做了 4 种张量并行模式;VeOmni 只有 FSDP 这一种底座,参数从一开始就是碎的,于是把功夫花在「按分片布局选路」和「把专家的分片维改到第 0 维」上。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。梯度裁剪的总体在 04 章,专家并行度与专家内 FSDP 度在 05 章,视觉学习率在 08 章。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `model.optimizer.type` · 优化器 | YAML 或命令行 | `adamw` | `muon`:二维、三维参数走 Muon,其余 AdamW;`anyprecision_adamw`:动量、方差与 Kahan 补偿 3 份状态都存 BF16,换掉 2 份 FP32 状态 | 日志 `Muon optimizer: N param(s) on Muon, M on AdamW.` |
| `model.optimizer.lr` · 主学习率 | YAML 或命令行 | 5e-5 | AdamW 组的峰值;Muon 学习率不设时也由它推;调度曲线的比例以它为分母 | wandb 或日志里的学习率 |
| `model.optimizer.weight_decay` · AdamW 权重衰减 | YAML 或命令行 | 0 | 只作用于 AdamW 组中不在不衰减名单里的参数 | 日志 `Parameters without weight decay: […]`(不开额外并行的 AdamW 路径) |
| `model.optimizer.no_decay_modules` / `no_decay_params` · 不衰减名单 | YAML 或命令行 | 空 | 按模块类名 / 参数名子串免衰减;**Muon 下命中的参数同时改走 AdamW** | 同上;Muon 下看 AdamW 参数计数 |
| `model.optimizer.betas` · AdamW 的 β | YAML 或命令行 | (0.9, 0.95) | 只作用于 AdamW 组 | 保存的配置 |
| `model.optimizer.lr_decay_style` · 曲线 | YAML 或命令行 | `constant` | `linear` / `cosine`;所有子优化器共用 | 学习率曲线 |
| `model.optimizer.lr_warmup_ratio` / `lr_start` · 预热 | YAML 或命令行 | 0 / 0.0 | 预热步数 = 总步数 × 比例,从 `lr_start` 线性爬到 `lr` | 前几步的学习率 |
| `model.optimizer.lr_min` / `lr_decay_ratio` · 余弦的地板与衰减段 | YAML 或命令行 | 1e-7 / 1.0 | 只有余弦读这两个;`lr_decay_ratio` < 1 时提前降到地板后停住 | 学习率曲线尾部 |
| `model.optimizer.muon_lr` · Muon 学习率 | YAML 或命令行 | 不设 | 不设:`match_rms_adamw` 下等于 `lr`,`original` 下是 25 × `lr` | 日志 `[Muon] … muon_lr=… (inherit optimizer.lr (match_rms_adamw))` |
| `model.optimizer.muon_adjust_lr_fn` · 按矩阵形状调步长 | YAML 或命令行 | `match_rms_adamw` | `original` 换成原版缩放,不设 `muon_lr` 时学习率还会跳到 25 倍 | 同上 |
| `model.optimizer.muon_weight_decay` · Muon 权重衰减 | YAML 或命令行 | 0.0 | 与 `weight_decay` 各管各的 | 日志 `[Muon] … muon_weight_decay=…, adamw_weight_decay=…` |
| `model.optimizer.muon_momentum` / `muon_nesterov` | YAML 或命令行 | 0.95 / 开 | Muon 的动量与是否 Nesterov | 同上 |
| `model.optimizer.muon_ns_implementation` · 正交化实现 | YAML 或命令行 | `gram_quack` | `gram` 纯 PyTorch 的同算法;`std` 与 PyTorch 自带 Muon 一致;`gram_quack` 要 Hopper 以上并装 `gram-newton-schulz` | 缺包时警告 `[Muon] ns_implementation=gram_quack requested but … Falling back to pure-PyTorch gram path.` |
| `model.optimizer.muon_ns_steps` / `muon_ns_coefficients` / `muon_eps` / `muon_gram_ns_reset_iterations` | YAML 或命令行 | 5 / [3.4445, -4.7750, 2.0315] / 1e-7 / [2] | Newton-Schulz 轮数、多项式系数、归一化 ε、Gram 版的重启位置;步数上限 99 | 保存的配置 |
| `model.optimizer.muon_expert_zero_comm` · 专家零通信 | YAML 或命令行 | 关 | 开:切分时把专家的 FSDP 分片维改成 0,专家的正交化不再通信;条件不满足自动回退 | 日志 `[muon_expert_zero_comm] ep: enabling Shard(0) …` 或 `… falling back to the default Shard(1) layout …` |
| `model.optimizer.muon_head_group_size` / `muon_head_split_modules` · 按头分块正交化 | YAML 或命令行 | 0 / 空 | 大于 0 时把列出的注意力投影按每块若干个头切行块各自正交化;必须同时列模块名,匹配到两个嵌套投影直接报错 | 日志 `[Muon] head split: …`;没匹配上时警告 `… matched no attention projection …` |

**观测。** 建优化器时 rank 0 打 3 类行:`Muon optimizer: …` 给出两边的参数个数和前 5 个参数名,拿它确认 embedding、输出头、norm 落在 AdamW;`[Muon] ns_implementation=…` 一行汇总正交化实现、两边学习率与学习率来源、两边权重衰减、动量;开了额外并行还有 `Muon split for ep: muon_ep=…, adamw_ep=…, muon_non_extra_parallel=…, adamw_non_extra_parallel=…`,专家那一组应当几乎全在 Muon。零通信有没有真的生效**不在这几行里**,要看切分阶段那行 `[muon_expert_zero_comm]`。每步日志里的学习率取的是第一个子调度器的最大值,Muon 下就是 Muon 组。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 2 机 16 卡训 Qwen3-30B-A3B,想用 Muon | 以 `configs/text/qwen3_moe_muon.yaml` 为底:`type: muon`、`lr: 1.0e-4`、`weight_decay: 0.1`、`muon_weight_decay: 0.1`、`lr_decay_style: cosine`、`muon_expert_zero_comm: true`、`ep_size: 8` | 本卡 16 个专家能被专家内 FSDP 度 2 整除,专家的正交化零通信;代价是前向聚合的是整块专家,专家内 FSDP 组的分片边界随之改变 |
| 稠密模型想试 Muon,又不想重扫超参 | `type: muon`,`muon_lr` 不设、`muon_adjust_lr_fn` 保持默认,`no_decay_modules: [Qwen3RMSNorm]` | Muon 沿用 AdamW 调好的学习率;每步多一轮 owner 式 all-to-all 和每矩阵一次正交化 |
| AdamW 微调,要预热和余弦 | `type: adamw`、`lr: 1.0e-5`、`lr_warmup_ratio: 0.03`、`lr_decay_style: cosine`、`lr_min: 1.0e-6`、`weight_decay: 0.1`、`no_decay_params: [bias]`、`no_decay_modules: [Qwen3RMSNorm]` | 前 3% 步线性爬升换开局稳定;偏置和 norm 不衰减需要显式列出 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 开了 Muon,loss 比 AdamW 基线差很多 | `[Muon]` 行里的 `muon_lr` 与来源;是不是在拿 Muon 微调一个 Adam 预训练的模型 | 回到 `match_rms_adamw` 且不设 `muon_lr`;微调优先沿用预训练时的优化器(知识库「优化器与学习率调度」) |
| 开了 `muon_expert_zero_comm`,优化器步耗时没变 | 切分阶段有没有 `[muon_expert_zero_comm] … falling back …` 警告 | 调专家并行度,让本卡专家数被专家内 FSDP 度整除 |
| 每步优化器很慢,日志有 `gram_quack requested but …` | 卡型是否 Hopper 以上;`gram-newton-schulz` 是否装上 | 装包,或接受纯 PyTorch 的 `gram` |
| Muon 下所有 rank 在优化器步挂住 | 是否有参数只在部分 rank 上拿到梯度 | 让这些参数每个 rank 都有梯度,或冻结它们 |
| 视觉塔的学习率设了没生效,日志有 `… ignores the provided param_groups argument …` | 是否 `type: muon` | 要分层学习率就改用 AdamW |
| 学习率曲线尾部停在奇怪的值 | 是不是线性曲线;`lr_min` 是不是被当成了绝对值 | 线性曲线不读 `lr_min`;余弦的地板是 `lr_min ÷ lr` 这个比例乘在各组初始学习率上 |
| MoE-LoRA 换成 Muon 后梯度范数明显变大、裁剪变频繁 | 是否开了专家并行且 LoRA 共享在专家之间 | 这是 Muon 路径少建了一份复制参数名单,对比时用 AdamW 基线 |

## 六、常见误区

- **以为 Muon 下偏置一定走 AdamW。** 会这么以为,是因为配置说明写的是「偏置和 norm 走 AdamW」。实际分法看维数:一维偏置走 AdamW,GPT-OSS 这种按专家堆起来的偏置是二维的 [专家数, 宽],会被当成一个矩阵正交化。想让它们走 AdamW,在 `no_decay_params` 里写上 `bias`。
- **以为不衰减名单只管衰减。** 在 AdamW 下是这样;Muon 下命中名单的参数同时被整个挪到 AdamW 组。把 `Qwen3RMSNorm` 这类一维模块写进去无害,把某个线性层的类名写进去,它就不再用 Muon 更新了。
- **以为 norm 和偏置默认不衰减。** 从 HF Trainer 过来的人习惯它自动排除 LayerNorm 与 bias。这里两份名单默认都是空的;`weight_decay` 默认 0 时看不出,一旦设成 0.1,norm 和偏置一起被衰减。
- **以为开了 `muon_expert_zero_comm` 就一定零通信。** 这个开关只是请求。条件不满足时切分阶段打一行警告就回退第 1 维,优化器照常跑、只是有全聚合;建优化器那行 `expert_zero_comm=True` 只是回显配置,不代表生效。
- **以为 `lr_min` 是绝对的最低学习率。** 它先被除以 `lr` 变成比例,再乘在每个组自己的初始学习率上:Muon 取 25 倍学习率时,Muon 组的地板也是 25 倍。线性曲线根本不读它,固定用 1e-7 ÷ `lr` 这个比例;`lr_decay_ratio` 也只有余弦读。
- **以为日志里的学习率就是 AdamW 的学习率。** 日志取第一个子调度器,Muon 下是 Muon 组。`muon_adjust_lr_fn: original` 又不设 `muon_lr` 时,日志里的数是 `lr` 的 25 倍,并不是配错了。
- **以为梯度是零的参数不会被 Muon 动。** Muon 只跳过梯度为空的参数;损失里乘了 0 的分支照样产生全零梯度,权重衰减照做。DeepSeek-V4 的 indexer 就为此在系数为 0 时直接不建那一项,而不是乘 0。

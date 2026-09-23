# LoRA:自带的实现与专家 LoRA

这一页回答一件事:大 MoE 模型只训低秩适配器时,适配器怎么挂到融合的专家张量上、怎么跟着专家并行切、怎么存成别人读得懂的格式。说法都在源码基准 `019b1c0276`(tag `v0.1.12`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

LoRA 本身是什么、为什么省显存,见知识库「LoRA」。这里的问题是它落到 VeOmni 这类框架上时哪里会断。

**全量微调的优化器状态放不下。** 树里 GPT-OSS-120B 的样例配置写得很直白:4 张 192 GB 的卡按专家并行度 4 切开,放得下 BF16 底座、梯度和激活,放不下全量的 AdamW 状态。能训的只剩一小撮参数,LoRA 就是这一小撮。

**专家权重不是线性层。** transformers v5 起,MoE 一层的全部专家存成专家模块上的两块三维参数:gate 与 up 拼在一起的 [E, 2I, H],以及 down 的 [E, H, I]。通用 LoRA 库按「模块名」找目标、把找到的线性层换掉,而这两块只是参数、不是模块,按模块名匹配根本够不着。用户照着稠密模型的习惯写上 gate、up、down,按模块名只会命中共享专家和稠密层里真正的线性层,路由专家原样冻着。

**包一层就改名,改名就对不上并行计划。** 通用 LoRA 库的做法是在模型外面套一层壳,所有参数名前面多出一段前缀,被换掉的层还要把原权重挪到下一级。而 VeOmni 的专家并行是模型按**裸参数名**自报的(05 章):不改写计划,切分时一个名字都匹配不上,专家要么不切、每张卡整份放下,要么在断言里挂掉;从 HF 读底座权重时,checkpoint 里的键也找不到新位置。

**元设备建模下没法当场初始化。** 模型先在元设备上建空壳(04 章),LoRA 矩阵这时也没有存储,「A 用 kaiming、B 置零」只能等物化之后再做;续训时又得分清哪些矩阵已经从文件读进来、哪些还要初始化,分错一个就会把读好的权重冲掉。

**专家并行下,「共享」的适配器会悄悄分叉。** 如果整层专家共用一对 A、B,开专家并行后每张卡只算到本卡那几个专家贡献的梯度。不跨卡求和,各卡的优化器按各自的半截梯度走,第一步之后几份「共享」适配器就不再相同,也不报错。

## 二、解法

最天然的直觉:**自己写一套,但在盘上长得和 PEFT 一模一样**。运行时的包装、注入、切分、存取全部自己管,这样能和 FSDP2、专家并行、融合专家算子一起设计;落盘的键名、配置文件字段照 PEFT 的格式写,训出来的适配器别的工具照样能读。整套代码不依赖 peft 包。

**包装是一层壳加两类替身。** 挂 LoRA 在模型运行时「冻结或挂 LoRA」那一步做(01 章),在切分与装载之前,模型还在元设备上。外壳内部再套一层容器装原模型,于是所有参数名带上和 PEFT 一样的前缀。然后按目标名单换两类东西:

- **线性层**换成带 LoRA 的线性层。原层原样挪到它下面,旁边挂一对 A [r, 输入维] 与 B [输出维, r],前向是原输出加上「B 乘 A 乘输入」再乘缩放系数。匹配规则照 PEFT:名单是列表时,模块全名等于某项或以「点加某项」结尾就算中;是单个字符串时当正则整串匹配;排除名单再剔掉一批。
- **专家模块**整个换成专家 LoRA 的包装。包装先检查专家模块确实是那两块三维参数,然后把它们**原对象**拿过来,挂到按 PEFT 布局排好的子模块下,不复制一个字节;再挂三对 A、B。

**融合的 gate+up 上挂的是两对,不是一对。** gate 与 up 之间夹着 SiLU,是非线性的,增量必须在激活之前分别加到 gate 半与 up 半上。所以包装在这块融合张量上分配两对秩为 r 的适配器,down 上再一对,一共三对:

![一层 MoE 专家 LoRA 的形状:左边是一个专家上的三对适配器,底座 gate 与 up 是同一块 1536 乘 2048 融合张量的上下两半,各挂一对 A 16 乘 2048、B 768 乘 16,down 底座 2048 乘 768 挂一对 A 16 乘 768、B 2048 乘 16;右边是 8 卡专家并行时每张卡持有的份额,底座与独立模式的适配器都按专家维切成每卡 16 个专家,共享模式的适配器每卡一份完整副本,反向后 8 卡对梯度求和](/opensource/VeOmni/11a-expert-lora-shapes.svg)

**两种模式。** 独立模式(默认)每个专家各有一对,三对 A、B 都带最前面那一维专家数;共享模式整层专家共用一对,没有专家维。以 Qwen3-30B-A3B 一层、秩 16 为例,独立模式约 1730 万个适配器参数,是这层专家底座(约 6.04 亿)的 2.9%;共享模式约 13.5 万,再小 128 倍。

**目标名由模型钩子翻译。** 用户在目标名单里照稠密模型的习惯写 gate、up、down 即可。模型类在注册表里取出时挂着一个翻译钩子(02 章那几种钩子之一):名单里出现 gate 或 up,就换成「各层专家的 gate+up 融合参数」这条通配模式;出现 down,就换成 down 那条;这三个名字从线性层名单里拿掉,其余名字不动。翻译出的模式如果一个参数都匹配不上,当场报错。没有这个钩子的模型(稠密模型,以及当前的 DeepSeek-V3)这三个名字保持原意,只匹配线性层。也可以不靠钩子,直接写参数的通配模式。

**并行计划由框架改写。** 切分前,框架拿模型自报的计划做两件事:给所有模式补上前缀;把指向专家裸参数的那两条,换成包装之后的新位置,每层一条,切法照旧。独立模式再多一步:把每层三对 A、B 全部登记进同一个专家并行组、同样沿专家维切,于是本卡持有的底座专家与本卡持有的适配器编号一一对上(上图右半)。共享模式的 A、B 不进计划,每卡一份完整副本;包装在第一次前向时给它们装一个钩子,梯度累加完就在专家并行组内求和。梯度裁剪也认得这组参数,只算一次范数,不按专家并行度重复累加;例外是选了 Muon 时这份名单没建,裁剪会偏大(12 章)。

**专家 LoRA 走融合的分组矩阵乘。** 底座专家的融合算子本来就是「按专家把 token 排好、按组做矩阵乘」。LoRA 版复用同一套 Triton 分组矩阵乘原语和同一条专家并行 all-to-all 流水:gate+up 的底座输出算完后,把两半的增量拼起来加上去,再切开过 SiLU;down 同理。增量按「先乘 A、再乘 B」算,不把 [E, O, H] 的完整增量矩阵物化出来。独立模式的每次 A、B 乘法本身也是一次分组矩阵乘,沿用底座的分组边界;共享模式退化成普通矩阵乘。Triton 版前向反向都是手写的;昇腾版把底座的分发、合并与分组矩阵乘拼起来,在分发后的 token 上加增量,反向交给自动求导。只有这两家绑定了 LoRA 版(03 章),选别的专家后端时包装退回逐专家循环的参考实现,而这条参考实现遇到专家并行直接报错:它拿全局专家号去索引只剩本卡一段的张量。

**哪些参数可训,由 LoRA 说了算。** 注入完成后统一过一遍:先把所有参数冻住,再只解冻名字里带 A、B 两段的参数;偏置默认不训,也可以选「全部偏置」或「只训挂了 LoRA 的层的偏置」。模型自己的冻结策略在 LoRA 之前跑,LoRA 最后覆盖,所以视觉语言训练里冻塔的开关在开 LoRA 时不起作用(08 章)。一个可训的适配器都没挂上,构建直接报错。

**初始化推迟到物化之后。** 元设备上的 A、B 不初始化;物化并读完权重后,凡是没从文件里读到的适配器矩阵才按 kaiming 与零初始化。专家包装按整块判断:全部缺才初始化,全都读到就不动,只读到一部分就报错拒绝,宁可停下也不冲掉读进来的那一半。

**存两份,各管各的。** 续训状态走 DCP,只存可训的参数与它们的优化器状态,冻结的底座不存,续训时照常从 HF 目录读(所以 LoRA 续训跳不过 HF 装载,04 章;目录布局见 13 章)。给推理用的导出走另一条:只取适配器张量,把专家维恢复成完整的 E 后用 DCP 并行写一份临时目录,rank 0 合并成一个 safetensors 权重文件,再写一份 PEFT 字段的适配器配置文件;专家 LoRA 的模式记在配置文件里一个 PEFT 不认识、会直接忽略的命名块里,不另起文件。读的时候也认 PEFT 旧式的 .bin 权重;遇到没有那个命名块的纯 PEFT 适配器,按盘上张量是三维还是二维推断是独立还是共享模式。

## 三、代价

**专家 LoRA 只认 v5 的融合布局。** 包装要求专家模块恰好有 gate+up 与 down 两块三维参数,否则直接报错;一个作业只能选一种专家模式,不能这几层独立、那几层共享。

**融合的 gate+up 多付一份参数。** 两对各自秩 r 的适配器,每个专家的参数量是 2r(H+I),挂一对的话是 r(H+2I);换来的是 gate 与 up 的增量不必挤在同一个 r 维子空间里。

**专家并行离不开融合算子。** 只有 Triton 与昇腾两家带 LoRA 版;选别的后端,不开专家并行时静默变慢,开了就在第一次前向报错。

**共享模式在专家并行下多一次通信。** 每层三对共享 A、B,每步反向后各做一次专家并行组内的 all-reduce;独立模式没有这一笔。

**导出的精度与时机是定死的。** 导出时 FP32 的适配器一律转成 BF16 再写盘;默认只在训练结束时导出一次,中间各步只有 DCP。

**合并只做了一半。** 把增量折回底座权重的接口只处理线性层,遇到专家 LoRA 直接报错;训练流程本身也不调用它,不会产出合并后的整模型。

**部分 PEFT 功能不支持。** 额外保留整块可训模块的那个功能当前直接报错(元设备建模下给这些模块种初值还没实现);DoRA、LoftQ 这类字段出现在读入的配置文件里只告警、被忽略。按秩与缩放的逐模块覆盖、LoRA 输入的 dropout 只对线性层生效,专家包装一律用全局的秩与缩放,不带 dropout。

**MFU 只会算 Qwen 家族。** LoRA 下的 FLOPs 估算只登记了 Qwen 系列的模型类型与投影名;别的模型或别的目标名,告警一次后把实际 FLOPs 与 MFU 报成 0,训练照跑。

## 四、和同类大厂框架不一样的地方

对照对象是 HF PEFT(版本 0.18.1)。

PEFT 是业界 LoRA 训练的事实标准库,VeOmni 的 LoRA 文档本身就是按「替代它、格式与它兼容」来写的;版本取仓库依赖锁定的那一版,它在 VeOmni 里只用于跑互通测试。另一个候选 Megatron-LM(基准 fb6a123a09)在训练侧没有 LoRA:它源码里带这个词的地方只是 MLA 的低秩投影、导出到 TensorRT-LLM 的参数,以及一处提到 NeMo 适配器的注释。

| 比较维度 | VeOmni | HF PEFT |
|---|---|---|
| 目标怎么找 | 线性层按模块名;专家的三维融合参数按参数通配模式,另有模型钩子把 gate、up、down 翻译过去 | 按模块名匹配,找的是模块子类;VeOmni 文档据此说它够不着专家模块上的三维参数 |
| 整层专家共用一对 | 有共享模式 | VeOmni 文档与源码注释都写明 PEFT 不原生支持 |
| 盘上格式 | 两个文件,键名带同样的前缀、去掉适配器名;配置文件多一个命名块 | 同样两个文件;读 VeOmni 的配置文件时忽略那个命名块;读专家 LoRA 的权重时,若没先装好 VeOmni 的包装,专家那一半键会被当作多余键丢掉 |
| 读旧格式 | 认 PEFT 的 .bin 权重,从张量维数推断专家模式 | 本身的格式 |

两边的取舍在于包装由谁定义。PEFT 的包装是通用的;VeOmni 的专家包装只为 v5 融合布局写,换来的是它能和专家并行的计划改写、融合算子、导出时的专家维恢复写在同一处,挂 LoRA 也被排进元设备建模之后、FSDP2 切分之前那个固定位置。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。LoRA 的开关就是 `model.lora_config` 这一整个字典:默认是空字典,空就不挂 LoRA。下表里 `lora_config.*` 都是它的键;YAML 里的老名字与 PEFT 名字两套都认。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `lora_config.rank`(或 `r`)· 秩 | YAML 或命令行 | 8(不写时) | 调大:适配器参数与优化器状态线性增长 | 日志 `Initialising VeOmni LoRA adapter from scratch: …` 里的 `r=` |
| `lora_config.alpha`(或 `lora_alpha`) | 同上 | 8(不写时) | 缩放系数 = alpha ÷ rank;开 `use_rslora` 后是 alpha ÷ √rank | 同上 |
| `lora_config.lora_modules`(或 `target_modules`) | 同上 | 无 | 列表按全名或「点加名字」结尾匹配线性层;单个字符串当正则整串匹配;在有翻译钩子的 MoE 模型上,`gate_proj`/`up_proj`/`down_proj` 改挂到专家 | `Injected dense LoRA into N nn.Linear module(s)`;一个都没匹配上时告警 `dense LoRA injected nothing` |
| `lora_config.target_parameters` · 专家参数的通配模式 | 同上 | 无 | 直接指定专家的 `gate_up_proj`/`down_proj`,模式要从 `model.` 起写全;一个都没匹配上直接报错 | `Injected independent MoE-LoRA into N experts module(s)`(或 `shared`) |
| `lora_config.share_expert_lora` | 同上 | false:独立模式 | true:整层一对,参数约少 E 倍,专家并行下每步多几次 all-reduce | 同上一行的 `independent` / `shared` |
| `lora_config.exclude_modules` | 同上 | 无 | 从命中的线性层里再剔掉 | 同 `lora_modules` 那行 |
| `lora_config.bias` | 同上 | `none` | `all`:全模型偏置都可训并导出;`lora_only`:只放开挂了 LoRA 的线性层的偏置 | 日志 `**** trainable parameters ****` 那一段 |
| `lora_config.lora_dropout` / `rank_pattern` / `alpha_pattern` | 同上 | 0.0 / 空 / 空 | 只作用于线性层;覆盖规则按正则搜索,第一个命中的生效 | 线性层的 `extra_repr` 里的 `r=`、`alpha=` |
| `lora_config.lora_adapter` · 从已导出的适配器目录起训 | 同上 | 不设 | 设了:包装按目录里的配置文件重建,YAML 里的秩、目标全被忽略;权重在装载阶段读入 | `Wrapping model with VeOmniLoraModel from …`、`also loading lora adapter weights from …` |
| `lora_config.is_trainable` | 同上 | true | false 会把适配器也冻住,随后因「没有可训的适配器」报错,训练里等于不可用 | 报错 `LoRA configuration produced no trainable adapters` |
| `lora_config.modules_to_save` | 同上 | 无 | 设了就报 `NotImplementedError` | 同左 |
| `model.ops_implementation.moe_implementation` | YAML 或命令行 | `fused_triton`(NPU 上默认改成 `fused_npu`) | 只有这两家带专家 LoRA 融合版;其他取值走逐专家循环,开专家并行时报错 | 报错 `eager forward does not support expert parallelism (EP)` |
| `model.accelerator.ep_size` | 同上 | 1 | 大于 1:底座与独立模式的适配器按专家维切,共享模式的适配器整份复制 | 日志 `Rewrote ExtraParallel plan group 'ep' for MoE-LoRA wrappers: …` |
| `model.broadcast_model_weights_from_rank0` / `model.ep_sharded_stream_load` | 同上 | true / false | 续训读适配器时:前者 rank 0 读后广播;关掉它再开后者,独立模式的专家适配器每卡只读自己那段 | 日志 `ep_sharded adapter: streamed … EP-sliced` |
| `train.checkpoint.save_hf_weights` | 同上 | true | LoRA 作业里它管的是适配器导出,不是整模型导出 | 日志 `LoRA adapter saved at … successfully!` |
| `train.checkpoint.hf_save_steps` / `hf_save_epochs` | 同上 | 0 / 0 | 都是 0 时只在训练结束导出一次;设了就按步或按轮导出 | 各步目录下有没有 `lora_ckpt/` |
| `train.checkpoint.load_path` | 同上 | 不设 | 断点续训:DCP 里的适配器、优化器状态、数据游标一起恢复;底座仍从 HF 读 | 04 章那行跳过日志不会出现 |

**观测。** 看 4 处。一是注入:线性层与专家各有一行注入日志,数字要和心里的层数对上,线性层那行只列前 5 个、专家那行只列前 3 个全名。二是可训参数:构建后打印的可训参数统计,开 LoRA 时应当只有适配器(与你选的偏置)。三是并行计划改写:开专家并行时应当出现 `Rewrote ExtraParallel plan group` 那一行,独立模式下「per-expert LoRA tensor(s) added」的数目等于专家层数乘 6。四是导出:每次导出后的 `lora_ckpt/` 下应只有 `adapter_config.json` 与 `adapter_model.safetensors`,专家 LoRA 的模式在配置文件的 `veomni_lora.moe_mode` 字段里。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。组合与推荐值是按语义推的起点,不是实测最优。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单机 8 卡、Qwen3-0.6B 这类稠密模型做 SFT | 以 `configs/text/qwen3_lora.yaml` 为底:`rank: 64`、`alpha: 32`,`lora_modules` 列满注意力与 MLP 的 7 个投影 | 用每层 7 对适配器的少量参数换全量微调的优化器状态;gate、up、down 在稠密模型上就是普通线性层 |
| 单机 8 卡、Qwen3-30B-A3B,专家也要适配 | 以 `configs/text/qwen3_moe_lora.yaml` 为底,改 `moe_implementation: fused_triton`、`ep_size: 8`、`share_expert_lora: false` | 每卡只放 16 个专家的底座与适配器,用专家并行的 all-to-all 换显存;独立模式每层约 1730 万适配器参数,换每个专家各自的适配能力 |
| 4 卡 192 GB、GPT-OSS-120B,只训注意力 | 照 `configs/text/gpt_oss_120b_lora_ep4.yaml`:`ep_size: 4`、`moe_implementation: fused_quack`、`lora_modules` 只写 q/k/v/o,混合精度参数 BF16、归约 FP32 | 专家不挂适配器,所以可以用不带 LoRA 版的 Quack 专家核;换来优化器状态只有注意力适配器那一点 |

| 症状 | 先查 | 然后 |
|---|---|---|
| 第一次前向报 `eager forward does not support expert parallelism (EP)` | `moe_implementation` 是不是 `eager` 或 `fused_quack`,而 `ep_size` 大于 1 且挂了专家 LoRA | 换成 `fused_triton`(GPU)或 `fused_npu`(昇腾);树里 `deepseek_v3_lora.yaml` 同时写着 `eager` 与 `ep_size: 8`,照抄就会撞上 |
| 写了 gate/up/down,专家却没有适配器 | 有没有 `Injected … MoE-LoRA` 那行;模型的 `__init__.py` 里有没有挂目标翻译钩子 | 没钩子的模型(DeepSeek-V3 等)改写 `target_parameters` |
| 构建时报 `LoRA target parameter pattern did not match any parameter` 或 `No 3D parameters in the model matched` | 通配模式的前缀:视觉语言模型要带 `model.language_model.` | 按 `named_parameters()` 打出来的全名改模式 |
| 续训报 `MoE LoRA wrapper … is only partially loaded` | 适配器文件是不是完整导出的那一份 | 换完整的适配器目录,或去掉 `lora_adapter` 从头初始化 |
| MFU 一直是 0 | 日志里有没有 `LoRA FLOPs are unavailable` | 非 Qwen 模型或目标名不在登记表里,只影响上报,不影响训练 |
| 训练完找不到适配器 | `save_hf_weights` 是否被关掉;`hf_save_steps` 是否为 0 | 打开导出,需要中间步就设 `hf_save_steps` |

## 六、常见误区

- **以为给 `lora_adapter` 起训就是断点续训。** 会这么以为,是因为两者都「从上次的适配器接着训」。实际上 `lora_adapter` 只读适配器权重:优化器状态、学习率调度、数据游标全部从零开始。要完整接上用 `train.checkpoint.load_path` 指向 DCP 目录。
- **以为续训时改 YAML 里的 `rank`、`lora_modules` 会生效。** 设了 `lora_adapter` 时,包装完全按适配器目录里的配置文件重建,YAML 里这些键一个都不读,也不告警。想换秩或换目标,只能从头训。
- **以为所有 MoE 模型写 gate/up/down 都会自动挂到专家上。** 只有登记了翻译钩子的 Qwen3-MoE、Qwen3.5-MoE、Qwen3-VL-MoE、Qwen3-Omni-MoE 会;DeepSeek-V3 支持专家 LoRA,但没有钩子,这三个名字只会挂到共享专家和稠密层的线性层上,路由专家原样冻着,日志里只是少一行专家注入。它的样例因此直接写 `target_parameters`。
- **以为 `rank_pattern`、`lora_dropout` 对专家也生效。** 会这么以为,是因为它们都在同一个 `lora_config` 里。实际上专家包装只读全局的 `rank`、`alpha` 与 `use_rslora`,逐模块覆盖和 dropout 只作用于线性层。
- **以为 `is_trainable: false` 能在训练作业里「只读挂载」一个适配器。** 文档把它写成推理用的选项,但训练运行时在挂完 LoRA 后会检查有没有可训的适配器,全冻就直接报错。
- **以为 PEFT 能原样读回 VeOmni 的专家 LoRA。** 文件确实能被 PEFT 打开,线性层那一半也能读进去;专家那一半要先装好 VeOmni 的包装,否则被当成多余键丢掉,推理时专家上的适配等于没有。
- **以为 `merge_and_unload` 能出一份合并好的整模型。** 它只合并线性层,碰到专家 LoRA 直接报错,训练流程里也没有任何地方调用它;专家 LoRA 只能以「底座加适配器」的形式交给推理侧。
- **以为 MFU 掉到 0 说明卡没在算。** LoRA 下的 FLOPs 估算只认 Qwen 家族与登记过的投影名,其余情况一律报 0,只打一次告警。看吞吐要看每步耗时与 token 数。

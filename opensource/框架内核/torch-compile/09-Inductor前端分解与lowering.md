# Inductor 前端:分解与 lowering

这一页回答一件事:AOTAutograd 交来的前向图与反向图里可能出现任意一个 ATen 算子,Inductor 怎么只写少量代码生成就把它们都接住,接不住的又怎么办。说法都在源码基准 `217579124a`(main,仓库自报 `2.15.0a0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

07 章交给 Inductor 的是 ATen 级别的前向图与反向图。ATen 的算子有 2000 多个,新算子还在不断加进来。给每一个都写一份 Triton 生成、一份 CPU 生成,工作量做不完;只写常用的那些,其余的一出现就只能调 eager 现成的 kernel。

调现成 kernel 本身不慢,慢在它两头。外部 kernel 只认显存里的实体张量:它的输入必须先完整写回显存,它的输出也是一块新写出来的显存。一条本来能融成一个 kernel 的逐元素链,中间夹一个这样的算子,就被切成前后两段,各多一次整张量的读写。覆盖不全,融合就断在覆盖不到的地方。

反过来,就算每个算子都有实现,如果「有实现」的意思是「每个算子产出一块显存」,那也只是把 eager 重写了一遍,中间结果照样一步一过显存。

还有粒度问题。layer norm、softmax 这种算子,当成一个黑盒就看不见里面的逐元素部分,没法和前后的算子融合;拆得太碎,同一个中间值被好几处使用,是存下来还是每处重算,又成了新的取舍。

所以这一章的问题是:**怎么用一小套原语接住所有算子,接住之后又不急着把每一步都写成显存,只在真正必须的地方落地;实在接不住的,代价要尽量小。**

## 二、解法

最天然的直觉是编译器的老办法:先把各式各样的指令规约到一个小指令集,只给小指令集写代码生成。Inductor 分三步做:录图时把大算子分解成原语;把每个原语降成一段「给定下标、算出这个位置的值」的函数,只在必须处把它落成缓冲;降不下去的留成调用 ATen 的外部节点。三步之间穿插几轮模式替换。PrimTorch 把算子规约到约 250 个、Inductor 的循环级表示只有约 50 个操作,这些数字知识库 TorchCompile 篇讲过,这里只讲现在的代码怎么走。

![以 softmax 后接一步采样为例:Dynamo 交来除以温度、softmax、multinomial 三个算子;录联合图时 softmax 按分解表展开成 amax、sub、exp、sum、div 五个原语,multinomial 不在表里原样保留;求导后的模式替换在 GPU 且后端为 Triton 时把 amax 与 sum 两次归约换成一次同时求出最大值与和的在线归约;降级后除以温度是一个不占显存的逐元素函数,被内联进在线归约与最后的逐元素函数,在线归约的最大值与和立即落成两块缓冲,最后的逐元素结果因为要交给外部 kernel 而落成第三块缓冲,multinomial 在显式回退清单上,成为调用 ATen eager kernel 的回退节点并写出第四块缓冲,调度器不与它合并](/opensource/torch-compile/09a-softmax-to-buffers.svg)

**分解在录图那一刻就做完了。** Inductor 把一张分解表交给 AOTAutograd,07 章录联合图时,每遇到表里的算子,就改跑它的 Python 分解,录下来的是分解之后的那几个小算子。表由三部分拼成:导出也在用的那套「分解到核心 ATen」的表,Inductor 自己加的一批(softmax、layer norm、batch norm、gelu 等),再减去刻意拿掉的一批。拿掉的理由都写在注释里:求和、split、squeeze 这类 Inductor 有更好的直接降法;silu 要用和 eager 一模一样的公式;addcmul、lerp 要保留融合乘加,数值才和 eager 的 CUDA kernel 对得上;两种 scatter 要留在图里给后面的重新就地化用。分解函数也可以在运行时表示「这次不管」,算子就原样留下,比如 layer norm 在 MTIA 上就不拆。还有一类算子在分发器上本来就是用别的算子拼出来的复合实现,录图时自动展开,不用进表。所以 Inductor 看到的图里已经没有 softmax,只有 amax、减法、指数、求和、除法。

**降级:每个算子变成一个下标函数。** Inductor 的降级器是一个 FX 解释器,按拓扑顺序逐个节点查降级表,调用对应的降级函数。逐元素算子的降级函数既不计算、也不分配显存,只造一个逐元素节点:它记着输出形状和一段函数,函数拿到下标,就用同样的下标去调每个输入的「读取器」,再把读到的值套上这个算子的运算。输入如果本身也是这样的节点,它的读取器就是它自己的函数,于是一串逐元素算子自然嵌套成一个大函数,值从头到尾没离开过寄存器。转置、reshape 这类视图连节点都不造,只是一个下标换算,读的时候把下标换一下再往下读。归约节点多记一段归约范围和归约类型。融合要的原料在这一层就已经备好,10 章只是决定把哪些函数放进同一个 kernel。

**缓冲只在必须时出现。** 每个降级结果先是一个装着函数的盒子,没有名字,不占显存。把它「落地」,就是把函数包成一块有名字、有布局的缓冲,之后别人读它就是一次真正的显存读取。GPU 上默认在这几种情况落地:

- 归约结果当场落地;归约长度小于 8 的例外,直接展开成逐元素
- 图的输出,包括前向留给反向的激活(哪些留下由 08 章决定)
- 要交给外部 kernel 的输入:矩阵乘、卷积、回退算子都只吃实体缓冲
- 一个结果被多处使用,而内联太贵:函数里读的缓冲超过 4 块,或者运算超过 30 个
- 一路内联下来,读的缓冲累计超过 8 块
- 函数体超过 100 个运算,再嵌套下去有递归过深的风险

不在这几种情况里的,就被每个使用者各自内联一份:宁可重算,不存。

**降不下去的就回退。** 查降级表查不到,先看是不是在「允许回退」的小名单上;不在,默认就当场生成一个隐式回退,打一行日志,继续往下走。另有 136 处显式登记的回退写死在降级表里:线性代数、直方图、multinomial 这类排序或采样、nonzero 与 unique 这类输出形状取决于数值的。回退节点是一个外部 kernel 节点:编译好的代码在这里直接调 ATen 的 eager kernel,它的输入先被落地,它的输出是一块新缓冲,调度器不会把它和任何节点合进同一个 kernel。反向图里的隐式回退更保守:ATen 算子没有标明布局要求时,一律要求输入是连续的,因为有的 eager kernel 碰到非连续输入会静默算错。

**三处模式替换。** 替换在三个时间点跑,各看到的图不一样:

- 求导前,在 Dynamo 交来的 torch 函数级图上。这张图还没函数化,原地修改与别名都在,源码注释自己也劝新写的替换放到后两处。默认在这里做的事不多,代表是采样写法的改写:对 softmax 结果除以指数分布噪声再取 argmax,换成对 logits 加 Gumbel 噪声再取 argmax,省掉整个 softmax;成批融合与转置合并都要显式打开
- 联合图上,切分前后向之前。推理图不录联合图,编前向时补跑这一轮。这里做常量折叠、删掉乘 1 加 0 与无意义拷贝这类空操作、给矩阵乘补齐形状,以及把 30 种手写注意力的写法认出来,换成一次 SDPA 调用;训练用的模式是连前向带反向一起录的,所以能在联合图里一次认出前后两半
- 求导后,在切好的前向图和反向图上各跑一遍。先删空操作,再依次跑三组模式;softmax 的最大值与求和两次归约换成一次在线归约,就在这一轮,只在 GPU 且后端是 Triton 时做;最后把函数化留下的拷贝尽量改回就地写

模式本身就写成普通的 Python 函数,和用户代码一样过一遍同一张分解表录成 ATen 图,所以它和图里实际出现的样子是同一种写法;常用的注意力与矩阵乘模式提前录好存在源码里,启动时不用现录。匹配到之后还要过额外检查,匹配范围跨过原地修改的边界、或跨了 CUDA stream,一律放弃。

**为什么实际不吃亏。** 分解只在录图时做一次;下标函数在代码生成之前不花任何运行时间;绝大多数算子都有直接的降级,回退是少数。而「宁可重算」对访存受限的逐元素链几乎是白送:重算的是几次寄存器里的运算,省下的是一次整张量的读写。

## 三、代价

**回退切断融合,而且默认悄无声息。** 一个回退算子夹在逐元素链中间,它前面的结果必须落地,它后面的节点要从它写的缓冲重新读,原来的一个 kernel 变成两个加一次外部调用。隐式回退只在日志里打一行 INFO,不报错,也不出现在默认输出里。反向图里的隐式回退还可能为了满足连续性多出一次拷贝。

**重算与存储的门槛是经验值。** 4 块、30 个、8 块这几个阈值对大多数图合适,但总有错的时候:一个多处使用、函数又不小的中间值被内联进每个使用者,运算重复了好几遍;或者反过来,一个便宜的值因为累计读数超了线被落地,多一次读写。softmax 在关掉在线归约时,指数就在求和与相除里各算一遍。

**分解之后,拿不到 eager 的专门实现。** 分解把一个算子换成几个原语,编译器生成的融合 kernel 通常更快,但原语的组合不一定等于 eager 那个手写 kernel 的算法。为此每个不适合分解的算子都要人工从表里拿掉、改成直接降级或回退,表是靠一条条经验维护的。

**数值和 eager 不完全一样。** 一串低精度的逐元素运算内联成一个函数后,中间值一直用 fp32 算,eager 在每一步之间的降精度再升精度被省掉了。结果通常更接近高精度参考,但和 eager 按位比对不上。随机数同理:默认用 Inductor 自己的随机数生成,序列和 eager 不同。

**编译期的开销。** 降级在 Python 里逐节点解释,图越大越久;模式替换第一次用到时要初始化,没有预录好的模式要现录一遍。

## 四、和同类大厂框架不一样的地方

对照对象是 XLA 与 TensorRT。前者是 Google 的机器学习编译器,JAX 的即时编译背后就是它;后者是 NVIDIA 推理部署的主力。两者都要面对同一个问题:前端交来的算子五花八门,后端的实现是有限的。说法按 openxla.org 与 docs.nvidia.com 的官方页核过。

| 要回答的事 | torch.compile 的 Inductor | XLA | TensorRT |
|---|---|---|---|
| 前端交来什么 | ATen 级别的 FX 图,算子集合是开放的,新算子随时出现 | StableHLO:一套带版本的操作集,作为各框架与编译器之间的可移植层 | 一层一层搭出来的网络:用 API 逐层添加,或由 ONNX 解析器把 ONNX 节点翻成层 |
| 怎么收窄到后端能处理的集合 | 录图时按分解表把大算子拆成原语,再降成循环级的下标函数 | 前端已经降到 StableHLO;XLA 在做与目标无关的优化时转成内部的 HLO | 没有单独的分解步骤:网络只能用 TensorRT 支持的层来搭,支持不了的层走插件 |
| 融合从哪来 | 原语的下标函数互相内联,调度器再按读写决定合并(10 章),对任意逐元素链通用 | 与目标无关的算子融合,之后后端再按自己的编程模型做一轮 | 构建阶段按一份已知模式的清单把几层合成一层,例如卷积接 ReLU |
| 覆盖不到怎么办 | 自动回退成调用 ATen eager kernel 的外部节点,默认不报错 | custom call 在 HLO 里描述一个外部操作,它的实现在运行时通过 FFI 注册 | 写插件(自定义层);ONNX 解析器遇到不认识的节点,会去插件注册表里找同名插件 |

一句话:XLA 要求前端先把自己翻译成一套封闭的操作集,覆盖问题在进编译器之前就解决了,外部实现要显式声明;TensorRT 按层实现、按模式清单融合,覆盖不到就写插件;torch.compile 接受 ATen 的开放算子集,靠分解表把它收窄,覆盖不到时自动退回 eager 的实现,换来「什么都能跑」,代价是回退处融合悄悄断开。

## 五、调参与观测

这一章的参数分两处:`torch._inductor.config` 里的全局配置(也能经 `torch.compile(options=...)` 或 `config.patch` 只作用于某次编译),在之后发生的编译里生效,已编好的不受影响;少数带环境变量的配置在导入 torch 时读一次,要在启动前设好。`mode="lite"` 一次改掉其中一组。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `implicit_fallbacks` | 全局配置 | `True` | `False`:遇到没有降级的算子直接报 `MissingOperatorWithDecomp` 或 `MissingOperatorWithoutDecomp`,不再悄悄回退 | 报错里带算子名与参数 |
| `fallback_random` | 全局配置 | `False` | `True`:随机类算子不分解、不用 Inductor 的随机数,回退到 ATen,结果与 eager 一致但更慢,dropout 也不再融合 | 生成代码里出现 `aten.native_dropout` 等外部调用 |
| `emulate_precision_casts` | 全局配置;环境变量 `TORCHINDUCTOR_EMULATE_PRECISION_CASTS` | `False` | `True`:低精度逐元素链在每步之间保留降精度再升精度,数值向 eager 靠拢,多几条转换指令 | 与 eager 的逐元素误差 |
| `pattern_matcher` | 全局配置 | `True` | `False`:三处的模式替换全部跳过,包括注意力识别与在线 softmax | `counters["inductor"]["pattern_matcher_count"]` 归零 |
| `use_pre_grad_passes`、`use_joint_graph_passes`、`use_post_grad_passes` | 全局配置;`TORCHINDUCTOR_LITE_MODE=1` 或 `mode="lite"` 一起关 | `True` | 分别关掉求导前、联合图、求导后三处 pass | tlparse 里对应阶段前后的两份图相同 |
| `online_softmax` | 全局配置;环境变量 `TORCHINDUCTOR_ONLINE_SOFTMAX` | `True` | `False`:softmax 的最大值与求和分成两次归约 | 生成代码里有没有 `online_softmax_reduce` |
| `realize_reads_threshold` | 全局配置 | `4` | 多处使用的中间值读的缓冲超过它就落地;调小落地更多、重算更少 | `TORCH_LOGS=ir_pre_fusion` 里 `ComputedBuffer` 的个数 |
| `realize_opcount_threshold` | 全局配置 | `None`,即 GPU 30、CPU 50 | 多处使用的中间值运算数超过它就落地 | 同上 |
| `realize_acc_reads_threshold` | 全局配置 | `None`,即 GPU 8、CPU 12 | 一路内联累计读的缓冲超过它就落地 | 同上 |
| `unroll_reductions_threshold` | 全局配置 | `8` | 归约长度小于它就展开成逐元素,不单独落地 | 同上 |
| `post_grad_custom_pre_pass`、`post_grad_custom_post_pass` | 全局配置 | `None` | 在求导后的内置模式之前、之后插入自己的图变换;要实现 `uuid()` 才能进编译缓存的键 | `TORCH_LOGS=post_grad_graphs` |
| `joint_custom_pre_pass`、`joint_custom_post_pass`、`pre_grad_custom_pass` | 全局配置 | `None` | 在联合图、求导前插入自己的变换;求导前的图未函数化,写起来最容易出错 | tlparse 的 `before_joint_graph` / `after_joint_graph` |
| `mode="lite"` | `torch.compile` | — | 除用户用区域编译显式标注的节点外,全部回退到 ATen;同时关掉三处 pass、重排与缓冲复用,只求数值与 eager 一致 | 生成代码几乎全是外部调用 |

**观测四处。** 回退:`TORCH_LOGS=inductor` 打出每一处 `Creating implicit fallback for`,后面跟着算子与参数;`TORCH_LOGS=+inductor` 再逐节点打出 `lowering ... via` 用的是哪个降级函数。模式替换:`torch._dynamo.utils.counters["inductor"]` 里 `pattern_matcher_count`、`pattern_matcher_nodes` 是命中次数与吃掉的节点数,`fuse_attention` 是注意力识别的次数;设 `TORCHINDUCTOR_PATTERN_MATCH_DEBUG` 再按模式分项计数。图:tlparse 报告里 `before_pre_grad_graph`、`after_pre_grad_graph`、`before_joint_graph`、`after_joint_graph`、`before_post_grad_graph`、`inductor_post_grad_graph` 依次是三处 pass 前后的图;`TORCH_LOGS=pre_grad_graphs,post_grad_graphs` 在终端里打。循环级节点:`TORCH_LOGS=ir_pre_fusion` 打出融合前的全部节点,默认不在 `+inductor` 里,要点名;每个 `ComputedBuffer` 是一块落地的缓冲,`FallbackKernel` 是一处回退。

**分段排除。** `backend="aot_eager_decomp_partition"` 跑 Dynamo、AOTAutograd、Inductor 的分解表与切分,但不降级、不生成代码:结果在这里就错,问题在分解;这里对、`inductor` 错,问题在降级或之后。

**典型配置**:

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 上线前确认没有算子悄悄回退 | 在 CI 里设 `torch._inductor.config.implicit_fallbacks = False` 跑一遍,线上保持默认;或线上开 `TORCH_LOGS=inductor` 收集回退清单 | 用一次报错换来回退点的完整清单,再决定是写分解、换写法还是接受 |
| 编译后精度与 eager 对不上,要定位 | 先 `backend="aot_eager_decomp_partition"`;分解没问题再用 `inductor` 加 `options={"emulate_precision_casts": True, "fallback_random": True}` | 用更慢的运行换来与 eager 可比的数值,排查完关掉 |
| 只想编译少数几段,其余保持 eager 的数值 | `torch.compile(model, mode="lite")`,再对要融合的区域做区域编译标注 | 放弃大部分融合,换来除标注区域外与 eager 一致的数值 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 某段逐元素链没融成一个 kernel | `TORCH_LOGS=inductor` 有没有这段里的算子在 `Creating implicit fallback for` 里 | 给它写一个分解到已有原语的函数(`register_decomposition` 注册进 Inductor 的表),或换成有降级的写法 |
| 报 `MissingOperatorWithDecomp` | 是不是关了 `implicit_fallbacks`,而这个算子其实有分解却不在 Inductor 的表里 | 按报错提示把它加进分解表,或打开隐式回退 |
| 手写注意力没被换成 SDPA | `counters["inductor"]["fuse_attention"]` 是否为 0 | 对照 `fuse_attention.py` 的 30 种写法调整,或直接调 `F.scaled_dot_product_attention` |
| 编译后 bf16 输出与 eager 差在末位 | 关 `emulate_precision_casts` 与开它各跑一次 | 开了就对上,说明只是省掉了中间的降精度,不是错误 |
| 训练中 dropout 的掩码与 eager 不同 | 是否依赖逐位复现 eager 的随机序列 | 开 `fallback_random`,只在调试时开 |
| 生成代码里同一个指数算了两遍 | `ir_pre_fusion` 里这个中间值是不是没落成 `ComputedBuffer` | 多数时候是划算的重算;确实太重时调小 `realize_opcount_threshold` |
| 自定义 pass 在求导前改图后结果偶尔出错 | pass 是不是碰到了原地修改或别名 | 把它挪到 `post_grad_custom_pre_pass`,那里的图已经函数化 |

## 六、常见误区

**「图里有 `aten._softmax`,生成代码里就会有一个 softmax kernel。」** 看 Dynamo 或 AOTAutograd 日志的人容易这么以为。实际 softmax 在录联合图时就被拆成了 5 个原语,GPU 上又被换成在线归约加一个逐元素函数,和前后的逐元素算子一起融进别的 kernel;代码里找不到叫 softmax 的东西,找得到 `online_softmax_reduce`。

**「编译没报错,说明每个算子都被编译了。」** 因为很多编译器缺实现就直接报错。这里缺降级默认悄悄生成隐式回退,只有 INFO 级日志;一个模型可能有好几处回退,每一处都切断一次融合,profiler 里看到的是一串和 eager 同名的 ATen kernel 夹在 Triton kernel 之间。

**「回退只是这一个算子慢一点。」** 回退的真正代价在它两边:它的输入必须先落成完整的缓冲,它的输出后面的节点要重新从显存读,调度器又不和它合并。一个回退能让原本一个 kernel 的链变成三段。

**「中间结果落地得越少越好。」** 落地少意味着内联多,一个被五处使用的中间值不落地,就在五个地方各算一遍。Inductor 用读数与运算数的阈值做取舍,大多数时候对,但对于又贵又被多处使用的中间值,落一次地反而更快。

**「编译后 bf16 的数和 eager 不一样,是编译器的 bug。」** 融合后的逐元素链中间值一直留在 fp32,比 eager 每一步都舍入到 bf16 更准,只是和 eager 按位不同。开 `emulate_precision_casts` 能对上;对上了就说明不是 bug。

**「开 `fallback_random` 就能让随机数和 eager 一样,别的不受影响。」** 它让随机类算子全部回退到 ATen,dropout 从融合的逐元素链里被拆出来单独调 kernel,性能明显下降;配置注释自己写着「慢,但调试有用」。默认下编译版的随机序列就是和 eager 不同,依赖逐位复现的测试要么只在调试时开它,要么改成比统计量。

**「写自定义图变换,放在最前面的求导前阶段最省事。」** 那张图最接近用户代码,看起来最好认。但它没有函数化,原地修改、别名、各种参数写法都还在,源码注释明确建议新 pass 写在联合图或求导后;`post_grad_custom_pre_pass` 看到的是已经分解、函数化、切好前后向的 ATen 图,模式也更稳定。

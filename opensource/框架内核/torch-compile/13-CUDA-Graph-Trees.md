# CUDA Graph Trees

这一页回答一件事:编译产物怎么被录成 CUDA graph 整段回放,一个训练步里前向、反向、多个编译区域录出的好几张图,又怎么共用同一块显存而不互相踩坏。说法都在源码基准 `217579124a`(main,仓库自报 `2.15.0a0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

10 章的融合把 kernel 数压下来了,但压不到一。小批量推理、小模型训练里,每个 kernel 只跑几微秒,CPU 逐个 launch 的开销和它同一量级,GPU 在两个 kernel 之间空等。CUDA graph 把整串 launch 录成一个对象一次提交,为什么这样能省、代价是地址和形状被冻结,知识库「CUDA Graph」篇讲过,这里不重复。

难的是把它套在编译产物上。torch.compile 编出来的不是一张图:训练时前向和反向各编一份(07 章);一处断图就把函数切成前后两段(05 章);每换一种形状又是一份。每一份都要录成自己的 CUDA graph,每张图的输入、输出和中间缓冲都得待在固定地址。

于是两头骨折。**每张图各开一个私有池**,地址互不干扰,但前向存给反向的激活在前向的池里,反向的池没法回收它,前向、反向、每个区域、每种形状各占一份,显存成倍上涨;官方文档点名的就是这个毛病。**所有图挤在一个池里**,手写的做法要求这些图永远按录制时的顺序调用,一旦顺序变了、或者用户手里还攥着上一张图的输出,后一张图就会把那块显存当成空闲的写掉,结果静默出错。

还有两件事让它更难。一是编译产物的输出要交还给用户的 Python 代码,用户可能把它存起来跨步使用,而下一次回放一定写回同一个地址。二是有些算子根本录不进图:要回 CPU 取值的、在 CPU 上算的、输出形状取决于数据的。

所以这一章的问题是:**一个进程里任意多张图怎么共用一块显存,既能让后面的图复用前面已经不用的地址,又不写坏任何还活着的张量;录不了的部分怎么办。**

## 二、解法

最天然的直觉是把池当成一条磁带:按执行顺序录,后一张图在前一张图留下的活张量旁边分配,回放时顺序不变,分配的样子也就不变。CUDA Graph Trees 就是这条磁带,只是允许它在某一处岔开:同一个位置之后走了不同的路,就在那里长出一个新分支,于是成了一棵树。

![同一个训练步里区域 1 前向、区域 2 前向、区域 2 反向、区域 1 反向四张图按执行先后排成一条链,区域 1 前向之后还挂着另一分支区域 2 前向′,两条分支共用同一段池内存;下方是同一个池里的四段地址:区域 1 存下的激活从区域 1 前向活到区域 1 反向,区域 1 的输出 h 不经拷贝直接交给区域 2 前向、死后地址被区域 2 反向写出的梯度复用,区域 2 存下的激活死后地址被区域 1 反向的输出复用,用户拿走的 loss 在下一步回放区域 2 前向时被写回同一地址覆盖。这是按源码规则推出来的示意,不是运行结果](/opensource/torch-compile/13a-one-pool-tree.svg)

**一份编译产物要调三次才进入快路径。** 第一次照常执行,叫热身:cuBLAS 初始化、Triton 调优这类只该做一次、又往往带着 CPU 同步的事都在这时做完,录制时就不会被录进图里;热身已经在共享池里分配,不多占一份。第二次录制:在共享池里捕获整段 kernel,捕获本身不执行,录完立刻回放一次拿到真结果。第三次起检查通过就只回放。热身期间地址还没定,所以一条路径上只要有一张图在热身,它后面的图也只能跟着热身。

**树上的一个节点是一份编译产物的一次录制。** 它的父节点是紧挨着它之前执行、而且输出还活着的那张图;调用时如果池里已经没有上一张图活着的输出,它就是一棵树的根。每个节点按「接下来调用的是哪份编译产物」存着自己的孩子。调用时在当前节点的孩子里找同一份产物,挨个检查三件事:从前面的图接过来的输入,地址和录制时一样;录制时已经死掉的输出,现在也已经死了;参数与缓冲的地址没变。全部成立就回放。都不成立,就在当前节点下再录一个孩子。断图后面跟着 if/else 的函数,两个分支各自长成一条枝;两条枝接在同一个父节点之后,用的是同一段内存,池的峰值取两条路径里较大的那条,不是两者之和。

**共用一个池靠两件事。** 第一,每块卡只有一个池和一条专用的流:缓存分配器只会把一段显存再分给当初分配它的那条流,流不同就复用不了。第二,每录完一个节点,把分配器的记账拍一张快照。回放只重放 GPU 上的 kernel,CPU 侧的记账不会跟着变;所以在回放过的路径上要再录一个新孩子时,先把分配器恢复到父节点那张快照,告诉它哪些块现在还活着,把录制之后已经死掉的块真正释放。新图就会绕开活着的张量,复用已死的地址。

**输入分三类进图。** 普通的 eager 输入每次回放前拷进池里的固定缓冲;参数与缓冲假定地址不变,不拷,只核对地址,变了就重录;前面的图的输出,本来就在池里固定的地址上,原样传进来,不拷。前向交给反向的激活属于第三类,这正是共用一个池省下来的东西:反向就在前向写下它们的地方读。

**一步何时结束,用「代」来判。** 每进入一次编译过的函数,全局代数加一。新的一代能不能开始,看有没有前向在等反向:前向跑完把标记置上,反向跑完清掉。能开始新一代时,树结束当前路径,上一代的输出不再挡路,下一次回放可以覆盖它们。推理里每次调用就是一步;训练里一次前向加一次反向是一步;启发式判断不了时,用户可以手动标记一步的开始。

**输出被覆盖,怎么让人知道。** 稳定回放时,每次返回的都是同一批张量对象,省掉重建的开销;所以上一步拿走的输出,在这一步回放之后就是新值,不会报错。报错只发生在另一种时刻:树在热身或录制的交界处结束一条路径时,会真正释放上一代留在池里的输出,并给它们挂上一条错误信息,再访问就抛出「已被后续运行覆盖」。另有一个可选的做法,在新一代开始前把用户可见的输出拷出池外。

**录不了的,整份跳过或者切开录。** 编译完成时先检查整份产物:改写了来自 eager 的输入、含 CPU 上的算子、跨多个设备、含输出形状取决于数据的算子、含一张固定清单里的禁录算子或被用户标成不安全的自定义算子、有非张量输入、缓存分配器被关掉,任一成立,这份产物就不录,照常执行并打一条警告。开源构建默认打开图切分,这时调度器在 CPU 算子、跨设备拷贝、控制流、输出形状取决于数据的算子和不安全的自定义算子处把图切开,只把 GPU 段各自录成图,切出去的部分夹在中间照常执行;切之前还会试着重排节点,把能挪的不可录算子挪到开头或结尾,估计峰值显存涨幅不超过 1.1 倍才采用。切分只处理算子,改写 eager 输入这类整份的原因照样让整份不录。

**形状一变,另录一套。** 一份编译产物如果带符号形状,按调用时那几个整数输入的取值分别录制,每个取值一套独立的树节点;超过 8 种取值打一条警告,也可以限定只录哪几种。

**为什么实际不吃亏。** 被录下来的那部分,内存都在一个池里按执行顺序复用,官方文档的说法是和 eager 相比没有额外的显存开销;回放时的检查只是比较一组地址和存活标记,在 C++ 里做,开销远小于逐个 launch。

## 三、代价

**普通输入每步都要拷一次。** 从 eager 代码传进来的张量地址每步都在变,只能先拷进池里的固定缓冲再回放。输入很大、而计算很小时,这次拷贝会吃掉一部分收益。

**前三次调用都不快。** 第一次热身、第二次录制,第三次才回放;每出现一种新形状、每长出一条新分支,这个过程再来一遍。而热身会传染:同一条路径上前面有图在热身,后面已经录好的图也只能跟着按热身跑。

**池里的显存不还给通用分配器。** 池只在它的图里复用,图外 eager 代码的分配走另一个池;两边都占得多时,总量比纯 eager 高,官方文档的对照表把这列为唯一会让显存上涨的情形。形状多时还有一笔账:官方文档写明,CUDA 12.4 与 550 版驱动之前,图里每个 kernel launch 要占 64 KB 显存。

**重录有上限,超了就回 eager。** 同一份产物在同一个父节点下,因为地址或存活情况对不上而重录,默认最多 128 次,超了这个位置就不再录,直接照常执行;参数地址变化引起的重录不计入这个数。前面的图的输出作为输入、地址却一再变化的,5 次之后改为把它拷进固定缓冲,不再为它重录。

**判断「一步结束」的启发式会错。** 训练循环里只跑前向、不跑反向的调用(开着梯度算一次指标),会让「有前向在等反向」的标记一直挂着,新的一代开始不了:后面的调用只能挂在当前路径下继续录,树越长越深,快路径走不进去,只留一条警告。

**切开录有两笔额外开销。** 切口处回到 eager 执行,切出去的算子越多,launch 越回到老样子。前向被切成多段时,段与段之间 eager 代码分配的张量不在池里,地址每步都变,反向不能再把它们当作地址固定的输入,只有参数还按固定地址处理,其余的每次回放都要按普通输入的方式检查和拷贝。

## 四、和同类大厂框架不一样的地方

对照对象是 XLA 的 GPU 后端与 TensorRT。前者是 Google 在 JAX 训练与推理里用的编译器,它把编译出的 kernel 与库调用序列转成一层运行时中间表示,在上面抽出 CUDA graph;后者是 NVIDIA 的推理优化器,执行上下文可以被用户录进 CUDA graph。以下关于它们的说法都只取自官方文档。

| 三家都得回答的问题 | torch.compile 的 CUDA Graph Trees | XLA 的 GPU 后端 | TensorRT |
|---|---|---|---|
| 谁决定录哪一段 | 编译器自动录每份编译产物;录不了的整份跳过,或开图切分只录能录的段 | 编译器在那层中间表示上抽出 CUDA graph 的边界,再整体编成 CPU 端可执行文件;文档说这部分仍在完善,只支持部分节点;另有一个开关列出哪几类命令被捕获进命令缓冲 | 用户自己捕获一次推理调用,之后用 CUDA graph 的接口回放 |
| 录不了怎么办 | 打警告后照常执行,不报错 | 文档只说目前支持部分节点,没写其余节点怎么处理,不作比较 | 循环、条件、数据相关形状让捕获失败并返回错误;这个上下文仍能不带图照常推理 |
| 输入形状变了 | 每种取值自动另录一套,池照旧共用 | 01 章已核:JAX 每遇到新形状就重新编译,图的边界跟着新的编译产物走 | 形状变了先正常调用一次刷新内部状态,再重新捕获;捕获下来的图只对那一组输入尺寸与上下文状态成立 |
| 多张图的显存 | 一个池,按执行先后复用,活着的张量有人追踪 | 文档没有写多张图之间怎么分显存,不作比较 | 输入输出缓冲与激活内存的地址都录进图;改动录图时用的上下文属于未定义行为;推荐一张图一个执行上下文,多个上下文共用一块由用户提供的激活内存 |

一句话:TensorRT 把录制和显存安排都交给用户,规则简单但一切靠人守;XLA 在编译期定好图的边界;torch.compile 面对的是会断图、会分叉、输出会被用户拿走的 Python 程序,只好在运行时维护一棵树和一套存活追踪,换来的是自动,代价是热身、重录与「一步结束」的启发式。

## 五、调参与观测

这一章的参数分四处:编译入口的 `mode`,包一次就定下来;Inductor 的全局配置 `torch._inductor.config`,在 Python 里改,对之后的编译生效,也可以经 `options` 只对一次编译调用生效(`mode` 与 `options` 不能同时传,见 10 章);少数键另有环境变量,导入时读取,要在进程启动前设好;运行期接口随时调用,立刻生效。表里 `triton.` 开头的都在 `torch._inductor.config.triton` 下。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `mode="reduce-overhead"` | `torch.compile` / `Module.compile` | `default`(不开) | 等价于 `triton.cudagraphs=True`;`max-autotune` 也开 | `torch._inductor.list_mode_options()` |
| `triton.cudagraphs` | 全局配置、`options`,或 `TORCHINDUCTOR_CUDAGRAPHS=1` | `False` | 总开关;配合 `options` 里其他键时用它代替 `mode` | `TORCH_LOGS=cudagraphs` 出现 `Recording cudagraph tree` |
| `triton.cudagraph_trees` | 全局配置 | `True` | `False` 退回每张图各录各的旧实现,池不共享,显存明显上涨 | — |
| `graph_partition` | 全局配置,或 `TORCHINDUCTOR_GRAPH_PARTITION` | 开源构建 `True` | 关掉:含不可录算子的整份产物不录;开着:切开只录 GPU 段 | `TORCH_LOGS=cudagraphs` 的 `Created N graph partitions: A cudagraphable, B non-cudagraphable` |
| `custom_should_partition_ops` | 全局配置 | `[]` | 列出的算子(`namespace::name`)强制在它处切开 | 同上,切分原因写 `custom partition op` |
| `triton.cudagraph_min_partition_size` | 全局配置 | 0(不检查) | 设成 N:kernel 数少于 N 的段不录,省掉录小段的开销 | 同上日志的 DEBUG 行 `below minimum size` |
| `triton.reorder_for_reducing_graph_partitions` | 全局配置 | `True`(lite 模式下 `False`) | 关掉:不再为减少段数重排节点 | 段数变化 |
| `triton.cudagraph_partition_memory_budget` | 全局配置 | 1.1 | 重排后估计峰值不超过原来的这个倍数才采用;调大段更少、峰值可能更高 | — |
| `triton.cudagraph_skip_dynamic_graphs` | 全局配置 | `False` | `True`:带符号形状的产物不录(开图切分时改为把用到符号的节点切出去) | 警告 `graph with symbolic shapes inputs` |
| `triton.cudagraph_capture_sizes` | 全局配置 | `None`(全录) | 列出要录的整数取值,其余取值照常执行 | `Recording cudagraph tree for symint key` |
| `triton.cudagraph_dynamic_shape_warn_limit` | 全局配置 | 8 | 不同形状数超过它打警告;设 `None` 不再提醒 | 警告 `We have observed N distinct sizes` |
| `triton.cudagraph_support_input_mutation` | 全局配置 | 开源构建 `True` | 允许改写参数、缓冲和前面图的输出;改写 eager 输入仍整份跳过 | 警告 `skipping cudagraphs due to mutated inputs` |
| `triton.cudagraph_unexpected_rerecord_limit` | 全局配置 | 128 | 同一父节点下同一份产物的重录上限,超了这个位置改走 eager | 警告 `exceeding max re-recording limit` |
| `triton.cudagraph_managed_input_rerecord_limit` / `triton.cudagraph_managed_input_rerecord_action` | 全局配置 | 5 / `"copy"` | 前面图的输出作为输入、地址变了这么多次后:`copy` 改为拷进固定缓冲,`skip` 这个位置改走 eager | `skip` 时警告 `re-recording threshold` |
| `triton.cudagraph_trees_generation_cloning` | 全局配置 | `None` | `"user_visible"`:新一代开始前把还活着的用户可见输出拷出池外,不再报覆盖错误;多一次拷贝 | 报错信息里会建议这个选项 |
| `triton.cudagraph_initial_mempool_allocation_gb` | 全局配置 | `None` | 建池时先占这么多 GiB 再释放给池,减少池分段增长带来的碎片 | `torch.cuda.memory_snapshot()` 里池的段数 |
| `triton.cudagraph_or_error` | 全局配置,或 `TORCHINDUCTOR_CUDAGRAPH_OR_ERROR=1` | `False` | `True`:任何跳过都直接抛错,用来确认没有漏录 | 抛出的 `RuntimeError` 就是跳过原因 |
| `triton.force_cudagraph_sync` | 全局配置 | `False` | 每次回放后同步,排查异步错误用,会变慢 | — |
| `torch.compiler.cudagraph_mark_step_begin()` | 运行期,每步开始前调用 | — | 手动开始新的一代,上一代的输出可以被覆盖 | 「无法走快路径」的警告消失 |
| `torch._dynamo.mark_static_address(t)` | 运行期,编译前对张量调用 | — | 把某个非参数张量当作地址固定的输入,不再每步拷贝;地址真的变了会重编译(`guard=True`)或重录 | `TORCH_LOGS=cudagraph_static_inputs` |
| `backend="cudagraphs"` | `torch.compile` | — | 只录 CUDA graph、不经 Inductor,用来二分问题出在编译还是出在录制 | — |
| `torch._dynamo.reset()` | 运行期调用 | — | 除了清编译缓存,也清掉所有 CUDA Graph Trees 与池 | 下一次调用重新热身、录制 |

**观测三处。** 为什么没录:跳过的原因以 `skipping cudagraphs due to ...` 打成警告,默认就能看到,次数累计在 `torch._dynamo.utils.counters["inductor"]["cudagraph_skips"]`,切分次数在同一处的 `cudagraph_partitions`。录了几次、为什么重录:`TORCH_LOGS=cudagraphs` 打出每次热身 `Running warmup function`、每次录制 `Recording function=..., mode=FORWARD/BACKWARD/INFERENCE`,以及每次重录的原因 `Re-recording function=..., reason=`,原因分三种:`cudagraph managed tensor data pointer changed`、`static input data pointer changed`、`expected dead indices before graph are live`。回放到底省了多少:profiler 里被录下的部分显示为整段图的 launch,和 eager 对照看 kernel 之间的空隙是否消失。

**典型配置**:

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 小批量推理,形状固定或已分桶 | `torch.compile(model, mode="reduce-overhead")`;先加 `TORCHINDUCTOR_CUDAGRAPH_OR_ERROR=1` 跑一遍确认没有跳过,再去掉;批大小变化时补到少数几档,或设 `triton.cudagraph_capture_sizes` | 用常驻的池和前三次调用的热身、录制,换掉每步的 CPU launch 开销 |
| 训练,循环里会插评估或攒梯度 | `mode="reduce-overhead"`;每步开头调 `torch.compiler.cudagraph_mark_step_begin()`;评估放在 `torch.no_grad()` 里;要攒梯度就在编译过的反向第一次运行前把 `.grad` 预分配好,不攒就每步前置 `None` | 多写两三行代码,换「一步结束」的判断不出错、树不越长越深 |
| 模型里有少量 CPU 算子或数据相关形状,又想吃到 CUDA graph | 保持 `graph_partition` 开着,`options={"triton.cudagraphs": True, "triton.cudagraph_min_partition_size": 3}` | 切口处的 eager 执行与段间张量的拷贝,换大部分 GPU 段能回放;太碎的小段不录,省掉录制开销 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 开了 reduce-overhead 速度几乎不变 | 警告里有没有 `skipping cudagraphs due to`;`cudagraph_skips` 计数 | 按原因处理:`mutated inputs` 改写法不改 eager 输入;`cpu device` 确认图切分开着;`incompatible op` 看是哪个算子 |
| `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` | 报错里带的栈是哪个输出 | 在编译区域外 `.clone()` 要跨步保留的输出;或每步前调 `cudagraph_mark_step_begin()`;或设 `triton.cudagraph_trees_generation_cloning="user_visible"` |
| 报错说梯度输出被覆盖 | 是不是在 reduce-overhead 下攒梯度,`.grad` 是在录制期间第一次分配的 | 按报错建议,在编译过的反向第一次跑之前把 `.grad` 分配好,或不攒梯度时每步置 `None` |
| 警告 `Unable to hit fast path of CUDAGraphs because outputs from a previous step still require backward` | 有没有只跑前向不跑反向的调用 | 那次调用放进 `torch.no_grad()`,或每步开头 `cudagraph_mark_step_begin()` |
| 显存比不开 CUDA graph 时高很多 | 形状种类数(`distinct sizes` 警告);`TORCH_LOGS=cudagraphs` 里录了多少次 | 补齐到少数几档;设 `triton.cudagraph_capture_sizes`;确认 `triton.cudagraph_trees` 没被关 |
| 每步都在重录 | `TORCH_LOGS=cudagraphs` 的 `reason=` | `static input data pointer changed`:参数被换了对象,查是否每步新建模块或重新赋值参数;`expected dead indices before graph are live`:用户拿着上一张图的输出跨过了下一张图 |
| 部分调用比别的明显慢 | 有没有 `exceeding max re-recording limit` | 那个位置已经永久走 eager;先消除重录原因,再 `torch._dynamo.reset()` 重来 |

## 六、常见误区

**「开了 reduce-overhead,整个模型就都在 CUDA graph 里了。」** 模式名让人以为这是全局开关。实际上每份编译产物各自判断,一份跳过只打一条警告就照常执行,程序不报错、只是没变快。上线前用 `TORCHINDUCTOR_CUDAGRAPH_OR_ERROR=1` 跑一遍,是确认没有漏录最省事的办法。

**「上一步的输出存在变量里,值就不会变。」** eager 下每次调用都分配新张量,这个直觉一直成立。这里稳定回放时每次返回的是同一批张量对象,下一次回放写回同一块显存,变量里的值就悄悄变成了新一步的,不报错。只有树结束路径、真正释放旧输出的那几个时刻才会抛「已被覆盖」的错误,所以这个错是偶然碰上的,静默读到新值才是常态。要跨步保留,在编译区域外 `.clone()`。

**「图切分要手动打开。」** 官方文档写的是「请设置 `graph_partition=True` 来启用」,很多人照做后以为自己改了什么。基准上开源构建里它默认就是开的(环境变量 `TORCHINDUCTOR_GRAPH_PARTITION` 缺省为 `1`),反倒是有人为了排查把它关掉后忘了开回来,含 CPU 算子的整份产物从此不录。

**「改写输入的函数要先打开 `cudagraph_support_input_mutation`。」** 文档的示例里显式设了它,提示框里写的配置路径还少了一层 `triton.`。实际上开源构建里它默认是 `True`,而且它放开的只是改写参数、缓冲和前面图的输出;改写从 eager 传进来的输入,开不开都整份跳过,要改写法。

**「攒梯度在 reduce-overhead 下照常工作。」** 攒梯度依赖 `.grad` 跨步保留。如果 `.grad` 是在反向第一次被录制时才分配的,它就落在池里,下一次回放反向会写回同一块地址,攒下来的值被覆盖。源码为这个场景单独写了一条报错,建议在编译过的反向第一次运行前把 `.grad` 分配好。

**「训练循环里顺手算个指标不影响。」** 开着梯度跑一次前向而不跑反向,「有前向在等反向」的标记就一直挂着,之后的调用开始不了新的一代,只能挂在当前路径下继续录,树越长越深。症状是显存和编译日志里的录制次数一起涨,只有一条 `Unable to hit fast path` 的警告提示。评估放进 `torch.no_grad()`,或者每步开头手动标记。

**「第二次调用就该快了。」** 01 章说编译要付「前一两次调用」的账,人们于是从第三次开始计时就觉得稳了。reduce-overhead 在编译之外还要热身一次、录制一次,每种新形状、每条新分支都重来,前面某张图在热身时后面的图也跟着按热身跑。测性能时,要在所有形状与分支都录过之后再计时。

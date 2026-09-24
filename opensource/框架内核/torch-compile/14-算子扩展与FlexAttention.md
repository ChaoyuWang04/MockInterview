# 算子扩展:自定义算子、用户 Triton kernel 与 FlexAttention

这一页回答一件事:编译器看不透的东西(自己写的 kernel、按张量的值走的分支与循环、五花八门的注意力变体)怎么作为图里的一个节点进来,而不是把图打断。说法都在源码基准 `217579124a`(main,仓库自报 `2.15.0a0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

Dynamo 能模拟的只有 Python 和 torch 的算子(02、03 章)。可模型里最值钱的那几段,偏偏常常是它看不透的。

**自己写的 kernel。** 一段调用 C++ 或 CUDA 扩展的函数,Dynamo 钻不进去,默认在调用处断图(05 章)。一段 Triton kernel,Dynamo 看得见调用它的那行 Python,却看不见 kernel 读了哪些张量、写了哪些张量;不知道写了谁,编译器就不敢重排、不敢复用缓冲。这类 kernel 往往正是热点,断在这里,热点两边的算子全都失去融合。

**按张量的值走的分支与循环。** 05 章讲过:这样的 if 会把图切开,这样的循环更糟,整个函数退回 eager。

**注意力变体。** 相对位置偏置、滑动窗口、文档打包掩码、分数上限截断,每一种都要一个手写的融合 kernel;FlashAttention 这类库只覆盖固定的几种。用普通 PyTorch 写,编译器只会照原样把「矩阵乘、改分数、softmax、矩阵乘」融成几个 kernel,中间那张长乘长的分数矩阵照样落显存:编译器不会自己换成分块加在线 softmax 的算法(知识库 TorchCompile 篇第八节)。长序列训练里,显存随序列长度的平方涨,先爆的就是这里。

于是两头骨折:**不进图**,图碎、热点孤立;**自己写 kernel**,又回到第一种情况。

所以这一章的问题是:**怎么让编译器看不透的计算作为图里的一个节点进来,并把形状、读写、反向这些信息交给编译器,让它在节点外照常优化;对注意力再进一步,让用户只写变体不一样的那一点,其余交给模板生成。**

## 二、解法

最天然的直觉是给黑盒套一层编译器看得懂的外壳:外壳上写清输出长什么样、改了哪些输入、反向怎么算,里面是什么编译器不必知道。torch.compile 按外壳的透明程度给了三种办法,FlexAttention 是第三种的招牌用法。

**自定义算子:外壳最厚,里面完全不看。** 把函数注册成一个带命名空间的算子,签名从类型注解推出来,同时声明它会改哪几个参数。再注册一个假实现:只根据输入的形状、类型、设备算出输出的样子,不碰数据,抓图时的假张量(02 章)走的就是它。注册之后,它在 Dynamo 眼里和 ATen 算子一样是一个 torch 算子,官方文档的说法是 torch.compile 永远不会钻进去,整个调用作为一个节点进图。反向要另外注册求导公式。到了 Inductor,它没有降级规则,走 09 章的隐式回退成为一个外部 kernel 节点;和内置算子不同,默认要求输入的步长与 eager 时一模一样,编译器不许替它换布局。

**会改输入的算子:先变成纯函数,再改回原地。** 07 章的函数化要把原地修改改写成返回新值,可它不知道一个黑盒的非原地版本是什么。办法是自动包一层高阶算子:语义是把要改的参数先拷一份,在拷贝上调原算子,把拷贝当新值返回。图因此是纯的。到 Inductor 求导之后的阶段再逐个判断:被改的张量(以及和它共享存储的视图)之后没人再读,并且它若是图的输入、原程序本来就要改它,就把拷贝删掉,直接原地调用;判断不了的保留拷贝。

**用户 Triton kernel:外壳透明,编译器看得见里面。** 在编译区域里直接按网格调用一个 Triton kernel,Dynamo 认得它,把这次调用换成一个高阶算子节点;kernel 对象本身放进一张旁表,图里只记编号、网格与参数。读写靠分析得出:让 Triton 把 kernel 编到它的中间表示,从每条写内存的指令往回追,看写的是哪个参数的指针;分析失败时保守地当作所有张量输入都被写。知道写了谁,就同样走「先函数化、后改回原地」。Inductor 把 kernel 的源码原样拷进自己生成的代码,和自己生成的 kernel 一起异步编译、一起启动,用户在调优装饰器里给的几组配置交给 Inductor 的启动器实测。想把它包成一个正式的算子(有名字、能导出、能给张量子类定义行为),另有一种注册方式:注册出来仍是自定义算子,但在 AOTAutograd 函数化时会被拆开,Inductor 看到的是里面的 kernel 调用,不是黑盒。

**控制流:把一段函数整体抓成图里的一个节点。** 条件、多路分支、while 循环、scan、map 这些算子的参数本身就是函数。Dynamo 碰到它们时开一个子追踪器,把每个分支或循环体单独模拟一遍,各抓成一张子图,挂在外层图上;调用处只是一个节点,两条分支都留在图里。子图的规矩比外层严:里面不许断图,不许改外层的 Python 对象,求梯度时不许原地改输入、输出不许和输入别名。谓词是 Python 常量时不建子图,直接特化成一支并给一条警告。到 Inductor,条件在主机侧读出谓词的值再启动对应分支的 kernel;while 循环也是主机侧的循环,每轮先算条件;scan 和 map 先被改写成 while 循环。同一套子图机制还有一个不为控制流、只为复用的用法:把结构相同的层标成嵌套编译区域,只编一次,每层复用(编译耗时见 16 章)。

**FlexAttention:只写变体的那一点。** 用户交出两个小函数:打分改写函数拿到一个格子的分数和它的批、头、查询位置、键位置,返回改写后的分数;掩码函数拿到后四样,返回这个格子算不算。

![FlexAttention 的一次前向:左边是用户写的两个小函数,打分改写函数对每个格子的分数做逐元素改写,掩码函数判断这个格子算不算,Dynamo 把两者各抓成一张子图;中间是按 128 乘 128 分块统计出的块掩码,以因果掩码、查询与键长度都是 512 为例,对角线 4 块是部分块,下三角 6 块是整块,上三角 6 块整块跳过,存成部分块与整块两张按行的索引表;右边是注意力模板生成的一个 Triton kernel,每个程序负责一个查询块,以第 1 行为例,先循环部分块,算完 QK 转置后依次执行内联的打分改写与掩码再做在线 softmax 累加,再循环整块,只执行打分改写不跑掩码,不在表里的块既不读 K、V 也不计算](/opensource/torch-compile/14a-flex-score-mod-block-mask.svg)

**两个函数怎么进图。** FlexAttention 本身是一个高阶算子。Dynamo 用 5 个 0 维张量调用打分改写函数,抓成一张子图;用 4 个调用掩码函数,抓成第二张;函数闭包里用到的张量(比如每个头一个斜率的偏置表)被提成子图的额外输入。

**块掩码决定哪些块根本不算。** 创建块掩码时,把掩码函数在整个查询乘键的网格上求一遍,按默认 128 乘 128 的块数一数每块里有多少格要算:全都要算的是整块,一部分要算的是部分块,一格都不算的不进任何表。结果存成两张按行排列的索引表,每个查询块一行,列出它要访问的键块。

**子图内联进模板。** Inductor 降级时,两张子图不当外部调用,而是降成循环级的表达式,填进注意力模板预留的两个插槽。模板本身就是分块加在线 softmax 的那套写法(知识库 FlashAttention 篇),每个程序负责一个查询块,按两张表各循环一遍:部分块上先跑打分改写、再跑掩码;整块上只跑打分改写;没列出的块,K、V 不读,矩阵乘也不做。反向同样由模板生成:对打分改写子图求导得到反向子图,再内联进反向模板。查询长度短于 128、批与头数是静态的解码场景,默认换一个专用模板,把键序列切成多段并行。模板的挑选沿用 12 章的实测机制,这里不重复。

**为什么实际不吃亏。** 外壳上的信息只在编译时用一次。运行时,自定义算子就是一次普通调用,用户 kernel 就是一次启动,FlexAttention 就是一个融合 kernel,分数矩阵从不落显存;因果掩码下序列越长,整块跳过的比例越接近一半。

## 三、代价

**自定义算子是融合的断点。** 它是外部 kernel 节点,前后的逐元素运算融不进来,输入要先完整写回显存(09 章)。默认的精确步长要求还可能在它前面多出一次拷贝,把编译器排好的布局改回 eager 的样子。

**声明错了,编译期看不出来。** 声明会改哪些参数这一项,文档字符串原话是必须准确,否则行为未定义:漏写一个,函数化以为它没被改,编译后的结果可能悄悄不对。假实现算错了形状,编译期一路照错的形状往下走,运行时才出问题。没注册求导公式,前向照常,反向时报错。输出形状取决于数据的算子,假实现里要申请一个无依据的符号(06 章)。

**自动函数化的拷贝不一定消得掉。** 被改的张量之后还被读,或两个参数共享存储,拷贝就留下:多一份显存加一次拷贝 kernel,只在 INFO 级日志里记一笔。

**用户 kernel 的读写分析是有边界的。** 它依赖 Triton 的中间表示,碰到不认识的写法就退回「全部输入都被写」,拷贝跟着变多。调用必须带网格;同一个 kernel 套多层调优装饰器、在调用处直接传 CTA 数都不支持;标成编译期常量的参数按值特化,换一个值就是另一份 kernel。默认也不把后面的逐元素尾巴融进用户 kernel。

**控制流算子规矩多,运行时要回主机。** 官方文档把这组算子标为原型特性。子图里任何一处断图,都不会像外层那样切开续跑,而是整个调用报错。谓词是张量时,要读回主机才知道跑哪一支,每次调用一次同步;while 循环的条件每一轮都要读回一次。分支与循环体各自编译,和外面的算子融不到一起。while 循环要求反向时,前向把每一轮的状态堆叠存下,显存随轮数线性涨。

**FlexAttention 的粒度是块。** 掩码的边界比 128 细的地方都是部分块,每个格子都要跑掩码函数;块太多是部分块时,收益会明显缩水。头数和每头维度被标成静态,换一次就重编;每头维度小于 16 不支持。块掩码必须和当前的查询、键长度一致,长度一变就要重建。创建块掩码的函数不编译时,会先把批乘头乘查询乘键的整张布尔掩码算出来。默认模式下每种形状只有一个前向配置,不实测;要挑配置得打开最大调优那档模式。不在编译区域里调用时,它退回一个把整张分数矩阵算出来的参考实现,只警告一次。

## 四、和同类大厂框架不一样的地方

对照对象是 JAX(FFI、Pallas 与结构化控制流原语)与 TensorFlow 的自定义算子。三家都要让编译器接纳自己看不透的计算,分歧在外壳上写什么、由谁来写,以及能不能让编译器看进 kernel 里面。关于两家的每一条都核自 docs.jax.dev 与 tensorflow.org 的官方页面。

| 三家都得回答 | torch.compile | JAX | TensorFlow |
|---|---|---|---|
| 现成的 C++ 或 CUDA kernel 怎么接进编译后的程序 | 注册成自定义算子,前端不钻进去,整个调用是图里一个节点,后端当外部 kernel 调 | 用 FFI 注册一个外部函数目标,在即时编译的函数里调用;文档说它对 JAX 是不透明的黑盒 | 用 C++ 注册一个新算子,再为 CPU、GPU 等分别实现 kernel |
| 输出的形状由谁给 | 注册一个假实现,按输入的元信息算出输出 | 每次调用时直接传入输出的形状与类型 | 注册算子时附一个形状函数,构图时推断输出形状、检查输入是否相容 |
| 反向怎么来 | 另外注册求导公式,没注册就在反向时报错 | 不会自动求导,求导规则要用户提供,批处理规则也要另给 | 另外注册梯度函数;用不到梯度时可以不注册 |
| 用 Python 写的 kernel | Triton kernel 在编译区域里直接调用,编译器分析它读写哪些张量,把源码并进自己生成的代码 | Pallas 用 JAX 的写法写 kernel,GPU 上降到 Mosaic GPU(只支持 Hopper 及更新的卡);也有 Triton 后端,但文档说只尽力维护、不推荐 | 文档未涉及,不作比较 |
| 按数值走的循环能不能求反向 | while 循环能求反向,代价是前向把每一轮的状态堆叠存下 | 文档的汇总表写明 while 循环只支持前向模式求导;定次循环要端点是静态的才支持反向 | 05 章已比过,不重复 |
| 注意力变体 | 用户写打分改写与掩码两个函数,编译器内联进注意力模板,按块掩码整块跳过 | 官方的注意力函数接受加性偏置与布尔掩码数组,以及因果、滑动窗口、序列长度等固定选项;实现可选 XLA 或 cuDNN,文档说 XLA 实现会生成掩码张量,cuDNN 会跳过因果之外的区域 | 文档未涉及,不作比较 |

一句话:JAX 与 TensorFlow 把外部 kernel 当成边界清楚的黑盒,形状与求导规则都由用户显式写全,换来简单可控;torch.compile 多给了一条路,让编译器看进用户的 Triton kernel、看进注意力变体的打分函数,换来的是能和模板融在一起,代价是读写分析、自动函数化与子图这几套额外机制。不用 FlexAttention 的做法,就是直接调 FlashAttention 这类手写库,变体受库支持的选项限制。两边的参数名都在第五节。

## 五、调参与观测

这一章的参数分四处:注册算子时的装饰器参数与注册函数,在模块导入时生效,改了要重启进程;调用 FlexAttention 时传的块掩码与内核选项,每次调用传入,变了会触发重编;全局配置与环境变量,在第一次编译前设好;编译入口的模式,决定模板要不要实测。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `mutates_args` | `torch.library.custom_op` / `torch.library.triton_op` 的必填参数 | 无默认 | 列出会被改的参数名;`"unknown"` 当全部输入都被改,拷贝变多;漏写是未定义行为 | `torch.library.opcheck` 的 `test_schema` |
| `@op.register_fake` | 注册函数 | 未注册;只改参数不返回、或带 `torch.Tag.inplace` / `torch.Tag.out` 标签的会自动生成 | 没有它,抓图时报 `There was no fake impl registered` | `opcheck` 的 `test_faketensor` |
| `op.register_autograd(backward, setup_context=...)` | 注册函数 | 未注册 | 没有它,前向照常,反向报 `no autograd formula was registered` | `opcheck` 的 `test_autograd_registration`;数值用 `torch.autograd.gradcheck` |
| `tags=torch.Tag.flexible_layout` 等布局标签 | `custom_op` 的 `tags` | 不带标签时取下一行的全局默认 | `flexible_layout` 允许编译器替它选布局,少一次拷贝;算子必须能吃任意步长 | `TORCH_LOGS=output_code` 里算子调用前的拷贝 |
| `torch._functorch.config.custom_op_default_layout_constraint` | 全局配置 | `"needs_exact_strides"` | 改成 `"needs_fixed_stride_order"` 或 `"flexible_layout"`,放宽所有未打标签的自定义算子 | 同上 |
| `torch._inductor.config.triton_kernel_default_layout_constraint` | 全局配置 | `"needs_fixed_stride_order"` | 用户 Triton kernel 的输入布局要求;`"flexible_layout"` 放宽 | 同上 |
| `torch._inductor.config.epilogue_fusion_user_defined_triton_kernel` | 全局配置 | `False` | `True`:kernel 只有一处写入、输出是新分配的空张量时,把后面的逐元素运算改写进那条写入语句 | `output_code` 里少一个逐元素 kernel |
| `create_block_mask(mask_mod, B, H, Q_LEN, KV_LEN, BLOCK_SIZE=128)` | 调用 | `BLOCK_SIZE=128`;`B`、`H` 传 `None` 按 1 广播 | 块越小,部分块越少、索引表越大,模板的块也要能整除它 | `print(block_mask)` 打出 `sparsity` 与块图 |
| `kernel_options={"BACKEND": ...}` | `flex_attention` 的实参 | `"AUTO"` | `"TRITON"` 固定走通用模板;`"TRITON_DECODE"` 强制解码模板,条件不满足报错;`"FLASH"` 走实验性的 CuTe DSL 实现,要另装 flash | `output_code` 里的 kernel |
| `kernel_options` 的 `BLOCK_M`、`BLOCK_N`、`num_warps`、`num_stages`(可加 `fwd_`、`bwd_` 前缀) | 同上 | 按卡型、数据类型、每头维度查表 | 钉死的项不再参与调优;`BLOCK_M`、`BLOCK_N` 必须整除块掩码的块大小,否则报错 | 报错里列出冲突的值 |
| `kernel_options` 的 `PRESCALE_QK` | 同上 | `False` | `True`:先把 QK 乘上缩放,略快,误差略大 | 与关掉时对比数值 |
| `kernel_options` 的 `ROWS_GUARANTEED_SAFE` | 同上 | `False` | `True`:去掉对行最大值的保护,略快;条件见第六节,不满足就出 NaN | 输出里的 NaN |
| `mode="max-autotune"` | 编译入口 | `default` | 默认每种形状只有 1 个前向配置,不计时;打开后多 6 个候选,实测挑最快 | stderr 的 `AUTOTUNE flex_attention(...)` |
| `torch._inductor.config.max_autotune_flex_search_space` / `TORCHINDUCTOR_MAX_AUTOTUNE_FLEX_SEARCH_SPACE` | 全局配置或环境变量 | `"DEFAULT"` | 只在调优模式下生效;`"EXHAUSTIVE"` 前向 144 个配置,编译时间成倍涨 | 同上 |
| `torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG` | 模块变量 | `False` | `True`:不编译调用时连内部那层编译也不做,能在两个函数里打断点;反向不可用,结果可能不对 | — |

**观测五处。** 自定义算子有没有进图:`TORCH_LOGS=graph_breaks` 里没有这一处断图;它在 Inductor 里是一处回退,`TORCH_LOGS=inductor` 会打 `Creating implicit fallback for`(09 章)。拷贝有没有消掉:`TORCH_LOGS=torch._inductor.fx_passes.reinplace` 逐个节点打出 `attempted to reinplace`、哪些没能改回原地,以及估算浪费的字节数。用户 kernel 的读写分析失败:`torch._dynamo` 日志的警告 `Encountered an exception in identify_accessed_tensors, assuming every input is mutated`。用户 kernel 进没进生成代码:`TORCH_LOGS=output_code` 里能看到它的源码,上方有一行注释 `# Original path:` 指回原文件与行号。控制流子图抓不下来:报错开头是 `This higher order operator doesn't work unless it is captured completely with torch.compile`,后面附着子图里的原始断图原因。FlexAttention 的块掩码好不好:`print(block_mask)` 给出跳过块的百分比和一张缩略块图。

**典型配置**:

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 把一个现成的 C++ 或 CUDA 扩展接进训练,不想断图 | `@torch.library.custom_op("mylib::op", mutates_args=())` 包住调用,`@op.register_fake` 写形状,`op.register_autograd(...)` 写反向;上线前跑 `torch.library.opcheck(op, args)` | 图不再断,换来算子两侧不融合、输入按 eager 的步长落地 |
| 自己写的 Triton kernel,要反向、要能导出 | `@torch.library.triton_op("mylib::op", mutates_args={})` 里用 `torch.library.wrap_triton(kernel)[grid](...)` 调用,再 `op.register_autograd(...)` | 多写一层注册,换来 Inductor 看得见 kernel、拷贝能消掉、导出时保留成一个算子 |
| 长序列文档打包训练(每个批次一条打包序列),注意力要挡住跨文档的格子 | 每步按文档编号张量 `block_mask = torch.compile(create_block_mask)(doc_mask, None, None, L, L)`;模型里 `flex_attention(q, k, v, score_mod=..., block_mask=block_mask)` 放在编译区域内;形状稳定后再试 `mode="max-autotune"` | 每步多一次建块掩码的开销,换来跨文档的整块不算、分数矩阵不落显存 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 抓图时报 `There was no fake impl registered` | 这个自定义算子有没有 `register_fake` | 补假实现;只改参数不返回的算子会自动生成,不必写 |
| 反向报 `no autograd formula was registered` | 有没有 `register_autograd` | 补上;确实不需要梯度就在调用前 `detach` |
| 编译后结果偶尔不对,eager 正常,图里有自定义算子 | `mutates_args` 是否漏了被改的参数 | 跑 `opcheck`,它的 `test_schema` 会抓出没声明的修改 |
| 自定义算子前面多出一个拷贝 kernel | 是否落在默认的精确步长要求上 | 算子能处理任意步长就加 `torch.Tag.flexible_layout` |
| 用户 kernel 周围拷贝变多,日志有 `assuming every input is mutated` | 同一条警告附带的异常栈 | 接受多出的拷贝,或把 kernel 的写入改成常规的指针加偏移写法 |
| 报 `This higher order operator doesn't work unless it is captured completely` | 报错里附带的子图断图原因 | 把那一句挪出分支或循环体;子图里不改外层对象 |
| 警告 `Pred is a Python constant`,只编进了一支 | 谓词是不是 Python 布尔 | 要保留两支,传单元素布尔张量或符号布尔 |
| 警告 `flex_attention called without torch.compile()`,显存随长度平方涨 | 调用是不是在编译区域外 | `torch.compile(flex_attention)`,或把调用放进被编译的模块 |
| 报 `block_mask was created for a smaller length` 或 `larger length` | 块掩码与当前长度 | 按当前长度重建;裁剪只对左上角成立的掩码有效 |
| 只有编译后输出出现 NaN | `ROWS_GUARANTEED_SAFE` 是否打开 | 关掉,或逐行核对第一个被调度的块里有没有可见的键 |

## 六、常见误区

**「注册成自定义算子,编译器就会优化它。」** 名字里有「算子」,又是给 torch.compile 用的,很容易这么想。实际上它对编译器是一个封闭调用:不钻进去、不融合,输入还被强制按 eager 的步长准备。注册换来的只是「不断图」。要编译器看得见 kernel,得是 Triton kernel,并用透明的那种注册方式。

**「`mutates_args` 写窄一点,少拷贝、更快。」** 这是把两个方向的后果搞反了。写宽了是安全的,只是多拷贝,而且 Inductor 常能把拷贝消掉;写窄了是未定义行为,函数化以为没被改的张量其实被改了,结果可能静默出错。拿不准就写全,或者用 `"unknown"`。

**「用户 Triton kernel 进了图,就会和前后的逐元素运算融在一起。」** 进图只解决断图和读写可见。融合默认是关的,打开 `epilogue_fusion_user_defined_triton_kernel` 之后也只处理一种情形:kernel 里只有一处写入,写的是一块新分配的空张量。其余时候它仍是一次独立的启动。

**「`torch.cond` 分支里写了抓不了的东西,顶多像外面一样断图回 eager。」** 外层的习惯让人这么以为。条件、while、scan、map 这些算子的子图不许断图,抓不下来就整个报错;因为分支只有整体进图才有意义,半张子图没法和外面拼起来。

**「`ROWS_GUARANTEED_SAFE` 只要每一行至少有一个能看到的键就能开。」** 字面意思确实像。内核选项的说明写得很清楚:要求更强,每行在它被调度到的第一个块里就得有可见的键,否则行最大值还是负无穷时就去做缩放,算出 NaN。因果掩码满足;滑动窗口这类掩码,靠后的行常常不满足。eager 下整行一起算,不受影响,所以 NaN 只在编译后出现,最难查。

**「不加编译调 `flex_attention`,只是慢一点。」** 它会退回参考实现,把整张分数矩阵算出来;长序列下不是慢,是显存直接按长度平方涨到爆。那条提示用编译的警告每个进程只打一次,多卡日志里很容易被略过。

**「`create_block_mask` 很便宜,每步调一次无所谓。」** 它的输出只有两张小索引表,看着便宜。不编译时它先按批乘头乘查询乘键把整张布尔掩码算出来,批 8、单头、长度 32768 就是 8 GiB;掩码不随批和头变化时,`B`、`H` 要传 `None`。官方的建议是把它也包进编译。

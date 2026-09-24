# 矩阵乘模板与 max-autotune

这一页回答一件事:矩阵乘和卷积不走逐元素代码生成那条路,Inductor 凭什么决定用现成库还是自己生成的模板,又怎么把后面的偏置、激活并进去。说法都在源码基准 `217579124a`(main,仓库自报 `2.15.0a0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

11 章那套代码生成擅长的是访存受限的逐元素与归约:按形状挑块大小,一趟循环读完写完。矩阵乘不是这种东西。它的快慢取决于分块、流水级数、Tensor Core 指令排布这一整套手艺(知识库「GEMM优化」篇),通用的循环生成器写不出 cuBLAS 那种水平。所以默认模式下 Inductor 根本不生成矩阵乘,直接调现成库;卷积同理,调 cuDNN。

直接调库有两处骨折。**第一,没有一套实现通吃所有形状。** 库内部靠启发式为每个形状挑实现,瘦长矩阵、K 远大于 M 和 N、维度不是 8 的倍数这些形状上未必挑得中;换一种分块、换一种切 K 的方式,同一张卡上可能就快一截。**第二,库是一个封闭的调用。** 10 章讲过,外部调用在调度器眼里是断点:矩阵乘之后的偏置、激活只能另起一个逐元素 kernel,把矩阵乘结果从显存读回来再写一遍。输出越大,这一趟往返越贵。

反过来,全交给自己生成的模板也不行:Triton 模板在某些形状上比 cuBLAS 慢,编译和挑选还要额外花时间。

所以这一章的问题是:**同一个矩阵乘,怎么在库和各种模板之间挑出这个形状上真正最快的那个,并在挑中模板时把后面的逐元素尾巴吃进去,又不让编译时间失控。**

## 二、解法

最天然的直觉:拿不准就都试一遍。Inductor 把这件事做成了一个开关:平时只有库这一个候选,不试;打开最大调优的那档模式后,同一个矩阵乘同时生成多种实现,在这台 GPU 上用同样形状的输入逐个计时,取最快的。

![同一个带偏置的 bf16 矩阵乘后接 ReLU,两种模式下怎么走。上半部分是默认模式:矩阵乘连同偏置直接调 cuBLAS,只有这一个候选,不计时;它的结果写回显存,再由一个单独的逐元素 kernel 读回来做 ReLU,一共 2 个 kernel,中间结果往返显存一次。下半部分是调优模式,分三步:第一步生成候选,包括 cuBLAS 的带偏置矩阵乘,以及至多 21 种分块配置、把偏置加在模板内的 Triton 矩阵乘模板,CUTLASS 等后端默认不在候选里;第二步在线程池里并行编译全部模板,再用同形状同步长的随机张量逐个计时,默认在编译进程里测,有 CUTLASS 候选或打开子进程开关时换到子进程,条形长度表示耗时;第三步到调度时才决定尾巴,拿排名最前的 Triton 模板把 ReLU 融进去实测一次,和全场最快的候选加一个单独 ReLU kernel 的耗时之和比,融合版更快就选它,中间结果不再落显存,否则保留全场最快的候选和单独的 ReLU](/opensource/torch-compile/12a-candidates-and-epilogue.svg)

**候选从哪来。** 矩阵乘在降级时(09 章)不直接变成一个节点,而是交给一个选择器,由各算子自己的降级函数往里填候选。候选分两类:一类是外部库调用,cuBLAS 的矩阵乘或带偏置的矩阵乘,偏置是步长为 0 的二维广播张量时再加一个走 cuBLASLt 的变体;另一类是模板,即一段带占位符的 Triton kernel 源码,每种分块配置(块的长宽深、流水级数、warp 数)实例化成一个候选。CUDA 上矩阵乘的默认配置表有 21 行,按实际的 M、N、K 缩小、去重后通常更少;还有一张穷举表,5 种块长的三维组合乘 5 种流水级数乘 3 种 warp 数,共 1875 行,要显式打开。K 比 M、N 都大 32 倍以上时,另加一种把 K 切成若干段分别乘再求和的候选,大到 64 倍时干脆不再放普通模板,源码注释说那种形状上模板几乎不可能赢。CUTLASS、CuTe DSL、NVIDIA 通用 GEMM 这几种后端也能作为候选纳入,但默认不在候选名单里,要自己加进后端列表并装好对应的库。

**什么时候真的有模板。** 模板要同时满足四件事:打开了调优;后端列表里有 Triton;输出是 fp16、bf16 或 fp32;GPU 至少有 68 个 SM。最后这条是写死的门槛,源码以 RTX 3080 为界,SM 不够的卡上调优模式照开,矩阵乘却仍然只有库这一个候选,只在日志里留一条警告。

**怎么测。** 先在线程池里把全部模板并行编译好,编译失败或超时的候选直接剔除;再按真实的形状与步长造随机张量,逐个计时。默认就在编译进程里测;候选里有 CUTLASS 时强制换到子进程,理由写在注释里:CUTLASS kernel 出错会留下粘滞的 CUDA 错误,把整个进程的上下文弄坏。子进程模式下每个候选有 60 秒的结果超时,超时或崩溃的记成无穷大,崩在非法地址这类错误上还会重启子进程。所有候选都失败时退回外部库调用。测出的结果按输入的形状、步长、类型存进调优缓存,下次同样的矩阵乘直接查表(缓存本身见 16 章)。

**尾巴留到调度时再决定。** 调优模式下,选择器并不当场拍板,而是交回一个「多候选缓冲」,把各候选的计时一起带进调度器。10 章的融合打分遇到它时,会多做一次实测:取单独跑最快的那个 Triton 模板,把后面的逐元素节点融进去编译、计时一次,再和「全场最快的候选(可能就是 cuBLAS)加上单独一个尾巴 kernel」的耗时之和比,融合版更快才采用。只有排名第一的模板参与这次比较;它单独跑就已经不快于那个和,连试都不试。融合全部结束后,还没定案的多候选缓冲按单独跑最快的那个定案。

**形状不对齐时先补齐。** 这一步不属于调优模式,默认就开(09 章点过它的位置)。Tensor Core 喜欢半精度下 8 的倍数、fp32 下 4 的倍数;某一维不整除时,联合图上的一个替换会考虑把 M、N、K 补零到整数倍。补不补也靠实测:先用算术强度粗筛,只处理算得上计算受限的矩阵乘;再把原样与补齐后(含补零本身的开销)各跑一次,补齐版要快过 1.1 倍才换。符号维不补,只补静态维。

**卷积与批量矩阵乘同一套。** 批量矩阵乘、分组矩阵乘、卷积都走同一个选择器,只是候选不同:卷积默认模式下只调 cuDNN,并且在 GPU 上把偏置剥出来单独加,注释说带偏置的 cuDNN 更慢;调优模式下前向多出 Triton 卷积模板,CUDA 上的两个反向仍然只用 cuDNN。输入是通道在后的 1×1 卷积,调优模式下直接改写成矩阵乘。

**为什么实际不吃亏。** 默认模式什么都不测,编译时间不变;调优只作用在矩阵乘和卷积这几个节点上,每个形状测一次就进缓存。尾巴融合那一步只拿第一名去试,每一对矩阵乘与尾巴只多编一个 kernel。

## 三、代价

**编译时间成倍拉长。** 默认配置下一个矩阵乘要编约 20 个模板再逐个计时,一个模型里有多少种不同形状的矩阵乘就重复多少遍,穷举表更是 1875 行起步。模板编译按 CPU 核数并行跑,计时在一张卡上是一个接一个的。这笔账只在第一次编译时付,调优结果进缓存之后,同样的形状不再重测。

**测出来的不一定是跑起来的。** 计时用的是随机张量,不是真实数据;默认在编译进程里测,GPU 上若同时有别的负载,计时噪声会让两次编译挑出不同的实现。相近的候选之间,每次编译可能选得不一样,数值结果也会因为累加顺序不同而有末位差异。

**模板的收益取决于形状。** 实测的结果完全可能是库本身最快,那样调优只是确认了默认选择,编译时间白花。收益集中在库没照顾到的形状,以及输出大、尾巴贵、融合能省下一整趟显存往返的场景。

**选了模板,尾巴也未必进得去。** 尾巴必须和矩阵乘输出元素数相同、不做原地修改,归约一般不行;融合版的实测也可能输给「库加单独尾巴」。输了就保留分开的两个 kernel,这是正确的结果,但会让人误以为融合没生效。另外,尾巴的这次实测只在默认开着的尾巴计时开关下做;关掉它,决定改为按估算耗时加寄存器溢出的启发式判断。

**动态形状上打折。** 计时用的是符号维的提示值,换一个长度,选出的实现未必还是最快。NVIDIA 通用 GEMM 干脆要求形状全静态;补齐只补静态维。

## 四、和同类大厂框架不一样的地方

对照对象是 XLA 的 GPU 后端与 TensorRT 的构建器。两者都在编译期对矩阵乘这类算子做实测挑选,差别在调优在什么时候发生、默认开不开、尾巴怎么处理。表里关于这两家的每一条都核自 openxla.org 与 docs.nvidia.com 上的官方页面。

| 每家都要回答 | torch.compile 的调优模式 | XLA 的 GPU 后端 | TensorRT 构建器 |
|---|---|---|---|
| 候选有哪些 | 外部库调用加多种配置的 Triton 模板;CUTLASS 等要自己加进名单 | 文档列出 cuBLAS、cuDNN、Triton 与自带的代码生成,在编译时对矩阵乘与卷积逐个实测 | 每层所有可用的实现(文档称 tactic)在构建阶段全部跑一遍,取最快 |
| 默认开不开 | 默认关,要选调优那档模式或打开对应开关;默认只调库 | 文档把编译期实测列为 GPU 上编译结果不确定的来源,要关掉得显式把调优级别设成 0,关掉后取第一个合法配置 | 构建就是实测;另有优化级别调节搜索多少实现,默认 3 |
| 计时怎么做 | 同形状的随机张量,默认在编译进程内,可换子进程 | 在本机 GPU 上实测 | 每个实现至少跑 4 次取平均,次数可调大 |
| 结果怎么复用 | 按输入形状、步长、类型进调优缓存 | 可指定按融合分文件的缓存目录,默认关;也能把全部结果导出、导入 | 计时缓存按层的配置记录每种实现的耗时,能序列化给下一次构建;不挂缓存时构建完即丢 |
| 矩阵乘后面的尾巴 | 调度时拿最快的 Triton 模板带尾巴实测一次,赢了才融 | 含矩阵乘的融合交给 Triton 生成,融合配置里直接写着选中的分块 | 按融合目录里的固定模式合层,再对合成后的层挑实现 |

一句话:TensorRT 和 XLA 把实测当成编译的常规步骤,换来的是构建或编译必然慢;torch.compile 默认不测,把这笔钱留给用户决定要不要付,付了之后尾巴融不融也按实测说话。

## 五、调参与观测

这一页的参数分三处:编译入口的 `mode`,改了要重新包一次;Inductor 的全局配置 `torch._inductor.config`,在 Python 里改,对之后的编译生效,也能通过 `torch.compile(..., options={...})` 只对这一次生效(`mode` 与 `options` 不能同时传);大多数键另有环境变量,在导入时读取,要在进程启动前设好。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `mode="max-autotune"` | `torch.compile` | `default` | 等于 `max_autotune=True`、`coordinate_descent_tuning=True`、`triton.cudagraphs=True`;后两项分别见 11、13 章 | `torch._inductor.list_mode_options("max-autotune")` |
| `mode="max-autotune-no-cudagraphs"` | 同上 | — | 同上但不开 CUDA graph | 同上 |
| `max_autotune` | 全局配置,`TORCHINDUCTOR_MAX_AUTOTUNE=1` | `False` | 矩阵乘、卷积的模板候选打开;少数逐元素 kernel 也多试几种配置(11 章) | stderr 出现 `AUTOTUNE mm(...)` 表 |
| `max_autotune_gemm` | 同上,`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1` | `False` | 只打开矩阵乘这类的调优,不带逐元素的调优与 CUDA graph | 同上 |
| `max_autotune_gemm_backends` | 同上,`TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS` | `ATEN,TRITON,CPP` | 去掉 `ATEN`:只剩模板,全失败时报 `NoValidChoicesError`;加 `CUTLASS`:多一类候选,计时强制进子进程;`CUTEDSL` 只作用于 Blackwell 上的分组矩阵乘;`NVGEMM` 要另装 `cutlass.operators` | 报错信息与 `AUTOTUNE` 表里的候选名 |
| `max_autotune_conv_backends` | 同上,`TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS` | `ATEN,TRITON`;两个反向的对应键在 CUDA 上默认只有 `ATEN` | 决定卷积的候选 | `AUTOTUNE convolution(...)` |
| `max_autotune_gemm_search_space` | 同上,环境变量同名大写 | `DEFAULT` | `EXHAUSTIVE`:Triton 配置换成 1875 行的穷举表,K 切分也列全;编译时间成倍再涨 | `AUTOTUNE` 表末行的候选数 |
| `autotune_in_subproc` | 同上,`TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1` | `False` | 计时挪进子进程:坏候选不会弄坏主进程的 CUDA 上下文,多一点进程间开销 | 表末行 `SubProcess` 或 `SingleProcess` |
| `max_autotune_subproc_result_timeout_seconds` | 全局配置 | 60 | 子进程里单个候选的计时超时;超时记为无穷大 | 警告 `Timed out benchmarking choice` |
| `autotune_multi_device` | 同上,`TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE=1` | `False` | 子进程模式下每张可见卡起一个计时进程并行测 | `autotuning` 日志的设备列表 |
| `precompilation_timeout_seconds` | 同上,`TORCHINDUCTOR_PRECOMPILATION_TIMEOUT_SECONDS` | 300 | 模板并行编译的总超时,超时的候选标为失败 | 警告 `Precompilation timeout after` |
| `benchmark_epilogue_fusion` | 同上,`TORCHINDUCTOR_BENCHMARK_EPILOGUE_FUSION` | `True` | 关掉:尾巴融不融改按估算与寄存器溢出判断,少一次编译和计时 | `TORCH_LOGS=fusion` 的 `can fuse (benchmark)` 与 `cannot fuse (benchmark)` |
| `max_epilogue_benchmarked_choices` | 全局配置 | 1 | 调大:排名前几的模板都带尾巴试一次,编译更慢 | 同上 |
| `shape_padding` | 同上,`TORCHINDUCTOR_SHAPE_PADDING` | `True` | 关掉:不对齐的维度不再补零 | 生成代码里矩阵乘前后多出的补零与切片 |
| `force_shape_pad` | 全局配置 | `False` | `True`:能补就补,跳过实测 | 同上 |
| `triton.decompose_k_threshold`、`triton.num_decompose_k_splits` | 同上,`TORCHINDUCTOR_DECOMPOSE_K_THRESHOLD`、`TORCHINDUCTOR_NUM_DECOMPOSE_K_SPLITS` | 32、10 | 决定什么形状加 K 切分候选、最多试几种切法;切法设成 0 即关 | `AUTOTUNE` 表里的 `decompose_k_mm_<切数>_split` 候选 |
| `cutlass.cutlass_dir` | `TORCHINDUCTOR_CUTLASS_DIR` | 源码树里的第三方目录,pip 安装的包里一般没有 | 指向 CUTLASS 源码根,CUTLASS 候选才能生成 | 警告 `Failed to import CUTLASS lib` |
| `autotune_num_choices_displayed` | 同上,`TORCHINDUCTOR_AUTOTUNE_NUM_CHOICES_DISPLAYED` | 10 | `none` 打出全部候选;0 不打 | stderr 的 `AUTOTUNE` 表 |
| `deterministic` | 同上,`TORCHINDUCTOR_DETERMINISTIC=1` | `False` | 不计时,直接取外部库调用;补齐也不做 | `AUTOTUNE` 表消失 |

**观测四处。** 选了谁:调优模式下每次实测后 stderr 打一张表,开头是 `AUTOTUNE mm(4096x4096, 4096x4096)` 这样的形状,下面按耗时列出前 10 名的名字、毫秒数、相对最快者的百分比与配置(块长、流水级数、warp 数),末行写这次用了 `SingleProcess` 还是 `SubProcess`、计时与预编译各花了多少秒、共几个候选;命中缓存的不打。花了多久:`torch._dynamo.utils.compile_times()` 里的 `mm_template_precompiling` 与 `mm_template_autotuning`(卷积、addmm 同理换名)。尾巴融没融:`TORCH_LOGS=fusion` 看模板与尾巴那一对的 benchmark 行;`TORCH_LOGS=output_code` 里选中模板的 kernel 名以 `triton_tem_fused_` 开头,名字里带着融进去的算子,选中库的是 `extern_kernels.mm` 这类调用,后面跟一个单独的逐元素 kernel。为什么没模板:日志里有 `Not enough SMs to use max_autotune_gemm mode` 就是卡太小;`TORCH_LOGS=autotuning` 打出候选的源码与参数。

**典型配置**:

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 长时间跑的推理服务,形状固定或已分桶,大卡 bf16 | `model.compile(mode="max-autotune")`,服务启动时按各档形状预热一遍 | 用首次编译多出的调优时间与 CUDA graph 常驻显存,换矩阵乘选中更快的实现、尾巴并进模板 |
| 训练,只想调矩阵乘,不要 CUDA graph | `torch.compile(model, options={"max_autotune_gemm": True})` | 用更长的编译换矩阵乘的实现;逐元素 kernel 与 CUDA graph 保持默认,少一层变数 |
| 离线打包、编译时间不敏感,追求最后一点 | `options={"max_autotune": True, "max_autotune_gemm_search_space": "EXHAUSTIVE", "autotune_in_subproc": True}` | 用成倍的编译时间换穷举配置;子进程让坏候选不拖垮主进程 |
| Hopper 上想试 CUTLASS | 设好 `TORCHINDUCTOR_CUTLASS_DIR`,再 `options={"max_autotune": True, "max_autotune_gemm_backends": "ATEN,TRITON,CUTLASS"}` | 用 CUTLASS 的编译与子进程计时开销,换多一类候选;尾巴默认不进 CUTLASS kernel |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 开了调优,矩阵乘还是全走 cuBLAS | 日志有没有 `Not enough SMs`;`AUTOTUNE` 表里 Triton 候选是不是都慢 | SM 不够就接受;都慢说明这个形状上库本来就最好,可以只关调优省编译时间 |
| 编译时间暴涨 | `compile_times()` 里 `*_template_autotuning` 与 `*_template_precompiling` 占多少 | 换成只调矩阵乘的开关;回到 `DEFAULT` 搜索空间;确认调优缓存命中(16 章) |
| 选中了模板,后面仍有一个单独的逐元素 kernel | `TORCH_LOGS=fusion` 里这一对是 `template epilogue not satisfied` 还是 `cannot fuse (benchmark)` | 前者看尾巴是不是归约、原地修改或元素数不同;后者说明融了反而慢,接受 |
| 编译时报 `Timed out benchmarking choice` 或进程崩在 CUDA 错误 | 是不是加了 CUTLASS 或某个模板配置在这张卡上出错 | 打开 `autotune_in_subproc`;必要时把出事的后端从名单里去掉 |
| 两次编译结果数值有末位差异、性能也不同 | `AUTOTUNE` 表里前几名是否几乎一样快 | 正常;要可复现就打开 `deterministic`,或复用同一份调优缓存 |
| 维度是奇数的矩阵乘编译后变快或变慢 | 生成代码里有没有补零 | 补齐由实测决定;要排除它的影响,关掉 `shape_padding` 对照一次 |

## 六、常见误区

**「max-autotune 的主要收益是矩阵乘。」** 名字和文档都这么引导。实际上这档模式同时打开了三件事:矩阵乘与卷积的候选实测、逐元素 kernel 的坐标下降调优(11 章)、CUDA graph(13 章)。开了之后变快,常常是 CUDA graph 省掉了 launch 开销;想单独看矩阵乘调优的收益,要用只调矩阵乘的那个开关对照。

**「开了调优,矩阵乘就一定会走 Triton。」** 调优只是把 Triton 放进候选,最后按实测选。库赢的形状并不少见,这时表里第一名就是 `mm`。SM 少于 68 的卡上连候选都没有 Triton,只有一条警告。

**「调优在子进程里跑,不会影响主进程。」** 很多人记得有个子进程调优的开关,就以为默认如此。默认它是关的,计时就在编译进程里做;只有出现 CUTLASS 候选时才强制换到子进程。在一张正在跑别的作业的卡上编译,计时噪声会直接影响选择。

**「融合打分已经决定了模板和尾巴要合。」** 10 章的打分让模板配对排在最前,人们就以为合是必然。调优模式下打分只是让这对候选进门,真正拍板的是那一次实测:排名第一的模板带尾巴跑一次,要赢过「全场最快加单独尾巴」才合。输了保留两个 kernel,这不是 bug。

**「补齐是调优模式的一部分,关掉调优就没有了。」** 补齐是联合图上的一个替换,默认模式下同样生效,也同样靠实测决定;它和调优开关无关。排查奇数维度的性能差异时,要单独关它对照。

**「把 CUTLASS 加进后端列表就能用上。」** pip 装的包里通常没有 CUTLASS 源码,不设源码目录只会打一条导入失败的警告然后静默跳过;设好了,候选数和编译时间都会明显上去,计时还被强制挪进子进程。CUTLASS 的尾巴融合另有开关,默认关。

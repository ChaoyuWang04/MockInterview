# 算力:低精度、融合与 CUDA graph

这一页回答一个问题:模型已经能训练,怎样减少同一条计算链上的算术、访存与CPU下发开销。说法都在源码基准 `fb6a123a09`(tag `core_v0.19.2`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

GPU没有跑满,不一定是矩阵乘太慢。小算子之间反复读写中间tensor、CPU逐个提交kernel,都可能让计算单元等数据或等任务。另一方面,大GEMM即使连续执行,仍可能受计算精度与数据字节数限制。

Megatron在既有训练链上提供三类优化:**低精度减少部分计算与数据的成本,fusion减少算子之间的边界,CUDA graph减少重复提交的成本**。先看时间花在哪里,才能知道哪个开关对症。

## 二、解法

低精度通过Transformer Engine的recipe控制量化格式、scale与执行上下文。模型选中的GEMM可以使用FP8或FP4,但参数存储、梯度和optimizer状态不因此一起变成同一种精度。计算时量化与初始化时保留量化参数是两条控制路径;还可以按模块匹配recipe,让不同位置采用不同精度。

fusion把相邻操作交给编译器或专用kernel一起执行。例如bias加法与GELU可以一起计算,减少中间结果反复往返显存。这里既有PyTorch编译路径,也有外部库的专用实现,不是把整个Transformer自动揉成一个kernel。

![同一条GEMM、bias、GELU、GEMM计算链的三个优化位置:低精度缩小选中操作数的数据payload,fusion消除bias和GELU之间的边界,CUDA graph保留kernel节点但用一次replay提交整个记录区域。](/opensource/Megatron-LM/12a-three-cost-layers.svg)

CUDA graph先warmup,再capture一段可重复的GPU工作,后续把新数据放入固定输入buffer并replay。Megatron既能按Transformer layer或其子区域capture,也能把训练iteration的forward/backward路径交给一张graph。**后者不包含optimizer step**,数据读取也先在graph外准备;optimizer有单独的graph入口。

这些机制可以组合,但作用不能互相替代。graph里的kernel仍然需要计算和访存;低精度也不会自动消除CPU提交之间的间隙。动态形状的MoE区域通常要留在graph外,或先用受支持的容量与padding路径固定形状。

## 三、代价

**低精度要付转换与数值误差的代价。** scale、量化、布局转换和对齐padding都不是免费;FP4的payload更小,也不表示整份模型显存按相同比例下降。recipe对Transformer Engine版本和GPU架构有要求,不能把“配置能写”理解成任意硬件都能执行。

**fusion与graph都带来执行约束。** fusion可能依赖dtype、shape和具体activation;graph还需要稳定的输入结构与buffer,并占用capture时间和额外显存。训练中的CPU同步、动态分配结果或不兼容的recompute/offload组合可能破坏capture,不是把范围选得越大就一定越快。

数值一致性还有单独的需求。batch-invariant kernel固定部分计算与归约顺序,用于减少batch组织变化引入的forward差异;它不是通用提速开关。当前实现对BF16、attention后端、dropout和并行组合有明确限制,不能把服务侧或logprob重算的需求推广成所有训练任务的默认选择。

## 四、和同类常见做法不一样的地方

PyTorch FSDP的分片路线主要回答状态怎样分布在设备上;本章的Megatron路径回答设备拿到数据后怎样执行。Megatron把Transformer Engine模块、recipe与训练调度中的capture/replay接在一起,所以检查优化是否生效,既要看模块后端,也要看训练循环实际包住了哪段工作。分片和本章的执行优化可以组合,但不能互相证明已经生效。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

下面区分参考训练CLI与直接构造Core配置的默认。CUDA graph先在固定shape的dense模型上建立基线;MoE的分发与执行边界见10章,offload与recompute见11章。

| 参数 / 在哪调 | 本基准默认 | 调整的作用与代价 | 怎么看 |
|---|---|---|---|
| CLI `--fp8-format` / `--fp8-recipe` | 无 / `delayed` | 启用TE FP8计算及缩放recipe;格式有 `e4m3`、`hybrid` | GEMM类型、量化开销、主loss与验证loss |
| CLI `--fp4-format` / `--fp4-recipe` | 无 / `nvfp4` | FP4格式为 `e2m1`;NVFP4要求Blackwell及以后、TE至少 `2.7.0.dev0`;不能与全局FP8同时开 | 硬件与TE兼容性、padding、数值稳定性 |
| CLI `--fp8-param-gather` / `--fp4-param-gather` | 关 / 关 | 映射Core `fp8_param` / `fp4_param`,控制相应参数存储与gather;不覆盖所有参数或optimizer状态 | 参数dtype、gather字节、总显存;还需满足optimizer/FSDP兼容条件 |
| CLI `--first-last-layers-bf16` | 关 | 开后首尾各1层保留BF16;层数可另调;与默认FP8 delayed recipe不兼容 | 首尾层计算类型与验证loss |
| CLI `--te-precision-config-file` | 无 | YAML按模块名顺序匹配recipe;第一个启用且匹配的规则生效 | INFO级别的模块匹配日志 |
| CLI `--no-bias-gelu-fusion` / `--no-bias-swiglu-fusion` | CLI默认允许融合 | 按activation映射Core `bias_activation_fusion`;直接Core默认关,CLI还会按模型条件修正 | trace中的算子边界,不是只看布尔值 |
| CLI `--no-bias-dropout-fusion` | CLI默认允许融合;Core默认关 | 关闭用于对照bias/dropout/residual支路的融合效果 | kernel与访存、稳态iteration时间 |
| CLI `--cuda-graph-impl` | `none` | `local`、`transformer_engine`为逐层路径;`full_iteration`包forward/backward,不含optimizer | capture日志、replay区域与稳态时间 |
| CLI `--cuda-graph-modules` | 空列表 | 逐层模式下为空表示整层;可选 `attn`、`mlp`、`moe`、`moe_router`、`moe_preprocess`、`mamba`;整轮模式必须空 | 动态区域是否仍在eager执行 |
| CLI `--cuda-graph-warmup-steps` | 3 | capture前warmup;TE内部还可能追加warmup,不是所有后端总共只跑3次 | 首次capture耗时与之后的稳态分开统计 |
| CLI `--optimizer-cuda-graph` | 关 | 单独capture optimizer step;不能由整轮模式推断已开启 | optimizer capture日志与step时间 |
| CLI `--batch-invariant-mode` | 关 | 固定部分forward执行顺序,不承诺更快;要求BF16参数、FlashAttention 3或4、CP=1及attention dropout为0等 | 同输入改变batch组织的输出对照 |

**capture范围要单独核对。**

| 实现 | 训练范围 | 关键边界 |
|---|---|---|
| `local` | MCore管理整层或所选子区域 | 自动接入现有调度;MoE router/preprocess会配对,完整MoE区域要求drop-and-pad等条件 |
| `transformer_engine` | TE创建整层或所选子区域的graph | 参考训练循环已接好helper;自定义循环要接capture与manual hooks,并非换个名字就完成接入 |
| `full_iteration` | 一次forward/backward调度,支持独立的validation forward-only graph | 不含数据读取与optimizer;模块列表必须空,CLI要求关闭loss/grad的NaN同步检查 |

训练模块选择与推理capture粒度是独立配置。推理graph只有 `local` 支持;本章不展开推理引擎。模块级训练/验证精度覆盖在文档中仍注明未验证与graph、activation recompute的组合,不要直接叠加当成已验证配方。

**配置起点**。假设已有固定sequence length与microbatch的dense decoder配方,使用Transformer Engine,不叠加offload或recompute。前两组适用于支持所选FP8 recipe的Hopper H100/H200环境,第三组用于已完成数值基线的CUDA环境;均是CLI增量,不是完整训练命令。

| 场景 | CLI增量 | 取舍 |
|---|---|---|
| 从BF16基线试FP8计算 | `--bf16 --transformer-impl transformer_engine --fp8-format hybrid --fp8-recipe delayed` | 保留默认参数存储策略,先观察计算收益与loss;不加首尾BF16例外 |
| CPU提交开销明显,先逐层capture | 在第一组上加 `--cuda-graph-impl local --cuda-graph-warmup-steps 3` | 固定dense整层capture;比较steady state,扣开capture成本 |
| 逐层graph仍有明显下发间隙 | 在原BF16基线上加 `--cuda-graph-impl full_iteration --cuda-graph-warmup-steps 3 --no-check-for-nan-in-loss-and-grad` | 扩大到forward/backward;失去该项同步检查,需先有稳定基线,且保持模块列表为空 |

组合与推荐值是按语义推的起点,不是实测最优。

**看哪里。** TE路径记录 `Start CUDA Graphs capture` 与capture耗时,整轮路径记录 `Capture CUDA graph for training` 和完成日志;optimizer有自己的capture日志。先确认进入目标路径,再用trace看GEMM、量化kernel、kernel间隙与graph replay,同时比较iteration时间、峰值显存和主/验证loss。仅有capture成功日志不能证明端到端加速。

| 症状 | 先查 | 然后 |
|---|---|---|
| FP8开了但显存降幅很小 | 是否只改计算精度,optimizer与activation还占多少 | 按状态类型拆账,不把payload比例当整卡比例 |
| fusion开着却看不到预期kernel | 实际模块后端、dtype、shape与activation | 对照关闭融合的同一配置trace |
| graph启动失败 | CPU同步、shape变化、capture范围和版本条件 | 先缩到可capture子区域,再逐项加入其他机制 |
| 第一轮特别慢,之后正常 | 是否包含JIT、warmup与capture | 单独报告启动成本和稳态收益 |
| capture成功却更慢 | 原来是否真受launch限制,新增padding和内存成本 | 对照eager基线,撤掉无收益的组合 |

## 六、常见误区

**看到FP8训练就按每参数1字节估总显存。** 名字让人以为所有状态一起转换,实际计算、参数存储、梯度与optimizer状态分别配置。先查张量类型及分项显存,再解释峰值。

**把“整轮”理解为整个Python训练循环只提交一次。** 名字省略了边界,实现包住的是forward/backward函数;数据准备、optimizer和其他循环逻辑不自动包含。trace里出现这些独立工作并不说明graph失效。

**只看Core默认就判断参考CLI没开fusion。** 两层配置的默认不同,CLI还会根据activation映射与修正。排查要追到最终构造出的配置和真正调用的模块。

**把graph capture成功当成数值和性能都通过。** capture只证明那次路径能录下来;新batch仍要走匹配的buffer与shape约束,低精度组合也仍要比较loss。验证范围应覆盖实际训练形状与配置,不能用一次启动替代。

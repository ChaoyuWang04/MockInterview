# Megatron-FSDP

这一页回答一个问题:怎样让完整参数只在计算需要时驻留,把省下的显存留给更大的模型。说法都在源码基准 `fb6a123a09`(tag `core_v0.19.2`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

即使 optimizer state 已分片,完整计算参数和梯度仍可能撑满显存。继续扩大数据并行规模,也不会自动缩小这些未分片的副本。想再省一层,就要让计算参数也按需聚合,而不是整个模型始终完整地留在每张卡上。

Megatron-FSDP 是仓库自己的实现,与 PyTorch FSDP 是两套代码。它既能接入 Megatron 训练循环,也提供独立的 PyTorch 模型接口。本页机制以参考入口默认实现为主:核心问题是**以哪个单元聚合参数,何时释放完整副本,通信能否赶上计算**。

## 二、解法

Megatron-FSDP 先选出一组模块作为管理单元,例如一个 Transformer layer。单元里的参数按 dtype、并行属性等分组,放入连续buffer后切片。每个rank持有的是这些buffer的一段,所以某个具体参数在本地可能只有部分数据,也可能完全没有。

全分片路径在单元计算前 all-gather 参数,就绪后才放行计算;普通前向结束后释放完整参数的临时存储,反向前再聚合。反向产生梯度后执行 reduce-scatter,各rank保留并累积自己的梯度片段,供本地optimizer更新。

![全分片策略下,两个rank分别常驻单元参数的一半,前向与反向前各自all-gather完整参数,用完释放临时完整副本,梯度reduce-scatter后按片段更新。](/opensource/Megatron-LM/07a-unit-residency.svg)

图中的释放有边界。属于明确管理单元的完整参数可以在计算结束后回收;单元外参数可能持续保留。activation checkpointing重算期间,刚聚合的参数还要紧接着供反向使用,释放会相应延后。开启持久buffer时,释放也可能只是归还可复用空间,不意味着GPU已分配内存数字立即下降。

为了减少等待,Megatron-FSDP 在处理当前单元时预取后续单元,并用独立CUDA stream安排参数聚合与梯度归约。预取越积极,临时驻留通常越多;这也是管理单元不能随意变大的原因。

## 三、代价

**省常驻显存,要付通信和临时buffer成本。** 全分片会在计算边界聚合参数,峰值仍包含正在计算或预取的完整单元、梯度临时空间和激活,不能把总显存简单除以数据并行度。CPU offload与激活策略见11章,低精度buffer见12章。

**单元划分是模型使用约定。** 模型必须在受管理的前反向边界内使用参数。自定义代码绕过模块调用、直接读取已经释放的参数,就可能破坏这个约定。单元过大抬高临时峰值,过小又可能增加通信调用成本。

**分片梯度改变了累积方式。** 在同时分片梯度的路径上,每次backward都要归约到梯度分片再累积;不能照搬“累积期间完全不通信”的预期。是否分片参数、是否分片梯度,是两个需要分别确认的选择。

## 四、和同类常见做法不一样的地方

PyTorch FSDP与Megatron-FSDP都能围绕模型模块安排分片,但配置入口和内部存储不能互换。本章更值得关注的是Megatron-FSDP怎样接回Megatron既有的并行与训练流程。

| 对比维度 | PyTorch FSDP的概括性路线 | Megatron-FSDP |
|---|---|---|
| 接入边界 | 在模型上选择分片管理范围,与调用方训练循环配合 | 独立实现外还有Megatron适配层,连接默认管理单元、并行组和参考循环 |
| 阅读状态生命周期的入口 | 从所选模块的参数分片与计算边界理解 | 从单元内连续buffer、参数切片及其聚集/释放理解 |
| 策略名称能否直接迁移 | 应按所用FSDP版本与接口核对语义 | 本实现可只分optimizer、再分梯度或连参数一起分,名称不代表永远全分片 |

比较显存时应先对齐分片对象与保留时间,再比较峰值。仅因两边都叫FSDP,不能推定相同配置会产生相同驻留行为。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

下表是Megatron参考训练入口的CLI。默认 `--megatron-fsdp-version 1` 是本页图的实现路径;参考入口会为它启用 `--use-distributed-optimizer`,但参数与梯度驻留由Megatron-FSDP管理,不能直接套用经典DDP路径的buffer解释。实验性version 2使用另一套实现,反而禁止该optimizer开关。

| 参数 / 在哪调 | 默认 | 调整的作用与代价 | 怎么看 |
|---|---|---|---|
| CLI `--use-megatron-fsdp` | 关 | 选用Megatron-FSDP包装与状态管理 | 启动最终配置 |
| CLI `--megatron-fsdp-version` | `1` | 选择实现版本;本页配置固定为1 | 避免把不同版本的限制混在一起 |
| CLI `--data-parallel-sharding-strategy` | `optim_grads_params` | 选择下表的驻留范围;更多分片可能增加计算边界通信 | 各rank显存峰值、iteration耗时 |
| CLI `--expert-data-parallel-sharding-strategy` | 无,跟随上一项 | MoE expert参数可单独选策略;此时上一项管非expert参数 | 启动两类策略与通信组 |
| CLI `--suggested-communication-unit-size` | 无,运行时派生 | 元素数量预算,影响参数预取与梯度归约队列;调大可能提高重叠也抬高临时占用 | profiler中all-gather与计算的重叠、峰值显存 |
| CLI `--fsdp-double-buffer` | 关 | 使用持久双buffer复用临时空间;增加常驻占用,不直接等于更省显存 | 预留内存与分配抖动 |
| CLI `--use-nccl-ub` | 关 | 注册通信buffer;参考入口会同时开启双buffer与手工注册,收益依赖通信环境 | 实际通信kernel与step耗时 |
| CLI `--ckpt-format` | 参考入口默认 `torch_dist`;本路径须改为 `fsdp_dtensor` | 选与Megatron-FSDP兼容的checkpoint表示 | 启动校验及一次保存/恢复验证 |
| 环境变量 `CUDA_DEVICE_MAX_CONNECTIONS` | 由环境决定 | 本路径拒绝值 `1`;官方建议不设置,给计算/通信stream并行留条件 | 启动环境与校验 |

**策略范围**。这是单一分片域的基本策略,表中“完整”描述buffer驻留,不意味着每个microbatch都执行相同通信;混合分片和expert单独策略需再分别看域与参数类别。

| 策略取值 | optimizer state / main weight | 梯度buffer | 计算参数 |
|---|---|---|---|
| `no_shard` | 完整 | 完整 | 完整 |
| `optim` | 分片 | 完整 | 完整 |
| `optim_grads` | 分片 | 分片累积 | 完整 |
| `optim_grads_params` | 分片 | 分片累积 | 分片常驻,管理单元按需聚合 |

独立API使用 `fully_shard_model` 与 `fully_shard_optimizer`,单元由 `fsdp_unit_modules` 指定,不能只包模型就忽略optimizer连接。默认Core适配层会选择受支持的层类型;独立API若没有合适的单元边界,就无法得到图中的逐单元释放收益。

**配置起点**。以下是在已验证的dense decoder训练命令上替换相应选项的增量;假设至少2个数据并行rank、NVIDIA GPU环境满足依赖,模型与batch能装下所选策略。先用BF16和普通buffer建立基线,不预设某个卡型的加速比。

| 场景 | CLI增量 | 取舍 |
|---|---|---|
| 计算参数能常驻,先省梯度与optimizer state | `--use-megatron-fsdp --megatron-fsdp-version 1 --data-parallel-sharding-strategy optim_grads --ckpt-format fsdp_dtensor` | 保留完整计算参数,减少反复按单元聚合的需求 |
| 计算参数也占不下,按单元全分片 | `--use-megatron-fsdp --megatron-fsdp-version 1 --data-parallel-sharding-strategy optim_grads_params --ckpt-format fsdp_dtensor` | 进一步省参数常驻显存,付出聚合与临时驻留成本 |
| 全分片已稳定,检查双buffer的收益 | 在上一组上加 `--fsdp-double-buffer` | 比较分配开销与峰值;不要同时改变精度、batch和分片域 |

组合与推荐值是按语义推的起点,不是实测最优。

**看哪里。** INFO级别启动日志有 `Number of FSDP Parameter Groups` 及buffer配置,先确认管理单元、参数分组和最终策略符合预期。然后固定模型、batch与精度,同时比较iteration耗时和各rank显存峰值;profiler中检查all-gather是否在本单元计算前造成等待,以及reduce-scatter能否与其他计算重叠。已经分片不代表通信已被隐藏。

| 症状 | 先查 | 然后 |
|---|---|---|
| 初始化成功,第一轮前向仍OOM | 单元大小、预取与临时完整参数 | 先收紧预取预算;独立API检查是否把整个模型当成单元 |
| 换成全分片后显存下降但更慢 | profiler中聚合是否暴露在计算前 | 再调预取或单元粒度,不要只增加DP |
| 开双buffer后预留显存上升 | 持久buffer占用是否符合预期 | 在相同负载下权衡稳定复用与占用 |
| 累积microbatch时仍有reduce-scatter | 是否选择了分片梯度的策略 | 按分片累积语义检查,不要直接判定no-sync失效 |
| 启动拒绝checkpoint或连接数 | checkpoint格式、环境中是否残留连接数1 | 改成本路径支持配置,恢复验证见13章 |

## 六、常见误区

**把两个同名实现的参数抄到一起。** PyTorch FSDP的教程不能直接当作Megatron-FSDP的配置表;连Megatron-FSDP内部的两种实现版本,optimizer接法也有差异。先确认入口、版本和最终配置。

**看到某个本地参数为空,就以为模型丢了权重。** 单元buffer整体分片之后,某个rank可能不持有这个参数的数据。不要拿本地shape当全局shape,更不能对这种非均匀片段随意调用要求各rank形状一致的collective。

**把通信预算当成硬显存上限。** 它按元素数给预取和归约排队提供建议,不是整张GPU的字节配额。dtype、单元边界、临时梯度和激活都会改变实际峰值。

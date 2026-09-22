# MoE:token 分发与专家通信

这一页回答一个问题:router 已经选好专家之后,token 如何跨卡送到正确的专家,再带着计算结果回到原来的序列位置。说法都在源码基准 `fb6a123a09`(tag `core_v0.19.2`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

**专家分散在不同卡上,输入却按原始序列排列。** 同一 token 可以选中多个专家,同一专家又会收到来自不同 rank 的 token。直接按原序列逐个执行,既难组织跨卡通信,也难把专家矩阵乘做大。

所以 router 给出映射和权重之后,还需要一个 token dispatcher:把「按序列排的 token」变成「按专家分组的 token—专家配对」,计算后再把这个变换还原。选哪些专家和如何均衡属于路由问题;这里关心**已经选好的配对怎样走完往返**。

## 二、解法

Megatron 把往返拆成准备、dispatch、专家前处理、专家计算、combine 和还原。以逐目的地交换的路径为例,本地先按目标专家重排 token 与 router 权重;一个 token 选中多个专家时,会出现多条配对记录。dispatch 把这些记录送到专家所在 rank,接收方再按本地专家分组,让同一专家的输入连续排列。

![两个 EP rank 各有一个 token 和一个专家,本地先展开 token—专家配对,all-to-all 送到专家,专家内部应用随行的 router 权重,逆向 all-to-all 返回后在原位置累加两份贡献。](/opensource/Megatron-LM/10a-token-expert-roundtrip.svg)

**router 权重随 token 一起走,但不必等返回后才乘。** grouped expert 的默认权重应用路径在激活之后、第二次线性计算之前应用权重;有输出 bias 时也按权重处理。因此 combine 收到的已是加权贡献,最后的逆重排把同一 token 的多份贡献相加,不会再乘一遍权重。图只画这个常规后乘路径,不涵盖可选的专家输入预乘。

**专家并行与专家张量并行分工不同。** 前者决定专家在哪个 rank,后者把一个专家的计算分给多卡。常规 all-to-all 路径先在 EP 组交换,专家 TP 大于 1 时再在该组 all-gather token,计算后 reduce-scatter,最后沿 EP 返回。这里用的是专家 TP 组,不应直接拿稠密层的 TP 度数替代。

库里有三种 dispatcher。默认的聚集路径把 token、映射与权重聚到专家 TP×EP 域,各 rank 选出本地专家要处理的部分,返回时 reduce-scatter;目的地交换路径按配对发送,拆开处理 EP 与专家 TP;灵活路径则把专家 TP×EP 当成统一通信域,再交给所选的融合通信后端。**dispatcher 和后端是两层选择**,不是把所有名字放在同一张候选名单里。

专家侧也有两种组织方式:顺序执行各本地专家,或把这些矩阵乘交给 grouped GEMM。dispatcher 提供每专家 token 数和已经重排的权重,专家实现消费这份布局,不重新做路由选择。

## 三、代价

**发出去的配对数和原 token 数不同。** 每个 token 的专家选择越多,重排、传输与专家执行的工作量通常越大。按目的地发送减少了不相关专家收到的数据,但还要付出配对展开、split 元数据和逆重排成本;并不能只看网络带宽。

**不丢 token,也不等于没有动态形状或 CPU 同步。** 常规路径即使知道总配对数,仍可能需要知道每个目的地收到多少。实现会安排 GPU→CPU 元数据搬运与同步点;融合后端也各有动态接收和固定容量模式。给专家容量并补齐可以让部分形状固定,但 padding 增加无效计算,容量不足还可能改变实际处理的配对。

**paged stash 是特定专家执行路径的保存激活机制。** 已接入的 TE grouped operation-fuser 路径按实际有效 token 数把反向所需激活存入页式缓冲,避免一直保存最大容量对应的内容。它默认关闭、要求 rank 接收容量配置,并与部分卸载设置互斥。GPU 内分配页面不代表整个作业无同步:迭代边界仍检查容量或 stash 溢出,必要时关闭该限制并重跑。这不是所有 dispatcher 自动获得的性质。

## 四、和同类常见做法不一样的地方

DeepSpeed MoE也有“整理输入—EP交换—专家计算—返回—还原”的链路。Megatron的dispatcher抽象把本地重排和跨rank通信方案分开,便于在同一MoE层下更换数据搬运路线。

| 对比维度 | DeepSpeed已核MoE层路径 | Megatron MoE路径 |
|---|---|---|
| 跨EP的数据流 | 整理专家输入后all-to-all,专家结果再all-to-all返回 | 可选聚集、目的地交换或融合dispatcher,不固定为同一通信路线 |
| 输入如何交给专家 | gating分发信息用于组织专家及capacity维度 | dispatcher提供按专家分组的输入与计数,专家实现消费这一约定 |
| 本地处理与通信 | 分发/组合与EP交换在MoE层主链衔接 | dispatcher负责重排及逆变换,融合后端可以接管其中部分阶段 |

本地重排、跨rank传输、grouped expert执行都可能影响收益。对照时固定路由结果与负载,分别观察阶段耗时,避免把所有收益归给all-to-all。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

训练 CLI 的字段来自 `TransformerConfig` 等配置对象。下面区分 dispatcher、通信后端、专家执行和容量;router 的选择规则沿用路由章。

| 参数与入口 | 默认 | 调整影响 | 怎么看 |
|---|---|---|---|
| CLI `--moe-token-dispatcher-type` | `allgather` | 可选 `allgather`、`alltoall`、`flex`;分别改变token聚集/定向交换/融合管理方式 | dispatch、combine耗时与显存峰值 |
| CLI `--moe-flex-dispatcher-backend` | `deepep` | 仅在flex路径选择DeepEP、HybridEP或NCCL-EP,取值为 `deepep`、`hybridep`、`ncclep`;需要对应依赖,不是普通alltoall的开关 | 初始化依赖检查与通信时间线 |
| CLI `--expert-model-parallel-size` | 1 | 把专家分布到更多rank,改变本地专家数与跨rank流量 | EP通信量、本地专家执行时间 |
| CLI `--expert-tensor-parallel-size` | 无,回落到TP度数 | 切分单个专家;普通alltoall路径增加专家TP域的all-gather/reduce-scatter | 专家TP通信与GEMM耗时 |
| CLI `--moe-grouped-gemm` | 关 | 将本地多个专家组织为grouped GEMM;收益取决于token分布和后端 | 专家计算区间与kernel数量 |
| CLI `--moe-permute-fusion` | 关 | 融合本地permutation/unpermutation等操作,要求相应TE kernel可用 | 重排kernel和临时显存 |
| CLI `--moe-expert-capacity-factor` | 无 | 默认不按每专家容量丢配对;设置后启用相应容量处理,可能影响训练内容 | 实际每专家token数、loss与通信形状 |
| CLI `--moe-pad-expert-input-to-capacity` | 关 | 要先设置每专家capacity;以padding换规则形状。flex+DeepEP不支持此组合,NCCL-EP当前不支持这套每专家drop/pad | 启动校验、padding开销 |
| CLI `--moe-expert-rank-capacity-factor` | 无 | HybridEP/NCCL-EP的每rank接收预算,不是上一项。HybridEP超预算会丢配对并由配套runner检查;NCCL-EP设置后选静态接收路径,需grouped GEMM与TE op-fuser | over-budget提示、重跑次数 |
| CLI `--moe-paged-stash` | 关 | 启用已接入专家路径的页式激活保存;必须有rank容量配置,不能与冲突的MoE卸载同时启用 | stash溢出/重跑日志、显存 |
| CLI `--moe-paged-stash-page-size` | 64 token | 页越大,页数较少但尾部可能浪费更多 | stash占用,固定负载对照 |

**典型配置**。以下是已有 MoE 训练命令的增量,假设使用支持 BF16 的 NVIDIA GPU,专家数能被 EP 度数整除,已有路由与训练配方不变。保持相同卡数、有效token数与global batch比较,未列出的模型和数据参数沿用原作业。

| 场景 | 配置增量 | 交换条件 |
|---|---|---|
| 单机 8 卡高速互联,先建立普通EP通信基线 | `--expert-model-parallel-size 8 --expert-tensor-parallel-size 1 --moe-token-dispatcher-type alltoall --moe-grouped-gemm` | 专家分散在8个rank,先去掉专家TP通信变量;不设置capacity时保持dropless |
| 上一组重排kernel和临时缓冲开销明显 | 加 `--moe-permute-fusion`,先确认TE融合kernel可用 | 用融合实现减少独立重排操作;算法与容量配置不变 |
| 多节点EP,已安装兼容DeepEP,希望比较融合传输 | 保持原EP/专家TP布局,改为 `--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep --moe-grouped-gemm` | 对照融合后端,不同时加入padding或paged stash;能否更快取决于拓扑和负载 |

组合与推荐值是按语义推的起点,不是实测最优。

**看哪里。** 用 profiler 按本地重排、dispatch通信、专家计算、combine通信和还原切开时间线,同时记录各rank的专家token分布与峰值显存。开了paged stash或静态rank预算时,检查 `Paged stash: rerunning forward-backward`、`NCCL EP: grew the receive capacity` 等日志;一次正常前向不足以证明完整迭代没有溢出或重跑。固定负载的step time是最终指标,某个kernel变快并不保证端到端收益。

| 症状 | 先查 | 然后 |
|---|---|---|
| 配了后端但流量路径没变 | dispatcher是否仍为 `allgather` 或 `alltoall` | 需要flex后端时同时选择 `flex`,确认依赖加载成功 |
| dropless仍出现CPU等待 | 动态接收split和每专家计数是否需要搬到CPU | 定位等待阶段,分别比较重排融合和后端,不要直接认定路由错误 |
| 专家GEMM很短,通信占比高 | 每专家token数、专家TP度数与EP拓扑 | 固定路由结果,比较grouped GEMM与更合适的专家布局 |
| padding后显存或耗时反升 | 有效token与容量缓冲差距 | 收紧容量或回到动态路径,同时检查是否丢配对 |
| 接收预算/页式缓冲频繁溢出 | 日志是否发生重跑,是rank预算还是stash耗尽 | 按峰值需求调整对应容量,不能把两种factor当同一参数 |

## 六、常见误区

**「默认聚集路径没有卡间token移动。」** 名字容易让人把它理解成本地筛选。实际上它先跨专家TP×EP域聚集token和映射,本地筛选发生在之后;通信量必须算进去。

**「combine要再乘一次router权重。」** 数学式看起来是在末尾加权求和,实现却可以把乘法放进专家计算。当前常规grouped路径已经加权,还原阶段只做回填和累加;重复乘会改变模型输出。

**「设置容量就一定能捕获CUDA graph,也不会丢token。」** 每专家容量、每rank接收预算和stash页面容量控制不同对象,支持组合还依赖后端。先确认具体分支与溢出行为,不要用一个容量开关推导整个MoE路径都固定形状或全程无同步。

# MoE:路由与负载均衡

这一页回答一个问题:router怎样选expert,又怎样避免token集中到少数expert。说法都在源码基准 `fb6a123a09`(tag `core_v0.19.2`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

只让每个token挑分数最高的expert,并不保证各expert收到相近的工作量。热门expert可能拖慢整层、抬高临时显存,冷门expert则很少获得训练信号。反过来,强行把数量配平也可能改变模型原本偏好的选择。

Megatron把这件事拆成两个问题:**本次选谁、以多大权重计算;后续怎样纠正负载偏斜**。本章到选择结果与权重为止,token怎样搬到对应设备、expert怎样执行见10章。

## 二、解法

router先把token的hidden state映射成各expert的logits,再产生稀疏选择与计算权重。常规默认路径先选top-k logits,再在选中的项之间做softmax;另一类路径先产生正分数,选中后再归一化。开启分组限制时,先筛expert组,再从保留的组内挑expert。

负载控制有两条常用路线。**aux loss路线**把expert命中数量与router分数结合起来,通过反向梯度影响router;可以按microbatch、单条sequence,或当前更新内累积的更大范围统计来计算。**expert bias路线**则根据实际命中计数调节选择偏置:过载expert的偏置下降,欠载expert的偏置上升,影响后续选择。

![sigmoid加expert bias的演示中,token A按加偏置后的分数选E0和E2,计算权重仍由未加偏置的分数得到;三个token形成2、1、3次expert命中,再反馈调整后续选择偏置。](/opensource/Megatron-LM/09a-routing-and-feedback.svg)

图里最关键的是两条分数不能混用。在支持expert bias的正分数路径中,**加偏置的分数只决定选谁**,最终权重取被选expert的原始分数。偏置影响入选机会,并不直接加进expert输出的混合权重;aux loss又有独立的无偏置统计路径,不能把它当成图中实际命中数的简单别名。

capacity是另一层选择。常规训练默认不设上限,保留router选出的分配;显式设置后,才按每个expert的容量筛掉超额分配。被筛掉的是token到某个expert的连接,并非从整个训练batch删除这个token。padding则可把不足的槽位补齐,两者不要混为一谈。

shared expert也是单独的支路,不参加这里的top-k竞争;它可以有自己的gate。它与被选expert的输出如何合并和重叠执行,留到10章。

## 三、代价

**均衡与模型偏好需要一起验证。** aux loss给训练目标增加约束,bias则改变候选排序。数量更均匀不自动等于质量更好,调整后要同时观察主loss、负载与吞吐。

**capacity限制改变了本次计算。** 超额连接被mask后对应权重归零,当前实现不会再把剩下的权重重新归一化。开启padding还可能增加空槽计算,所以“形状固定”不等于“有效工作更多”。

**统计范围决定反馈在回答什么问题。** 每条sequence都均衡与整个更新内均衡不是同一个目标。expert bias在global batch边界更新,并不能保证眼前每个microbatch都已经配平。

## 四、和同类常见做法不一样的地方

DeepSpeed的MoE gating也有aux loss、capacity和token dropping,这些不是Megatron独有的能力。对比应围绕选择、权重与负载反馈分别进行,而不是把“开了MoE”当成一套固定行为。

| 对比维度 | DeepSpeed已核top-k gating路径 | Megatron本章router路径 |
|---|---|---|
| 交给后续执行的信息 | gating结果包含组合权重、分发信息、辅助损失与专家计数 | router给出选择关系和计算权重,dispatcher据此整理token |
| 容量与丢弃 | capacity与是否drop token共同决定哪些分配保留 | 容量约束与选择策略分别配置,丢弃还会改变实际计算量 |
| 负载反馈的阅读重点 | 查看gating中的aux loss及capacity处理 | 分开查看梯度式aux loss和训练步末bias更新;选择bias不直接成为最终权重 |

同样的top-k并不保证相同的有效token分配。比较质量或吞吐前,应先对齐容量、丢弃策略和实际专家负载。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

以下是参考训练入口的CLI;未列出的模型、并行和数据参数沿用已有可运行配方。表内默认值绑定本章源码基准,不是所有MoE模型的共同默认。

| 参数 / 在哪调 | 默认 | 调整的作用与代价 | 怎么看 |
|---|---|---|---|
| CLI `--moe-router-topk` | 2 | 每token选择的expert数;增大增加expert计算与分配数量 | 每层命中数与吞吐 |
| CLI `--moe-router-score-function` | `softmax` | 另有 `sigmoid`、`sqrtsoftplus`;改变评分与权重计算 | 主loss、routing分布 |
| CLI `--moe-router-pre-softmax` | 关 | softmax路径改为先对全部expert归一化再选择,选中权重不等同默认路径 | 权重与主loss |
| CLI `--moe-router-load-balancing-type` | `aux_loss` | 可选microbatch、sequence、global统计的aux方式,也有其他算法或 `none` | 对应层级的均衡指标 |
| CLI `--moe-aux-loss-coeff` | 0.0 | 默认类型存在但系数0不施加aux loss;非零时控制约束强度 | aux指标与主loss一起看 |
| CLI `--moe-router-enable-expert-bias` | 关 | 启用动态选择偏置;只支持 `sigmoid`、`sqrtsoftplus` | 各expert命中与bias变化 |
| CLI `--moe-router-bias-update-rate` | 0.001 | 每次global batch反馈的偏置调整幅度 | 负载是否持续偏斜或来回摆动 |
| CLI `--moe-expert-capacity-factor` | 无 | 启用容量筛选;容量为向上取整的token数×top-k×factor÷expert数 | 有效连接数、被筛掉比例 |
| CLI `--moe-token-drop-policy` | `probs` | 容量不足时按路由权重筛选;另有 `position` | 对照筛选前后routing map |
| CLI `--moe-pad-expert-input-to-capacity` | 关 | 补齐容量槽位;必须先设capacity factor | padding开销与执行形状 |
| CLI `--moe-router-num-groups` / `--moe-router-group-topk` | 无 / 无 | 先限制expert候选组,再选expert;两项要匹配模型分组 | 选择覆盖范围与吞吐 |
| CLI `--moe-router-dtype` | 无,沿用输入dtype | 可设 `fp32`、`fp64`提高router计算精度,增加开销 | logits稳定性、主loss |
| CLI `--moe-z-loss-coeff` / `--moe-per-layer-logging` | 无 / 关 | 前者约束router logits规模,后者开启逐层aux与z-loss观测 | 逐层异常,不是只看全模型平均 |

**配置起点**。假设已有8个routed expert的decoder训练配方,在能容纳其工作量的NVIDIA GPU上测试;固定模型、batch、精度与dispatcher,只改变路由选择。以下都不是完整训练命令,也不承诺某种卡型的性能。

| 场景 | CLI增量 | 取舍 |
|---|---|---|
| 建立dropless aux基线 | `--moe-router-topk 2 --moe-router-load-balancing-type aux_loss --moe-aux-loss-coeff 0.01 --moe-per-layer-logging` | 不设置capacity;通过aux梯度改善均衡,同时检查主loss |
| 尝试无aux的bias均衡 | `--moe-router-topk 2 --moe-router-score-function sigmoid --moe-router-load-balancing-type none --moe-aux-loss-coeff 0 --moe-router-enable-expert-bias --moe-router-bias-update-rate 0.001` | bias按负载改变后续入选机会;不保证每个microbatch严格均匀 |
| 已确认超额负载,试验有容量路径 | 在第一组上加 `--moe-expert-capacity-factor 1.25 --moe-token-drop-policy probs` | 限制expert接收连接,但实际计算被改变;先比较丢弃率与质量,再决定是否padding |

组合与推荐值是按语义推的起点,不是实测最优。

**看哪里。** aux路径记录 `load_balancing_loss`、`seq_load_balancing_loss`、`global_load_balancing_loss`,z-loss另记 `z_loss`。这些日志值会去掉配置系数,不能直接当作加进主loss的数值。逐层loss也不等于逐expert命中数;判断负载要再看routing map、bias计数或router diagnostics,并比较capacity筛选前后的连接数。当前router diagnostics要求本地保有完整sequence,不支持已沿TP或CP切开的sequence,不要把诊断入口当成所有并行组合都可用的开关。

| 症状 | 先查 | 然后 |
|---|---|---|
| 配了aux类型,负载约束却没生效 | aux系数是否仍为0 | 显式给非零系数,对照主loss和负载 |
| 开bias启动失败 | score function是否仍是默认softmax | 选择支持的评分函数,重新验证权重与质量 |
| 某些token少于top-k条有效连接 | 是否设置了capacity | 对照drop前后map,区分容量筛选与router选择 |
| 开padding后更慢 | 空槽比例、capacity是否过大 | 比较不padding的同负载基线 |
| 总aux正常但个别expert持续拥挤 | 统计范围是否掩盖局部偏斜 | 看逐层实际命中与bias,再调整策略 |

## 六、常见误区

**以为bias越大,expert输出权重就直接越大。** bias可能让expert入选,但选中后的计算权重来自未加bias的分数。排查时把排序分数与混合权重分开打印。

**把默认aux类型读成默认开启均衡损失。** 类型和系数是两项独立配置;本基准系数默认0。只看到类型名就停止检查,会漏掉这一点。

**以为所有MoE都必然drop token。** Megatron常规训练默认capacity为空;设置容量才进入这条筛选路径。对已有模型添加上限是计算语义变化,不能只按显存优化处理。

**把routing replay当成重放整份权重。** replay主要固定选中的expert索引,权重仍从当前分数取出。它能复用路由选择,不等于复用上一次router的全部输出。

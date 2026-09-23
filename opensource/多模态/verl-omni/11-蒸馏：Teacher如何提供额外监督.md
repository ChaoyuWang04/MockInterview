# 蒸馏:Teacher 如何提供额外监督

结果分数只告诉模型做得好不好,Teacher 则能在学生实际走过的状态上给出更细的监督。说法都在源码基准 `b708218d5b`(tag `v0.2.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

一张图的 OCR Reward 可以告诉 Actor 文字是否正确,却没有直接给出每个去噪状态下应当怎样改变预测。Policy Gradient 仍能利用结果分数学习,但如果手上已经有擅长这个任务的模型,就可以把它的预测也用起来。

**在线策略蒸馏让 Teacher 看学生走过的状态。** 训练数据来自学生自己的 Rollout,Teacher 不另外生成一批“标准图片”让学生照抄。这样监督落在学生当前确实会遇到的状态上,而不是只落在 Teacher 擅长走出的轨迹上。

这里必须分清角色:Reward Model 评价任务结果;Reference Model 用来约束相对参考策略的变化;Teacher 提供要学习的预测目标。即便后两者都冻结、只做前向,它们进入损失的用途也不同。

## 二、解法

**扩散 Teacher 重放保存下来的去噪状态,返回下一步转移分布的均值。** Actor 在同一状态上产生自己的均值,蒸馏损失比较两者,并用该步的噪声方差归一。监督逐个保存的训练步进入更新,不是为最终图片再添一个分数。它也不要求保存完整生成链的每一步。

训练有两种组合:只用 Teacher 蒸馏作为主目标,或者在原来的任务目标上加一项蒸馏损失。前一种即便仍计算任务 Reward,分数也可以只用于观察;后一种才是任务反馈和 Teacher 指导共同推动 Actor。Reference Model 的约束还可以独立叠加。

**多个 Teacher 按样本来源分工,不对同一样本投票。** 单 Teacher 接收全部样本;多 Teacher 时,管理器按路由列拆批,补齐各自并行前向所需的行数,先发出请求再收结果,最后去掉填充并恢复原始顺序。每条样本只拿到它对应 Teacher 的目标。

![扩散蒸馏监督流:学生 Rollout 保存的状态分别交给 Actor 与按来源选中的冻结 Teacher,两者转移均值进入蒸馏损失;Teacher 可与 Actor 共卡或独立部署,Reward 与 Reference Model 分别提供任务反馈和参考约束。](/opensource/verl-omni/11a-teacher-supervision.svg)

## 三、代价

**额外监督要付额外前向和模型状态的钱。** 默认扩散 Teacher 放在 Actor worker 内,多 Teacher 就要在那里维护多份模型。独立 GPU 池能隔离显存,但默认仍在训练步骤内等待 Teacher 完成,搬到另一组卡不会自动变成流水线。

专门的错步调度能让下一批 Teacher 前向与当前批 Actor 更新重叠,但要求 V1 分卡异步、独立 Teacher 和足够预热。Teacher 冻结,不会因为迟收结果而变成“旧 Teacher”;**学生样本却更早取出**,因此相同取样年龄门槛不能直接当作相同的更新时年龄保证。

**Teacher 不是任意模型都能替换。** 扩散路径继承学生的模型结构配置,加载另一个完整 Checkpoint,并检查解析后的噪声调度配置一致。模型家族与轨迹语义要匹配;Teacher 不加载 LoRA adapter,已有 LoRA Teacher 需要先合并。默认路径为空,不是自动拿 Actor 当前权重当 Teacher。

## 四、与上游 verl 的关系

上游 verl 的 Teacher 运行时围绕 Token log-prob:通过推理服务给学生已生成的序列计算 Teacher 概率。VeRL-Omni 的全模态自回归蒸馏复用这套管理器,补上图像、视频、音频输入与 vLLM-Omni 服务适配。它与扩散蒸馏共享“在学生状态上监督”的思路,但输出不是去噪转移均值,部署也不是 Actor 内的扩散前向引擎。

扩散线因此自己实现了 Teacher 路由与前向交接,再接回已有训练损失入口。当前这条在线蒸馏路径覆盖 Policy Gradient 训练器与 FSDP/FSDP2,不能因为 V1 支持在线 Direct Preference,就认为 Teacher 也自动覆盖该算法族。

**DMD 要单独划界。** 当前仓库包含分布匹配的数值工具、配置和损失:比较 Teacher 与学生分布估计模型的预测,构造给学生的更新方向。它需要另外训练一个估计学生分布的模型,与冻结 Teacher 的普通在线蒸馏不同。本基准的这些损失还没有接上生产所需张量和交替训练的完整引擎流程,不能仅换损失选项就把它当作可运行的 DMD 配方。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

以下以扩散在线蒸馏为主,`A` 表示 `actor_rollout_ref.actor`, `D` 表示 `distillation`。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `D.enabled` | 扩散启动配置 | `false` | 建立 Teacher;必须同时启用消费其输出的损失 | 开关与损失不配套时启动报错 |
| `D.teacher_models.teacher_model.model_path` | 扩散启动配置 | `null`,必填 | 指向同家族完整 Teacher Checkpoint | 缺路径或噪声调度不匹配时报错 |
| `A.diffusion_loss.loss_mode` | Actor 启动配置 | 跟随模型算法 | 设 `distill_kl` 以蒸馏作主目标 | `actor/distill_kl_loss` 与任务 Reward 分开看 |
| `A.use_distill_loss` / `A.distill_loss_mode` / `A.distill_loss_coef` | Actor 启动配置 | `false` / `distill_kl` / `1.0` | 在主损失上叠加 Teacher 项,系数改变其贡献 | 蒸馏损失与 Actor 梯度、任务指标 |
| `D.teacher_key` | 扩散启动配置 | `data_source` | 多 Teacher 的样本路由列;单 Teacher 忽略 | 缺列、未知值会报错,不是自动回退 |
| `D.nnodes` / `D.n_gpus_per_node` | 扩散启动配置 | `0` / `0` | 节点数为 `0` 时共卡,大于 `0` 时建立独立 Teacher 池 | GPU 布局与 Teacher 阶段时间 |
| `D.teacher_models.<名称>.world_size` | 多 Teacher 启动配置 | `0` | 独立池中各 Teacher 占用卡数之和须等于总池;单 Teacher 自动占满 | 配置校验 |
| `D.scheduler` | 扩散启动配置 | `inline` | `one_step_off` 重叠 Teacher 前向与 Actor 更新 | `timing_s/teacher` 或 `timing_s/wait_prev_teacher` |
| `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | Reference 形状的推理配置 | `null` | Teacher 沿用此微批设置,继续回退到 ref 的训练微批或 Actor 微批 | Teacher 显存与耗时 |

**配置方向。** 从官方 SD3.5 Medium OCR 蒸馏或多 Teacher 完整配方出发,保留其模型、轨迹与 Reward 配置;表格只表达改动方向。

| 场景 | 在完整配方上的调整 | 拿什么换什么 |
|---|---|---|
| 验证 Teacher 是否提供有效目标 | 给定已合并的 Teacher 路径,启用蒸馏并选 `distill_kl` 主损失 | 用模仿 Teacher 代替任务奖励驱动,Reward 仅作独立观察 |
| 保留任务探索,加 Teacher 约束 | 主目标保留 `flow_grpo`,打开 `A.use_distill_loss`,先沿用系数 `1.0` | 同时接受两种监督,需观察任务质量而非只看蒸馏下降 |
| Teacher 前向明显占用关键路径 | 独立池上用 `one_step_off`;要求 V1 `separate_async`、`sync_compatible=false`、预热批数至少 `2` | 多占 GPU 与提前取样,换取前向重叠 |

组合与推荐值是按语义推的起点,不是实测最优。

**观测。** 扩散侧分别看 Teacher 损失、任务 Reward 和 Teacher 等待。错步模式的等待接近零只说明该阶段被隐藏,不证明整步更快或质量更好。全模态侧使用 `distillation/loss`、`distillation/loss_min`、`distillation/loss_max`,它们是 Token 监督指标,不能与扩散均值 KL 直接比较数值。

| 症状 | 先查 | 然后 |
|---|---|---|
| 开了 Teacher 却启动失败 | 是否启用了对应损失,训练器是否属于支持范围 | 对照报错修正组合,不要只追查模型下载 |
| 增加 Teacher 后似乎只有新模型生效 | 是否把默认 `teacher_model` 与新增条目并列 | 多 Teacher 使用新的条目名;默认占位项会被移除 |
| 多 Teacher 少量样本也很慢 | 路由后的子批大小与补齐量 | 调整微批或资源布局,别按原整批大小估计 Teacher 负载 |
| 换独立池后速度没有提升 | 是否仍为 `inline` | 先确认显存隔离收益,有足够生成余量再尝试重叠 |

## 六、常见误区

- **以为“复用 Actor 配置”意味着不必指定 Teacher 权重。** 复用的是结构和前向组织,Teacher 路径仍必填;误用同一份未微调权重还可能让初始蒸馏信号很小。
- **以为多 Teacher 会给每条样本综合意见。** 实际是路由选一个;不同领域知识通过不同样本汇入同一个 Actor,没有对多个 Teacher 预测取平均。
- **以为有损失类就有可训练算法。** 扩散在线 Teacher 只生产转移均值;另一个预测 MSE 损失缺配套生产端,DMD 也缺完整训练串接。检查输入由谁产生比检查注册名更关键。
- **以为 Teacher 损失下降就说明任务能力变强。** 它只证明学生更接近给定目标;目标本身是否符合任务,仍要用独立任务指标判断。

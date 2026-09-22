# 全模态训练：自回归路径与 verl 复用

多模态输入怎样回到 Token 级训练流程,以及哪些工作能沿用 verl、哪些仍须自己完成。说法都在源码基准 `b708218d5b`(tag `v0.2.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

**输入包含图像、视频或音频,不等于训练循环必须按扩散方式重写。** 自回归策略仍然逐 Token 生成,训练需要的也是回答序列、对应 log-prob 和有效位置。因此在线打分、Advantage 计算、Actor 更新这些环节,有机会直接沿用 verl。

真正容易接错的是模型边界:生成服务返回的内容未必就是被训练的策略序列,标准语言模型的输入也未必足以重放那次生成。比如训练语音生成阶段时,除了策略 Token,还可能需要保留声学轨迹和条件。若只把多模态输出压成一段文本,训练侧前向就可能对应另一条轨迹。

## 二、解法

**在线路径保留 verl 的训练器,把模型差异收在边缘。** 同步模式直接继承 verl 的 V1 同步训练器,主要补上模型自己的 Tokenizer、Processor 和生成恢复钩子;分卡异步模式继承对应的异步训练器,再接入能够同步 LoRA 的 worker 与权重管理器。在线 Policy Gradient 的损失与 Advantage 仍走 verl 原有机制。

训练侧的 FSDP 引擎先准备标准语言模型输入,再让适配器补充模型专用字段;模型加载后,由适配器选择训练阶段和配置前向。生成侧则把媒体条件送入 vLLM-Omni,取回策略 Token、可选 log-prob 及附加数据。普通 Thinker 文本回答可以使用 verl 的单轮 AgentLoop;需要把输出映射成 Talker 策略序列时,才使用本仓扩展的 AgentLoop,校验序列、掩码和 log-prob 是否逐项对齐。注册契约见 02 章。

![全模态训练路径图:在线同步与分卡异步分别继承 verl 的对应训练器,经 Rollout 和 Reward 获得更新样本;离线偏好直接读取样本对,走本仓独立循环;异步中断的单阶段请求保留前缀并在新权重下续跑。](/opensource/verl-omni/09a-online-offline-paths.svg)

**离线偏好是另一条循环。** 入口将离线 Direct Preference 交给本仓独立训练器。它读取已给定的 chosen/rejected 回答,计算 Actor 与 Reference Model 的 log-prob,再计算偏好损失,没有在线 Rollout 或 Reward 服务。当前这条训练器只接受离线 DPO,不应把配置类里关于在线偏好的注释当成完整实现。样本转换见 10 章,Teacher 的独立职责见 11 章。

## 三、代价

**异步减少等待,同时增加了策略版本管理。** 分卡路径让独立 Rollout 与 Actor 更新重叠,但权重同步仍会打断正在生成的请求。服务先中断请求并暂停接单;客户端保留已生成 Token,恢复后把它们接到原提示词后继续生成,同时扣除已用的回答长度预算。一个回答因此可能跨越权重版本,框架记录它的起止版本,并由采样缓冲区限制陈旧程度。

这个恢复边界不能扩大解释:当前单阶段 Thinker 路径有对应处理,多阶段自回归中断仍有上游限制。分卡也不意味着训练 GPU 上完全没有 Rollout 副本;共卡副本参与初始采样和验证,训练阶段再让出显存。

**离线省去生成,却保留了成对训练的约束。** chosen 与 rejected 必须始终相邻;任意打乱行会破坏比较关系。使用 LoRA 时可以关闭适配器,在 Actor 内取得基础模型的 Reference log-prob;其他情况下需要 Reference 模型资源。减少在线服务并不等于只做一次 Actor 前向。

## 四、和上游 verl 的关系

| 共同任务 | verl-omni 的选择 | 这意味着什么 |
|---|---|---|
| 在线取样、Advantage 与更新 | 继承固定依赖版本的 verl V1 训练流程 | 不把这些能力归为本仓重新实现 |
| 多模态模型前向与生成 | 增补模型输入、阶段选择、输出重组与权重同步 | Token 训练骨架相同,模型接口仍须适配 |
| 离线偏好训练 | 本仓独立组织数据批、Reference 前向和 DPO 更新 | 不能按在线父类去寻找这条主循环 |
| Megatron 训练 | 另有配置继承 verl 的 Megatron 训练配置,并选择其语言模型引擎 | 配置复用不证明本仓 FSDP 的全部 omni 能力在 Megatron 上对等 |

同步与分卡异步是本章核实的 omni 在线训练器扩展。verl 本身还有其他执行模式,但不能仅因配置里保留其字段,就认定它们完成了同样的模型与恢复适配。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `algorithm.trainer_type` / `algorithm.sample_source` | omni 主配置 | `policy_gradient` / `online` | `direct_preference` + `offline` 进入独立偏好循环 | 是否启动 Rollout 与 Reward 服务 |
| `trainer.v1.trainer_mode` | 在线启动配置 | `omni_sync` | `omni_separate_async` 增加独立 Rollout 资源池 | 资源布局、生成等待与同步耗时 |
| `trainer.v1.separate_async.parameter_sync_step` | 分卡配置 | `4` | 一轮同步之间完成更多本地更新;须满足训练批量等于该值乘 Actor mini-batch | `timing_s/update_weights` 与陈旧度 |
| `trainer.v1.separate_async.num_warmup_batches` | 分卡配置 | `1` | 提前提交更多提示词批次 | 首轮等待和待生成请求量 |
| `trainer.v1.sampler.max_off_policy_threshold` | 在线采样配置 | `8` | 放宽或收紧缓冲区可接受的版本差 | `training/off_policy/*` |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | Rollout 配置 | `naive` | 分卡必须改用支持跨池传输的后端 | 启动断言与权重传输日志 |
| `actor_rollout_ref.actor.omni_loss.beta` | 离线偏好配置 | `0.1` | 缩放相对 Reference 的偏好差值,不控制在线 Policy Gradient | `actor/dpo_loss`、`val/reward_margin` |
| `actor_rollout_ref.actor.omni_loss.average_log_prob` | 离线偏好配置 | `false` | 从有效 Token 求和改为平均,改变序列长度对目标的影响 | 按模态的验证指标与回答长度 |
| `algorithm.paired_preference` | 离线偏好配方 | 主配置 `false`,成对配方设为 `true` | 以偏好对展开批量并保护成对顺序 | chosen/rejected 相邻性与 shuffle 警告 |

**配方方向。** 从对应模型的官方完整脚本开始,下表只说明应调整哪条路径。

| 场景 | 调整方向 | 拿什么换什么 |
|---|---|---|
| 首次验证 Thinker 在线训练 | 使用 Qwen3-Omni Thinker GSPO 同步配方 | 接受阶段轮换,先核对 Reward 与 Token log-prob |
| 已确认 Rollout 长尾占主导 | 使用 Thinker 分卡异步配方,同时分配独立 GPU 和权重传输后端 | 增加资源与陈旧度管理,争取生成和训练重叠 |
| 已有多模态偏好对 | 使用 Qwen3-Omni 离线偏好 LoRA 配方 | 省去在线采样,接受数据覆盖面及成对批处理约束 |

组合与推荐值是按语义推的起点,不是实测最优。

异步先同时观察 `timing_s/gen`、`timing_s/update_weights` 和 `training/off_policy/trajectory_spans/*`、`training/off_policy/trajectory_staleness/*`:生成等待下降但版本跨度变大,并不能只凭吞吐判定配置更好。离线验证看 `val/reward_accuracy`、`val/reward_margin` 及其按模态分组结果;这里的 Reward 是由策略与 Reference log-prob 差构造的偏好指标,不是在线评分器的任务得分。

| 症状 | 先查 | 然后 |
|---|---|---|
| GSPO 调整 DPO 系数后没有变化 | 是否仍是 `policy_gradient` | 调整在线 `policy_loss` 配置,不要改离线损失块 |
| 分卡一启动就被断言挡住 | GPU、非 `naive` 后端与批量关系 | 按完整分卡配方对齐,不能只换模式名 |
| 权重同步后生成为空或挂住 | 是否使用多阶段 AR;查看 abort 的报错 | 回到已覆盖的单阶段 Thinker 边界,不要先增加超时时间 |
| 离线验证 accuracy 异常 | chosen/rejected 顺序、掩码、Reference log-prob | 再检查模态分组,区分数据错配与训练目标问题 |

## 六、常见误区

- **看见全模态就寻找扩散轨迹。** 输入模态与策略生成方式是不同维度;本章路径训练的是自回归 Token 序列,模型专用回放字段另行保留。
- **把异步恢复理解为重新生成整个回答。** 客户端会保留前缀与对应 log-prob,只补剩余部分;排查策略版本时需要看回答跨越的范围。
- **以为继承 verl 就能打开所有优化。** omni FSDP 引擎显式拒绝 Liger 与 fused kernel 开关;部分冻结模型使用 FSDP1 时也有原始参数约束,不能照搬普通语言模型配方。
- **见到 Megatron 配置就推断后端可无缝替换。** 这份配置转向 verl 的语言模型引擎,不是给本仓 omni FSDP 实现换一个名称。

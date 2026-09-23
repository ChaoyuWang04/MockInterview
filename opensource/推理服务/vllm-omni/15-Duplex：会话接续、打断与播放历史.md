# Duplex：会话接续、打断与播放历史

这一页解决连续语音交互中，用户插话或连接中断后，模型应该继续听谁、保留哪段回答的问题。

说法都在源码基准 `4d877780d3`(tag `v0.29.0rc1`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

语音回答生成完，并不意味着用户听完。模型可能还在生成、网络正在传输，而客户端已经积压了一段待播放音频；这时用户插话，若把整个回答写进历史，下一轮模型就会默认用户听过根本没有播出的内容。

连接中断又会放大这个问题：直接重开会话丢掉上下文，直接重发音频可能重复播放，旧连接晚到的输出还可能混入新一轮。因此 Duplex 的核心是**让模型执行、传输连接和已播放历史分别有自己的边界，再把它们对齐**。

## 二、解法

会话留在 engine 内，由会话管理器持有；WebSocket 只是可以替换的连接。恢复时先验证凭据与事件日志的连续性，再接续 engine lease、轮换凭据并重放客户端未确认的事件。新连接接管后，旧连接不能再关闭新连接的会话。它保住的是仍然活着的服务内会话，不是把整个模型状态持久化到磁盘。

输入侧先决定谁负责轮次。模型原生 Duplex 可以边听边决定何时说话；服务端 VAD 则判断语音起止，并驱动打断与提交。Overlap 策略再决定说话期间到来的输入是继续听、延后处理还是 barge-in。Barge-in 不只取消任务，还推进轮次的 epoch；epoch 是拒收旧输出的代际编号，使异步执行里晚到的旧回答不能重新冒出来。

![客户端播放确认决定历史保留边界，打断推进 epoch，断线重连则替换连接并继续同一 engine 会话。](/opensource/vllm-omni/15a-duplex-session.svg)

输出侧用 playback ledger 记录音频进度，并按 response 保存播放与历史快照。客户端 ACK 给出实际播放位置，系统才据此提交或截断回答。生成、输出记账和实际播放是不同事实；当前实现会同时推进生成与发送游标，这两个值都不能当作客户端送达回执。

## 三、代价

会话脱离连接生存，就必须保留输入、模型状态、事件日志和历史快照，并为断线设置宽限期。宽限期越长，临时掉线更容易恢复，也越久占着会话容量；事件日志一旦超出保留范围，服务会要求重新同步，而不会假装可以无损续传。当前恢复凭据与日志保存在 API 进程本地，因此 Duplex 接口限制单个 API worker。

播放确认也不是强制对齐器。模型提供音频与文字位置标记时可以据此裁剪；没有标记时存在按时长比例估计的路径。明确声明文字与音频未对齐的输出，只有完整音频结束且播放确认覆盖全部内容后才能提交整段文字，不能承诺字级精确截断。

插件能力决定模型状态能续到哪一层。MiniCPM-o 4.5 支持原生持续输入与可恢复的核心请求；Qwen3-Omni 的接入按提交后的轮次工作，可以并行收音和打断回答，但会重新编码上下文。会话恢复不自动等于 KV lease，更不等于任何模型都能回滚内部状态。

## 四、与 sglang-omni 的 Realtime 历史处理对照

SGLang-Omni 的 Realtime 会话实现把 WebSocket 与逐轮对话放在同一个会话对象内：取消的回答不进入历史，客户端截断已完成的回答时会删除对应 assistant item。vLLM-Omni 这条 Duplex 路径把 engine 会话与连接分开，并进一步维护按 response 的播放游标和历史位置，尝试保留已确认部分。

这说明两者对“哪些回答算听过”采取了不同粒度的处理；不能只看到都支持 Realtime 和打断，就认为连接恢复、历史截断与模型持续输入具有相同语义。

## 五、调参与观测

这里有部署配置、会话配置与逐事件控制。部署 YAML 改后重启；会话参数通过 `session.update` 更新，需等 `session.updated` 确认，模型插件仍可拒绝不支持的修改。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `session_mode` | deploy YAML | `turn`；本页显式用 `duplex` | 选择 Duplex 会话入口与已注册插件 | `session.created` 中的模型能力；普通模型目录存在不代表插件已接入 |
| `duplex_session.max_sessions` | deploy YAML | 通用 `1`；MiniCPM-o 4.5 配方为 `4` | 增加准入容量，也增加模型状态与输入缓存占用 | `resource_exhausted`；该上限不是吞吐保证 |
| `duplex_session.idle_ttl_s` / `disconnect_grace_s` | deploy YAML | `300` / `30` 秒 | 延长空闲寿命或断线恢复窗口，占槽更久 | `session.expired` 原因；普通 heartbeat 不会清除断线状态 |
| `duplex_session.resume_replay_ttl_s` / `resume_replay_max_bytes_per_session` | deploy YAML | `60` 秒 / `8388608` 字节 | 扩大事件重放范围，增加 API 进程内存占用 | `session.resync_required` 的 `journal_gap` / `journal_overflow` |
| `duplex_session.server_vad_model_path` | deploy YAML | `None` | 指定本地 Silero ONNX；显式路径失效不偷偷换后端 | `unsupported_turn_detection` 与 VAD 加载错误 |
| `turn_detection` | `session.update.session`，也接受嵌套音频输入字段 | 未提供则不修改；显式 `null` 关闭服务端 detector | `server_vad` 派生 `barge_in_on_speech`；`null` 派生 `listen_only`，轮次行为再由插件决定 | `input_audio_buffer.speech_started` / `speech_stopped` 与 `overlap.decision` |
| `turn_detection.threshold` / `min_speech_duration_ms` | 上述 VAD 对象 | `0.5` / `96` 毫秒 | 提高阈值或延长最短语音通常更保守，也可能更迟识别短语音 | 语音起点与误触发；阈值需大于 `0.15` 且不大于 `1` |
| `turn_detection.silence_duration_ms` / `prefix_padding_ms` | 上述 VAD 对象 | `500` / `300` 毫秒 | 延长静音等待推迟轮次提交；padding 向前扩展语音起点时间戳 | `speech_stopped`、`audio_start_ms`；不把 padding 当额外播放缓冲 |
| `turn_detection.create_response` / `interrupt_response` | 上述 VAD 对象 | `true` / `true` | 前者控制自动提交后是否请求回答；后者当前不接受 `false` | 配置拒绝错误或后续 `response.created` |
| `playback_commit_policy` | 会话配置 | 配置类为 `commit_all_on_done`，当前 runner 启动覆写为 `ack_only` | 使用 ACK 决定历史边界；后续更新应核对会话实际值 | `playback.acknowledged` 的 `history_committed` 与播放游标 |
| `played_ms` / `committed_ms` / `truncate` | `playback.ack`，带 `response_id` | 未给提交位置则采用播放位置；`truncate` 默认不启用 | 报告累计播放位置；显式截断可缩回历史提交边界 | `playback_item_mismatch`、`playback_ack_too_late` 和 ACK 响应 |

VAD 语音起点分支可以立即触发 barge-in，不应把通用 overlap 的 `overlap_barge_in_ms=1200` 当作这条路径必等的延迟。显式 `turn_detection: null` 也不等于所有模型都会自己答复：Qwen3 插件明确要求 VAD 或客户端提交，不接受原生自动回答开关。

| 场景 | 怎么配置 | 拿什么换什么 |
|---|---|---|
| 大显存 CUDA 单卡、MiniCPM-o 4.5，模型决定何时说话 | `vllm-omni serve openbmb/MiniCPM-o-4_5 --omni --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --trust-remote-code`；用模型客户端 preset 提供真实 `ref_audio`，会话设置 `turn_detection: null` | 按模型原生听说节奏持续输入；音频输出需要参考音频，仍须发送播放 ACK |
| 沿官方双卡配置起 Qwen3-Omni，以服务端 VAD 驱动可打断轮次 | `vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --deploy-config vllm_omni/deploy/qwen3_omni_duplex.yaml`；会话 `turn_detection: {"type":"server_vad","threshold":0.5,"silence_duration_ms":500}` | 用 VAD endpointing 驱动提交和打断；这是逐轮处理，不是原生持续追加 KV；需先满足 VAD 依赖与卡容量 |
| 已运行的任一支持恢复的 Duplex 会话，客户端遇到短时断网 | 重连 URL 加 `resume=1`，首事件为 `session.resume`，携带原 `session_id`、最新 `resume_token`、实际收到的 `last_received_server_event_seq`；保持默认断线与重放窗口作为起点 | 复用存活会话并补事件；客户端按序号去重，凭据或日志边界失效时不能无条件续传 |

组合与推荐值是按语义推的起点,不是实测最优。

`session.event_ack` 确认收到了哪些事件，可以回收重放日志；`playback.ack` 确认音频播到哪里，影响历史。收到音频包就发送播放 ACK，会把网络接收进度错误地当成用户听过的内容。客户端应依据播放器实际进度回报，并在下一轮用户输入提交前及时确认；已有历史占位的回答允许后续更新原位置，未占位的旧回答晚到 ACK 则可能被拒绝。

| 症状 | 先查 | 然后 |
|---|---|---|
| 回答播完但下一轮好像不知道自己说过 | 是否发送了对应 response 的播放 ACK | 查看 `history_committed`；未对齐音频还需完整结束及全量播放确认 |
| 用户说话立即打断，改大 overlap 时长却无效 | 是否走 VAD `speech_started` 分支 | 调 VAD 判定并检查能力声明，不把通用时长当硬门槛 |
| 模型一直听、不回答 | 使用的是原生插件还是逐轮 Qwen3 插件 | 核对 VAD / 显式 commit、response 创建以及模型 listen 事件 |
| 重连得到新会话或没有历史 | 是否意外使用带模型名的自动开会话连接 | 使用 resume 握手，检查原会话是否仍存活 |
| 收到重新同步要求 | 日志 TTL、字节上限及最后接收序号 | 区分事件缺口与 engine lease 过期，不只延长网络超时 |
| 增加 API worker 后 Duplex 被拒绝 | `multi_api_duplex_unsupported` | 保持单 API worker；增加会话容量需按模型与 worker 资源核对 |

## 六、常见误区

**看到生成结束事件，就把全文写入“用户已听过”的历史。** 生成比播放快时，这会让打断后的对话凭空多出一段回答。当前 runner 启动采用 ACK-only 记账，完成事件不能替代播放器回报。

**看到发送游标增长，就认定网络已经送达。** 当前模型输出处理会同时推进 `generated_ms` 与 `sent_ms`，随后才发事件；掉线时事件还可能只留在日志里。诊断延迟要分别观察生成、接收和播放，不能用这两个内部游标计算真实网络时延。

**把长连接恢复理解为任意 API worker 都能接回。** engine 会话确实独立存在，但凭据和重放日志仍属于本地 frontend。单 worker 限制有实际状态归属原因，不能只在反向代理层放开连接数解决。

**看到 PersonaPlex 模型与客户端 preset，就认定支持本页全部能力。** 当前统一插件路径应以 pipeline 的 `duplex_plugin` 与实际能力返回为准；MiniCPM-o 4.5 和 Qwen3 已有具体插件，模型目录或旧 adapter 的存在不能代替这个检查。

**认为播放毫秒数天然对应一个精确文字前缀。** 未对齐的模型输出没有这种保证；截断可能依赖位置标记、比例估计，或等待完整播放后整段提交。排查历史缺字时先核对模型的音频文字契约，而不是直接放宽 ACK。

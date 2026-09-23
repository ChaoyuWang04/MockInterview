# 服务面：OpenAI 兼容接口与多模态输出

这一页解释客户端提交的多模态请求怎样变成模型输入,以及普通响应、音频流和视频任务为什么需要不同的完成与交付方式。说法都在源码基准 `4d877780d3`(tag `v0.29.0rc1`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

文本对话可以等完整答案,也可以边生成边返回;语音客户端需要能播放的音频,长视频则可能要先拿任务标识,随后查询和下载。如果把这些任务都当作一次文本响应,客户端会在错误的地方等待,或者把“任务已经接收”误认为“媒体已经生成”。

模型差异又加了一层约束。专用 TTS 接收文本、speaker 与参考音频,omni 对话模型可能要求经过聊天模板的上下文;能输出音频,并不表示两者能吃同一种请求。

**服务层要统一接入方式,但必须保留每类任务的输入契约和结果生命周期。** 这比给所有模型挂上相同路径更重要。

## 二、解法

**先按运行引擎与模型能力装配服务,再处理请求。** API server 基于 vLLM 的应用建立路由,替换需要 omni 处理的入口,初始化各类 serving handler,最后应用模型声明的端点限制。单阶段 Diffusion 和多阶段模型走不同的初始化分支;路由存在并不保证当前模型能执行对应任务。

**输入转换留在 serving adapter。** Chat 先渲染聊天模板、整理多模态输入,检查请求的输出模态,再组装各阶段采样参数并交给引擎。专用 TTS 则按部署的模型选择 adapter,由它校验 voice 和任务类型、构造 prompt、调整采样预算与输出策略。以 Qwen3-TTS 为例,checkpoint 的语音任务类型会参与校验;不能靠改一个请求字段让预设音色模型变成克隆模型。

**输出按使用方式交付。** Chat 封装完整 JSON 或 SSE;Speech 封装音频字节或音频事件;Image 生成等待结果后返回 base64 或文件。文件响应可以分块发送已生成的字节,不代表图片在生成过程中逐步变清晰。

长视频另走任务生命周期:创建请求登记任务并启动后台生成,客户端先得到任务标识。后台收到引擎开始执行的通知后更新状态,生成结束还要编码和保存文件,成功后才把任务标记为完成。后续查询读任务记录,下载则同时需要任务记录与磁盘文件。

![同步响应沿原HTTP连接交付结果,异步视频先返回任务标识,后台生成保存文件后才完成,查询读取进程内记录而下载还依赖独立文件](/opensource/vllm-omni/14a-response-job-lifecycle.svg)

OpenPI 保留独立的 WebSocket 协议:先发送模型的 policy 配置,再接收 MessagePack observation,等待该次引擎请求完成后返回 action。它不是聊天事件的另一种输出模态。持续语音会话及全双工的状态推进留给第 15 章。

## 三、代价

**兼容的是明确的接口行为,不是所有字段与模型的任意组合。** 服务端仍要检查模型名、输入格式、输出模态和 adapter 能力。音频编码格式、图片响应格式与流式方式也各有约束,不能复制一份请求体到所有端点。

**媒体转换占用请求链路。** 图像编码、音频封装和视频文件写入都在模型输出之外。视频任务的完成耗时包含生成与保存等工作,不能直接拿它当 GPU kernel 时间;任务标识返回得快也不表示 denoise 变快。

**视频任务当前不是持久任务队列。** 元数据与后台 task 存在 API 进程内,生成文件另存本地目录。重启后保留文件不等于恢复任务;只共享文件目录也不能让多个 API worker 共享查询和取消状态,所以当前异步视频接口拒绝多 worker 访问。

**文件过期和任务删除是两件事。** 可选 TTL 清理磁盘文件,不负责同步清理内存中的完成记录。客户端可能查到已完成任务,下载时却发现文件不在。主动删除运行中任务则先尝试有界中止,再取消前端 task;这不保证正在执行的 GPU 批次立刻停止。

## 四、和 sglang-omni 的语音入口对照

这里核对 sglang-omni 基准 89e60d0bf2 的 Speech 入口。两边都先校验与整理请求,再交给内部生成接口,不是把 HTTP 请求原样递给模型。

sglang-omni 的共享 SpeechRequestValidator 接收部署能力配置,构造带语音任务 metadata 的 GenerateRequest;vLLM-Omni 在共享 serving 流程中调用按模型注册的 TTS adapter,将模型校验、prompt 与采样覆写集中到该 adapter。差异是模型专用逻辑在哪个接入点进入,不能据此说某边没有统一请求层。

还有一个直接影响客户端迁移的差异:所核的 sglang-omni HTTP 语音流返回原始 PCM,vLLM-Omni 的普通语音流开关默认选择 SSE,另有显式的原始音频选项。同一个“流式”意图,并不自动对应相同的响应格式;客户端必须按实际媒体类型解码。

## 五、调参与观测

本页同时涉及启动配置和请求字段:环境变量与启动参数改后重启服务,请求字段只影响本次调用。文件保留、传输格式和生成成本应分开调整。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| Speech `stream` | `/v1/audio/speech` JSON | `false` | `true` 默认返回 SSE 音频事件;能否提前播放还取决于模型分块 | `Content-Type: text/event-stream` 与首个音频事件 |
| Speech `stream_format` | 同上 | 未设置 | `audio` 选择原始音频流;`sse` 选择事件流 | 响应媒体类型与客户端解码方式 |
| Speech `response_format` | 同上 | `wav` | 选择音频编码;HTTP 流式仅接受 `pcm`、`wav` | 参数校验错误、媒体类型与播放结果 |
| Image `response_format` | `/v1/images/generations` JSON | `b64_json` | `file` 改为文件响应,改变传输封装,不减少 denoise | JSON 的 `data` 或响应文件;`url` 当前校验拒绝 |
| Image `n`、`size` | 同上 | 1、模型默认 | 增加数量或尺寸通常增加生成和传输成本 | 输出数量、`metrics.stage_durations`、`metrics.peak_memory_mb` |
| Image `num_inference_steps` | 同上 | 模型默认 | 改变本次 denoise 步数及质量/耗时取舍 | 固定 seed 的结果和阶段耗时 |
| `VLLM_OMNI_SERVER_STORAGE__PATH` | 启动环境变量 | `/tmp/storage` | 改变视频生成文件目录,不持久化任务元数据 | `persisted` 日志、目录容量与下载结果 |
| `VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY` | 启动环境变量 | 4 | 改变保存/删除文件的并发上限,不是 GPU 请求并发 | 磁盘 I/O 与保存延迟 |
| `VLLM_OMNI_SERVER_STORAGE__FILE_TTL` | 启动环境变量 | 不设 TTL | 设正秒数后清理过期文件,缩短下载保留窗口 | job `expires_at`、文件是否存在 |
| `VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL` | 启动环境变量 | 未设 TTL 时未设置;设 TTL 后默认300秒 | 增大降低扫描频率,允许过期文件更久后才被删除 | 清理日志与磁盘占用;不是严格到秒删除 |
| `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` | 启动环境变量 | 600秒 | 改变 `/v1/videos/sync` 的等待上限 | 超时返回504;不影响异步 job 的完成速度 |
| `--robot-openpi-idle-timeout` | 命令行 | 30秒 | 调整 OpenPI 等下一条消息的空闲超时;0 关闭 | idle timeout 日志与连接关闭 |

**先检查服务能力,再检查生成状态。** `GET /health` 和 `GET /v1/models` 用于确认服务及模型;任务是否支持还要看实际 handler 的错误信息。Speech 的 `[SpeechE2E]` 日志包含请求标识、状态和总耗时;异步视频看 `status`、`error`、`stage_durations` 与文件保存日志。视频 `inference_time_s` 计时到文件保存后,不等于纯 GPU 推理时间。

**视频客户端的顺序**是 `POST /v1/videos` 提交 multipart,用返回的 `id` 查询 `GET /v1/videos/{id}`,完成后访问 `GET /v1/videos/{id}/content`,不再保留时调用 `DELETE /v1/videos/{id}`。图像生成当前沿本次请求等待返回,不采用这套视频 job 查询协议。

以下配方只确定服务与响应行为。图像以单卡 A100 80 GB 的 Qwen-Image 为起点;TTS 与视频使用已按各模型部署说明验证容量的 CUDA 环境,不据此承诺任意卡型都能装下模型。

| 场景 | 怎么起与怎么请求 | 拿什么换什么 |
|---|---|---|
| 单请求图像下载,先验收输出 | `vllm serve Qwen/Qwen-Image --omni --port 8091`;向图像生成端点提交 `{"prompt":"A red cup on a table","response_format":"file"}` | 保持整次请求等待,得到可直接保存的文件,不额外建立视频式任务状态 |
| Qwen3-TTS 预设音色,客户端直接播放PCM | `vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091`;向语音端点提交 `{"input":"你好。","voice":"vivian","stream_format":"audio","response_format":"pcm"}` | 用持续读取音频流换取分块播放机会,客户端需按模型采样率处理PCM |
| 长视频与有限文件保留窗口 | `VLLM_OMNI_SERVER_STORAGE__FILE_TTL=3600 vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091`;使用异步视频端点提交并轮询 | 接受客户端维护任务标识与下载流程,避免长连接等待;文件约1小时后进入可清理范围,元数据仍不持久 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 有音频能力,Speech 请求却失败 | 是专用 TTS 还是需要聊天模板的 omni 模型 | Qwen3-Omni 等按提示走 Chat 并请求音频,不要绕过输入契约 |
| 流式语音交给播放器后是乱码 | 响应是 SSE 还是原始PCM/WAV | 按实际类型解析,需要原始流时显式设置 `stream_format` |
| 不支持的 completions 或 image edits 返回400 | 模型 endpoint restriction 的理由 | 这是替换后的拒绝 handler,不要当作404路由缺失排查 |
| 视频创建成功但没有可下载内容 | job 是否仍 `queued`、`in_progress` 或已 `failed` | 完成后再下载;失败时读取 `error`,不是重复下载 |
| 视频显示完成,下载却404 | 文件TTL、目录与文件存在性 | 区分元数据仍在和文件已过期,不要把完成状态当永久存储承诺 |
| 多 API worker 下视频接口返回409 | 是否使用进程内 job store 的接口 | 使用单API worker承载这套任务生命周期,共享磁盘不能解决元数据归属 |
| OpenPI 连上后读不出 JSON | 是否按MessagePack与NumPy标记解析二进制帧 | 使用匹配的OpenPI客户端,先读policy配置握手 |

## 六、常见误区

**认为 OpenAI 兼容意味着所有路径都是 OpenAI 标准。** 客户端看到熟悉的前缀很容易复用同一套协议处理。这里同时提供兼容接口和项目扩展;OpenPI 的二进制 observation/action 与 Chat SSE 是不同协议,流式视频和全双工也各有契约。

**认为图像文件响应就是渐进生成。** 返回类型用了 StreamingResponse,容易误解为边 denoise 边出图。该路径先拿完整图片并编码,再按块发送文件;它优化的是交付形式,不是生成时机。

**把视频任务创建的200当成生成成功。** 创建接口已经返回,后台仍可能排队、失败或保存文件失败。只有任务进入 `completed` 才表示这次生成与保存已完成,而下载仍受文件生命周期约束。

**把持久目录当成任务恢复方案。** 文件在重启后还看得见,看似只差重新启动服务。实际查询先查进程内任务记录,没有记录就无法通过原job接口定位该文件;本地文件存储不等于持久job数据库。

**为节省磁盘只设置TTL,却期待任务列表同步变短。** 清理器按文件修改时间扫描目录,没有同步删除job记录。应用若需要明确的保留与列表语义,仍要在下载后主动删除任务,不能把文件TTL当任务TTL。

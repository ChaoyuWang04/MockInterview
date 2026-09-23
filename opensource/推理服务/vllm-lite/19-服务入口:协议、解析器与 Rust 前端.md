# vLLM 19|服务入口:协议、解析器与 Rust 前端

推理模型吐回来的是一条夹着思考标记和工具调用标记的文字流,API 层得按这个模型的格式把它切成三路、装进协议字段;切错了输出就是坏的,而这些活全压在一个 Python 进程上,并发一高它先满。

说法都在源码基准 `94f4170df3`(tag `v0.30.0`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

Agent 类负载每一轮都长一个样:客户端带着工具清单发来多轮对话,要拿回三样分开放好的东西——模型想了什么、对用户说了什么、要调哪个函数带什么参数。模型只会吐 token。一段回答在它那里是首尾相接的一条流:思考开始标记、思考内容、思考结束标记、一句正文、调用开始标记、函数名与参数、调用结束标记。每个模型家族的标记和参数写法都不一样,vLLM 登记在册的工具调用格式有 51 个名字,思考段格式有 34 个。

API 层切错,坏的是**输出本身的正确性**,而且不报错。工具调用那一段原样留在正文里,工具调用列表是空的,结束原因写着正常结束,客户端以为模型不打算调工具;思考段没切出来,用户看见一大段草稿;反过来配了思考切分,模型这轮没写结束标记或被长度上限截断,整段回答都算成思考,正文是空的。多轮时客户端把上一轮带回来,切错的内容又被模板渲染进下一轮的提示词,错误一路往后传。

另一个问题在量上。切分逐条请求、逐个流式块地做,和 01 章的反切词、判停止串挤在同一个 API 进程里。老写法的解析器每来一块都拿到从开头到现在的全文重扫一遍:一条 4000 token 的回答每块推 1 个 token,要扫 4000 遍,累计扫过的文字约合 800 万个 token。并发一高,GPU 还没满,API 进程的 CPU 先满,客户端看到的出字间隔跟着拉长。

## 二、解法:一条路进,三路出,切法按名字登记

最天然的直觉:协议有几种就写几套处理,模型格式有几种就写几个切分器,让用户指定用哪个。vLLM 前一半没这么做,后一半正是这么做的。

**几种协议并成一条路。** OpenAI 聊天接口是主干。Anthropic Messages 接口自己不干活:收到请求先翻译成一条 OpenAI 聊天请求——系统提示变成系统消息,历史里的思考块变成助手消息的推理内容,工具定义一一对应,「任意工具」对应「必须调用」——然后原样走聊天那条路,结果再翻回 Anthropic 的内容块。Responses 接口有自己的一套请求与事件格式,但渲染和切分用的是同一个渲染器、同一套解析器。老的补全接口不套模板也不切分。gRPC 是另一回事:打开后 HTTP 服务整个不起,换成一个外部包提供的服务,它只拿到引擎客户端,本章的渲染器与解析器都不交给它。

**请求字段怎么变成采样参数。** 请求里写了的优先;没写的取模型目录里生成配置文件的默认值,启动时会打一行警告说默认值被模型覆盖了;两边都没有才用中性值,温度 1、top-p 1。输出长度不写就是上下文剩下的全部。一条请求最多带 4 个停止串。

**模板在哪渲染、谁来选。** 在 API 进程里、切词之前,由渲染器把消息列表拼成提示词。渲染器按模型架构选:DeepSeek V3.2、V4、V4.1 与 Inkling 用代码逐条拼,不走模板;Kimi K3 有专门的渲染器,切词时把结构标记直接当特殊 token 处理;带官方分词器文件的 Mistral 模型交给 Mistral 自己的库;其余都走 Hugging Face 的 Jinja 聊天模板。模板本身分四级取:启动时指定的最先,其次是多模态处理器自带的(请求带工具时跳过这一级),再次是分词器自带的,最后是 vLLM 内置的兜底模板。请求里自带模板默认直接拒掉,要显式打开信任开关才收。思考开不开是模板变量:服务端可以设默认值,请求可以覆盖,请求里的推理力度字段也会被翻成这个变量。渲染完 API 进程还要看一眼提示词,模板已经把思考段关上了,就告诉引擎「思考已结束」,14 章的格式约束从第一个 token 起生效。

**解析器按名字登记,启动时由人指定。** 两张表,一张工具调用、一张思考段,都是「名字 → 文件与类」,用到才导入。启动时给两个名字,每条请求把两者合成一个统一解析器对象。Python 这边不猜:不给名字就不切。工具调用还多一道闸,只给名字、不打开「自动工具选择」开关,工具解析器根本不装。

**流式下怎么增量切。** 统一解析器分两个阶段:思考阶段只归推理解析器管,它认出思考结束标记后转入下一阶段;此后调用开始标记之前的文字算正文,之后交给工具解析器,它先吐函数名,再把参数一段段往后吐。切分器有新旧两种写法。旧接口每来一块都拿到累积全文,自己跟上次比出差量,Hermes 格式那个解析器的注释直说每次从头重扫。新写法是一台声明式解析引擎:每种格式只声明有哪些标记、在哪个状态遇到哪个标记跳到哪个状态、每个状态里的普通文字算哪一路;引擎自带增量词法器,只吃新来的那一块,状态留在自己身上,特殊 token 按编号认,不靠拼出来的文字。51 个工具调用名字里有 15 个、34 个思考段名字里有 14 个已经挂在这台引擎上,新模型要求先写成这种。

![一条推理模型的工具调用请求在 API 层走一圈:OpenAI 聊天、Responses、Anthropic 三个入口汇到同一个渲染与切词站,再送进 EngineCore;回来的增量文字是一条首尾相接的流,紫色思考段、绿色正文、黄色工具调用,推理解析器管到思考结束标记为止,之后调用开始标记前的文字原样当正文,调用段交给工具解析器,三路分别进推理内容、正文、工具调用列表;红色标出两种切错的去向:工具解析器没开时调用文字留在正文里,等不到思考结束标记时整段进推理内容而正文为空](/opensource/vllm-lite/19a-request-roundtrip-three-way-split.svg)

**三路放在响应的哪里。** 思考段进推理内容字段,正文进内容字段,工具调用进工具调用列表,此时结束原因改成「工具调用」;Anthropic 那边对应思考块、文本块、工具块。推理内容字段改过名,请求里还收老名字并就地改成新名字,响应里只出新名字。多轮时客户端把思考段带回来,vLLM 把它连同工具调用一起交给模板,要不要渲染回提示词由模板决定;交错思考的模型(在两次工具调用之间还要想)靠的就是这一点。

**必须调工具时改成约束解码。** 工具选择设成「必须调用」或点名某个函数时,不再指望模型按自己的格式写:工具参数的 schema 被翻成结构化输出约束,模型直接吐一个 JSON 列表,切的时候按列表解析。自动模式默认放模型自由写、事后切;有工具声明了严格模式,或服务端把严格级别调高,才给调用外壳加一层按这个模型格式生成的结构标签约束。约束在引擎里怎么生效归 14 章。

**Rust 前端:整个 API 进程换掉。** 01 章说过 API 进程可以多开;另一条路是换成一个 Rust 写的进程。它不是加速某一段,而是从 HTTP 一路替换到和 EngineCore 通信的那一层。自下而上 6 层:ZMQ 加 MessagePack 的引擎客户端,说的是和 Python API 进程同一套协议,EngineCore 一行不改;一个只收发 token 的薄门面;分词与增量反切词;聊天层,管模板渲染和思考、工具切分;axum 写的 OpenAI 兼容 HTTP 服务;最上面是命令行。启动仍归 Python:启动命令照常拉起 EngineCore,把已经绑好的监听套接字和两端地址交给 Rust 子进程,自己留下来盯进程死活。Rust 进程内部是多线程的,分 3 个线程池:HTTP 收发一个;渲染、切词、组装请求这些重活挪到另一个,免得堵住 HTTP;和引擎的 ZMQ 收发再一个。所以它只开 1 个进程,多开 API 进程的设置被忽略。值得换的场景很窄:API 进程的 CPU 先满而 GPU 没满,流量只走聊天与补全,用到的解析器 Rust 这边也有。

## 三、代价

- **解析器靠人配对。** Python 侧不按模型自动挑。模型换一代往往换一种格式(DeepSeek V3、V3.1、V3.2、V4、V4.1 各一个名字),配成上一代的名字不会报错,只是切不出来。解析器内部抛了异常也不让请求失败,按「没调工具」处理,连指标都记成没调工具。
- **流式要扣字。** 一块文字的结尾像某个标记的开头,比如只来了一个左尖括号,解析器只能先扣住,等下一块确认不是标记才吐,流式输出偶尔停一拍。旧写法的平方级重扫前面已经算过,挂在新引擎上的名字没有这笔账。
- **Anthropic 多绕一圈。** 流式时先生成完整的 OpenAI SSE 文本,再逐行解析回对象、重包成 Anthropic 事件,每块多一次序列化和反序列化。回给客户端的思考块签名是随机生成的,不能拿来校验。
- **Responses 的会话状态在进程内存里,默认不存。** 要按上一轮的编号续对话,得打开存储开关;打开后没有淘汰,内存只涨不降,多开 API 进程时各存各的。
- **Rust 前端缺东西。** 官方自称实验性、功能不全。它只服务聊天、补全、模型列表、切词与反切词,加上打开开关后的 token 进出接口和开发接口;Responses、Anthropic、嵌入与打分、语音转写都没有。Python 这边有 47 个参数它认得但没实现,带上就启动失败,另有 7 个接受了但不起作用。能用的解析器名字也少,工具调用 26 个、思考段 22 个(各含 5 个同时管两件事的统一解析器)。模板渲染换成了 Rust 的 minijinja,和 Python 的 Jinja2 不是同一份实现。

## 四、和 SGLang、TensorRT-LLM 比:解析器谁来挑,前端往哪换

对照对象是 SGLang(基准 826d5170ae)与 TensorRT-LLM(基准 59f5c47f2e)。三家都把 Anthropic 请求翻成自家的 OpenAI 聊天请求走同一条路,也都靠一张按名字登记的解析器表切输出;分歧在不给名字时谁来挑,以及 Python 前端扛不住时往哪换。

| 必须有人做的事 | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| 登记了几种格式 | 工具调用 51 个名字,思考段 34 个 | 工具调用 40 个,思考段 32 个 | 工具调用 14 个,思考段 15 个 |
| 不给名字时 | Python 前端不切;单独跑的 Rust 前端按模型名里的子串猜,由 Python 拉起时也不猜 | 默认不切;设成自动时读聊天模板猜 | 默认不切;设成自动时读模型目录里配置文件的模型类型,不是本地目录或猜不出就启动报错 |
| Anthropic 接口 | 翻成聊天请求,流式时把 SSE 解析回来重包;Rust 前端没有 | 翻成聊天请求,始终挂着;Rust 前端没有 | 翻成聊天请求;gpt-oss 那条路上不挂 |
| Python 前端扛不住时 | 多开 API 进程,或换成独立的 Rust 进程,隔着原来的 ZMQ 边界接 EngineCore | Rust 前端嵌进调度器进程,调度器直接从进程内的环形缓冲区取请求 | 仓库里没有 Rust 代码 |
| gRPC | 打开后替换 HTTP,服务由外部包提供;单独跑的 Rust 前端可在 HTTP 之外另开一个 gRPC 端口 | 有单独的 gRPC 入口,跨实例网关另有一条 gRPC 管线(18 章) | 原型阶段,打开后替换 HTTP,两种协议可选 |

vLLM 的取舍是把格式知识做成一张要人填的表,Python 侧宁可不切也不替用户猜;前端的出路是整个换掉 API 进程,而不是把 Rust 塞进引擎进程。好处是 Rust 前端出了问题,EngineCore 不受牵连,换回 Python 只要去掉一个环境变量;代价是两边中间仍隔着一次 ZMQ 收发,Rust 这边的功能也得一样样从 Python 搬过去。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。请求级能改的是:`tools`、`tool_choice`(带了工具不写时默认 `auto`)、`parallel_tool_calls`(默认开)、每个工具的 `strict`、`include_reasoning`(默认开)、`reasoning_effort`、`chat_template_kwargs`(覆盖服务端默认值,比如 `{"enable_thinking": false}`),以及采样字段;Anthropic 接口里的 `output_config.effort` 翻成 `reasoning_effort`。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--tool-call-parser` · 工具调用按哪种格式切 | 启动 | 无 | 按模型卡填;名字不在表里且开了下一项时启动报错 | 启动日志 `"auto" tool choice has been enabled.` |
| `--enable-auto-tool-choice` · 让模型自己决定调不调工具 | 启动 | 关 | 不开:上一项形同虚设,请求带工具就回 400;开了必须同时给上一项 | 报错 `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` |
| `--reasoning-parser` · 思考段按哪种格式切 | 启动 | 空 | 同时交给引擎,决定格式约束何时生效、思考预算收不收(14 章);用量里多一项思考 token 数 | 响应 `usage.completion_tokens_details.reasoning_tokens` |
| `--tool-parser-plugin`、`--reasoning-parser-plugin` · 从文件加载自定义解析器 | 启动 | 空 | 文件里登记的名字就能填进上面两项;Rust 前端不支持 | — |
| `--chat-template` · 替换模型自带模板 | 启动 | 无 | 文件路径或单行模板,优先级最高;内置的工具调用模板在仓库 `examples/` 下 | 启动时先校验一遍 |
| `--chat-template-content-format` · 消息内容按字符串还是分块列表交给模板 | 启动 | `auto`,解析模板语法树判断 | 判错时多段内容或多模态内容渲染不对,手动定成 `string` 或 `openai` | 用 `/tokenize` 看渲染结果 |
| `--trust-request-chat-template` · 允许请求自带模板 | 启动 | 关 | 不开时请求带模板直接 400;Rust 前端不支持 | 报错 `Refused request with untrusted chat template` |
| `--default-chat-template-kwargs` · 模板变量的服务端默认值 | 启动,JSON | 空 | `{"enable_thinking": false}` 让 Qwen3 默认不思考;请求里同名键覆盖它 | 思考 token 数 |
| `--tool-strict-level` · 工具调用结构标签的最低严格度 | 启动 | `auto` | `function`:自动模式下也约束调用外壳与函数名;`parameter`:再按 schema 约束参数,相当于每个工具都严格;新 schema 首次要编译 | 首个带新工具集的请求 TTFT |
| `VLLM_ENFORCE_STRICT_TOOL_CALLING` · 结构标签总开关 | 环境变量 | 开 | 关:一律不加结构标签;不影响「必须调用」与点名调用的 JSON 约束 | — |
| `--exclude-tools-when-tool-choice-none` · 不调工具时也不把工具写进提示词 | 启动 | 关 | 开:这类请求提示词变短、模型看不到工具;默认下工具定义照写 | prompt token 数 |
| `--generation-config` · 采样默认值从哪来 | 启动 | `auto`,读模型目录 | `vllm`:忽略模型自带默认值,只用中性值 | 启动警告 `Default vLLM sampling parameters have been overridden` |
| `--grpc` · 用 gRPC 服务替代 HTTP | 启动 | 关 | 开:要装 `vllm[grpc]`;单进程,不走本章的渲染与解析 | 日志 `vLLM gRPC server started on` |
| `VLLM_MAX_STOP_STRINGS` · 一条请求最多几个停止串 | 环境变量 | 4 | 超了请求校验失败回 400 | — |
| `VLLM_ENABLE_RESPONSES_API_STORE` · Responses 接口存响应 | 环境变量 | 0 | 设 1:`previous_response_id` 才找得到;存进程内存,不淘汰 | 启动警告 `This may cause a memory leak` |
| `VLLM_USE_RUST_FRONTEND` · API 层换成 Rust 进程 | 环境变量 | 0 | 设 1:API 进程固定 1 个;不支持的参数带上就启动失败 | 日志 `Launching Rust frontend:` |
| `VLLM_RUST_FRONTEND_PATH` · Rust 二进制在哪 | 环境变量 | `auto`,找随包安装的 `vllm-rs` | 自编的二进制填路径 | 报错 `the vllm-rs binary was not found` |
| `VLLM_RS_REQUEST_WORKER_THREADS` · Rust 前端做渲染与切词的线程数 | 环境变量 | 可用核数,上限 32 | 调高:重活并行度高;和 EngineCore 抢核 | 日志 `capping request runtime worker threads` |
| `TOKIO_WORKER_THREADS` · Rust 前端 HTTP 线程数 | 环境变量 | 可用核数,上限 32 | 一般不用动 | 日志 `capping tokio worker threads` |
| `VLLM_RS_ZMQ_WORKER_THREADS` · Rust 前端和引擎收发的线程数 | 环境变量 | 4 | 一般不用动 | — |

**怎么看。** 一是响应:`message.reasoning`、`message.content`、`message.tool_calls` 与 `finish_reason` 是否为 `tool_calls`,流式看 `delta.reasoning`;切错时最典型的样子是 `tool_calls` 为空而 `content` 里有调用标记。二是指标:配了工具解析器才注册 `vllm:tool_call_parser_invocations_total`,标签 `mode` 分流式与非流式、`outcome` 分 `tool_call` 与 `no_tool_call`、`request_type` 分聊天与 Responses;非流式每条一次,流式每块一次;解析器内部出错也记成 `no_tool_call`。三是看渲染结果:`/tokenize` 收和聊天接口一样的 `messages`、`tools`、`chat_template_kwargs`,加 `return_token_strs: true` 就能看到模板拼出来的每个 token,思考段开没开、工具定义写没写进去一目了然。Rust 前端的日志里 `using tool parser`、`using reasoning parser`、`reasoning parsing disabled` 各打一次,说明它最后选了谁。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 交错思考的 agent,MiniMax-M2,4 卡 | `vllm serve MiniMaxAI/MiniMax-M2 -tp 4 --enable-auto-tool-choice --tool-call-parser minimax_m2 --reasoning-parser minimax_m2`;客户端每轮把 `reasoning` 连同 `tool_calls` 原样带回 | 两个名字都挂在解析引擎上,流式切分只吃增量;带回的思考段让提示词变长,换来工具调用之间的推理不丢 |
| 单卡 H100 跑 Qwen3-8B 做对话,默认不思考、个别请求打开 | `vllm serve Qwen/Qwen3-8B --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking": false}'`;要思考的请求带 `chat_template_kwargs: {"enable_thinking": true}` | 默认省掉思考 token、TTFT 到首个正文字更短;打开思考的请求按需付出思考长度 |
| 2 卡 H100 跑 Qwen3-Coder-30B-A3B 做代码 agent,高并发短请求把 API 进程 CPU 打满 | `VLLM_USE_RUST_FRONTEND=1 vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct -tp 2 --enable-auto-tool-choice --tool-call-parser qwen3_xml` | 用 1 个多线程 Rust 进程换多个 Python API 进程的 CPU;失去 Responses、Anthropic 接口,不能再带 `--stream-interval` 这类未实现参数 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| `tool_calls` 为空,`content` 里是一段带标记的调用文字,`finish_reason` 是 `stop` | 启动日志有没有 `"auto" tool choice has been enabled.`;解析器名字是不是这一代模型的 | 补 `--enable-auto-tool-choice`;按模型卡换 `--tool-call-parser` |
| 带工具的请求回 400,说 `tool_choice` 需要 `--tool-call-parser`,可明明给了 | 是否只给了名字、没开 `--enable-auto-tool-choice` | 两个一起给 |
| `content` 是空的,整段回答在 `reasoning` 里 | 是否被 `max_tokens` 截断;这轮模型是否根本没写思考结束标记 | 调大输出上限;用 `/tokenize` 看模板有没有把思考段预先关上 |
| 思考文字混在 `content` 里 | 是否配了 `--reasoning-parser` | 按模型配上;客户端读 `reasoning`,不要读老名字 |
| 指标里 `no_tool_call` 的比例在模型升级后突然涨 | 新模型的调用格式是否换了 | 对照模型卡换解析器名;用 `/tokenize` 看模板 |
| 点名调用或 `required` 的第一条请求慢几秒 | 结构化输出在编译这个 schema | 正常,之后按内容缓存(14 章) |
| API 进程 CPU 100%、GPU 没满,流量以流式工具调用为主 | 解析器是不是旧写法(非引擎) | 同一模型有挂在引擎上的名字就换;多开 API 进程(01 章);再不够换 Rust 前端 |
| 开 Rust 前端后启动失败,列出 `not implemented in Rust frontend yet` | 列出的是哪些参数 | 去掉;离不开就留在 Python 前端 |
| 开 Rust 前端后启动报解析器名字不可用 | Rust 这边的表里有没有这个名字 | 换成它有的名字,或退回 Python 前端 |
| 开 Rust 前端后 `/v1/messages`、`/v1/responses` 返回 404 | Rust 前端没有这两条路由 | 这类客户端走 Python 前端 |
| Responses 带 `previous_response_id` 返回找不到 | 存储开关是否打开;请求是否落到了另一个 API 进程 | 设 `VLLM_ENABLE_RESPONSES_API_STORE=1` 且单 API 进程,或客户端自己带全量历史 |
| 请求带自定义模板回 400 | 报错是不是 `untrusted chat template` | 服务端改用 `--chat-template`;确需按请求换再开 `--trust-request-chat-template` |

## 六、常见误区

- **以为给了 `--tool-call-parser` 就开了工具解析。** 参数说明写的是「按模型选工具调用解析器」,名字一填看着就生效了。实际不开 `--enable-auto-tool-choice`,解析器根本不装,也不校验名字对不对;请求一带工具(不写 `tool_choice` 时默认 `auto`)就回 400,写成 `required` 时报错还偏偏说「需要设置 `--tool-call-parser`」,让人以为名字填错了。
- **以为 vLLM 会按模型自动挑解析器。** SGLang 和 TensorRT-LLM 都有「自动」这个取值,Rust 前端的 README 也写默认按模型自动识别,于是只起一个 `vllm serve <模型>` 就等着工具调用出来。Python 前端没有自动这一说,不给名字就不切;由 `VLLM_USE_RUST_FRONTEND` 拉起的 Rust 前端也跟 Python 对齐,默认不切。只有单独跑 `vllm-rs serve` 才按模型名里的子串猜,模型用本地路径起、名字里不带型号,照样猜不到。
- **以为响应里的思考字段叫 `reasoning_content`。** 早期版本和不少教程都这么写,客户端代码也就照着读。现在响应里只有 `reasoning`,老名字只在请求里还收;客户端读老名字不会报错,只是永远读到空,看起来像是模型不思考了。
- **以为 `include_reasoning: false` 只是不回传思考段。** 它还让 API 进程告诉引擎「思考已结束」,于是带 `response_format` 或结构化输出的请求从第一个 token 就被格式掐住,思考段被压没了,答案质量跟着掉。只想不看思考,就在客户端丢掉 `reasoning`;想让模型不思考,用 `chat_template_kwargs` 关。
- **以为 Anthropic 接口里给的思考预算会生效。** Anthropic 客户端习惯在请求里带思考开关与预算,vLLM 返回 200、响应里也有思考块,看上去一切正常。实际协议定义里没有这个字段,整段被静默丢掉;要控制思考,只能用 `output_config.effort`(翻成 `reasoning_effort`)或 `chat_template_kwargs`。返回的思考块签名是随机数,客户端若拿它做校验会失败。
- **以为 Rust 前端是无损替换。** README 自称可以直接顶替,切换又只是一个环境变量,于是在生产配置上直接加 `VLLM_USE_RUST_FRONTEND=1`。它是实验性的:47 个 Python 参数带上就启动失败,01 章排障时先调的 `--stream-interval` 就在其中;Responses 与 Anthropic 路由不存在,工具调用解析器的名字是 26 对 Python 的 51。先拿实际的启动参数和客户端试一遍,再谈吞吐。
- **以为 Responses 接口会像 OpenAI 那样替你存对话。** OpenAI 那边 `store` 默认开,客户端就只传 `previous_response_id`。vLLM 默认不存,而且对请求里的 `store` 不报错、直接当作没开,下一轮拿着编号来就是找不到;打开存储后又是每个 API 进程一份、只进不出的内存字典。多轮 agent 最稳的做法仍是客户端自己带全量历史。

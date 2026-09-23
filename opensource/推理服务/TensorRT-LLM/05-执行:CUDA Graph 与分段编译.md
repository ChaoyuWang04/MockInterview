# TensorRT-LLM 05|执行:CUDA Graph 与分段编译

decode 一步的 GPU 活很短,CPU 却要按层把 kernel 一个个发出去;prefill 的 token 数每步都变,整张图对不上形状。

说法都在源码基准 `59f5c47f`(tag `v1.2.1`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

纯 decode、小批、小模型时,坏的是**每个 token 的间隔**:一层 10–20 次下发,叠 40–80 层就和计算同一个量级,GPU 一半时间在等 CPU 派活。重叠调度藏得住「上一步的停词」,藏不住这一步的下发本身。

prefill 坏的是**图的命中**:prompt 长短、切块大小、混合批里的 context 行,都会改输入形状。CUDA Graph 要求形状和地址固定,整图要么根本抓不住,要么桶对不上只能退回逐个下发。短追加、分块后的最后一小块,计算已经薄到和 decode 一样被下发压住,却没有能重放的整图。

## 二、解法:decode 整图抓批大小,prefill 把注意力留在 eager

最天然的直觉:把一步前向录成一张图,以后只发一次重放。纯 decode 每个请求 1 个 token,形状只由批大小决定,这条路走得通。本项目默认就给 decode 配一份图配置:按批大小生成一组桶,启动时从大桶到小桶空跑再抓,运行时只允许没有 context 行的批进图。补齐默认关,批大小必须落在桶上才重放;打开补齐才向上凑最近的桶。

prefill 不能整图。形状是 token 总数,注意力的元数据(每条多长、KV 在哪)步步变。解法是把前向切开:注意力现场 eager,其余线性段按 token 桶各录一张。切法有两条。一条先走编译器,按注册过的注意力类算子把计算图切开,每段再抓 CUDA Graph。另一条不编译,录模型身体时遇到注意力就断开,和可断图同一思路。两条都要求显式打开 prefill 后端;字段默认是关,编译配置默认也是空,文档里「推荐开分段」不是启动默认。

![纯 decode 按批大小抓一张整图,注意力也在图里;批不在桶上整步退回 eager。prefill 把线性段按 token 桶录进图,注意力用虚线留在 eager;分段后端默认关](/opensource/TensorRT-LLM/05a-decode-full-prefill-piecewise.svg)

为什么实际不吃亏:decode 的桶从 1 铺到默认上限 128,小批几乎总能对上;大桶先抓,小桶复用同一块图显存池。prefill 只让每层注意力那一次下发留在图外,其余十来次省掉。注意力不进图,序列长度才能变。

## 三、代价

**启动要真跑前向。** decode 每个桶先热身再抓;分段还要按 token 列表再抓一轮。桶越多,就绪越慢。

**显存先被扣走。** 配置注释按每张图最多约 200 MB 量;补齐打开后桶更密,账更大。分段的 token 列表越长,中间激活也越多,能并发的请求被挤下去。

**超桶或形状不对就断崖。** decode 批大于最大档、或补齐关着却没落在桶上,整步 eager,吞吐不是平滑下降。prefill 没开后端时,短 context 永远吃满下发。

**录进去的东西改不了。** 桶列表、补齐、prefill 后端都是启动配置。图外现场出错还有 Python 栈;崩在重放里没有。

**分段绑编译器时,追踪失败整条路废。** 可断那条不编译,但和 LoRA、编译配置、多模态互斥,覆盖面窄,官方标成实验。

## 四、decode 整图怎么抓、prefill 形状一变为什么抓不住

这一页比的是 decode 整图怎么抓、prefill 为什么形状一变就抓不住、分段之后注意力留不留在 eager。对等列是 vLLM 和 SGLang。

| 维度 | TensorRT-LLM | vLLM | SGLang |
|---|---|---|---|
| decode 默认可不可用图 | 默认就有一份 decode 图配置,纯 decode 才进整图 | 启动录 decode 图,按 token 数建表 | CUDA 上 decode 默认整图,按批大小分桶 |
| prefill 默认可不可用图 | 字段默认关,要显式选分段或可断 | 分段图借道编译器,注意力留 eager | CUDA 上默认可断图,不追踪不编译 |
| 图按什么分桶 | decode 按批大小;分段按 token 数 | 两边都按 token 数 | decode 按请求数,prefill 按 token 数 |
| 形状不在桶上 | 补齐默认关,对不上就整步 eager;分段侧补齐是强制的 | 建表时把整数指向上一桶,查表即补齐 | 运行时找不小于实际值的桶,prefill 补太远就不用图 |
| 注意力进不进图 | decode 整图进;分段和可断都把注意力留在 eager | 分段图把注意力留在 eager | 可断图按层上的标记断开注意力 |
| 编译器是不是必经 | 分段那条要编译配置;可断那条禁止编译配置 | 分段图走编译器 | CUDA 默认可断,编译分段是备选 |

重心在本项目:decode 整图默认就在,但补齐默认关;prefill 要另外开门,门开了注意力仍不进图。不要把文档推荐的分段写成字段默认。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `cuda_graph_config` · decode 整图 | 启动(LLM 或 extra yaml) | 有对象:`max_batch_size` 落到 128,`enable_padding` 关,按 1–31、32、64、128 生成桶 | 设 `null`:decode 也不抓图,小批 TPOT 变长;打开 `enable_padding`:批向上凑桶,垫片行换命中率 | 日志 `Running CUDA graph capture for N batch sizes`;批对不上时这一步回到 eager |
| `cuda_graph_config.enable_padding` · decode 批是否凑桶 | 启动 | 关 | 开:落在两桶之间也能重放,多算垫片行;关:必须精确命中 | 同上;高并发批大小乱跳时看是不是整步 eager |
| `cuda_graph_config.max_batch_size` / `batch_sizes` · 最大档与桶列表 | 启动 | 未写则最大档 128,自动生成桶 | 调高:更大 decode 批仍有图,显存涨;只给列表则以列表最大值为档 | `capture for` 后面的桶个数 |
| `prefill_cuda_graph_backend` · prefill 用哪条图 | 启动,原型 | `disabled` | `piecewise`:走编译器分段,注意力 eager;`breakable`:不编译可断,与 LoRA、编译配置、多模态互斥 | 日志 `Run prefill CUDA graph capture for num tokens=`;默认看不到这行 |
| `prefill_capture_num_tokens` · 分段 token 桶 | 启动,原型 | 后端非关且未写时:2 的幂到 128,再 256 步进到 3072 | 砍桶:启动快、显存少,短 prefill 更常 padding 或 miss | 同上 `num tokens=` |
| `torch_compile_config` · 编译器 | 启动,原型 | 空,不编译 | 选 `piecewise` 时若为空会自动建一份;可断后端遇到非空直接校验失败;`enable_piecewise_cuda_graph` 已弃用,写了会警告并改写成 `piecewise` | 启动警告 deprecated;追踪失败起不来 |
| `encoder_cuda_graph_config` · 编码器图 | 启动,原型 | 空,不抓 | 编解码器才有意义,本页主干是 decoder | 不要和 decode 那份配置搞混 |

**怎么看。** 一是启动日志:decode 热身或抓图打 `Running CUDA graph warmup/capture for N batch sizes`;prefill 打开后才有 `Running prefill CUDA graph warmup...` 和按 token 的 capture 行。二是就绪后:纯 decode 且批在桶上,下发次数应接近「一步一次图」而不是按层爆炸;混进 context 行则走 prefill 路径,没开门就是整步 eager。三是显存:抓完后可用显存掉一截,注释按每张图最多约 200 MB 估,不是实测。

**典型配置。** 三套能直接抄走的起法,参数名和默认值都来自上表。

| 什么场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| 单卡在线 decode,批大小经常对不齐桶 | `trtllm-serve Qwen/Qwen2.5-7B-Instruct --config decode-graph-pad.yaml`,yaml 写 `cuda_graph_config: {enable_padding: true, max_batch_size: 256}` | 用垫片行和更多图显存换 decode 命中;prefill 仍默认关 |
| 短 prompt、切块 prefill,要压 context 下发 | `trtllm-serve Qwen/Qwen2.5-7B-Instruct --config piecewise-prefill.yaml`,yaml 写 `prefill_cuda_graph_backend: piecewise` 和 `torch_compile_config: {enable_userbuffers: false}` | 用启动编译和 token 桶显存换 prefill 下发;注意力仍 eager |
| 排障,要对齐逐个下发 | `trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --config cuda-graph-off.yaml`,yaml 写 `cuda_graph_config: null` 且不要写 prefill 后端 | 用整步 eager 换 Python 栈;小批间隔会变差 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 小批 decode 仍按层下发 | `cuda_graph_config` 是否被设成 `null`;这一步有没有 context 行 | 纯 decode 才能整图;混批走 prefill 路径 |
| 批大小 40 永远 miss | 补齐是否关着;默认桶在 32 和 64 | 打开补齐,或把 40 写进桶列表 |
| 启动很慢、显存掉一截 | capture 行的桶个数和 token 列表 | 砍 `batch_sizes` 或 `prefill_capture_num_tokens` |
| 以为开了分段,日志没有 prefill capture | `prefill_cuda_graph_backend` 是不是还是 `disabled` | 显式写成 `piecewise` 或 `breakable`,不要只抄文档「推荐」 |
| 可断后端起不来,提到 LoRA 或 compile | 那两条互斥 | 关掉 LoRA 和 `torch_compile_config`,或改用 `piecewise` |
| 重放结果错、eager 时对 | 先关图对齐 | 图里没有 Python 栈;对齐后再逐项开回 |

## 六、常见误区

- **以为有了 `cuda_graph_config` 对象就等于 prefill 也在抓图。** 那份默认只服务纯 decode。prefill 另有后端字段,默认关。
- **以为文档推荐分段等于字段默认。** 文档示例会写 `piecewise` 和 token 列表;基准上后端默认 `disabled`,编译配置默认空。
- **以为补齐默认开。** 字段写通常是净赢,默认却是关。关着时批必须落在生成出来的桶上。
- **以为 default_factory 有对象不等于一定在重放。** 配置在,但这一步有 context 行、批不在桶上、或统计开关打开,都会退回 eager。
- **以为分段之后注意力也进了图。** 分段和可断都把注意力留在 eager,图只吃线性段。
- **以为 `enable_piecewise_cuda_graph` 还是正门。** 已弃用,写了会警告并改写成 prefill 后端 `piecewise`。
- **以为旧引擎编译链还能把整网编进 TensorRT。** 已经删掉。今天的「编译」是 PyTorch 侧为分段图服务的,不是那条引擎。

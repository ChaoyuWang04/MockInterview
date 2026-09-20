# SGLang 07|执行与 CUDA graph

CUDA graph 要解决的是下发不是计算、录下来的东西不许变的五条硬约束、按桶补齐、显存不贵启动时间贵,vLLM 解读的「Worker 与 ModelRunner」一章已经讲透。这一页只说 SGLang 遇到的问题、它的解法、代价、以及和 vLLM 行为上不一样的地方。02 章讲过采样 token 直接写进显存格子,03 章讲过补齐用的垫片行和垫片页,本章只回指不重讲。每一条对应源码的哪个文件与符号,见代码索引页。

## 一、核心问题

decode 一步的 GPU 计算只有 5–30 毫秒,但一层要下发 10–20 个 kernel,80 层就是 1000–2000 次下发,每次 5–10 微秒,加起来和计算同一个量级。02 章的重叠调度把「处理上一步的结果、挑下一批」那段 CPU 时间藏起来了,藏不住的是下发本身:kernel 是 CPU 逐个发出去的,GPU 发一个算一个。小模型、小批、大张量并行(每张卡上的计算被切薄了)这三种负载下,TPOT 和吞吐被下发压住,GPU 利用率上不去。

prefill 的麻烦不一样:每步的 token 数每步都变,而图要求形状固定。长 prompt 每个 kernel 都算得够久,下发不是瓶颈;短追加(多轮对话只补 20–100 个 token)、推测解码的验证批、分块之后的最后一小块,这些 prefill 小到和 decode 一样被下发压住,却没有图可用。

## 二、解法:两个阶段各录各的,prefill 那台不碰编译器

最天然的直觉是把一步前向录成一张图,以后每步只发一次「重放」。decode 每个请求恰好 1 个 token,形状只由批大小决定,所以按批大小分桶,每个桶把整个前向连注意力一起录成 1 张图。桶从 1、2、4、8、12 起,到 256 每 8 个一档,到 512 每 16 个一档,再往上每 32 个一档;最大档默认按显存定,80 GB 的卡上是 256,张量并行 4 路以上抬到 512,24 GB 的卡只有 48。运行时把实际批大小向上补到最近的桶,补的行就是 03 章的垫片行。批大小超过最大档,整批退回逐个下发。

prefill 的 token 数固定不住。SGLang 在 CUDA 上默认用「可断图」:仍然按 token 数分桶(4 到 32 每 4 个一档,到 256 每 16 个,到 512 每 32 个,到 1024 每 64 个,到 4096 每 256 个,再往上每 512 个,最大档默认等于分块 prefill 的块大小),但录的时候遇到注意力就把图断开,注意力现场发,发完再接着录下一段。注意力元数据(每个请求多长、KV 在哪)在图外算,所以形状可以变。断点不止注意力一处:线性注意力、MoE 的 all-to-all、Mamba 层、稀疏索引器也都是断点。

关键在断点是怎么来的。可断图不追踪、不编译,断点是写在层里的一个标记——被标记的函数在录图时自动跳出图外现场执行,平时什么也不做。另一条路「编译分段图」走的是 vLLM 的路子:先用 torch.compile 把前向完整追踪成一张计算图,按注册过的注意力算子切开,每段各录一张。同一层前向,两条路得到的分段形状差不多,来路却差着一整条编译链。

![一层前向被录成图的两条来路:上面一条是默认的可断图,源码里的注意力、线性注意力、MoE all-to-all、Mamba 各挂一个标记,录图时直接按标记断开,标记之外的算子连成一段一段录进去;下面一条是编译分段图,同一份源码要先过 dynamo 追踪出计算图,再按注册的注意力类算子切段,每段再交给编译器出一份 kernel,最后才逐段录图,链路长出三步,任何一步失败整条路就废了;两条路录出来的分段形状相同,右侧标出每个断点在重放时多一次下发](/opensource/sglang/07a-two-roads-to-segments.svg)

为什么实际不吃亏:补齐的浪费有上限,decode 在 256 以下最多补 7 行;prefill 补齐后超过实际 token 数 2 倍就干脆不用图,所以只有 1 个 token 的追加会走回逐个下发。所有桶共用一个显存池,先录最大的桶,小桶复用大桶的池,图的中间激活只按最大那张算。断点只让每层 1 次注意力下发留在图外,其余 10 多次都省了。

## 三、代价

- **启动变慢。** 每个桶都要真跑前向:空跑 2 遍热身,再录 1 遍。decode 最大档 256 时 36 个桶,prefill 块大小 8192 时 58 个桶,两组各在启动日志里打一行耗时。
- **显存先被扣走。** 显存比例是自动算的时候,启发式先扣一笔给图:decode 按最大批大小每个单位扣 2 MB(256 就是 512 MB),prefill 每个桶 8 MB、MLA 模型一律 1.5 GB,DP attention 再按 dp 数每单位加 3 MB。手动指定过比例之后这笔账不再自动扣,图吃的是余量,调大最大档就要同时把比例降下来(03 章)。
- **超桶断崖。** decode 批大小超过最大档,整批退回逐个下发,吞吐不是平滑下降而是掉一截。高并发场景要把最大档抬到并发上限。
- **录进去的东西改不了。** 桶列表、后端、批大小上限在启动时定死,运行时没有任何接口能改;换权重后想重录,要在权重更新请求里显式要求重录。
- **崩在图里没有现场。** 重放时出错看不到 Python 调用栈。有个调试开关能让每个算子仍然走录图与重放的那条路径,但逐个执行,用来复现。
- **录失败直接起不来。** 捕获抛错时进程退出,错误信息里给 3 条建议:降显存比例、把 decode 最大档降到 16、关掉 decode 图(不推荐)。

## 四、和 vLLM 不一样的六处

![CUDA graph 这件事上两边都必须回答的 6 个问题各自由什么机制回答:6 行从上到下是用不用图谁说了算、变长的 prefill 怎么录、图按什么分桶、形状不在桶上怎么办、图吃多少显存这个数从哪来、权重换了图怎么办;vLLM 一侧用一个方框同时盖住分桶与不在桶上两行,因为建表时把区间铺平了,查表就是分桶机制本身;SGLang 一侧用一个方框同时盖住谁说了算与 prefill 怎么录两行,因为选了哪个后端就等于选了怎么录;最后一行 vLLM 侧是灰的,参照项目那一章没有这条路径。图上不写参数名,两边的参数名都在第五节](/opensource/sglang/07b-vllm-diff-six-questions.svg)

1. **按阶段各配一个后端。** vLLM 是一个档位枚举,复合档在配置期声明 decode 与混合批各用哪一档。SGLang 把 decode 和 prefill 当两个独立阶段,各自有后端(整图、可断图、编译分段图、关掉四选一)、最大档和桶列表。整图给 prefill 用不是不行,但是实验性的:自动推导那条路永远不会选到它,只有显式指定才到得了,到了还会打一行警告。
2. **prefill 默认不走编译器。** vLLM 的分段图借道 torch.compile。SGLang 在 CUDA 上默认可断图,不追踪不编译,靠层里的标记在注意力处断开;编译分段图是备选,也是 AMD 和昇腾上的默认。官方文档自己就说,这条路的大部分问题出在追踪失败。
3. **decode 桶按请求数,prefill 桶按 token 数。** vLLM 两边都按 token 数。SGLang decode 一个请求 1 个 token(推测解码时是草稿 token 数),桶就是批大小;prefill 桶是 token 总数,每个桶的请求数上限由请求表的行数定。
4. **补齐靠查桶,不铺表。** vLLM 建表时把桶之间的整数都登记成指向上一桶,查表一次命中。SGLang 运行时二分找最小的不小于实际值的桶,prefill 还多一道「补齐超过 2 倍就不用图」的判断。
5. **显存是启发式预扣的,不是探测出来的。** vLLM 录最大的两张图实测,再按单价外推。SGLang 在算显存比例时按最大档乘系数预扣,录完不回填;有个环境变量能让它录完再量一次、把余量还给 KV 池,默认关。
6. **图能在运行时重录。** RL 训推一体换权重后,权重更新请求里要求重录,就地重录 decode 图;PD 分离的节点在运行时切换角色时也会补录。参照项目那一章没有这条路径。

## 五、调参与观测

**参数在哪调。** 全部是启动参数,改了要重启。四层写法,后面的压过前面的:`--disable-cuda-graph` 这类老旗标(已弃用,等于两个阶段都关)、`--disable-prefill-cuda-graph` 与 `--disable-decode-cuda-graph` 两个单阶段关闭开关、`--cuda-graph-backend-decode` 这些按阶段的旗标、`--cuda-graph-config` 一段 JSON 按阶段写全。**显式指定过 prefill 后端就跳过全部自动关闭规则**,指什么都算——包括那两个老旗标,因为它们也算显式指定。少数内部开关是环境变量。运行时没有能改这一页任何参数的接口,`/set_internal_state` 只放行流水线微批大小与推测解码阈值;唯一的运行时动作是权重更新请求里的 `recapture_cuda_graph`。

| 参数 · 一句话中文说明 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--cuda-graph-max-bs-decode` · decode 图录到多大的批 | 启动 | 按显存分档,括号里是张量并行 4 路及以上:20 GB 以下 8;20–35 GB 48(160);35–60 GB 32(160);60–160 GB 256(512);更大 512;探测不到显存时 160 | 调高:大批 decode 不再退回逐个下发,高并发下吞吐不掉;显存预扣按每单位 2 MB 涨,比例手动指定时要同步降 | 就绪前 `Capture target decode CUDA graph begin` 行的 `bs=[…]`;`Decode batch` 行 `cuda graph: False` 而 `#running-req` 大于最大档 |
| `--cuda-graph-bs-decode` · 直接给桶列表 | 启动 | 自动生成 | 砍桶:启动快、显存少,落不进桶的批补更多垫片行;列表里的最大值同时成为最大档 | 同上 `bs=[…]` |
| `--cuda-graph-backend-decode` · decode 图用哪种后端 | 启动 | `full` | `breakable`:注意力断开,用于排障或某个算子录不进;`disabled`:每步多 1000–2000 次下发,TPOT 明显变长;`tc_piecewise` 没实现,退回 `full` 并警告一次 | `Capture … begin` 行的 `backend=` |
| `--cuda-graph-backend-prefill` · prefill 图用哪种后端 | 启动 | CUDA 上 `breakable`,其他平台 `tc_piecewise`;一串规则会自动改成 `disabled` | 显式指定就跳过自动关闭;`tc_piecewise` 用 torch.compile,启动慢、失败多;`full` 是实验性的,只能这样指定才到得了;`disabled` 短 prefill 的 TTFT 变长 | `Capture target prefill CUDA graph begin` 行的 `backend=`;`Disable prefill CUDA graph because …` 与 `Breakable CUDA graph is incompatible with …` 两种行 |
| `--cuda-graph-max-bs-prefill` · prefill 图录到多少 token | 启动 | 分块大小;MLA 模型 2048;Llama-2 封 4096;流水线并行下可断图封 8192;不超过 `--max-total-tokens` | 调低:桶少启动快,超过的块走逐个下发;调高到块大小以上没意义,一块不会超过它;DP attention 下会按每卡的块大小再夹一次 | `begin` 行的 `num_tokens=[…]`;`Prefill batch` 行的 `cuda graph:` |
| `--cuda-graph-bs-prefill` · 直接给 token 桶列表 | 启动 | 自动生成 | 同 decode 的桶列表;DeepEP 下自动对齐到 8 的倍数 | 同上 |
| `--cuda-graph-prefill-max-context` · prefill 图的上下文长度上限 | 启动 | 不设 | 只对 DeepSeek-V4 这类要按上下文开元数据的模型有意义:按页对齐,超过这个长度的批走逐个下发 | 启动日志 `Prefill CUDA graph max context size:` |
| `--cuda-graph-tc-compiler` · 编译分段图用什么编译 | 启动 | `eager` | `inductor`:每段再过一遍 inductor,首次启动更慢,缓存在 `~/.cache/sglang/torch_compile_cache`;只有 `tc_piecewise` 读它 | 启动耗时 |
| `--disable-cuda-graph-padding` · 不补齐,只在批大小正好命中时用图 | 启动 | 关 | 开:decode 桶变成 1 到最大档每个都录,启动更慢、显存更多,换来没有垫片行;和 `--enable-torch-compile` 互斥,同时给会启动报错 | `bs=[…]` 变成连续整数 |
| `--debug-cuda-graph` · 走录图与重放的路径但每个算子逐个执行 | 启动 | 关 | 开:性能等于没有图,用来定位录进图之后才出的错;只支持 CUDA 与 ROCm,别的设备会自动关掉 | 启动警告 `Debug mode for CUDA graph is enabled`;错误有了 Python 调用栈 |
| `--enable-torch-compile` · 整模型 torch.compile | 启动 | 关 | 开:decode 批大小不超过 `--torch-compile-max-bs`(32)的桶先编译再录,小模型小批快一点;首次启动要等自动调优,缓存在 `TORCHINDUCTOR_CACHE_DIR`;开了它 prefill 的 `tc_piecewise` 自动关;文档标注已不维护 | 启动耗时;`SGLANG_TORCH_COMPILE_MODE` 改编译模式,默认 `max-autotune-no-cudagraphs` |
| `--enable-profile-cuda-graph` · 录 decode 图时开 profiler | 启动 | 关 | 开:只作用于 decode 那组图,看哪个桶最贵;`SGLANG_GRAPH_BATCH_CAPTURE` 再按桶各出一份 | profiler 输出目录 |
| `--enable-cudagraph-gc` · 录图期间允许垃圾回收 | 启动 | 关(录图期间冻结 GC) | 开:录图变慢,只在录图期间内存涨到不可接受时用 | 启动耗时 |
| `--mem-fraction-static` · 权重加 KV 占多少 | 启动 | 自动算时已扣图的预算 | 手动指定后图吃余量;调大最大档后要降它(03 章) | `Capture … end` 行的 `mem usage=` 与 `avail mem=` |
| `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` · 录完再量显存定 KV 池 | 环境变量 | 关 | 开:预扣的图预算不再留死,KV 池按录完的余量定;MLA、统一内存、非 CUDA 设备、decode 上下文并行、fp4 KV、显存让出模式下一律不生效 | 就绪前 `max_total_num_tokens` 是否变大 |
| `SGLANG_USE_BREAKABLE_CUDA_GRAPH` · 文档说它让断点标记生效 | 环境变量 | 关 | 这个基准上没有任何代码读它,设了等于没设;断点只看当前是不是在可断图后端里录 | 无 |
| `SGLANG_VIT_ENABLE_CUDA_GRAPH` · 多模态视觉编码器录图 | 环境变量 | 关 | 开:视觉编码器按序列长度分别录图,每种长度第一次见到时现录,长度种类多则显存涨;文档只列了 Qwen2.5-VL 与 Qwen3-VL 两种,代码里另有 InternVL 与 Kimi-K3-VL 两处读点 | 首张图片慢、同尺寸第二张快 |
| `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH` · 关掉草稿模型扩展步的图 | 环境变量 | 关 | 开:推测解码的草稿扩展步走逐个下发,省下这张图的显存池;注释点名 DeepEP MoE 那种录图池比图省得还多的场景 | 启动日志少一组 `Capture draft extend …` |
| `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE` · 录图全过程出 trace | 环境变量 | 关 | 开:排查录图卡住或慢;它压过按桶那一份 | trace 文件 |

**典型配置。** 下面三套是起点,各项组合按参数语义推出,未经实测标定。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| H100/H200 上跑 Llama、Qwen 这类稠密模型,高并发短请求 | 图相关的一个都不给,只把并发上限和最大档对齐:`--max-running-requests 256`(最大档默认也是 256) | 拿启动时间和 512 MB 显存换掉 decode 的全部下发开销;两个数对齐,就没有超桶断崖 |
| 并发要上到 512 的 80 GB 单卡部署 | `--cuda-graph-max-bs-decode 512 --mem-fraction-static 0.82` | 拿显存换高并发下的吞吐:最大档翻倍,预扣从 512 MB 涨到 1 GB,而手动写了比例这笔账就没人替你扣,所以比例要自己降 |
| 重放结果和逐个下发对不上,要定位 | `--debug-cuda-graph`,必要时再 `--cuda-graph-backend-prefill disabled` | 拿全部性能换一个 Python 调用栈;注意第二个旗标同时关掉了 prefill 侧的全部自动关闭规则判断,排完障要去掉 |

**怎么看。** 启动日志里每组图两行:`Capture target decode CUDA graph begin. backend=full, num_tokens_per_req=1, bs=[…], avail mem=… GB` 和 `… end. elapsed=… s, mem usage=… GB, avail mem=… GB`;prefill 那组把 `bs` 换成 `num_tokens`;推测解码时 decode 那组叫 `target verify`,草稿模型另有 `draft decode` 与 `draft extend`。`end` 行的 `mem usage` 就是图真正吃掉的显存,拿它和预扣的预算比。prefill 图被自动关掉时,配置期那批规则打 `Breakable CUDA graph is incompatible with …`,起模型时那批打 `Disable prefill CUDA graph because …`,原因都写在句尾。稳态看 `Decode batch` 与 `Prefill batch` 行里 `cuda graph: True/False` 这一位(在吞吐那几项前面),它是这一步实际有没有重放。开了 `/metrics` 之后同一位是 `sglang:is_cuda_graph`,累计数是 `sglang:cuda_graph_passes_total`,按 `mode` 分成 decode/prefill 各走图与不走图四类;录图耗时是 `sglang:startup_cuda_graph_time_seconds`。

| 症状 | 先查 | 然后 |
|---|---|---|
| 启动时 `Capture cuda graph failed` | 错误信息里那 3 条建议 | 先把 `--mem-fraction-static` 降到 0.8;不行降 `--cuda-graph-max-bs-decode` 到 16 |
| 启动时 prefill 那组录图抛错 | 错误信息里的后端名 | 换另一种 prefill 后端;仍失败就 `disabled` 并报 issue |
| 并发一上去 `Decode batch` 行变 `cuda graph: False` | `#running-req` 是否大于最大档 | 抬 `--cuda-graph-max-bs-decode` 到并发上限,同时降比例 |
| 短 prompt 的 TTFT 比预期高 | 两种 prefill 图关闭行 | 看原因:LoRA 后端、多模态、KDA 混合模型、上下文并行、两批重叠;能换的换,不能换的接受 |
| `Prefill batch` 行总是 `cuda graph: False` 而启动时录了 | 每步 token 数是否超过最大档或补齐超 2 倍;是否带 logprob 或输入 embedding | 前者调最大档或桶列表;后者是设计限制 |
| 启动比以前慢了 1 分钟以上 | 两行 `elapsed=` | 砍桶列表;`--enable-torch-compile` 是否误开 |
| 就绪前可用显存低于 5 GB | 两行 `mem usage=` | 图吃多了:降最大档或降比例 |
| 重放结果错、逐个下发时对 | `--debug-cuda-graph` 复现 | 有 Python 栈了再定位;报 issue |
| RL 换权重后 decode 变慢或输出异常 | 权重更新请求有没有带 `recapture_cuda_graph` | 带上;同时 flush_cache(03 章) |
| DP attention 下某张卡一直 `cuda graph: False` | 各卡批大小是否被补到全局最大 | 这是设计:补到各卡中最大的那个批;看是否有卡超过最大档 |
| 开了 HiCache 之后 prefill 图没了 | 自动关闭规则里的 CPU offload 与分层缓存那条 | 这条只对 `tc_piecewise`,CUDA 默认的 `breakable` 不受影响;显式指定后端可绕过 |

## 六、常见误区

- **以为 prefill 默认走 torch.compile 的分段图。** 文档页标题就叫 Piecewise CUDA Graph,正文写着它默认开、用 `--cuda-graph-backend-prefill=disabled` 关。这个基准上 CUDA 的默认是 `breakable`,不追踪不编译;`tc_piecewise` 是 AMD 和昇腾的默认。看 `begin` 行的 `backend=` 而不是看文档。
- **看到重叠调度就以为下发也被藏起来了。** 02 章说 CPU 和 GPU 并行,大家以为 CPU 的活全被藏了。重叠藏的是「处理上一步、挑下一批」,下发 kernel 是发射前向那一刻的活,藏不住,只有图能省。
- **把 `--cuda-graph-max-bs-decode` 当成并发上限。** 名字像。它只定图录到多大,超过它的批照跑,只是逐个下发;并发上限是 03 章的 `--max-running-requests`。两个数不对齐时高并发有一个吞吐断崖。
- **调大最大档之后 OOM,以为是 KV 池的事。** 比例自动算时图的预算是按最大档预扣的;手动指定过比例这笔账就没人替你扣。调档和调比例要一起动。
- **以为显式指定 prefill 后端只是「选一种」。** 指定的同时跳过全部配置期自动关闭规则:LoRA、多模态、KDA 混合模型这些本来会关的场景都不再关,录失败直接起不来。想「只是选一种」就别指定,让它自己定。
- **以为整图那一档 prefill 用不了。** 命令行帮助里还写着「整图只给 decode」,那是过期的说法:允许表里 prefill 四种后端都在,只是自动推导永远不会选到整图,必须显式指定,而且会打一行实验性警告。
- **以为 `--disable-cuda-graph` 还在。** 它已弃用但还能用,效果是两个阶段都关,而且这是它唯一的命令行入口。想只关一边,有 `--disable-prefill-cuda-graph` 和 `--disable-decode-cuda-graph` 两个专门的开关。
- **以为推测解码把 decode 图关掉了。** 文档的不兼容列表里写着推测解码,那说的是 `tc_piecewise` 那条路的 prefill 图。decode 图照录,只是每个请求的 token 数变成草稿数、桶列表更密;草稿模型另有自己的两组图。
- **以为 LoRA 和图不能共存。** decode 图一直支持 LoRA;prefill 图在 `breakable` 和 `full` 下也支持,只有 `tc_piecewise` 关掉它,以及 DP attention 下 LoRA 的 prefill 全走逐个下发。
- **看到 `Decode batch` 行 `cuda graph: False` 就去查后端。** 先看 `#running-req` 是不是超过了最大档,这是最常见的原因;其次是这一步有没有输入 embedding 覆盖、DP attention 下有没有某张卡超档。后端配置错在启动时就报了。
- **以为把桶列表砍到只剩最大档能省显存。** 显存池按最大那张图算,砍小桶省的只是每张图的元数据;砍掉之后小批要补到最大档,每步算一堆垫片行。桶列表是用启动时间换补齐浪费,不是换显存。
- **以为录一个桶就跑一遍前向。** 每个桶先空跑 2 遍把 kernel 加载和一次性初始化做完,第 3 遍才是真录。桶列表长一倍,启动时间也长一倍。
- **以为 `--enable-torch-compile` 是 `tc_piecewise` 的开关。** 两者无关:前者是整模型编译、只作用于 decode 小批、已不维护;后者是 prefill 的一种图后端。而且前者开了会把后者自动关掉。
- **按文档 export 了 `SGLANG_USE_BREAKABLE_CUDA_GRAPH=1`,以为断点靠它开。** 可断图的文档页写着这个变量是断点生效的前提。这个基准上全树只有变量定义和 `--debug-cuda-graph` 时置 1 两处,没有任何读取点;断点生效只看后端是不是 `breakable`。
- **换权重后没重录图,以为图和权重无关。** 图录的是 kernel 序列和地址,权重就地覆盖时地址没变,重放不会报错;但换了结构不同的适配器或做过显存释放再恢复,地址就可能变。权重更新请求里带 `recapture_cuda_graph`。

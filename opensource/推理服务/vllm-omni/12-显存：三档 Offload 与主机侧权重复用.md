# 显存：三档 Offload 与主机侧权重复用

这一页解决 DiT 权重无法全部常驻显存、同机 worker 又重复准备主机权重的问题。

说法都在源码基准 `4d877780d3`(tag `v0.29.0rc1`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

视频 DiT 的权重与 activation 一起挤显存，增加分辨率或帧数后，即使权重能装下，也可能没有余量执行。把权重移到 CPU 可以让作业启动，但每个 denoising step 都要重新搬运；再增加同机 worker，主机侧重复加载和布局转换又会抬高启动耗时与内存峰值。

这里有两个独立问题：**运行时留多少权重在 GPU，启动时能否复用别人已经准备好的主机权重**。前者由 Offload 管，后者由 Host Weight Runtime 管，不能用其中一个开关替代另一个。

## 二、解法

先选能满足容量的最粗粒度。模型级 Offload 在 encoder 与 DiT 等组件之间切换驻留，适合整个 DiT 尚能放下的情况。层级 Offload 把 CPU 权重保留下来，计算当前 block 时预取下一个，算完释放设备副本，不必每次再把不变的权重搬回 CPU。

分布式层级 Offload，简称 DLO，进一步复用固定的设备缓冲区。它可以在各 rank 上分摊主机权重，再通过 AllGather 还原当前 block；也可以让每个 rank 独立搬运自己的完整本地布局，避免为权重传输等待其他请求。这里的本地布局仍可能已被 TP 切分。

![DLO 的设备槽轮换决定权重驻留，而主机权重运行时让多个 worker 映射同一份已完成布局的权重。](/opensource/vllm-omni/12a-weight-residency.svg)

Host Weight Runtime（HWR，主机侧权重运行时）复用的已经是完成 loader 转换后的张量，而非仅下载好的 checkpoint。它把源文件身份、模型契约、dtype、最终布局与影响布局的并行坐标一起作为匹配条件；命中后恢复模型张量，随后仍交给 DLO 搬往 GPU。当前接入是符合契约的 BF16 DiT 路径，不能据基础设施的通用接口认定所有模型都可复用。

## 三、代价

粒度越细，每轮 denoising 的传输越频繁。预取只有在计算时间足够覆盖传输时才有收益；低计算量请求仍可能被 PCIe 带宽限制。DLO 的固定双槽约束的是流式 block 权重，常驻层、非 block 模块、activation、VAE 与通信暂存还要另外算，不能把它当成整个进程的显存上限。

AllGather 减少各 rank 私有的主机权重，但引入 collective 顺序约束。DP 请求或 cache 命中让不同 rank 执行不同 block 时，不能随意跳过权重 collective。独立搬运更适合执行进度不一致的服务负载，却不再靠额外的 DP 分片节省主机内存。

HWR 用磁盘或内存文件系统上的 artifact、身份校验和生命周期管理换取热启动复用。共享映射仍消耗主机内存和存储空间，GPU 上各 worker 的执行副本也不会合并。首次未命中仍要正常加载；已开始恢复但提交失败的模型必须丢弃，不能在部分改写的对象上继续补加载。

## 四、与 sglang-omni 的 Offload 接入边界对照

以 SGLang-Omni 的 engine-stage 配置入口作对照，它把 CPU Offload 容量交给 SGLang 引擎参数；vLLM-Omni 在自己的 Diffusion 栈里直接管理组件切换、block 预取、权重 collective 与主机表示恢复。这里比较的是框架承担的职责，不能把 SGLang 引擎的容量设置视为本页 DLO 的同义开关，也不能据此判断哪边更快。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

| 参数 | 在哪调 | 默认 | 调了之后 | 怎么看 |
|---|---|---|---|---|
| `--diffusion-offload-config` | CLI JSON | `None` | 选择 `mode: module/layer` 与 `components: [dit, text_encoder]`，降低所选权重驻留 | `Enabling offloader backend` 确认实际 backend |
| `layer_options.<组件>.weight_transfer` | 上述 JSON | `rank-local` | `allgather` 选择 DLO，以通信换主机权重分片 | 主机占用、请求延迟及各 rank 的参与情况 |
| `layer_options.dit.resident_layers` | 上述 JSON | `0` | 增加前部常驻层，以显存换更少传输；要求 rank-local 与模型驻留计划 | 常驻层日志、GPU 峰值和 denoising 延迟 |
| `pin_memory` | 上述 JSON | 沿用 `pin_cpu_memory=True` | pinned host memory 有利于异步 H2D，增加锁页资源需求 | 传输等待、主机内存 |
| `--enable-cpu-offload`、`--enable-layerwise-offload`、`--enable-distributed-layerwise-offload` | CLI 兼容开关 | 均关闭 | 旧入口冲突时按 DLO、层级、模型级优先；不用于叠加策略 | 实际 backend 日志 |
| `--dlo-no-use-allgather` | CLI | 旧 DLO 默认开启 AllGather | 关闭权重 collective，保留 loader 的 rank-local 布局；HWR 接入要求此路径 | 传输模式日志和 rank 间等待 |
| `--host-weight-runtime-mode` | CLI | `disabled` | `preferred` 命中则恢复，未命中正常加载并尝试发布；`required` 在适用路径上要求命中 | HWR plan、回退及发布失败日志 |
| `--host-weight-runtime-root` | CLI | `None` | 指定同机共享的可写目录；预热与服务保持同一路径 | artifact 是否存在、身份是否一致 |
| `--dlo-host-registration-limit-gib` | CLI | `0.0`，不加额外上限 | 限制每 worker 注册 HWR 映射的 GiB；不满足条件则回到有界 staging | 主机注册日志、H2D 等待；不代表禁止 HWR |

**HWR 的适用范围先于命中策略。** 只有不使用 AllGather 的 DiT DLO 才会进入该消费者；`required` 不会把其他 Offload 路径自动转换过来。当前要求默认 loader、模型 restore contract、专属 DiT 权重源，拒绝 HSDP、量化和已配置 LoRA；BF16 表示允许显式保留 FP32 参数，不支持任意 dtype，另有 PP、CFG parallel、EP 的所有权限制。当前显式声明契约的实现包括 Flux2 Klein 与 MiniMax H3。

compact 配置 `--diffusion-offload-config` 与开启的 HWR **不能同时使用**。因此普通 Offload 配方使用 compact 入口，HWR 配方使用兼容 DLO 入口。

| 场景 | 怎么起 | 拿什么换什么 |
|---|---|---|
| CUDA 单卡、Wan2.2，DiT 放得下但组件共驻超限 | `vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --diffusion-offload-config '{"mode":"module","components":["dit","text_encoder"]}'` | 组件交替驻留，增加阶段切换等待；卡容量仍需容纳活动 DiT 与 activation |
| CUDA 单卡、Wan2.2 视频，DiT 本身难以常驻 | `vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --diffusion-offload-config '{"mode":"layer","components":["dit"]}'` | 每轮按 block 搬运，用传输换权重容量；不承诺目标帧数一定不 OOM |
| 同机反复启动、BF16 Flux2 Klein 基础模型，CUDA 卡和主机均已完成容量检查 | `vllm serve black-forest-labs/FLUX.2-klein-4B --omni --dtype bfloat16 --enable-distributed-layerwise-offload --dlo-no-use-allgather --host-weight-runtime-mode preferred --host-weight-runtime-root /tmp/vllm-omni-hwr` | 首次正常加载后尝试发布，后续复用最终布局；目录需持久保留，重启系统清空后要重新预热 |

组合与推荐值是按语义推的起点,不是实测最优。

HWR 命中时检查 `DLO host-weight plan active` 中的 `host_weight_runtime`，不要把 `checkpoint_mmap` 当成同一件事。前者恢复最终布局，后者是直接 checkpoint 映射计划。DLO 的 `Allocated 2 shared device buffers` 说明流式设备槽分配；它不能证明请求峰值显存或总主机内存满足预算。需要把冷启动和热启动耗时、主机共享/私有内存、GPU 峰值及稳定态请求延迟分别记录。

| 症状 | 先查 | 然后 |
|---|---|---|
| HWR 开关已开但每次都正常加载 | 是否确实选中 rank-local DiT DLO，以及是否出现 ineligible 日志 | 核对模型契约、dtype、loader 与源前缀，再查共享目录和 artifact 身份 |
| 首次以 `required` 启动失败 | 同一身份是否已发布 artifact | 先用 `preferred` 完成加载与发布，再以相同配置启动 |
| DLO 跑到 collective 卡住 | DP rank 是否都进入同一组权重 collective | 检查请求参与和 cache 分支，必要时切 rank-local |
| 已 offload 仍在 VAE 阶段 OOM | VAE、activation 和未选组件的占用 | 单独处理这些预算，不继续盲目减小流式权重驻留 |
| HWR 恢复报错后又发生完整加载 | 是否出现 `discarding the model` / `fresh canonical` | 区分允许回退的恢复故障与真正的命中，保留首个异常查原因 |

## 六、常见误区

**看到 DLO 的双缓冲日志，就把整卡预算写成两个 block。** 这个数字容易被当成容量公式，但它只描述流式权重槽，AllGather 的 shard buffer 和执行 activation 都在之外；常驻层开启后还会继续增加占用。

**照旧文档判断 HWR 只有基础接口，于是从来不看启动参数。** 基准源码已经有 CLI 和 Diffusers loader 的具体接入；应沿日志核对命中、回退和发布，不能拿模块设计的边界声明当成“尚未集成”的证据。

**认为 `preferred` 可以吞掉任何不支持的组合。** 它允许明确的未命中与部分恢复失败回退，并不是总括异常处理。compact 配置互斥、缺少 root、身份构建中的不支持布局仍可能直接失败；应先满足消费者契约。

**认为同一个模型名一定能共享，目录变大就是缓存失效。** TP rank、最终布局或源内容不同都可能产生不同 artifact 身份；这正是在阻止错误权重复用。检查这些身份差异后再决定清理，不能只按模型名合并文件。

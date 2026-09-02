# 任务队列

**严格按优先级排序,从上往下做。**

维护规则(三条,必须遵守):

1. **做完就删掉那一行**——这是队列不是日志,不留「已完成」记录,历史去看 git log
2. **替换式更新**,不是增量堆积:新任务插到该在的优先级位置,顺手删掉已失效的
3. **只写任务,不写理由**:为什么这么排、怎么写、有什么取舍,一律在对应手册里(见 `00-START.md` 的地图)

---

## P0 · Infra 量化三篇

> 面经里量化是整整一页的考点,当前一篇没有。

- [ ] `04-Infra/01-原理/量化.md` —— 旧稿重写为基础篇(量化公式、per-tensor/channel/head/token 粒度与参数 shape、PTQ/QAT、int8/fp8/fp4 数据类型)
- [ ] `04-Infra/01-原理/权重与激活量化.md` —— GPTQ/AWQ/SmoothQuant、outlier 处理
- [ ] `04-Infra/01-原理/KVCache量化.md` —— 动态 vs 静态、attention 算子怎么改、校准集、CUDA Graph 与分布式下取量化参数、与推测解码交互、fp8 vs int8

## P1 · Infra 分布式与通信四篇

- [ ] `04-Infra/01-原理/并行策略.md` —— 旧稿重写(DP/TP/PP/EP/SP/CP 各自通信算子与通信量、DDP vs DP、跨机限制、64 卡并行度分配的数学约束、推理侧谁提吞吐谁降时延)
- [ ] `04-Infra/01-原理/集合通信.md` —— AllReduce/AllGather/ReduceScatter/All2All、Ring vs Tree、通信量估算、与计算 overlap
- [ ] `04-Infra/01-原理/MoE并行与DeepEP.md` —— dispatch/combine、四种方案通信量对比、切分方式、负载不均衡的 trace 表现与解法
- [ ] `04-Infra/01-原理/GPU互联与组网.md` —— PCIe/NVLink/NVSwitch/RDMA、零拷贝、多机 topo

## P2 · Infra 原理章补齐四篇

- [ ] `04-Infra/01-原理/显存管理与OOM.md`
- [ ] `04-Infra/01-原理/Prefill与Decode的矩阵形状.md`
- [ ] `04-Infra/01-原理/TorchCompile.md`
- [ ] `04-Infra/01-原理/00-总览.md` —— 全章写完后再写,收口用

## P3 · Infra 剩余旧稿翻新三篇

- [ ] `04-Infra/01-原理/ZeRO.md`(lint:图 14 节点 + 嵌套 subgraph + 代码 26 行)
- [ ] `04-Infra/01-原理/推理加速.md`(lint:图 14 节点)
- [ ] `04-Infra/01-原理/解码策略.md`

## P4 · Infra 框架与引擎 13 篇

> 只讲架构与用法,原理一律引到 01-原理 章。

- [ ] 新写 10 篇:`verl` `slime` `ROLL` `OpenRLHF` `Megatron` `FSDP` `vLLM` `SGLang` `NCCL` `00-总览`
- [ ] 旧稿翻新 3 篇:`DeepSpeed` `RL框架对比` `推理引擎对比`(后两者 lint 报嵌套 subgraph)

## P5 · 模型结构章旧稿翻新 15 篇

> lint 报违规最集中的一章(Norm位置 22 节点、KV共享注意力 19 节点等)。

- [ ] `00-总览` `KV共享注意力` `MLA` `SWA` `稀疏注意力` `线性注意力` `Hybrid注意力` `注意力配件` `RoPE` `MoE基础` `MoE路由` `Norm位置` `FFN与激活` `残差流` `MTP`
- [ ] 同章新写 3 篇:`MagiAttention` `Tokenizer` `新架构追踪`

## P6 · 强化学习章

- [ ] 旧稿翻新 6 篇:`00-总览` `PPO` `GRPO` `DPO` `RLHF与RM` `KL散度`
- [ ] 新写 3 篇:`GRPO变体` `多模态RL` `AgenticRL`

## P7 · 预训练与微调章

- [ ] 旧稿翻新 6 篇:`预训练流程` `ScalingLaws` `数据工程` `优化器` `SFT` `LoRA`
- [ ] 新写 3 篇:`00-总览` `蒸馏` `长上下文训练`

## P8 · 多模态章 33 篇

- [ ] 旧稿翻新 3 篇:`Diffusion` `FlowMatching` `VLM结构`
- [ ] 新写 30 篇:根目录共用范式(`00-总览` `VAE` `DiT`)+ 六个子领域(视觉理解 / 图像生成 / 视频生成 / 音频 / 3D生成 / 世界模型)

## P9 · 应用章 11 篇

- [ ] 新写:`00-总览` + RAG 5 篇 + Agent 5 篇

## P10 · 开源项目解读 28 个

- [ ] 按 `06-开源解读流程.md` 的状态表推进,建议顺序:`nano-vllm` → `vllm` → `Mooncake` → `verl` → `Megatron-LM`

## 待用户输入(阻塞中,不占优先级)

- [ ] 提供 18 处 `🖼️ 占位` 的配图(`grep -rn "🖼️ 占位" knowledge/` 列清单)
- [ ] 提供 P2 之后各批次对应的面经原件(截图),否则无法准确列高频问法、无法判断该不该加 🔴
- [ ] (可选)用官方数据校准 LeetCode 清单:见 `07-LeetCode清单.md` 的 curl + `npm run hot100:sync`

# 任务队列

**严格按优先级排序,从上往下做。** 本文件只回答“下一步做什么”;文章边界见 04,写作标准见 03 和 05。

维护规则(四条,必须遵守):

1. **做完就删掉那一行**——这是队列不是日志,不留「已完成」记录,历史去看 git log
2. **替换式更新**,不是增量堆积:新任务插到该在的优先级位置,顺手删掉已失效的
3. **只写任务,不写理由**:为什么这么排、怎么写、有什么取舍,一律在对应手册里(见 `00-START.md` 的地图)。题库联动任务写文章名或 topic;完成后删任务。写前查真题、写后逐题验收见写作契约第九节
4. **不保存动态统计**:题量、文章状态和面试池规模用对应命令现场查看

---

## 范式库与周会

- [ ] 首轮官方范式收集:Claude Code 与 Codex 的官方文档与博客,按 12 收进 `paradigms/`,并对 `AGENTS.md`、`docs/`、`.claude/skills/` 做一次影响扫描
- [ ] 下次周会按 13 出第一份纪要;攒满 3 份后决定 `meetings/` 与 `paradigms/` 要不要网页入口
- [ ] 手册去案例化,每次一份:把具体失误、决策过程、「实做后修正」的经过提炼成一句原则或删掉;06、10 最长,先做

## 开源贡献

- [ ] `sglang-omni` 接入收尾:维护者给私有实验仓设置 `CLAUDE_CODE_OAUTH_TOKEN` 后,在演练 PR 上各打一次 `review:code`、`review:pr` 验收两个评审,再清掉演练分支与 worktree;hlab 注册 `oss-sglang-omni`(运维线,按 homelab-control 守则)
- [ ] `sglang-omni` 选题:按 14 读 Roadmap 与相关 issue,出候选进 `contrib/sglang-omni.md`

## P6 · 开源解读与报告解读

- [ ] 报告解读批次:`reports/index.md` 已登记的 **余 63 条 ⬜**,每条先取原件再按 10 手册第三节动笔,验收按第七节。下一批做「检索与 RAG」三条:`Qwen3-Embedding`(arXiv 2506.05176)、`BGE-M3`(arXiv 2402.03216)、`jina-embeddings-v4`(arXiv 2506.18902),做完该方向清零;标「原件待核」的 8 条(Gemini-3、Llama-4、Muse-Spark、MAI-Thinking-1、Grok-4.6、Qwen3.5、Qwen3.7、SmolLM3)取件前先换渠道核存在性与首发日
- [ ] 全库扫一遍 Mermaid 标签里未加引号的括号(已知一处:`reports/Z.ai/IndexCache.md` 第 576 行)
- [ ] 按 `06-开源解读流程.md` 完成 `vllm`:总览与 01–17 章全部成稿,2026-09-18 已把 01–10 章统一成 11 章那种多 Part 结构(底稿第七节末行有改造记录);**下一步逐章通读挑问题,从总览开始**——总览缺阅读顺序、术语表、相关链接、组件登记表,特性表标题写「十五个」实际十八行且行序是追加序,表后「详见列空着」是施工残留;另外底稿第九节 12/15 章的延迟区还没兑现、`99-代码索引.md` 缺 15 章一节;**下一批讲稿等用户发 18 的截图**;`os:check` 报的基准过期是预期噪音不用管(底稿第八节已拍板)
- [ ] `sglang`:**总览与 01–18 章全部按 2026-09-20 的标准返工完成**,34 张手写 SVG,`npm test` 与 `scripts/svg-check.mjs` 全绿。**等用户通读验收**。两处留给下次的:底稿覆盖率核对表里 `mem_cache/hybrid_cache`(134 KB)与 `mem_cache/storage/umbp`(126 KB)标「待定」,不是平台移植也不是算子层,该归 03 还是 05 没定;08 章那条「默认 target-only 是否与标准拒绝采样等价」在底稿里标着待查,要把 kernel 的拒绝分支读完才能升回肯定句
- [ ] `sglang/02` 第四段与 `TensorRT-LLM/02` 对照表把 vLLM 的异步调度写成「实验性开关、默认关」,vLLM `94f4170df3` 上已默认开(`vllm/config/vllm.py` 异步调度默认分支);由各自那条线复核后改;另 `sglang/05` 对 vLLM 下放的描述(「CPU 层跟着进程走、跨实例得换通路」「从不等」「文件系统满了才换下去」)在该基准上也不成立,见 `vllm-lite/05` 第四段;`TensorRT-LLM/06` 写「查表和独立草稿模型默认不支持重叠调度」,其基准 `59f5c47f2e` 上独立草稿已归一体模式、支持重叠,只有 NGram 这类会关;`sglang/08` 留的「vLLM 贪心草稿被拒后从哪重采」已核清(从目标分布扣掉被拒 token 再采,见 `vllm-lite/09`);`sglang/11` 说 vLLM 有「16 个注册名」(实为 17)、「网关先发 P 再发 D」(只对默认的拉成立);`sglang/10` 说 vLLM 单副本开专家并行「仍走 all-gather 加 reduce-scatter」(实际不走 all-to-all,一次 all-reduce);`TensorRT-LLM/08` 说 vLLM 的均衡「是把热专家迁到轻载卡」(vLLM 也用冗余槽复制热专家);`sglang/13` 说 vLLM「每步用 100 微秒超时取编译结果」(已改为非阻塞检查);`sglang/14` 说 vLLM「默认切开占位区、每块只编本块」(实为整项编完、后续块取缓存)与「视频剪枝率是请求级」(实为启动配置);`sglang/15` 说 vLLM「每条请求一个线程组」(实为按适配器槽位排序后每个适配器一段);`TensorRT-LLM/11` 对照表里 vLLM 一列「默认一套,另有实验路径」「可一次采多个再适配掩码」不准(默认 V2,「多个」只指推测解码时每条填 1+K 行);`sglang/16` 说 vLLM「只有原生与 Transformers 后备两条路」(还有 `terratorch`);`sglang/17` 页头说「参照项目 vLLM 的解读里没有对应的一章」(已有 `vllm-lite/18`)
- [ ] vLLM 四个未覆盖主题按三档融入规则补进已有章:01 章加「API 进程里还有什么」Part、06 章加「编译:分段图之外」Part、05 章 Part 5 后加「状态空间模型」Part、03 章显存见底那组加 `simple_kv_offload` 一节;第二档变体一半只补索引行(gemma4、eplb、ubatch wrapper 等);覆盖地图初稿与脚本在 `tmp/vllm-coverage/`(未入库),索引页 85 个短文件名要补完整路径
- [ ] `torch.compile`(框架内核,按 06「大单体仓库按子系统拆」立项):源码在 `projects/框架内核/pytorch`,与 FSDP 共用基准 `217579124a`;快照 `/tmp/torchcompile-base`,底稿 `opensource/框架内核/torch-compile/_登记表.md`;序章已过门;01 章已验收;**02–04 章已成稿并过主线程验收,等用户逐章验收**;过门后派 05–07,每批 3 章,口径见底稿第六节。`os:check` 按 `projects/<主题>/<项目>` 定位仓库,子系统解读的基准与证据核对会被跳过(FSDP 同样),要补得让脚本从底稿读源码映射
- [ ] 通信三项,按序:`nccl` → `DeepEP` → `nixl`(都在 `projects/通信/`)
- [ ] 算子项目,按序:`FlashAttention`(参考项目,通用矩阵切法只在这里讲)→ `FlashInfer` → `DeepGEMM` → `FlashMLA` → `triton` → `Triton-distributed` → `TransformerEngine` 补 13 章量化核、14 章通信-GEMM 重叠核;全部按 06 的「算子项目的变体」写
- [ ] `TensorRT-LLM` 总览补一段「和 vLLM 真正不一样的几处」
- [ ] `TransformerEngine` 01–12 章第四段对照组返工:现为 Apex 与 FlashAttention,多格「不适用」,换成大厂大规模在用的对照或删成短列表
- [ ] `ray`:只做 Ray Core(actor、placement group、对象存储)
- [ ] `Dynamo`(`projects/推理服务/dynamo`)
- [ ] `torchtitan`
- [ ] Agent 应用层第一轮,按序:`LangGraph` → `MCP` → `OpenHands` → `mem0` → `LlamaIndex` → `E2B`
- [ ] 知识库扩到非 AI 方向:先在 04 定新章节与文章边界;每个方向的原理文章先于该方向第一个开源解读成稿
- [ ] 通用基础设施第一轮,每个方向先做一个代表,按序:`Linux 内核`(按子系统拆)→ `Redis` → `PostgreSQL`(按子系统拆)→ `etcd` → `Kubernetes`(按子系统拆)
- [ ] AI 长尾,按序:`llm-compressor` → `SpecForge` → `speculators` → `checkpoint-engine` → `AReaL` → `ktransformers` → `diffusers` → `llama.cpp`
- [ ] Agent 应用层第二轮,按序:`OpenAI Agents SDK` → `Codex CLI` → `OpenClaw` → `browser-use` → `Dify` → `GraphRAG` → `Temporal` → `DSPy` → `Langfuse` → `A2A`
- [ ] TPU 与编译器,按序:`JAX` → `XLA` → `maxtext` → `LLVM / MLIR`(按子系统拆)→ `TVM`
- [ ] 通用基础设施第二轮,按序:`RocksDB` → `Kafka` → `containerd` → `Envoy` → `gRPC` → `Nginx` → `Prometheus` → `Cilium`
- [ ] 数据与存储,按序:`TiDB / TiKV` → `ClickHouse` → `DuckDB` → `Spark` → `Flink` → `Arrow` → `Ceph` → `3FS`
- [ ] 其余硬件与系统,按序:`MLX` → `IREE` → `Firecracker` → `Terraform` → `systemd` → `NixOS / nixpkgs`
- [ ] AMD 栈,按序:`HIP + clr` → `composable_kernel` → `RCCL`(后两个在 rocm-libraries / rocm-systems 聚合仓里,按子目录稀疏克隆)
- [ ] 昇腾栈,按序:`torch_npu + op-plugin` → `CANN 算子库`(ops-transformer、catlass)→ `CANN GE` → `HCCL` → `vllm-ascend`;主仓在 gitcode,GitHub 是镜像
- [ ] 图计算,按序:`PyG` → `NebulaGraph` → `Neo4j` → `DGL`
- [ ] 区块链与加密货币,按序:`Bitcoin Core` → `go-ethereum` → `reth` → `rust-libp2p` → `Agave` → `Optimism` → `SP1`
- [ ] 密码学与安全,按序:`OpenSSL` → `rustls` → `liboqs`
- [ ] 语言与运行时,按序:`CPython`(按子系统拆)→ `Go 运行时` → `V8`(按子系统拆)→ `rustc`
- [ ] 前端,按序:`React` → `Vue` → `Node.js` → `Next.js` → `Vite`
- [ ] 后端与服务端框架,按序:`FastAPI + Starlette` → `Gin` → `axum + hyper` → `Django` → `Spring Boot`
- [ ] 语言生态库,按序:`Tokio` → `NumPy` → `Abseil` → `Polars` → `uv + ruff`
- [ ] 网络协议栈,按序:`quic-go` → `DPDK`
- [ ] 搜索与向量检索,按序:`Lucene` → `Faiss` → `Milvus`
- [ ] 音视频,按序:`FFmpeg`(按子系统拆)→ `LiveKit`
- [ ] 形式化验证,按序:`Lean 4` → `TLA+`
- [ ] 金融量化,按序:`Qlib` → `NautilusTrader` → `vn.py` → `Lean`
- [ ] 开源解读主页加方向标题(AI 基础设施、编译器、云原生、数据库与存储等),只改 `lib/opensource.ts` 的分组顺序;**等另一位同事的前端改动完成后再做**

## 模拟面试系统(可用,剩余项不阻塞)

- [ ] 面试中途刷新会丢:落盘只在结束时发生,可改成每轮增量写
- [ ] 英文档

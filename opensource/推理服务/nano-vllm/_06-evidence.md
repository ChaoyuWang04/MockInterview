# 06 启动期 · 证据表

基准:commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`,本地 `projects/推理服务/nano-vllm/`,只读。路径省略前缀 `nanovllm/`;`engine/` 下文件直接写文件名。厨房依赖 torch 与 GPU,本页无干跑,数字为默认店规下的手算。

## 一、图上元素对代码

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| ① | 建店规 | `llm_engine.py:18-21`;`config.py:20-25` | `config = Config(model, **config_kwargs)` `Sequence.block_size = config.kvcache_block_size` / `assert self.kvcache_block_size % 256 == 0` `assert 1 <= self.tensor_parallel_size <= 8` / `self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)` | 同步 |
| ② | 开分店 | `llm_engine.py:22-30` | `ctx = mp.get_context("spawn")` / `process = ctx.Process(target=ModelRunner, args=(config, i, event))` `process.start()` | 子进程异步启动,主进程不等 |
| ③ | 建菜谱装权重 | `model_runner.py:26-33`;`utils/loader.py:13-28`;`models/qwen3.py:187-193` | `dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)` / `self.model = Qwen3ForCausalLM(hf_config)` `load_model(self.model, config.model)` / `packed_modules_mapping = getattr(model, "packed_modules_mapping", {})` | 同步;NCCL 入组要等齐所有 rank |
| ④ | 试做最大一单 | `model_runner.py:34`,`:91-101` | `seq_len = min(max_num_batched_tokens, max_model_len)` `num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)` / `seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]` / `self.run(seqs, True)` | 同步 |
| ⑤ | 摆桌 | `model_runner.py:35`,`:103-121` | `block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize` / `config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes` / `self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)` | 同步;写回的是本进程手里的 config |
| ⑥ | 录流水线 | `model_runner.py:36-37`,`:223-257` | `if not self.enforce_eager: self.capture_cudagraph()` / `self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))` / `for bs in reversed(self.graph_bs):` … `with torch.cuda.graph(graph, self.graph_pool):` | 同步;每档 synchronize |
| ⑦ | 建领班,开门 | `llm_engine.py:32-35`;`scheduler.py:15` | `self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)` `config.eos = self.tokenizer.eos_token_id` `self.scheduler = Scheduler(config)` / `self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)` | 同步 |
| 分店进程启动 | 拿到 config 副本与 Event | `llm_engine.py:26-27` | `event = ctx.Event()` `process = ctx.Process(target=ModelRunner, args=(config, i, event))` | spawn 以 pickle 传参,config 是副本 |
| 传菜窗口开张 | barrier 后 attach,loop | `model_runner.py:41-48`,`:61-66` | `self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)` `dist.barrier()` / `dist.barrier()` `self.shm = SharedMemory(name="nanovllm")` `self.loop()` / `while True: method_name, args = self.read_shm()` | 分店 `__init__` 从此不返回 |
| 边 店规就绪 | ① 到 ② | `llm_engine.py:20-24` | 见 ①、② | |
| 边 自己建厨房 | ② 到 ③ | `llm_engine.py:31` | `self.model_runner = ModelRunner(config, 0, self.events)` | rank 0 在主进程内构造 |
| 边 spawn | ② 到分店 | `llm_engine.py:28` | `process.start()` | |
| 边 权重在卡上 | ③ 到 ④ | `model_runner.py:32-34` | `load_model(self.model, config.model)` `self.sampler = Sampler()` `self.warmup_model()` | |
| 边 峰值已知 | ④ 到 ⑤ | `model_runner.py:108-109` | `peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]` `current = torch.cuda.memory_stats()["allocated_bytes.all.current"]` | |
| 边 桌数写回 | ⑤ 到 ⑥ | `model_runner.py:113-114` | `config.num_kvcache_blocks = ...` `assert config.num_kvcache_blocks > 0` | |
| 边 enforce_eager 跳过 | ⑤ 到 ⑦ | `model_runner.py:36` | `if not self.enforce_eager:` | example.py:9 显式 True |
| 边 流水线就绪 | ⑥ 到 ⑦ | `model_runner.py:38-39`,`llm_engine.py:31-32` | `torch.set_default_device("cpu")` `torch.set_default_dtype(default_dtype)` / 构造返回后 `self.tokenizer = ...` | |
| 边 TP>1 建窗口后 barrier | ⑥ 到传菜窗口 | `model_runner.py:41-44` | `if self.world_size > 1: if rank == 0: self.shm = SharedMemory(...)` `dist.barrier()` | |
| 边 同样跑完 ③ 到 ⑥ | 分店泳道 | `model_runner.py:17-39` | 同一个 `__init__`,`rank` 不同 | |

## 二、图外但正文用到的证据

| 断言 | 位置 | 原样引用 |
|---|---|---|
| Config 只收认识的字段 | `llm_engine.py:18-19` | `config_fields = {field.name for field in fields(Config)}` `config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}` |
| 默认 dtype 与 device 改到卡上再改回 | `model_runner.py:28-30, 38-39` | `torch.set_default_dtype(hf_config.dtype)` `torch.set_default_device("cuda")` / `torch.set_default_device("cpu")` |
| 试做时没有桌位则槽位映射为空 | `model_runner.py:149-150` | `if not seq.block_table:    # warmup` `continue` |
| 试做时 k_cache 为空跳过写入 | `layers/attention.py:57, 62-63` | `self.k_cache = self.v_cache = torch.tensor([])` / `if k_cache.numel() and v_cache.numel(): store_kvcache(...)` |
| 每 rank kv 头数按 TP 切 | `model_runner.py:110-111` | `num_kv_heads = hf_config.num_key_value_heads // self.world_size` `head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)` |
| 逐层绑定 K、V 视图 | `model_runner.py:116-121` | `if hasattr(module, "k_cache") and hasattr(module, "v_cache"): module.k_cache = self.kv_cache[0, layer_id]` `module.v_cache = self.kv_cache[1, layer_id]` |
| graph 最大批与最多桌数 | `model_runner.py:226-227` | `max_bs = min(self.config.max_num_seqs, 512)` `max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size` |
| 内存池取自第一张图 | `model_runner.py:244-246` | `if self.graph_pool is None: self.graph_pool = graph.pool()` `self.graphs[bs] = graph` |
| 分片装载走参数自带的 weight_loader | `utils/loader.py:22-23, 27-28` | `weight_loader = getattr(param, "weight_loader")` `weight_loader(param, f.get_tensor(weight_name), shard_id)` / `weight_loader = getattr(param, "weight_loader", default_weight_loader)` |
| 桌大小进客人类后被谁用 | `sequence.py:15, 56-65` | `block_size = 256` / `return (self.num_tokens + self.block_size - 1) // self.block_size` |
| 退出路径 | `llm_engine.py:35-41`;`model_runner.py:50-59, 65-66` | `atexit.register(self.exit)` / `self.model_runner.call("exit")` … `p.join()` / `self.shm.close()` … `if self.rank == 0: self.shm.unlink()` … `dist.destroy_process_group()` |
| 用户脚本的 main 保护 | `example.py:32-33` | `if __name__ == "__main__":` `main()` |

## 三、状态所有权行

| 状态 | 持有者 | 位置 | 原样引用 |
|---|---|---|---|
| `config.num_kvcache_blocks` | 各进程各自的店规;经理只用 rank 0 那份 | `model_runner.py:113`;`llm_engine.py:34` | `config.num_kvcache_blocks = int(...) // block_bytes` / `self.scheduler = Scheduler(config)` |
| `config.eos` | 经理在开门前填 | `llm_engine.py:33` | `config.eos = self.tokenizer.eos_token_id` |
| `Sequence.block_size` | 客人类属性,经理在 ① 写 | `llm_engine.py:21` | `Sequence.block_size = config.kvcache_block_size` |
| `kv_cache`、`graphs`、`graph_pool`、`graph_vars` | 每个厨房各一份 | `model_runner.py:115, 235-236, 250-257` | `self.kv_cache = torch.empty(...)` / `self.graphs = {}` `self.graph_pool = None` / `self.graph_vars = dict(...)` |
| 共享内存句柄 | rank 0 创建并 unlink,分店只 attach | `model_runner.py:43, 47, 53-55` | `SharedMemory(name="nanovllm", create=True, size=2**20)` / `SharedMemory(name="nanovllm")` / `if self.rank == 0: self.shm.unlink()` |

## 四、设计取舍行

| 取舍 | 证据位置 | 原样引用 |
|---|---|---|
| 试做量峰值而不是公式估算 | `model_runner.py:92-93, 108-109, 113` | `torch.cuda.reset_peak_memory_stats()` / `peak = ...["allocated_bytes.all.peak"]` / `config.num_kvcache_blocks = int(total * ... - peak + current) // block_bytes` |
| 预录 36 档流水线 | `model_runner.py:234, 238-243` | `self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))` / `outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup` `with torch.cuda.graph(graph, self.graph_pool):` |
| 桌数各算各的 | `model_runner.py:113`;`llm_engine.py:27, 34` | 分店拿到的是 `args=(config, i, event)` 的 pickle 副本;经理用自己手里的 `config` 建领班 |
| 地址与名字写死 | `model_runner.py:26, 43` | `"tcp://localhost:2333"` / `SharedMemory(name="nanovllm", ...)` |
| spawn 而不是 fork | `llm_engine.py:24` | `ctx = mp.get_context("spawn")` |

## 五、archify 校验回执

命令:`node ~/.claude/skills/archify/bin/archify.mjs validate workflow diagrams/06-startup.json --quality showcase --json`

- 第 1 轮:版式通过,`composition/desktop-readability` 1 错:副标签「Config 三条断言;桌大小写进客人类」在 1440 视口投影 5.69px。修复:九个节点的副标签压到 12 字宽以内。
- 第 2 轮:见主线程交付记录(validate、deliver、visual-check 均由主线程执行)。
- 图规模:9 节点、10 边、3 泳道、3 阶段、2 组,主路径 ① 到 ⑦。

## 六、疑问与冲突

1. **桌数不一致的风险**只是推断:代码没有广播 rank 0 的桌数,也没有断言各 rank 相等;真实后果要在多卡上实测。
2. **同机多实例撞名**是推断:共享内存名与端口写死是事实,撞上后的具体异常未实测。
3. **spawn 的原因**代码没有注释;「不继承 CUDA 上下文」是通行解释,标为推断。
4. 本页由主线程亲写:两次派发的子 agent 都在起步阶段因基础设施停滞,第三次派发被分类服务超时挡回。

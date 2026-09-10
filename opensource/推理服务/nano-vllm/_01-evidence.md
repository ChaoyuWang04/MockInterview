# 01 指挥循环 · 证据表

基准:commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`,本地 `projects/推理服务/nano-vllm/`,只读。路径省略前缀 `nanovllm/`;`engine/` 下文件直接写文件名。引用为原样代码,最多 3 行,用 ` / ` 分隔不相邻的行。

干跑说明:本机无 `torch`、`transformers`、`xxhash`、`numpy`。用 `types.ModuleType` 打桩这四个模块(xxhash 用 blake2b 8 字节替代,numpy 只给 `array(...).tobytes()`),绕过 `nanovllm/__init__.py`,直接导入 `engine/sequence.py`、`engine/scheduler.py`、`engine/block_manager.py`,以 `SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=512, eos=-1, kvcache_block_size=256, num_kvcache_blocks=64)` 构造 `Scheduler`;A=`range(1000,1300)`、B=`range(2000,2600)`,`SamplingParams(temperature=0.6, max_tokens=3)`;厨房用递增假 token 代替。脚本在 `/tmp/nanovllm_dryrun_01.py`,不入仓库。轨迹:5 个 step,prefill 300 / 512 / 88,decode 2 轮;pickle 字节数 A prefill 982、B prefill 1884、decode 83 与 85。

## 一、图上元素对代码

### 参与者

| ID | 图上元素 | 位置 | 原样引用 | 备注 |
|---|---|---|---|---|
| user | 用户脚本 · example.py | `example.py:9,24` | `llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)` / `outputs = llm.generate(prompts, sampling_params)` | 主线单卡、eager |
| engine | 经理 · LLMEngine | `llm_engine.py:15`;`llm.py:4` | `class LLMEngine:` / `class LLM(LLMEngine):` | `LLM` 只是别名 |
| scheduler | 领班 · Scheduler | `scheduler.py:8`;`llm_engine.py:34` | `class Scheduler:` / `self.scheduler = Scheduler(config)` | 主进程 |
| blocks | 座位管理员 · BlockManager | `block_manager.py:26`;`scheduler.py:15` | `class BlockManager:` / `self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)` | 领班私有 |
| runner | 厨房 rank 0 · ModelRunner | `llm_engine.py:31` | `self.model_runner = ModelRunner(config, 0, self.events)` | 主进程内构造 |
| shm | 传菜窗口 · SharedMemory + Event | `model_runner.py:43`;`llm_engine.py:26` | `self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)` / `event = ctx.Event()` | 1 MiB;每个子进程一个 Event |
| runnerN | 分店厨房 · ModelRunner rank 1..N | `llm_engine.py:24-28` | `ctx = mp.get_context("spawn")` / `process = ctx.Process(target=ModelRunner, args=(config, i, event))` / `process.start()` | spawn 子进程;进 `__init__` 末尾 `self.loop()` |

### 消息(编号同图)

| 编号 | 消息 | 调用点 → 定义/返回点 | 原样引用 | 参数与返回类型;同步/异步 |
|---|---|---|---|---|
| ① | generate(prompts, sp) | `example.py:24` → `llm_engine.py:60-65, 88-90` | `def generate(self, prompts: list[str] \| list[list[int]], sampling_params: SamplingParams \| list[SamplingParams], use_tqdm: bool = True,` `) -> list[str]:` / `outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]` / `return outputs` | 同步阻塞;实际返回 `list[dict]` |
| ② | add(seq) ×2 | `llm_engine.py:69-70` → `:43-47` → `scheduler.py:22-23` | `self.add_request(prompt, sp)` / `seq = Sequence(prompt, sampling_params)` `self.scheduler.add(seq)` / `self.waiting.append(seq)` | 同步;返回 None |
| ③ | schedule() | `llm_engine.py:50` → `scheduler.py:25` | `seqs, is_prefill = self.scheduler.schedule()` / `def schedule(self) -> tuple[list[Sequence], bool]:` | 同步 |
| ④ | allocate(A, 0) → [0, 1];B 留下 | `scheduler.py:35-39, 44-45` → `block_manager.py:58-73, 75-92`;B:`scheduler.py:42-43` | `num_cached_blocks = self.block_manager.can_allocate(seq)` / `self.block_manager.allocate(seq, num_cached_blocks)` / `if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq` `break` | `can_allocate -> int`(-1 为不够);`allocate` 写 `seq.block_table`;同步 |
| ⑤ | ([A], True) | `scheduler.py:46-55` | `seq.num_scheduled_tokens = min(num_tokens, remaining)` / `seq.status = SequenceStatus.RUNNING` / `if scheduled_seqs: return scheduled_seqs, True` | 返回 `tuple[list[Sequence], bool]` |
| ⑥ | call("run", [A], True) | `llm_engine.py:52` → `model_runner.py:85-89` | `token_ids = self.model_runner.call("run", seqs, is_prefill)` / `method = getattr(self, method_name, None)` `return method(*args)` | 同进程同步 |
| ⑦ | write_shm + Event.set(TP 分支) | `model_runner.py:86-87` → `:76-83`;瘦身 `sequence.py:72-74` | `if self.world_size > 1 and self.rank == 0: self.write_shm(method_name, *args)` / `data = pickle.dumps([method_name, *args])` … `for event in self.event: event.set()` / `last_state = self.last_token if not self.is_prefill else self.token_ids` `return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)` | 单向;写完立刻回到 `:88` 本地执行,不等分店 |
| ⑧ | read_shm → call("run")(TP 分支) | `model_runner.py:48` → `:61-66` → `:68-74`;不回传 `:216, 218`;`layers/embed_head.py:62-65` | `method_name, args = self.read_shm()` `self.call(method_name, *args)` / `self.event.wait()` … `self.event.clear()` / `token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None` / `dist.gather(logits, all_logits, 0)` `logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None` | 子进程阻塞在 `event.wait()`;`run` 返回 None,`loop` 不接收返回值 |
| ⑨ | token_ids=[tA1] | `model_runner.py:214-220`;H2D `:164-168`;取末位 `embed_head.py:58-60` | `def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:` / `input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)` / `last_indices = context.cu_seqlens_q[1:] - 1` | 返回 `list[int]`,长度 = len(seqs);`.tolist()` 同步 GPU |
| ⑩ | postprocess([A], [tA1]):第一个 token 写回 | `llm_engine.py:53` → `scheduler.py:81-88` → `sequence.py:67-70` | `seq.num_cached_tokens += seq.num_scheduled_tokens` / `if is_prefill and seq.num_cached_tokens < seq.num_tokens: continue` / `seq.append_token(token_id)` | 同步;返回 None;300 == 300 不 continue |
| ⑩ 补 | 本 step 输出为空 | `llm_engine.py:54` | `outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]` | 只有 FINISHED 才进 outputs |
| ⑪ | schedule() → B 拿 512 | `scheduler.py:30-32, 42-43, 46` | `remaining = self.max_num_batched_tokens - num_batched_tokens` / `if remaining < num_tokens and scheduled_seqs:` / `seq.num_scheduled_tokens = min(num_tokens, remaining)` | `scheduled_seqs` 为空,不 break |
| ⑫ | allocate(B, 0) → [2, 3, 4] | `block_manager.py:90-92` | `for i in range(num_cached_blocks, seq.num_blocks):` `seq.block_table.append(self._allocate_block())` `seq.num_cached_tokens = num_cached_blocks * self.block_size` | 桌位一次分齐 |
| ⑬ | ([B], True) · 512;B 仍 WAITING;A 空等 | `scheduler.py:48-51, 54-55` | `if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:` / `if scheduled_seqs: return scheduled_seqs, True` | 0+512≠600 不 popleft;`:57` decode 分支未到达 |
| ⑭ | call("run", [B], True),不带 block_tables | `model_runner.py:139-142, 162-163` | `start = seq.num_cached_tokens` `seqlen_q = seq.num_scheduled_tokens` / `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache` `block_tables = self.prepare_block_tables(seqs)` | 512 == 512,`block_tables` 保持 None |
| ⑮ | token_ids=[tB?] | `model_runner.py:216-218` | `temperatures = self.prepare_sample(seqs) if self.rank == 0 else None` / `token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None` | `run` 无"本轮是否采样"分支 |
| ⑯ | postprocess:丢弃 tB? | `scheduler.py:83-87`;盖章 `block_manager.py:110-120` | `self.block_manager.hash_blocks(seq)` `seq.num_cached_tokens += seq.num_scheduled_tokens` / `if is_prefill and seq.num_cached_tokens < seq.num_tokens:` `continue` | 512 < 600 → continue;token 不落任何地方 |
| ⑰ | schedule() → B 剩 88,转 RUNNING | `scheduler.py:40-41, 48-51` | `num_tokens = seq.num_tokens - seq.num_cached_tokens` / `seq.status = SequenceStatus.RUNNING` `self.waiting.popleft()` `self.running.append(seq)` | 512+88 == 600 |
| ⑱ | ([B], True) · 88 | `scheduler.py:54-55` | `if scheduled_seqs:` `return scheduled_seqs, True` | |
| ⑲ | call("run", [B], True),带 block_tables | `model_runner.py:162-163, 169`;`layers/attention.py:64-70` | `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache` / `set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)` / `if context.block_tables is not None:    # prefix cache` `k, v = k_cache, v_cache` | k 长 600 > q 长 88 |
| ⑳ | token_ids=[tB1] | `model_runner.py:218` | 同 ⑨ | |
| ㉑ | postprocess → B + tB1 | `scheduler.py:86-88` | `if is_prefill and seq.num_cached_tokens < seq.num_tokens:` `continue` `seq.append_token(token_id)` | 600 == 600 不 continue |
| ㉒ | schedule() → decode | `scheduler.py:54-59` | `if scheduled_seqs:` (为空,跳过) / `# decode` `while self.running and len(scheduled_seqs) < self.max_num_seqs:` `seq = self.running.popleft()` | |
| ㉓ | can_append ×2 · 不加桌 | `scheduler.py:60, 67-70` → `block_manager.py:103-108` | `while not self.block_manager.can_append(seq):` / `seq.num_scheduled_tokens = 1` `seq.is_prefill = False` `self.block_manager.may_append(seq)` / `return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)` | 301%256=45、601%256=89 |
| ㉔ | ([A, B], False) | `scheduler.py:71-73` | `assert scheduled_seqs` `self.running.extendleft(reversed(scheduled_seqs))` `return scheduled_seqs, False` | 原序放回 running 队首 |
| ㉕ | call("run", [A, B], False);瘦身只带 last_token | `model_runner.py:172-181`;`sequence.py:73` | `input_ids.append(seq.last_token)` `positions.append(len(seq) - 1)` / `last_state = self.last_token if not self.is_prefill else self.token_ids` | decode 时 `is_prefill=False` |
| ㉖ | token_ids=[tA2, tB2];eager | `model_runner.py:197-198` | `if is_prefill or self.enforce_eager or input_ids.size(0) > 512:` `return self.model.compute_logits(self.model(input_ids, positions))` | example.py `enforce_eager=True` |
| ㉗ | postprocess → 各 +1 | `scheduler.py:88` | `seq.append_token(token_id)` | step 5 同形 |
| ㉘ | deallocate(A)、(B);FINISHED | `scheduler.py:89-92` → `block_manager.py:94-101` | `if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:` `seq.status = SequenceStatus.FINISHED` `self.block_manager.deallocate(seq)` `self.running.remove(seq)` / `seq.num_cached_tokens = 0` `seq.block_table.clear()` | `num_completion_tokens` 见 `sequence.py:43-45` |
| ㉙ | [{text, token_ids}] ×2 | `llm_engine.py:73, 84-89`;`scheduler.py:19-20` | `while not self.is_finished():` / `for seq_id, token_ids in output: outputs[seq_id] = token_ids` / `return not self.waiting and not self.running` | 同步返回 |

### 图外但正文用到的证据

| 正文断言 | 位置 | 原样引用 |
|---|---|---|
| 经理持有子进程句柄与 Event | `llm_engine.py:22-23, 29-30` | `self.ps = []` `self.events = []` / `self.ps.append(process)` `self.events.append(event)` |
| 退出路径 | `llm_engine.py:35-41`;`model_runner.py:65-66` | `atexit.register(self.exit)` / `self.model_runner.call("exit")` … `for p in self.ps: p.join()` / `if method_name == "exit": break` |
| 分店与总店的会合点 all_reduce | `layers/linear.py:152-156` | `if self.tp_size > 1:` `dist.all_reduce(y)` |
| step 用符号区分吞吐 | `llm_engine.py:51, 76-79` | `num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)` / `if num_tokens > 0: prefill_throughput = ...` `else: decode_throughput = -num_tokens / ...` |
| decode 分支 assert 失败面 | `scheduler.py:37-38, 71` | `if num_cached_blocks == -1: break` / `assert scheduled_seqs` |
| preempt 打回 WAITING | `scheduler.py:75-79` | `seq.status = SequenceStatus.WAITING` `seq.is_prefill = True` `self.block_manager.deallocate(seq)` `self.waiting.appendleft(seq)` |
| 共享内存写入方式 | `model_runner.py:80-81` | `self.shm.buf[0:4] = n.to_bytes(4, "little")` `self.shm.buf[4:n+4] = data` |
| 分店进程内 `temperatures=None` | `model_runner.py:216` | `temperatures = self.prepare_sample(seqs) if self.rank == 0 else None` |
| decode H2D 同样 pinned + non_blocking | `model_runner.py:182-185` | `input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)` |
| 本轮工单 set/reset | `utils/context.py:21-27`;`model_runner.py:219` | `_CONTEXT = Context(is_prefill, cu_seqlens_q, ...)` / `_CONTEXT = Context()` / `reset_context()` |
| SamplingParams 默认 | `sampling_params.py:6-8` | `temperature: float = 1.0` `max_tokens: int = 64` `ignore_eos: bool = False` |
| 预算默认 | `config.py:9` | `max_num_batched_tokens: int = 16384` |

## 二、状态所有权行

| 状态 | 持有者 | 位置 | 原样引用 |
|---|---|---|---|
| `waiting`、`running` | 领班 | `scheduler.py:16-17` | `self.waiting: deque[Sequence] = deque()` `self.running: deque[Sequence] = deque()` |
| `token_ids`、`status`、`block_table`、四个计数、`is_prefill` | 客人 Sequence | `sequence.py:19-28` | `self.status = SequenceStatus.WAITING` `self.token_ids = copy(token_ids)` / `self.num_cached_tokens = 0` `self.num_scheduled_tokens = 0` `self.is_prefill = True` `self.block_table = []` |
| `blocks`、`hash_to_block_id`、`free_block_ids`、`used_block_ids` | 座位管理员 | `block_manager.py:30-33` | `self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]` `self.hash_to_block_id: dict[int, int] = dict()` `self.free_block_ids: deque[int] = deque(range(num_blocks))` |
| tokenizer、领班、rank 0 厨房、`ps`、`events` | 经理 | `llm_engine.py:22-34` | `self.model_runner = ModelRunner(config, 0, self.events)` `self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)` `self.scheduler = Scheduler(config)` |
| `model`、`sampler`、`kv_cache`、`graphs`、`graph_vars`、`shm` | 厨房(每个 rank 各一份) | `model_runner.py:31-33, 43/47, 115, 235, 250` | `self.model = Qwen3ForCausalLM(hf_config)` `self.sampler = Sampler()` / `self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, ...)` |
| 本轮工单 `_CONTEXT` | 厨房进程内全局 | `utils/context.py:16` | `_CONTEXT = Context()` |
| `config.num_kvcache_blocks` | rank 0 开店时写回 config | `model_runner.py:113` | `config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes` |
| 子进程收到的 Sequence 副本 | 无人持有,算完即弃 | `model_runner.py:63-64`;`sequence.py:76-83` | `method_name, args = self.read_shm()` `self.call(method_name, *args)`(返回值未赋值) / `def __setstate__(self, state):` |

## 三、设计取舍行

| 取舍 | 证据位置 | 原样引用 |
|---|---|---|
| prefill 优先、decode 让路 | `scheduler.py:54-55` | `if scheduled_seqs:` `return scheduled_seqs, True` |
| 分块只给队首 | `scheduler.py:42` | `if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq` |
| 子进程不回传 | `embed_head.py:63-65`;`model_runner.py:218`;`:63-64` | `all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None` / `token_ids = ... if self.rank == 0 else None` |
| 瘦身序列化 | `sequence.py:72-74` | `last_state = self.last_token if not self.is_prefill else self.token_ids` |
| 分块中途也采样、再丢 | `model_runner.py:218`;`scheduler.py:86-87` | `token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None` / `if is_prefill and seq.num_cached_tokens < seq.num_tokens:` `continue` |
| 主循环同步单线程 | `llm_engine.py:73-75` | `while not self.is_finished():` `t = perf_counter()` `output, num_tokens = self.step()` |
| H2D 异步、`.tolist()` 同步 | `model_runner.py:164-168, 218` | `.cuda(non_blocking=True)` / `.tolist()` |
| 单向广播无 ack | `model_runner.py:82-83, 88` | `for event in self.event: event.set()` / `method = getattr(self, method_name, None)` |

## 四、archify 校验回执

命令:`node ~/.claude/skills/archify/bin/archify.mjs validate sequence diagrams/_01-command-loop.json --quality showcase --json`

- 第 1 轮:`ok: false`,1 错 0 警。`composition/desktop-readability`:viewBox 宽 1440 缩放到 930px 桌面视口后,参与者副标(源字号 7px)投影 4.52px < 6px。修复:viewBox 改为 `[1080, 1150]`。
- 第 2 轮(最终):`ok: true`;`checks` 9 项全部 `ok: true`(single_svg、finite_svg、orthogonal_arrows、label_route_clearance、relationship_crossings、relationship_corridors、container_border_runs、route_rhythm、legend_clearance);`composition.profile: showcase`,`status: pass`,`summary: {errors: 0, warnings: 0}`;metrics:`minLabelRouteClearance 8`,`desktopReadabilityIssues 0`,`minSegmentPx 130.2`,`properCrossings 0`。
- 图规模:7 个参与者,29 条消息(其中 ⑦⑧ 为 TP=2 分支,dashed),4 个时间段,10 条激活条,3 个 guided view,3 张卡。只做了 validate,未 deliver / preview / visual-check。

## 五、疑问与冲突

1. **参与者 ID 不一致**:底稿第四节给厨房的 ID 是 `runner`,任务卡也用 `runner`;但主线程已发布的 `diagrams/00-overview.json` 用的是 `runner0`。本图按任务卡用 `runner`。建议底稿冻结一个,后续图统一。
2. **`generate` 的返回类型注解与实现不符**:`llm_engine.py:65` 注解 `-> list[str]`,`:89` 实际返回 `list[dict]`(`{"text", "token_ids"}`)。页面按实际写。
3. **主线到底按谁的默认**:底稿第五节列 `enforce_eager=False` 为店规默认,但 `example.py:9` 显式 `enforce_eager=True`;任务卡说"主线按 example.py 的单卡"。本页 decode 走读按 eager 路径,并注明 CUDA graph 路径存在但未走。建议底稿明确"示例甲 = example.py 的实参 + 预算 512 + max_tokens=3"。
4. **哈希索引不随退桌清理**:干跑到两人结账后 `free_block_ids` 回到 64,但 `hash_to_block_id` 仍有 3 条;条目只在该块被 `_allocate_block` 重新分配时删除(`block_manager.py:47-48`)。不影响本页,提示 03/04(拼桌)页处理。
5. **分块 prefill 第二轮仍重发整段 token_ids**:`__getstate__` 只看 `is_prefill`,不看 `num_cached_tokens`,B 两轮都是 1884 字节。是事实描述,不是 bug 判断。
6. **1 MiB 上限的失败形态未实测**:页面只写"写不下会失败",没有断言异常类型。
7. **Sequence 的 `is_prefill` 字段**在 prefill 完成后不会立刻翻成 False(`:49-51` 只改 status),要等第一次被 decode 调度(`:68`)才翻。A 在 step 2–3 空等期间 `is_prefill` 仍为 True,但期间没被 pickle,不影响本页。
8. **`step()` 里 `outputs` 只在 FINISHED 时产出**:用户脚本永远看不到中间 token;页面已说明,但底稿比喻表里"经理……收结果"可加一句"只收结账单"。

## 六、对底稿与任务卡的改进建议(试跑)

1. **sequence 图给尺寸约束**:showcase 要求参与者副标在 1440 桌面视口投影 ≥ 6px,推出 viewBox 宽 ≤ 1085;建议底稿直接写"sequence 用 `[1080, H]` + `column_fit: spread`",省一轮修复。H 约等于最后一条消息 y + 100。
2. **消息条数给上限**:底稿"12 个节点"约束不管消息;本图 29 条已经是可读上限,建议写"一张 sequence 图 ≤ 30 条消息,同形 step 只画一次并在 note 里说'同形'"。
3. **相邻泳道的标签长度**:1080 宽、7 个参与者时列距约 130px,相邻泳道的标签建议 ≤ 12 个汉字或 ≤ 22 个拉丁字符,细节放 `note`。本图未触发 label_route_clearance,但已接近。
4. **统一证据表列头**:建议底稿给出固定列头(编号 | 图上元素 | 调用点 → 定义/返回点 | 原样引用 | 参数与返回类型;同步/异步),各页一致便于汇总。
5. **给出干跑打桩模板**:本机没有 torch/transformers/xxhash/numpy,底稿只说了缺 xxhash/numpy 时手算;实际还要绕过 `nanovllm/__init__.py`(它会导入 torch)。建议把打桩脚本放进底稿附录或 `tools/`,03/04 页直接复用。
6. **页面骨架为"骨架代码"留编号**:任务卡要求在第三节前加"骨架代码",骨架里没有编号位,本页用了无编号的 `## 骨架代码`。建议骨架改成"二、术语卡 / 三、骨架代码 / 四、走读 …"或明确允许无编号插入。
7. **"每一跳同步还是异步"的答法**:建议底稿规定三分法——同进程同步调用 / 单向广播(fire-and-forget)/ GPU 异步(non_blocking、`.tolist()` 同步点),各页用同一套词。
8. **参与者 ID 冻结表要和总图对齐**(见冲突 1)。

# 05 一次前向 · 证据表

基准:commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`,本地 `projects/推理服务/nano-vllm/`,只读。路径省略前缀 `nanovllm/`;`engine/` 下文件直接写文件名。厨房依赖 torch 与 GPU,本页无干跑;页面里的槽位与累计长度按源码算式手算。页面草稿与两张规格由子 agent 完成,证据表由主线程按页面引用逐条核对后补写。

## 一、图上元素对代码

### 图 a:拍平与工单

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| ① | 一批客人 seqs | `llm_engine.py:52` → `model_runner.py:214-215` | `token_ids = self.model_runner.call("run", seqs, is_prefill)` / `input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)` | 同步 |
| ② | prefill 读法 | `model_runner.py:138-161` | `start = seq.num_cached_tokens` `seqlen_q = seq.num_scheduled_tokens` `end = start + seqlen_q` / `input_ids.extend(seq[start:end])` `positions.extend(range(start, end))` / `slot_start = seq.block_table[i] * self.block_size` | 无桌位表时 `:149-150` continue |
| ③ | prefill 张量 ×5 | `model_runner.py:164-168` | `input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)` / `cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)` / `slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)` | H2D 异步;`max_seqlen_q/k` 是 Python int(`:147-148`) |
| ④ | 桌位表(条件) | `model_runner.py:162-163`,`:186`,`:123-127` | `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache` `block_tables = self.prepare_block_tables(seqs)` / `block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]` | prefill 条件建;decode 必建 |
| ⑤ | 本轮工单 | `model_runner.py:169`,`:187`;`utils/context.py:21-23` | `set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)` / `set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)` / `_CONTEXT = Context(is_prefill, cu_seqlens_q, ...)` | 进程内全局变量 |
| ⑥ | 交前向 | `model_runner.py:216-217` | `temperatures = self.prepare_sample(seqs) if self.rank == 0 else None` `logits = self.run_model(input_ids, positions, is_prefill)` | 只传 ids 与 positions |
| ⑦ | decode 读法 | `model_runner.py:177-181` | `input_ids.append(seq.last_token)` `positions.append(len(seq) - 1)` `context_lens.append(len(seq))` `slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)` | 不读 `token_ids` |
| ⑧ | decode 张量 ×4 | `model_runner.py:182-185` | `input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)` / `context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)` | H2D 异步 |
| ⑨ | 传菜窗口 | `model_runner.py:86-87`,`:76-83`;`sequence.py:72-74` | `if self.world_size > 1 and self.rank == 0: self.write_shm(method_name, *args)` / `data = pickle.dumps([method_name, *args])` / `last_state = self.last_token if not self.is_prefill else self.token_ids` | 单向广播;瘦身副本 |

### 图 b:前向两条路

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| ① | 前向入口 run_model | `model_runner.py:195-197` | `@torch.inference_mode()` `def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):` `if is_prefill or self.enforce_eager or input_ids.size(0) > 512:` | 三条件任一为真则现做 |
| ② | 静态缓冲 graph_vars | `model_runner.py:203-210`,`:250-257` | `graph_vars["slot_mapping"].fill_(-1)` `graph_vars["slot_mapping"][:bs] = context.slot_mapping` / `graph_vars["context_lens"].zero_()` / `graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables` | 设备内拷贝 |
| ③ | 回放 replay | `model_runner.py:202`,`:211-212` | `graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]` / `graph.replay()` `return self.model.compute_logits(graph_vars["outputs"][:bs])` | 桶向上取整;取前 bs 行 |
| ④ | 出 logits lm_head | `layers/embed_head.py:56-66` | `if context.is_prefill: last_indices = context.cu_seqlens_q[1:] - 1` `x = x[last_indices].contiguous()` / `dist.gather(logits, all_logits, 0)` `logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None` | prefill 取末位;TP 时 gather |
| ⑤ | 出菜 Sampler | `layers/sampler.py:7-12`;`model_runner.py:218-219` | `logits = logits.float().div_(temperatures.unsqueeze(dim=1))` `probs = torch.softmax(logits, dim=-1)` `sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)` / `token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None` `reset_context()` | `.tolist()` 同步点;rank≠0 为 None |
| ⑥ | 现做 eager | `model_runner.py:197-198` | `return self.model.compute_logits(self.model(input_ids, positions))` | prefill 永远走这里 |
| ⑦ | 灶台 Attention | `layers/attention.py:59-75` | `if k_cache.numel() and v_cache.numel(): store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)` / `if context.block_tables is not None:    # prefix cache` `k, v = k_cache, v_cache` / `o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context.context_lens, block_table=context.block_tables, ...)` | prefill varlen;decode with_kvcache |
| ⑧ | KV cache 大张量 | `model_runner.py:115-121`;`layers/attention.py:21-30` | `self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)` / `module.k_cache = self.kv_cache[0, layer_id]` / `slot = tl.load(slot_mapping_ptr + idx)` `if slot == -1: return` … `cache_offsets = slot * D + tl.arange(0, D)` | 槽位是拍平后的绝对行号 |
| ⑨ | 切菜刀 all_reduce | `layers/linear.py:152-156`;`layers/embed_head.py:34-42`;`models/qwen3.py` 中 o_proj 与 down_proj 为 RowParallelLinear | `y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)` `if self.tp_size > 1: dist.all_reduce(y)` / `y = mask.unsqueeze(1) * y` `dist.all_reduce(y)` | 会合点:embedding 1 次、每层 2 次、末尾 gather 1 次 |
| ⑩ | 分店厨房 | `model_runner.py:61-66`,`:216-218` | `method_name, args = self.read_shm()` `self.call(method_name, *args)` / `temperatures = ... if self.rank == 0 else None` / `token_ids = ... if self.rank == 0 else None` | 不采样、不回传 |

### 图外但正文用到的证据

| 断言 | 位置 | 原样引用 |
|---|---|---|
| 桶表按 16 步进,最大批 512 | `model_runner.py:226, 234` | `max_bs = min(self.config.max_num_seqs, 512)` / `self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))` |
| 录制时工单指向静态缓冲切片 | `model_runner.py:240-243` | `set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])` `outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup` `with torch.cuda.graph(graph, self.graph_pool):` |
| 灶台三条断言 | `layers/attention.py:36-39` | `assert key.stride(-1) == 1 and value.stride(-1) == 1` `assert key.stride(1) == head_dim and value.stride(1) == head_dim` `assert k_cache.stride(1) == D and v_cache.stride(1) == D` `assert slot_mapping.numel() == N` |
| Triton 程序数等于 token 数 | `layers/attention.py:40` | `store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)` |
| 温度不能为 0 | `sampling_params.py:11` | `assert self.temperature > 1e-10, "greedy sampling is not permitted"` |
| 工单字段 | `utils/context.py:6-14` | `is_prefill: bool = False` `cu_seqlens_q: torch.Tensor \| None = None` … `block_tables: torch.Tensor \| None = None` |
| lm_head 不在图里 | `model_runner.py:198, 212, 241-243` | `self.model.compute_logits(self.model(input_ids, positions))` / `self.model.compute_logits(graph_vars["outputs"][:bs])` / 录制的只有 `self.model(...)` |

## 二、状态所有权行

| 状态 | 持有者 | 位置 | 原样引用 |
|---|---|---|---|
| 本轮工单 `_CONTEXT` | 厨房进程内全局,`run` 末尾清空 | `utils/context.py:16`;`model_runner.py:219` | `_CONTEXT = Context()` / `reset_context()` |
| `graph_vars` 六个静态缓冲 | 厨房 | `model_runner.py:250-257` | `self.graph_vars = dict(` `input_ids=input_ids, positions=positions, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, outputs=outputs,` |
| `kv_cache` 与每层视图 | 厨房建、灶台持视图 | `model_runner.py:115-121` | `module.k_cache = self.kv_cache[0, layer_id]` `module.v_cache = self.kv_cache[1, layer_id]` |
| `temperatures`、`token_ids` | 只在 rank 0 存在 | `model_runner.py:216, 218` | `if self.rank == 0 else None` |

## 三、设计取舍行

| 取舍 | 证据位置 | 原样引用 |
|---|---|---|
| prefill 不走图 | `model_runner.py:197` | `if is_prefill or self.enforce_eager or input_ids.size(0) > 512:` |
| 分桶回放与补齐 | `model_runner.py:202, 206-209` | `next(x for x in self.graph_bs if x >= bs)` / `graph_vars["slot_mapping"].fill_(-1)` `graph_vars["context_lens"].zero_()` |
| 工单用全局变量 | `utils/context.py:21-23`;`layers/attention.py:60` | `global _CONTEXT` / `context = get_context()` |
| 槽位 -1 哨兵 | `layers/attention.py:22-23` | `slot = tl.load(slot_mapping_ptr + idx)` `if slot == -1: return` |
| pinned 加 non_blocking,唯一同步点 | `model_runner.py:164-168, 218` | `.cuda(non_blocking=True)` / `.tolist()` |
| 指数竞赛采样 | `layers/sampler.py:11` | `probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)` |
| lm_head 不进图 | `model_runner.py:212, 233` | `return self.model.compute_logits(graph_vars["outputs"][:bs])` / `outputs = torch.zeros(max_bs, hf_config.hidden_size)` |

## 四、archify 校验回执

两张规格由子 agent 校验通过后落盘,主线程复验:`validate dataflow … --quality showcase --json` 均 `ok: true`,checks 9/9,`composition.summary {errors: 0, warnings: 0}`。图 a 9 节点,图 b 10 节点。deliver、export、visual-check 见主线程交付记录。

## 五、疑问与冲突

1. 页面里「桶表有洞」一条是按代码推断:`graph_bs` 到 `max_bs` 为止按 16 步进,`max_num_seqs` 不是 16 的倍数且大于 8 时,decode 批可能落在最大桶之外而 `next(...)` 抛 `StopIteration`;默认 512 安全,未实测。
2. 子 agent 在收尾阶段被基础设施停滞终止,其汇报里提到的两处收尾(一处副标题编辑与一处笔误)经主线程检查:两张规格校验通过,页面里未找到笔误,视为已完成。

# 04 KV 块与前缀缓存 · 证据表

基准:`projects/推理服务/nano-vllm/`,commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`,行号均对此 commit。图:`diagrams/_04a-kv-blocks.json`(账本流)、`diagrams/_04b-kv-blocks.json`(内容流)。本页所有结论只写当前状态,不写演进史。

## 一、图上元素对代码

### 图 a 节点(8 个)

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| ① | 空闲队列 · free_block_ids | 定义 `nanovllm/engine/block_manager.py:32`;出队 `:44`;入队 `:56`;中间取走 `:87` | `self.free_block_ids: deque[int] = deque(range(num_blocks))` / `block_id = self.free_block_ids.popleft()` / `self.free_block_ids.append(block_id)` | deque[int];主进程同步 |
| ② | 分配或重置 · _allocate_block | 定义 `block_manager.py:43-51`;调用点 `:91`(allocate 的新桌)与 `:108`(may_append) | `assert block.ref_count == 0` / `if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:` / `del self.hash_to_block_id[block.hash]` + `block.reset()` + `self.used_block_ids.add(block_id)` | 无参,返回 int 桌号;同步 |
| ② | reset 本体 | `block_manager.py:20-23` | `self.ref_count = 1` / `self.hash = -1` / `self.token_ids = []` | 无返回;同步 |
| ③ | 桌号单 · seq.block_table | 定义 `nanovllm/engine/sequence.py:28`;写入 `block_manager.py:89, 91, 101, 108` | `self.block_table = []` / `seq.block_table.append(block_id)` / `seq.block_table.append(self._allocate_block())` / `seq.block_table.clear()` | list[int];客人对象上的字段 |
| ④ | 盖章 · hash_blocks | 定义 `block_manager.py:110-120`;调用点 `nanovllm/engine/scheduler.py:83` | `start = seq.num_cached_tokens // self.block_size` / `end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size` / `if start == end: return` | (seq) → None;postprocess 里最先调用,同步 |
| ④ | 盖章的链式起点与登记 | `block_manager.py:114-120` | `h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1` / `block.update(h, token_ids)` / `self.hash_to_block_id[h] = block.block_id` | 后写覆盖同章条目 |
| ⑤ | 登记簿 · hash_to_block_id | 定义 `block_manager.py:31`;写 `:120`;读 `:65, 81`;唯一删除点 `:47-48` | `self.hash_to_block_id: dict[int, int] = dict()` / `block_id = self.hash_to_block_id.get(h, -1)` / `del self.hash_to_block_id[block.hash]` | dict[int, int] |
| ⑥ | 拼桌判定 · can_allocate | 定义 `block_manager.py:58-73`;调用点 `scheduler.py:36` | `for i in range(seq.num_blocks - 1):` / `if block_id == -1 or self.blocks[block_id].token_ids != token_ids:` + `break` / `if len(self.free_block_ids) < num_new_blocks: return -1` | (seq) → int 命中桌数或 −1;同步 |
| ⑥ | 在座命中才减新桌数 | `block_manager.py:68-70` | `num_cached_blocks += 1` / `if block_id in self.used_block_ids:` / `num_new_blocks -= 1` | 空出带章的命中桌不减 |
| ⑦ | 共享或复活 · allocate 命中分支 | 定义 `block_manager.py:75-92`;调用点 `scheduler.py:45` | `if block_id in self.used_block_ids: block.ref_count += 1` / `else: block.ref_count = 1` + `self.free_block_ids.remove(block_id)` + `self.used_block_ids.add(block_id)` / `seq.num_cached_tokens = num_cached_blocks * self.block_size` | (seq, num_cached_blocks) → None;同步 |
| ⑧ | 退桌 · deallocate | 定义 `block_manager.py:94-101`;调用点 `scheduler.py:78`(preempt)与 `:91`(结账) | `for block_id in reversed(seq.block_table):` / `block.ref_count -= 1` / `if block.ref_count == 0: self._deallocate_block(block_id)` | (seq) → None;同步 |
| ⑧ | _deallocate_block 不擦章 | `block_manager.py:53-56` | `assert self.blocks[block_id].ref_count == 0` / `self.used_block_ids.remove(block_id)` / `self.free_block_ids.append(block_id)`(函数体全文,无 hash、token_ids、登记簿操作) | 同步 |

### 图 a 边(9 条)

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| f1 | ① → ② popleft | `block_manager.py:44` | `block_id = self.free_block_ids.popleft()` | int |
| f2 | ② → ③ 桌号 | `block_manager.py:91`、`:108` | `seq.block_table.append(self._allocate_block())` | int 追加进 list |
| f3 | ③ → ④ 整桌 token | `block_manager.py:116-117`;`sequence.py:63-65` | `block = self.blocks[seq.block_table[i]]` / `token_ids = seq.block(i)` / `return self.token_ids[i*self.block_size: (i+1)*self.block_size]` | list[int],256 个 |
| f4 | ④ → ⑤ hash→桌号 | `block_manager.py:118-120` | `h = self.compute_hash(token_ids, h)` / `block.update(h, token_ids)` / `self.hash_to_block_id[h] = block.block_id` | dict 写入 |
| f5 | ⑤ → ⑥ 查簿比对 | `block_manager.py:63-66` | `token_ids = seq.block(i)` / `h = self.compute_hash(token_ids, h)` / `block_id = self.hash_to_block_id.get(h, -1)` | int 或 −1 |
| f6 | ⑥ → ⑦ 命中 n 桌 | `scheduler.py:36, 45` | `num_cached_blocks = self.block_manager.can_allocate(seq)` / `self.block_manager.allocate(seq, num_cached_blocks)` | int 经领班转交 |
| f7 | ⑦ → ③ 共享桌号 | `block_manager.py:81, 89` | `block_id = self.hash_to_block_id[h]` / `seq.block_table.append(block_id)` | 同一编号进第二张桌号单 |
| f8 | ③ → ⑧ 结账或请离席 | `scheduler.py:75-78`、`:89-91`;`block_manager.py:95` | `self.block_manager.deallocate(seq)`(两处) / `for block_id in reversed(seq.block_table):` | 倒序遍历 |
| f9 | ⑧ → ① 归零回队尾 | `block_manager.py:98-99, 56` | `if block.ref_count == 0:` / `self._deallocate_block(block_id)` / `self.free_block_ids.append(block_id)` | append 队尾 |

### 图 b 节点(7 个)

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| ① | 桌号单 · seq.block_table | 厨房读取 `nanovllm/engine/model_runner.py:124-125, 149, 154, 181` | `block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]` / `if not seq.block_table:    # warmup` / `slot_start = seq.block_table[i] * self.block_size` | list[int];TP>1 时随 Sequence 序列化到分店(`sequence.py:74`) |
| ② | 算槽位 · prepare_prefill / prepare_decode | `model_runner.py:151-161`(prefill)、`:181`(decode) | `slot_start = seq.block_table[i] * self.block_size` / `if i == start_block: slot_start += start % self.block_size` / `slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)` | list[int] → int32 张量,pinned 内存,`non_blocking` 拷贝(`:168, 184`),GPU 异步 |
| ② | 末桌的上界 | `model_runner.py:157-160` | `if i != end_block - 1: slot_end = seq.block_table[i] * self.block_size + self.block_size` / `else: slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size` | 半桌只展开到 end |
| ③ | 本轮工单 · Context | 定义 `nanovllm/utils/context.py:5-14`;写 `model_runner.py:169, 187`;清 `:219` | `slot_mapping: torch.Tensor \| None = None` / `set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)` / `reset_context()` | 厨房进程内全局 dataclass;每 step 重建 |
| — | 本轮算出的 K、V | `nanovllm/layers/attention.py:34-35, 59` | `N, num_heads, head_dim = key.shape` / `D = num_heads * head_dim` / `def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):` | 张量 [N, H_kv, d];由 `models/qwen3.py:86` `o = self.attn(q, k, v)` 传入 |
| ④ | 写桌 · store_kvcache + Triton kernel | 定义 `attention.py:10-40`;调用点 `:62-63` | `if k_cache.numel() and v_cache.numel():` / `store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)` / `store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)` | 网格 (N,),一 token 一程序;GPU 异步 |
| ④ | kernel 跳过 −1 与物理偏移 | `layers/attention.py:21-23, 28-30` | `slot = tl.load(slot_mapping_ptr + idx)` / `if slot == -1: return` / `cache_offsets = slot * D + tl.arange(0, D)` + `tl.store(k_cache_ptr + cache_offsets, key)` | 要求 `k_cache.stride(1) == D`(`:38`) |
| ⑤ | 大张量 · kv_cache 与每层视图 | `model_runner.py:112-121`;默认设备与 dtype `:29-30` | `block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize` / `self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)` / `module.k_cache = self.kv_cache[0, layer_id]` + `module.v_cache = self.kv_cache[1, layer_id]` | 桌数 `config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes`(`:113`);占位符 `attention.py:57` `self.k_cache = self.v_cache = torch.tensor([])` |
| ⑥ | 读桌 · flash-attn | `attention.py:64-74` | `if context.block_tables is not None:    # prefix cache` + `k, v = k_cache, v_cache` / `o = flash_attn_varlen_func(q, k, v, ... causal=True, block_table=context.block_tables)` / `o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context.context_lens, block_table=context.block_tables, ...)` | 返回 o;GPU 异步 |

### 图 b 边(7 条)

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| g1 | ① → ② 桌号列表 | `model_runner.py:153-154, 181` | `for i in range(start_block, end_block):` / `slot_start = seq.block_table[i] * self.block_size` | list[int] |
| g2 | ② → ③ slot_mapping | `model_runner.py:168-169`、`:184, 187` | `slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)` / `set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)` | int32 张量;GPU 异步拷贝 |
| g3 | ③ → ④ 槽位 | `attention.py:60, 63` | `context = get_context()` / `store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)` | 张量引用 |
| g4 | K、V → ④ 本轮 K、V | `attention.py:26-27, 33` | `key = tl.load(key_ptr + key_offsets)` / `value = tl.load(value_ptr + value_offsets)` / `def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):` | 每程序搬 D 个数 |
| g5 | ④ → ⑤ 按槽位写 | `attention.py:28-30` | `cache_offsets = slot * D + tl.arange(0, D)` / `tl.store(k_cache_ptr + cache_offsets, key)` / `tl.store(v_cache_ptr + cache_offsets, value)` | 写入本层视图 |
| g6 | ⑤ → ⑥ 整张 cache | `attention.py:61, 66, 72` | `k_cache, v_cache = self.k_cache, self.v_cache` / `k, v = k_cache, v_cache` / `o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,` | 整张视图传入,由 block_table 圈定 |
| g7 | ③ → ⑥ block_tables | `model_runner.py:162-163, 186`;`attention.py:70, 73` | `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache` + `block_tables = self.prepare_block_tables(seqs)` / `block_tables = self.prepare_block_tables(seqs)` / `cache_seqlens=context.context_lens, block_table=context.block_tables,` | int32 矩阵,右填 −1(`:125`) |

### 卡片与走读表里的断言

| 编号 | 图上元素 | 调用点与定义点 | 原样引用 | 参数与返回类型;同步或异步 |
|---|---|---|---|---|
| 卡 a1 | 「桌 0 第 2 步 ref_count 加到 2、第 10 步 remove 复活」 | 干跑第 2、10 步;`block_manager.py:83-88` | 干跑输出 `allocate(B): 命中桌 0 在座 ref+1 -> ref=2` / `allocate(B): 命中桌 0 空出带章 free.remove 复活 -> ref=1` | 见第六节 |
| 卡 a2 | 「桌 7 从未带章;删登记只在 popleft 到带章的桌时触发」 | 干跑第 2、3 步与补充场景;`block_manager.py:47-48` | 干跑输出 `_allocate_block: popleft 桌 7(无章,不删) reset ref=1` / 补充场景 `_allocate_block: popleft 桌 6 并删登记(:47-48) reset ref=1` | 见第六节 |
| 卡 a3 | 「退桌倒序遍历,前缀桌最晚入队」 | `block_manager.py:95`;干跑第 9 步 | `for block_id in reversed(seq.block_table):` / 干跑输出 `free_block_ids=[2, 1, 0, 7, 6, 5, 4, 3]` | — |
| 卡 b1 | 「B 的 50 个 token 落在槽 1792..1841;C 第 1025 个 token 落在槽 1792」 | 按 `model_runner.py:151-161, 181` 复算;干跑第 2、3 步 | 干跑输出 `B 写槽 1792..1841(桌 [7],50 个)` / `C 写槽 1792(桌 7 偏移 0),context_len=1025` | 厨房不能干跑,算式逐行复刻 |
| 卡 b2 | 「slot 为 −1 的行用于 CUDA graph 回放的空批槽」 | `model_runner.py:206-207`;`attention.py:23` | `graph_vars["slot_mapping"].fill_(-1)` / `graph_vars["slot_mapping"][:bs] = context.slot_mapping` / `if slot == -1: return` | — |
| 卡 b3 | 「warmup 时不写桌」 | `attention.py:57, 62`;`model_runner.py:34-35, 149-150` | `self.k_cache = self.v_cache = torch.tensor([])` / `if k_cache.numel() and v_cache.numel():` / `self.warmup_model()` 先于 `self.allocate_kv_cache()` | — |
| 走读 4 | 「第 2 步 C 也改读整张 cache」 | `model_runner.py:162-163`;`attention.py:65-66` | `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:` / `if context.block_tables is not None:` + `k, v = k_cache, v_cache` | 干跑输出 `cu_seqlens_k[-1]=1586 > cu_seqlens_q[-1]=1074 -> block_tables [[3, 4, 5, 6], [0, 1, 7, -1]]` |
| 走读公式 | 链式哈希的字节拼法 | `block_manager.py:37-41` | `h = xxhash.xxh64()` / `if prefix != -1: h.update(prefix.to_bytes(8, "little"))` / `h.update(np.array(token_ids).tobytes())` + `return h.intdigest()` | 本机 numpy 2.5.3 下 `np.array(int)` 为 int64,每桌 2048 字节 |

## 二、状态所有权行

| 状态 | 持有者 | 位置 | 原样引用 |
|---|---|---|---|
| ref_count | Block(座位管理员的卡片) | 写 `block_manager.py:12, 21, 84, 86, 97`;读 `:46, 54, 98` | `self.ref_count = 0` / `block.ref_count += 1` / `block.ref_count -= 1` |
| hash | Block | 写 `block_manager.py:13, 17, 22`;读 `:47, 114` | `self.hash = -1` / `self.hash = hash` / `h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1` |
| token_ids 副本 | Block | 写 `block_manager.py:14, 18, 23`;读 `:66` | `self.token_ids = []` / `self.token_ids = token_ids` / `self.blocks[block_id].token_ids != token_ids` |
| free_block_ids | BlockManager | `block_manager.py:32, 44, 56, 87`;读 `:71, 104` | `deque(range(num_blocks))` / `popleft()` / `append(block_id)` / `remove(block_id)` |
| used_block_ids | BlockManager | `block_manager.py:33, 50, 55, 88`;读 `:69, 83` | `self.used_block_ids: set[int] = set()` / `self.used_block_ids.add(block_id)` / `self.used_block_ids.remove(block_id)` |
| hash_to_block_id | BlockManager | 写 `block_manager.py:120`;删 `:48`;读 `:47, 65, 81` | `self.hash_to_block_id[h] = block.block_id` / `del self.hash_to_block_id[block.hash]` |
| block_table | Sequence | 写 `sequence.py:28`;`block_manager.py:89, 91, 101, 108`;读 `scheduler.py:35, 44`;`model_runner.py:124-125, 149-160, 181` | `seq.block_table.append(block_id)` / `seq.block_table.clear()` |
| num_cached_tokens | Sequence | 写 `sequence.py:25`;`block_manager.py:92, 100`;`scheduler.py:84`;读 `block_manager.py:111-112`;`model_runner.py:139` | `seq.num_cached_tokens = num_cached_blocks * self.block_size` / `seq.num_cached_tokens = 0` / `start = seq.num_cached_tokens` |
| block_size(桌容量) | Config → Sequence 类属性 / BlockManager / ModelRunner | `config.py:17, 22`;`llm_engine.py:21`;`sequence.py:15`;`block_manager.py:29`;`model_runner.py:20` | `kvcache_block_size: int = 256` / `assert self.kvcache_block_size % 256 == 0` / `Sequence.block_size = config.kvcache_block_size` |
| num_kvcache_blocks(桌数) | 厨房算出写回 Config,领班读 | `config.py:18`;`model_runner.py:113-114`;`scheduler.py:15` | `num_kvcache_blocks: int = -1` / `config.num_kvcache_blocks = int(...) // block_bytes` / `BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)` |
| kv_cache 大张量 | ModelRunner(每个厨房进程各一份) | `model_runner.py:115` | `self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)` |
| k_cache / v_cache 视图 | Attention(每层一对) | `attention.py:57`;`model_runner.py:117-121` | `self.k_cache = self.v_cache = torch.tensor([])` / `module.k_cache = self.kv_cache[0, layer_id]` |
| slot_mapping / block_tables / context_lens | Context(本轮工单) | 写 `model_runner.py:169, 187`;清 `:219`;读 `attention.py:63, 70, 73`;graph 备份 `model_runner.py:206-210, 250-255` | `set_context(...)` / `reset_context()` / `graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables` |

「登记簿只增不删,唯一的删除点在重新分配时」的证明链:`hash_to_block_id` 全文只出现在 `block_manager.py:31, 47, 48, 65, 81, 120` 六处,`del` 只在 `:48`,且被 `:47` 的两个条件守着。

「共享桌不会再被写」的证明链:拼桌命中只在 `allocate` 的前 `num_cached_blocks` 桌(`:78-89`),而 `can_allocate` 只查前 `num_blocks - 1` 桌(`:62`),所以最后一桌与 `may_append` 追加的桌(`:108`)都来自 `_allocate_block`,ref_count 恒为 1;prefill 写入从 `num_cached_tokens` 起(`model_runner.py:139`),decode 只写 `block_table[-1]`(`:181`),都落不到共享桌上。

## 三、设计取舍行

| 取舍 | 证据位置 | 原样引用 |
|---|---|---|
| 链式哈希而不是单块哈希 | `block_manager.py:38-39, 64, 80, 114-118` | `if prefix != -1: h.update(prefix.to_bytes(8, "little"))` / `h = self.compute_hash(token_ids, h)`(h 在循环里滚动) / `h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1` |
| 整桌才参与拼桌;最后一桌永远不查 | `block_manager.py:62, 111-113`;`scheduler.py:39` | `for i in range(seq.num_blocks - 1):` / `if start == end: return` / `num_tokens = seq.num_tokens - num_cached_blocks * self.block_size`(据此至少算 1 个 token,原因为推断,代码无注释) |
| 退桌惰性失效而不是立刻擦章 | `block_manager.py:53-56` 对比 `:47-49` | `_deallocate_block` 函数体三行无 hash 操作 / `if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id: del self.hash_to_block_id[block.hash]` |
| 复活的代价:线性 remove | `block_manager.py:87` | `self.free_block_ids.remove(block_id)` |
| 先进先出回收而不是 LRU 结构 | `block_manager.py:32, 44, 56, 95` | `deque(range(num_blocks))` / `popleft()` / `append(block_id)` / `for block_id in reversed(seq.block_table):`(倒序还桌,前缀桌最晚入队) |
| ref_count 共享而不是复制 | `block_manager.py:83-84, 97-99`;`:46, 54` | `if block_id in self.used_block_ids: block.ref_count += 1` / `if block.ref_count == 0: self._deallocate_block(block_id)` / 两处 `assert ... ref_count == 0` |
| 不需要 copy-on-write | 见第二节「共享桌不会再被写」 | — |
| 哈希只当索引,命中还要比对整桌 token | `block_manager.py:66` | `if block_id == -1 or self.blocks[block_id].token_ids != token_ids:` |
| 整批一起改读 cache,而不是逐条判断 | `model_runner.py:162-163`;`attention.py:65-66` | `if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache` / `k, v = k_cache, v_cache` |

## 四、archify 校验回执

- 命令:`node ~/.claude/skills/archify/bin/archify.mjs validate dataflow opensource/推理服务/nano-vllm/diagrams/_04a-kv-blocks.json --quality showcase --json` 与同命令对 `_04b-kv-blocks.json`。
- 结果(两图相同):`ok: true`;checks 全过:single_svg、finite_svg、orthogonal_arrows、label_route_clearance、relationship_crossings、relationship_corridors、container_border_runs、route_rhythm、legend_clearance;`composition.status: pass`,`composition.summary: {errors: 0, warnings: 0}`。
- 图 a 修复轮次:
  1. 初稿(空闲队列与分配同在 row 0、桌号单在其下、退桌在 row 2、复活→桌号单走顶部通道):6 条错误,包括顶部通道下落时穿过「分配」节点、桌号单→退桌的横线穿过「盖章」、3 个标签压在节点上、1 个 7px 微段。
  2. 改布局:空闲队列 (0,0)、分配 (0,1)、桌号单 (1,1)、盖章 (2,1)、登记簿 (2,0)、退桌 (3,2)、拼桌判定 (4,1)、复活 (4,0);桌号单→退桌用 `fromSide: bottom` + `via [[315, 385]]`;退桌→空闲队列用 `via [[745, 460], [20, 460], [20, 157]]` 从左侧进;竖向边标签用 labelAt 放到行间:剩 2 条(「hash→桌号」与「查簿比对」标签互压,且离登记簿→判定的竖线 1.9px)。
  3. 「hash→桌号」移到竖线左侧 [470, 214],「查簿比对」移到横段上方 [770, 250]:通过。
- 图 b 修复轮次:
  1. 初稿:工单→读桌走 vertical-channel,横穿「大张量」节点并与写桌→大张量共用 59px 走廊,连带 3 条标签间距错误。
  2. 工单→读桌改 `route: top-channel`,「槽位」标签移到竖线左侧 [470, 214]:通过。
- 几何备忘(供后续页):节点 112×58,列距 215(stage 0..4 的 x 起点 44、259、474、689、904),行距 114(row 0..2 的 y 起点 128、242、356),行间标签放 y=214 附近可避开节点;viewBox 1080 宽在 1440 桌面下缩放 0.86,副标题超过约 14 个汉字宽会被缩到 6px 以下触发 desktop-readability 错误(上一版草稿的 `_allocate_block · 删旧登记` 即因此不过)。
- 只做了 validate,未做 deliver、preview、visual-check。

## 五、疑问与冲突

1. **03 页与 03 证据表把「空出带章的命中桌不减新桌数」称为「保守」**,本页认为这是精确计数:复活时该桌会从空闲队列里 `remove`(`:87`),确实消耗一张空桌,所以 `len(free_block_ids) < num_new_blocks` 的比较恰好等于「复活之后剩下的空桌够不够新桌」。建议 03 页第三节末段与 `_03-evidence.md` 第三节最后一行把「(保守)」改成「(精确:复活也要占一张空桌)」;底稿第十二节第 5 条的措辞本身没错,可不动。
2. **底稿示例乙写「第 4 到 8 步 B 在等位队首查桌位一直回 −1」**,干跑里第 9 步的查询也回 −1(查询发生在 A、C 结账之前),共 6 次;03 证据表写的是「第 4–9 步」。建议底稿改成「第 4 到 9 步」。
3. **示例乙里 ② 的「删登记」分支不触发**:桌 7 从未盖章,桌 2 也从未写满。本页用补充干跑(D 在第 16 步之后到店)展示该分支,已在页面与卡片里注明是补充场景,不是示例乙的一部分。
4. **「最后一桌不参与拼桌」的原因是推断**:代码与注释都没说。页面第七节明确写了「代码没写原因,从行为推断」。
5. **`np.array(token_ids)` 的 dtype 取决于平台**(本机 numpy 2.5.3 为 int64,每桌 2048 字节);不同平台哈希值不同,但同机内命中逻辑不受影响。页面公式写的 int64 以本机为准。
6. **TP>1 时每个分店各自算一次 `num_kvcache_blocks`**(`model_runner.py:113` 在每个进程里执行,config 是 pickle 副本),若各卡剩余显存不同,分店的桌数可能与 rank 0 不一致,而领班只用 rank 0 写回的数;代码没有校验。属 06 页范围,本页未展开。
7. **`Block.token_ids` 保存的是切片副本**(`sequence.py:65` 切片产生新 list),每张带章的桌在 CPU 上多占 256 个 int;本页未画。
8. **上一版未完成的草稿**:开工时 `diagrams/` 下已有 `_04a-kv-blocks.json`、`_04b-kv-blocks.json`(14:50、14:53,无对应页面与证据表,图 a 校验不过),判断为本卡早先中断的产物,已按本卡文件名覆盖;`/tmp/nanovllm04_dryrun.py` 也是早先版本,本次另写 `/tmp/nanovllm04_dryrun_v2.py`,未使用旧脚本。

## 六、干跑记录

- 脚本:`/tmp/nanovllm04_dryrun_v2.py`,不进仓库。复用 03 的打桩方案(`importlib` 直接加载 `sequence.py`、`block_manager.py`、`scheduler.py`、`sampling_params.py`,桩掉 `nanovllm/__init__.py` 与 `nanovllm.config`),并按底稿第十二节第 1 条用 `uv run --with xxhash --with numpy python /tmp/nanovllm04_dryrun_v2.py` 以真实 xxhash 4.0.1 与 numpy 2.5.3 运行;脚本保留了 blake2b/struct 桩作为无网络时的退路,本次未用到。
- 探针:包装 `_allocate_block`、`_deallocate_block`、`can_allocate`、`allocate`、`deallocate`、`hash_blocks`、`may_append`、`preempt`,每步打印桌 0、桌 7 的 ref_count、是否带章、在空闲队列第几位、被谁持有、登记簿是否指向自己;槽位与 block_tables 按 `model_runner.py:123-127, 151-163, 181` 的算式在 schedule 之后、postprocess 之前复算(厨房依赖 torch 与 GPU,不能运行)。
- 店规与客人:`SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=16384, eos=151645, kvcache_block_size=256, num_kvcache_blocks=8)`;A = 512 个共享 token(1000..1511)+ 100 个、B = 同样 512 个 + 50 个、C = 1024 个不同的 token;max_tokens=8;厨房每轮给每位客人返回 `9000 + step`。
- 主线轨迹与 03 页第三节的表逐步一致:第 1 步 A 得 [0, 1, 2],盖章 0、1;第 2 步 C 得 [3, 4, 5, 6] 并盖章四桌,B 命中 2 桌得 [0, 1, 7]、只算 50 个、不盖章;第 3 步 preempt(B) 收回 [0, 1, 7](桌 0、1 的 ref_count 2→1,桌 7 归零),C 的 may_append 拿到桌 7;第 4 到 9 步 can_allocate(B) = −1;第 9 步 A、C 结账后空闲队列为 [2, 1, 0, 7, 6, 5, 4, 3],登记簿仍有 6 条;第 10 步 can_allocate(B) = 2,桌 0、1 走 free.remove 复活,popleft 得桌 2,block_table=[0, 1, 2],num_cached_tokens=512,本轮 51 个;第 16 步 B 结账,共 16 步。
- 内容流复算:第 1 步 A 写槽 0..611,不建 block_tables;第 2 步 C 写槽 768..1791、B 写槽 1792..1841,cu_seqlens_k 1586 > cu_seqlens_q 1074,block_tables = [[3, 4, 5, 6], [0, 1, 7, −1]];第 3 步 A 写槽 612、C 写槽 1792;第 10 步 B 写槽 512..562,block_tables = [[0, 1, 2]];第 11 到 16 步 B 写槽 563..568。
- 补充场景:主线结束后 D(300 个不同 token,max_tokens=1)到店,空闲队列 [7, 6, 5, 4, 3, 2, 1, 0]:popleft 桌 7(无章,不删)、popleft 桌 6 并触发 `:47-48` 删登记;D 的槽位为 1792..2047 与 1536..1579,说明槽位顺序跟着桌号单走,不必单调。
- 哈希核对:A 与 B 第一桌的章都是 `0xa6e9cbce06c61a00`,第二桌带前缀的章都是 `0x2228cf3136cedbb4`;同一桌 token 不带前缀算出 `0xc408cf50addef910`,与带前缀的不同。数值依赖上面的合成 token 与本机 int64,只用于说明「相等才命中」。

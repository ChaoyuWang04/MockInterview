# 03 请求状态机 · 证据表

基准:`projects/推理服务/nano-vllm/`,commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`,行号均对此 commit。图:`diagrams/_03-request-lifecycle.json`。

## 一、图上元素对代码

### 状态(5 个)

| 图上 ID | 图上名 | 位置 | 原样引用 | 说明 |
|---|---|---|---|---|
| queued | ①等位 · WAITING · block_table 为空 | `nanovllm/engine/sequence.py:20,28` | `self.status = SequenceStatus.WAITING` / `self.block_table = []` | 建单即此状态 |
| queued | 同上,进队尾 | `nanovllm/engine/scheduler.py:22-23` | `def add(self, seq: Sequence):` / `self.waiting.append(seq)` | 新客排队尾 |
| chunking | ②大桌分批上菜中 · WAITING · 已有 block_table | `nanovllm/engine/scheduler.py:44-46` | `if not seq.block_table:` / `self.block_manager.allocate(seq, num_cached_blocks)` / `seq.num_scheduled_tokens = min(num_tokens, remaining)` | 分桌后若 :48 条件不成立,status 不变、仍在 waiting[0] |
| running | ③在座 · RUNNING | `nanovllm/engine/scheduler.py:48-51` | `if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:` / `seq.status = SequenceStatus.RUNNING` / `self.waiting.popleft()` + `self.running.append(seq)` | 唯一置 RUNNING 的地方 |
| running | 副标题「每轮上一道菜」 | `nanovllm/engine/scheduler.py:67-68` | `seq.num_scheduled_tokens = 1` / `seq.is_prefill = False` | decode 每轮 1 个 token |
| finished | ④结账 · FINISHED · 桌位全部收回 | `nanovllm/engine/scheduler.py:89-92` | `if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:` / `seq.status = SequenceStatus.FINISHED` / `self.block_manager.deallocate(seq)` + `self.running.remove(seq)` | 唯一终态 |
| preempted | 请离席 · preempt · 收桌位,回队首 | `nanovllm/engine/scheduler.py:75-79` | `seq.status = SequenceStatus.WAITING` / `seq.is_prefill = True` / `self.block_manager.deallocate(seq)` + `self.waiting.appendleft(seq)` | 瞬时状态,画成可恢复失败 |

### 迁移(6 条,另 2 条未画)

| 图上 ID | 从 → 到 | 触发事件 | 位置 | 原样引用 |
|---|---|---|---|---|
| t-seat-whole | 等位 → 在座 | 领班叫号,拼桌查询非 -1,剩余预算 ≥ 要算的 token 数 | `scheduler.py:35-39,44-51` | `num_cached_blocks = self.block_manager.can_allocate(seq)` / `num_tokens = seq.num_tokens - num_cached_blocks * self.block_size` / `seq.status = SequenceStatus.RUNNING` |
| t-seat-chunk | 等位 → 分批上菜中 | 预算不够整份且本轮还没叫到别人 | `scheduler.py:42-43,46` | `if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq` / `break` / `seq.num_scheduled_tokens = min(num_tokens, remaining)` |
| t-chunk-done | 分批上菜中 → 在座 | 最后一段的 num_cached + num_scheduled == num_tokens | `scheduler.py:40-41,48` | `else:` / `num_tokens = seq.num_tokens - seq.num_cached_tokens` / `if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:` |
| t-finish | 在座 → 结账 | append_token 后出了 eos(未 ignore)或生成数 == max_tokens | `scheduler.py:88-92` | `seq.append_token(token_id)` / `if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:` / `seq.status = SequenceStatus.FINISHED` |
| t-preempt | 在座 → 请离席 | decode 分支里 can_append 为假:先抢 running 尾,空了抢自己 | `scheduler.py:59-65` | `seq = self.running.popleft()` / `while not self.block_manager.can_append(seq):` / `if self.running: self.preempt(self.running.pop()) else: self.preempt(seq) break` |
| t-preempt | 触发条件本体 | 长度除以 256 余 1 且没有空桌 | `block_manager.py:103-104` | `def can_append(self, seq: Sequence) -> bool:` / `return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)` |
| t-requeue | 请离席 → 等位 | preempt 本体:回队首、重置标记、收桌位 | `scheduler.py:75-79` | `seq.status = SequenceStatus.WAITING` / `seq.is_prefill = True` / `self.waiting.appendleft(seq)` |
| t-requeue | 收桌位细节 | 逆序减引用,归零回空闲队列,计数归零,桌号单清空 | `block_manager.py:94-101` | `block.ref_count -= 1` / `seq.num_cached_tokens = 0` / `seq.block_table.clear()` |
| 未画:decode 自环 | 在座 → 在座 | 每轮 decode 出 1 个 token,未结账 | `scheduler.py:66-70,84-88` | `seq.num_scheduled_tokens = 1` / `seq.num_cached_tokens += seq.num_scheduled_tokens` / `seq.append_token(token_id)` |
| 未画:等位自等 | 等位 → 等位 | 拼桌查询回 -1,留在队首,本轮转 decode | `scheduler.py:36-38` | `num_cached_blocks = self.block_manager.can_allocate(seq)` / `if num_cached_blocks == -1:` / `break` |

### 卡片与副标题里的断言

| 断言 | 位置 | 原样引用 |
|---|---|---|
| 「分批只许队首」 | `scheduler.py:42` | `if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq` |
| 「deallocate 不清哈希」 | `block_manager.py:53-56` | `def _deallocate_block(self, block_id: int):` / `self.used_block_ids.remove(block_id)` / `self.free_block_ids.append(block_id)`(无 hash 与 token_ids 重置) |
| 「只在被别人重置时才丢章」 | `block_manager.py:47-49` | `if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:` / `del self.hash_to_block_id[block.hash]` / `block.reset()` |
| 「空出但有章的桌回来时能复活」 | `block_manager.py:83-88` | `if block_id in self.used_block_ids: block.ref_count += 1` / `else: block.ref_count = 1` / `self.free_block_ids.remove(block_id)` |
| 「留:token_ids、num_tokens、seq_id、采样参数」 | `scheduler.py:75-79`,`block_manager.py:94-101` | preempt 与 deallocate 只改 status、is_prefill、num_cached_tokens、block_table 与队列,未触及 token_ids |
| 「B 回来命中 2 桌只重算 51;C 自抢自回来只重算 1」 | 干跑场景一第 10 步、场景二第 10 步 | 见第六节 |

## 二、状态所有权行

| 字段 | 写入点(全部) | 引用 | 读取点 |
|---|---|---|---|
| status | `sequence.py:20`;`scheduler.py:49,76,90` | `self.status = SequenceStatus.WAITING` / `seq.status = SequenceStatus.RUNNING` / `seq.status = SequenceStatus.FINISHED` | `sequence.py:41` `return self.status == SequenceStatus.FINISHED`;`llm_engine.py:54` `if seq.is_finished` |
| is_prefill | `sequence.py:27`;`scheduler.py:68,77` | `self.is_prefill = True` / `seq.is_prefill = False` / `seq.is_prefill = True` | `sequence.py:73` `last_state = self.last_token if not self.is_prefill else self.token_ids` |
| num_cached_tokens | `sequence.py:25`;`block_manager.py:92,100`;`scheduler.py:84` | `seq.num_cached_tokens = num_cached_blocks * self.block_size` / `seq.num_cached_tokens = 0` / `seq.num_cached_tokens += seq.num_scheduled_tokens` | `scheduler.py:41,48,86`;`block_manager.py:111-112`;`model_runner.py:139` `start = seq.num_cached_tokens` |
| num_scheduled_tokens | `sequence.py:26`;`scheduler.py:46,67,85`;`model_runner.py:99`(warmup 的临时客人) | `seq.num_scheduled_tokens = min(num_tokens, remaining)` / `seq.num_scheduled_tokens = 1` / `seq.num_scheduled_tokens = 0` | `scheduler.py:47-48,84`;`block_manager.py:112`;`model_runner.py:140`;`llm_engine.py:51` |
| block_table | `sequence.py:28`;`block_manager.py:89,91,101,108` | `seq.block_table.append(block_id)` / `seq.block_table.append(self._allocate_block())` / `seq.block_table.clear()` | `scheduler.py:35,44`;`block_manager.py:95,114,116`;`model_runner.py:124-125,149-160,181` |
| token_ids / num_tokens / last_token | `sequence.py:21-23,67-70` | `def append_token(self, token_id: int):` / `self.token_ids.append(token_id)` / `self.num_tokens += 1` | `block_manager.py:63,79,117`(`seq.block(i)`);`model_runner.py:143,178` |
| 队列归属 | `scheduler.py:23,50-51,79,92` | `self.waiting.append(seq)` / `self.waiting.popleft()` + `self.running.append(seq)` / `self.waiting.appendleft(seq)` / `self.running.remove(seq)` | `scheduler.py:19-20,30-31,58-59,72` |
| eos 的来源 | `llm_engine.py:33`;`scheduler.py:13` | `config.eos = self.tokenizer.eos_token_id` / `self.eos = config.eos` | `scheduler.py:89` |
| max_tokens / ignore_eos 默认 | `sampling_params.py:7-8` | `max_tokens: int = 64` / `ignore_eos: bool = False` | `sequence.py:30-31` |

「只有 running 里的会被抢」的证明链:preempt 仅两处调用(`scheduler.py:62,64`),参数分别是 `self.running.pop()` 与从 `self.running.popleft()` 取出的 seq;decode 分支的入口条件是 prefill 分支没叫到人(`scheduler.py:54-55` `if scheduled_seqs: return scheduled_seqs, True`);分批中的客人在 waiting[0] 且 `num_tokens - num_cached_tokens > 0`、`remaining ≥ 1`,prefill 分支必定把他放进 scheduled_seqs(`scheduler.py:41,46,52`),于是不会进入 decode 分支。

## 三、设计取舍行

| 取舍 | 证据 | 原样引用 |
|---|---|---|
| 只有三个枚举值,细分状态靠字段推断 | `sequence.py:8-11`;`scheduler.py:35,40`;`sequence.py:44-45` | `class SequenceStatus(Enum): WAITING = auto() RUNNING = auto() FINISHED = auto()` / `if not seq.block_table:` / `return self.num_tokens - self.num_prompt_tokens` |
| 请离席回队首,新客进队尾 | `scheduler.py:79` 对比 `:23` | `self.waiting.appendleft(seq)` 对比 `self.waiting.append(seq)` |
| 队首堵车的代价 | `scheduler.py:31,36-38` | `seq = self.waiting[0]` / `if num_cached_blocks == -1:` / `break` |
| 重算而非保存进度:preempt 不搬 K/V | `scheduler.py:75-79` | 函数体只有四行,无任何张量操作 |
| 先抢最晚入座的 | `scheduler.py:62` | `self.preempt(self.running.pop())` |
| 收桌位不擦章,回座可复活 | `block_manager.py:53-56,83-88` | 见第一节「卡片与副标题里的断言」 |
| 复活的桌仍按新桌计入空桌需求(精确计数:复活要 free.remove,确实占一张空桌,见 04 页) | `block_manager.py:66-72` | `num_cached_blocks += 1` / `if block_id in self.used_block_ids: num_new_blocks -= 1` / `if len(self.free_block_ids) < num_new_blocks: return -1` |

## 四、archify 校验回执

- 命令:`node ~/.claude/skills/archify/bin/archify.mjs validate lifecycle opensource/推理服务/nano-vllm/diagrams/_03-request-lifecycle.json --quality showcase --json`
- 结果:`ok: true`;checks 全过:single_svg、finite_svg、orthogonal_arrows、label_route_clearance、relationship_crossings、relationship_corridors、container_border_runs、route_rhythm、legend_clearance;`composition.status: pass`,`composition.summary: {errors: 0, warnings: 0}`。
- 修复轮次:
  1. 初稿(5 主态 + 决策节点 + 8 条迁移,标签放边上):13 条 layout/constraint 错误,全是主带相邻节点之间 36px 装不下标签。
  2. 去掉决策节点(4 主态)、标签用 labelAt 放到主带下方 y=206、请离席标签放到竖线中段 y=240:剩 1 条(结账标签贴着顶部通道竖线 x=402)。
  3. 结账标签移到 [485,206];同时试着给每条迁移加 note 记录 file:line:错误反弹到 12 条,原因是 note 会作为可见文字渲染、标签矩形随之膨胀。去掉全部 note、把在座副标题缩短到 110px 内:通过。
- 附带试验:`from == to` 的自环被 clean-flow/endpoint-side-direction 拒绝;`viewBox` 不改变列距(节点 118px 宽、列距 154px、主带 y=126–188、打断带 y=278–336、顶部通道 y=98);`--layout-json` 对 lifecycle 不可用,几何信息只能从 diagnostics 里的 rect 读。
- 只做了 validate,未做 deliver、preview、visual-check。

## 五、疑问与冲突

1. 底稿说 lifecycle 有「事件带」与「终态带」,本图只用两条带:主带(含唯一终态 FINISHED,画成 success)与打断带(preempt)。代码里没有第二个终态(无取消、超时、报错),单独开一条终态带会是空的;若主线程坚持三带,可把 FINISHED 移到终态带 col 2,但主路径就断在在座。
2. 任务卡要求画出的四个状态与 preempt 都在;为了标签能放下,去掉了「出菜后查单」决策节点,decode 自环也画不出(工具拒绝自环),两者都改写进节点副标题与正文。
3. 示例乙按底稿字面(A、B 前 512 相同)不一定命中拼桌:章在 postprocess 才盖,A、B 若同一轮叫号则 B 命中 0 桌。本页把到店顺序定为「A 先一轮,C、B 随后且 C 在前」,并把三人 max_tokens 定为 8(底稿未定);04 页若也用示例乙,需与此对齐。
4. 底稿比喻表写 preempt「把队尾客人的桌位收回」,精确说法是 running 名单尾(最晚入座者),名单只剩自己时收回自己的;建议补「或自己」。
5. 未兜住的角落:`scheduler.py:63-65` 请走自己后 `break`,若本轮无人可叫,`scheduler.py:71` `assert scheduled_seqs` 触发。只在单条序列超过全部桌位容量时出现。
6. RUNNING 且 is_prefill 仍为 True 的客人(刚入座、还没 decode 过)会被 preempt,且在 TP>1 时会按全量 token_ids 序列化;01/05 页若讲序列化瘦身,应注明这一点。
7. `embed_head.py:58` 也读 `context.is_prefill`,那是本轮工单的字段,不是客人的 is_prefill;本页未画。

## 六、干跑记录

- 脚本:`/tmp/nanovllm03_dryrun.py`(逐步打印)与 `/tmp/nanovllm03_table.py`(压缩成表),不进仓库。用 `importlib` 直接加载 `nanovllm/engine/sequence.py`、`block_manager.py`、`scheduler.py` 与 `sampling_params.py`,绕开会 import torch 的包入口;config 用 `types.SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=16384, eos=151645, kvcache_block_size=256, num_kvcache_blocks=8)`。
- 本机三个 Python(3.14、3.12、3.11)都没有 xxhash 与 numpy,未安装;用 `hashlib.blake2b(digest_size=8)` 桩替 `xxhash.xxh64`,用 `struct.pack("<{n}q")` 桩替 `numpy.array(...).tobytes()`。桩只改变哈希数值,不改变「相等才命中」的逻辑,轨迹与手算一致。
- 厨房是假的:每轮给每位被叫到的客人返回 `9000 + step`,永远不等于 eos;场景四另行验证 eos 与 ignore_eos。
- 场景一(主线,页面第三节的表):A 第 1 步到店;C、B 第 2 步到店(C 在前);共 16 步;第 3 步 preempt(B) 收回 [0, 1, 7];第 4–9 步 can_allocate(B) = -1;第 10 步 can_allocate(B) = 2、allocate 后 block_table=[0, 1, 2]、num_cached_tokens=512、本轮 51 个 token。
- 场景二:B 在 C 前到店;第 3 步 preempt(C) 收回 [4, 5, 6, 7];第 4–9 步 can_allocate(C) = -1(需 5 张,空 4 张);第 10 步 can_allocate(C) = 4、block_table=[4, 5, 6, 7, 2]、num_cached_tokens=1024、本轮 1 个 token;共 16 步。
- 场景三(max_num_batched_tokens=512,max_tokens=2):第 1 步后 C 为 WAITING、block_table=[0, 1, 2, 3]、num_cached_tokens=512、仍在 waiting[0];第 2 步转 RUNNING;第 3–4 步 A 分批(512 + 100),C 两轮未 decode;第 5 步两人 decode 并结账;共 5 步。
- 场景四:D(ignore_eos=False)第 2 步收到 eos 即结账,completion=2;E(ignore_eos=True)跑满 max_tokens=8。

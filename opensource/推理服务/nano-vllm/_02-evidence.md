# 02 调度决策 · 证据表

源码基准:commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`(本机核对 `git rev-parse HEAD` 一致,工作树干净)。路径均相对 `projects/推理服务/nano-vllm/`。

## 一、图上元素对代码

### 图 a `diagrams/_02a-scheduling.json`(12 节点、13 边、3 泳道、2 组)

泳道按阶段而不是按角色:这张图回答的是「先走哪条分支、什么时候换分支」,prefill / decode / 抢占三条泳道让「prefill 优先」和「抢占是 decode 的异常出口」一眼可见;座位管理员的四个调用都以双名制写进节点副标题,各自仍有证据行。若按角色分,prefill 与 decode 的十个节点会挤在同一条「领班」泳道里,6 列放不下。

| 元素 | 图上文字 | 路径:行 | 原样引用 |
|---|---|---|---|
| lane prefill | 领班 · prefill 分支(看 waiting 队首) | `nanovllm/engine/scheduler.py:29-31` | `# prefill` / `while self.waiting and len(scheduled_seqs) < self.max_num_seqs:` / `seq = self.waiting[0]` |
| lane decode | 领班 · decode 分支(轮 running) | `scheduler.py:57-59` | `# decode` / `while self.running and len(scheduled_seqs) < self.max_num_seqs:` / `seq = self.running.popleft()` |
| lane preempt | 抢占 · preempt(请离席重排) | `scheduler.py:75-79` | `def preempt(self, seq: Sequence):` … `self.waiting.appendleft(seq)` |
| group g_prefill_loop | prefill 循环:每位 waiting 客人 | `scheduler.py:30-52` | 同 lane prefill;循环体到 `scheduled_seqs.append(seq)` |
| group g_decode_loop | decode 循环:每位 running 客人 | `scheduler.py:58-70` | 同 lane decode;循环体到 `scheduled_seqs.append(seq)` |
| p1 | ① 看队首 / waiting[0];算 remaining | `scheduler.py:31-34` | `seq = self.waiting[0]` / `remaining = self.max_num_batched_tokens - num_batched_tokens` / `if remaining == 0: break` |
| p2 | ② 算要做多少 / 首次 can_allocate;续做算余量 | `scheduler.py:35-41` | `if not seq.block_table:` `num_cached_blocks = self.block_manager.can_allocate(seq)` … `num_tokens = seq.num_tokens - num_cached_blocks * self.block_size` / `else: num_tokens = seq.num_tokens - seq.num_cached_tokens` |
| p2(被调) | can_allocate | `nanovllm/engine/block_manager.py:58-73` | `for i in range(seq.num_blocks - 1):` … `if len(self.free_block_ids) < num_new_blocks: return -1` / `return num_cached_blocks` |
| p3 | ③ 分块判定 / 非队首且预算不够 → 停 | `scheduler.py:42-43` | `if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq` / `break` |
| p4 | ④ 排入批次 / allocate;min(需求, 余额) | `scheduler.py:44-47` | `if not seq.block_table: self.block_manager.allocate(seq, num_cached_blocks)` / `seq.num_scheduled_tokens = min(num_tokens, remaining)` / `num_batched_tokens += seq.num_scheduled_tokens` |
| p4(被调) | allocate | `block_manager.py:75-92` | `assert not seq.block_table` … `seq.block_table.append(self._allocate_block())` / `seq.num_cached_tokens = num_cached_blocks * self.block_size` |
| p5 | ⑤ 状态迁移 / 做完进 running;否则留队首 | `scheduler.py:48-52` | `if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:` / `seq.status = SequenceStatus.RUNNING` / `self.waiting.popleft()` `self.running.append(seq)` |
| p6 | ⑥ 返回 prefill 批 / (scheduled, True) | `scheduler.py:54-55` | `if scheduled_seqs:` / `return scheduled_seqs, True` |
| d1 | ⑦ 取 running 队首 / prefill 没排上才来 | `scheduler.py:54, 58-59` | `if scheduled_seqs:`(为假才往下)/ `seq = self.running.popleft()` |
| d2 | ⑧ 能续桌? / can_append:len%256==1 | `scheduler.py:60`;`block_manager.py:103-104` | `while not self.block_manager.can_append(seq):` / `return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)` |
| d3 | ⑨ 排入 decode 批 / scheduled=1;may_append | `scheduler.py:66-70`;`block_manager.py:106-108` | `else:` `seq.num_scheduled_tokens = 1` `seq.is_prefill = False` `self.block_manager.may_append(seq)` `scheduled_seqs.append(seq)` / `if len(seq) % self.block_size == 1: seq.block_table.append(self._allocate_block())` |
| d4 | ⑩ 返回 decode 批 / extendleft 放回;False | `scheduler.py:71-73` | `assert scheduled_seqs` / `self.running.extendleft(reversed(scheduled_seqs))` / `return scheduled_seqs, False` |
| x1 | ⑪ 抢队尾 / running.pop() → 退桌重排 | `scheduler.py:61-62, 75-79` | `if self.running:` `self.preempt(self.running.pop())` / `seq.status = SequenceStatus.WAITING` `seq.is_prefill = True` `self.block_manager.deallocate(seq)` `self.waiting.appendleft(seq)` |
| x2 | ⑫ 自抢 / running 空 → preempt 自己 | `scheduler.py:63-65` | `else:` / `self.preempt(seq)` / `break` |
| e_p1_p2 | 有人且有余额 | `scheduler.py:30, 33` | `while self.waiting and …` / `if remaining == 0:`(为假才继续) |
| e_p2_p3 | (无标签) | `scheduler.py:39-42` | `num_tokens = …` 之后紧接 `if remaining < num_tokens and scheduled_seqs:` |
| e_p3_p4 | 可排入 | `scheduler.py:42-44` | 条件为假不 break,落到 `if not seq.block_table:` |
| e_p4_p5 | (无标签) | `scheduler.py:47-48` | `num_batched_tokens += …` 之后紧接 `if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:` |
| e_p5_p6 | 收工且有人 | `scheduler.py:52-55` | `scheduled_seqs.append(seq)` → 循环退出 → `if scheduled_seqs: return scheduled_seqs, True`。**简化**:循环有四个出口(见第五节),图上合成这一条边与 e_p1_d1 |
| e_p1_d1 | 没排上任何人 | `scheduler.py:54-58` | `if scheduled_seqs:` 为假 → `# decode` → `while self.running …` |
| e_d1_d2 | 问桌位 | `scheduler.py:59-60` | `seq = self.running.popleft()` / `while not self.block_manager.can_append(seq):` |
| e_d2_d3 | 能续 | `scheduler.py:60, 66` | `while not …can_append(seq):` 条件为假 → `else:` 子句执行 |
| e_d3_d4 | 轮完 | `scheduler.py:58, 71-72` | `while self.running and …` 为假 → `assert scheduled_seqs` / `self.running.extendleft(…)` |
| e_d2_x1 | 缺桌,还有别人 | `scheduler.py:60-62` | `while not …can_append(seq):` / `if self.running:` / `self.preempt(self.running.pop())` |
| e_x1_d2 | 退桌后再问 | `scheduler.py:60` | 内层 `while` 回到条件再次调用 `can_append(seq)` |
| e_d2_x2 | 缺桌,只剩自己 | `scheduler.py:63-64` | `else:` / `self.preempt(seq)` |
| e_x2_d4 | break | `scheduler.py:65, 58, 71-73` | `break` 跳出内层 → 外层 `while self.running` 因 running 已空而结束 → `assert scheduled_seqs` → 返回。**前提**:批里已有人,否则 assert 抛异常(第五节) |
| card 预算的两个变量 | | `scheduler.py:27, 32-34, 47` | `num_batched_tokens = 0` / `remaining = …` / `num_batched_tokens += seq.num_scheduled_tokens` |
| card 两个 deque 的方向 | | `scheduler.py:31, 50-51, 62, 72, 79` | `self.waiting[0]` / `self.waiting.popleft()` `self.running.append(seq)` / `self.running.pop()` / `extendleft(reversed(…))` / `self.waiting.appendleft(seq)` |
| card 示例甲前 5 步 | | 干跑输出(第四节) | step 1-5 `is_prefill=True`,batch 分别 [A]、[B]、[B]、[C]、[C] |

### 图 b `diagrams/_02b-postprocess.json`(11 节点、10 边、4 泳道、1 组)

泳道按角色:postprocess 是一条几乎线性的记账流程,没有分支竞争,值得看的是「谁改了谁的字段」,所以经理 / 领班 / 座位管理员 / 客人状态四条泳道各占一层。

| 元素 | 图上文字 | 路径:行 | 原样引用 |
|---|---|---|---|
| lane engine | 经理 · LLMEngine.step() | `nanovllm/engine/llm_engine.py:49-53` | `def step(self):` … `token_ids = self.model_runner.call("run", seqs, is_prefill)` / `self.scheduler.postprocess(seqs, token_ids, is_prefill)` |
| lane scheduler | 领班 · Scheduler.postprocess() | `scheduler.py:81` | `def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):` |
| lane blocks | 座位管理员 · BlockManager | `block_manager.py:26` | `class BlockManager:` |
| lane guest | 客人状态 · Sequence | `nanovllm/engine/sequence.py:8-11, 14` | `class SequenceStatus(Enum): WAITING / RUNNING / FINISHED` / `class Sequence:` |
| group g_loop | 逐位客人(与 batch 同序) | `scheduler.py:82` | `for seq, token_id in zip(seqs, token_ids):` |
| e1 | ① 收回 token_ids / 与 batch 同序 | `llm_engine.py:52-53`;`nanovllm/engine/model_runner.py:218` | `token_ids = self.model_runner.call("run", seqs, is_prefill)` / `token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None` |
| s1 | ② 先登记满桌 / hash_blocks(seq) | `scheduler.py:83` | `self.block_manager.hash_blocks(seq)` |
| b1 | hash_blocks / 整桌算哈希并登记 | `block_manager.py:110-120` | `start = seq.num_cached_tokens // self.block_size` / `end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size` / `if start == end: return` … `self.hash_to_block_id[h] = block.block_id` |
| s2 | ③ 计数推进 / cached += scheduled | `scheduler.py:84-85` | `seq.num_cached_tokens += seq.num_scheduled_tokens` / `seq.num_scheduled_tokens = 0` |
| s3 | ④ 分块做完了吗 / cached < num_tokens? | `scheduler.py:86-87` | `if is_prefill and seq.num_cached_tokens < seq.num_tokens:` / `continue` |
| s4 | ⑤ append_token / eos 也追加 | `scheduler.py:88`;`sequence.py:67-70` | `seq.append_token(token_id)` / `self.token_ids.append(token_id)` `self.last_token = token_id` `self.num_tokens += 1` |
| s5 | ⑥ 结账判定 / eos 或满 max_tokens | `scheduler.py:89` | `if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:` |
| b2 | ⑦ deallocate 退桌 / 倒序 ref_count − 1 | `scheduler.py:91`;`block_manager.py:94-101` | `self.block_manager.deallocate(seq)` / `for block_id in reversed(seq.block_table):` `block.ref_count -= 1` `if block.ref_count == 0: self._deallocate_block(block_id)` |
| q_wait | 留在 waiting 队首 / 采样丢弃,下轮续做 | `scheduler.py:86-87`(不追加)+ `scheduler.py:31, 48-51`(未换队) | `continue` / 迁移条件为假时没有 `popleft`,客人仍在 `waiting[0]` |
| q_run | 留在 running / 下轮 decode | `scheduler.py:89`(为假)+ `scheduler.py:51, 72` | 条件为假则无后续语句;客人由 `self.running.append(seq)` 或 `extendleft` 留在 running |
| q_fin | ⑧ FINISHED 出队 / running.remove | `scheduler.py:90, 92`;`llm_engine.py:54` | `seq.status = SequenceStatus.FINISHED` / `self.running.remove(seq)` / `outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]` |
| e_e1_s1 | zip(seqs, token_ids) | `scheduler.py:82` | `for seq, token_id in zip(seqs, token_ids):` |
| e_s1_b1 | 登记 | `scheduler.py:83` | `self.block_manager.hash_blocks(seq)` |
| e_b1_s2 | 登记完 | `scheduler.py:83-84` | `hash_blocks` 返回后紧接 `seq.num_cached_tokens += …` |
| e_s2_s3 | (无标签) | `scheduler.py:85-86` | `seq.num_scheduled_tokens = 0` 之后紧接 `if is_prefill and …` |
| e_s3_q_wait | 没做完 → continue | `scheduler.py:86-87` | `continue` |
| e_s3_s4 | 做完,或 decode | `scheduler.py:86, 88` | 条件为假(`is_prefill` 为 False 或已做完)→ `seq.append_token(token_id)` |
| e_s4_s5 | (无标签) | `scheduler.py:88-89` | `append_token` 之后紧接 `if (not seq.ignore_eos and …)` |
| e_s5_b2 | 结账:status = FINISHED | `scheduler.py:90-91` | `seq.status = SequenceStatus.FINISHED` / `self.block_manager.deallocate(seq)` |
| e_b2_q_fin | 退桌后出队 | `scheduler.py:91-92` | `deallocate(seq)` 之后 `self.running.remove(seq)` |
| e_s5_q_run | 未结账 | `scheduler.py:89` | 条件为假,循环进入下一位 |
| card 顺序很重要 | | `scheduler.py:83-84, 88-89` | 见 s1/s2/s4/s5 |
| card 分块没做完时 | | `scheduler.py:86-87`;`model_runner.py:216-218` | `continue` / `temperatures = self.prepare_sample(seqs) …` `token_ids = self.sampler(…)`(采样照跑) |
| card 结账三连 | | `scheduler.py:90-92`;`block_manager.py:53-56` | `FINISHED` → `deallocate` → `running.remove` / `_deallocate_block` 只动 `used_block_ids` 与 `free_block_ids`,不清 hash、不删登记簿(这四行里没有这两个名字) |

## 二、状态所有权行

| 状态 | 持有者 | 谁写 | 路径:行 | 原样引用 |
|---|---|---|---|---|
| `waiting`、`running` | 领班 | 领班:`add`、`schedule`、`preempt`、`postprocess` | `scheduler.py:16-17, 23, 50-51, 72, 79, 92` | `self.waiting: deque[Sequence] = deque()` / `self.running: deque[Sequence] = deque()` |
| `scheduled_seqs`、`num_batched_tokens`、`remaining` | 无(`schedule()` 局部变量) | 领班 | `scheduler.py:26-27, 32` | `scheduled_seqs = []` / `num_batched_tokens = 0` / `remaining = …` |
| `max_num_seqs`、`max_num_batched_tokens`、`eos`、`block_size` | 领班(从店规抄来) | 只读 | `scheduler.py:11-14` | `self.max_num_seqs = config.max_num_seqs` … `self.block_size = config.kvcache_block_size` |
| `blocks`、`free_block_ids`、`used_block_ids`、`hash_to_block_id` | 座位管理员 | 座位管理员 | `block_manager.py:29-33` | `self.blocks: list[Block] = …` / `self.hash_to_block_id: dict[int, int] = dict()` / `self.free_block_ids: deque[int] = deque(range(num_blocks))` / `self.used_block_ids: set[int] = set()` |
| `seq.status` | 客人 | 领班 | `sequence.py:20`;`scheduler.py:49, 76, 90` | `self.status = SequenceStatus.WAITING` / `seq.status = SequenceStatus.RUNNING` / `seq.status = SequenceStatus.WAITING` / `seq.status = SequenceStatus.FINISHED` |
| `seq.block_table`、`seq.num_cached_tokens` | 客人 | 座位管理员(`allocate`、`may_append`、`deallocate`)与领班(`postprocess :84`) | `sequence.py:25, 28`;`block_manager.py:89-92, 100-101, 108`;`scheduler.py:84` | `seq.block_table.append(block_id)` / `seq.num_cached_tokens = num_cached_blocks * self.block_size` / `seq.num_cached_tokens = 0` `seq.block_table.clear()` |
| `seq.num_scheduled_tokens` | 客人 | 领班 | `sequence.py:26`;`scheduler.py:46, 67, 85` | `seq.num_scheduled_tokens = min(num_tokens, remaining)` / `= 1` / `= 0` |
| `seq.is_prefill` | 客人 | 领班 | `sequence.py:27`;`scheduler.py:68, 77` | `seq.is_prefill = False` / `seq.is_prefill = True`;用途 `sequence.py:73` `last_state = self.last_token if not self.is_prefill else self.token_ids` |
| `seq.token_ids`、`num_tokens`、`last_token` | 客人 | 客人自己(`append_token`),由领班触发 | `sequence.py:67-70`;`scheduler.py:88` | `self.token_ids.append(token_id)` `self.last_token = token_id` `self.num_tokens += 1` |
| `is_prefill`(整批) | 无(返回值) | 领班返回,经理转交厨房 | `scheduler.py:55, 73`;`llm_engine.py:50-53`;`model_runner.py:215` | `return scheduled_seqs, True` / `return scheduled_seqs, False` / `input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)` |

## 三、设计取舍行

| 取舍 | 代码事实(路径:行) | 原样引用 | 推断部分(页面里已标为取舍而非定理) |
|---|---|---|---|
| prefill 优先于 decode | `scheduler.py:54-55`;`model_runner.py:215` | `if scheduled_seqs: return scheduled_seqs, True` / `… if is_prefill else self.prepare_decode(seqs)` | TTFT 变短、decode 批更大;在座客人每来一位新客多等一轮。示例甲干跑:A 第 1 步得首 token,第 6 步才得第二个 |
| 只允许队首分块 | `scheduler.py:42-43`;续做判定 `scheduler.py:35, 40` | `if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq` / `if not seq.block_table:` … `else:` | 任一时刻最多一位半桌客人且必在队首,续做只看 `block_table` 空不空;代价是余额浪费(示例甲第 1 步剩 212、第 3 步剩 424) |
| 重算式抢占,不换出 KV | `scheduler.py:75-79`;`block_manager.py:94-101`;`engine/` 目录 grep `swap|offload` 无命中,`cpu` 只命中 `model_runner.py:38` 的 `torch.set_default_device("cpu")`(初始化后复位默认设备,与 KV 无关) | `self.block_manager.deallocate(seq)` / `self.waiting.appendleft(seq)` | 无 PCIe 搬运、无第二套表;代价是全序列重算。验证 1 干跑:受害者重排时 `num_scheduled=257`(prompt 256 + 已生成 1)。验证 3:整桌命中时只重算 1 个 token(`_deallocate_block :53-56` 不清哈希) |
| 抢队尾,不抢自己 | `scheduler.py:61-65`;队尾含义由 `:51` 与 `:72` 决定 | `self.preempt(self.running.pop())` / `self.preempt(seq)` `break` / `self.running.append(seq)` / `extendleft(reversed(…))` | 队尾 = 本轮还没轮到的人里最晚入座者;被抢者 `appendleft` 回队首优先重排。验证 1 干跑:后到者 Y 被抢,先到者 X 拿到 Y 退出的桌 |

## 四、干跑与 archify 校验回执

### 干跑

- 脚本 `/tmp/nanovllm-dryrun/dryrun_02.py`(未入库)。运行方式 `uv run --no-project --with xxhash --with numpy python dryrun_02.py`(本机 `python3` 缺 `xxhash` 与 `numpy`,uv 临时环境解决,仓库无改动)。
- 真实导入:`nanovllm/engine/scheduler.py`、`block_manager.py`、`sequence.py`、`sampling_params.py`。两处桩:`nanovllm/__init__.py`(它 `from nanovllm.llm import LLM` 会拉起 torch)用空包对象跳过;`nanovllm.config`(它 `from transformers import AutoConfig`)用空 `Config` 类替代。`config` 用 `types.SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=512, eos=0, kvcache_block_size=256, num_kvcache_blocks=16)`。token 用伪随机整数 ∈ [1, 10^6),永不等于 eos = 0;三位客人 prompt 互不相同,不触发拼桌。
- 示例甲结果:共 7 步;step 1-5 `is_prefill=True`,batch 依次 [A]、[B]、[B]、[C]、[C],`num_scheduled` 依次 300、512、88、512、488;step 6-7 `is_prefill=False`,batch [A, B, C];桌位 A [0, 1]、B [2, 3, 4]、C [5, 6, 7, 8];空桌 16 → 14 → 11 → 7 → 16;第 7 步三人 FINISHED,`completion=3`。与手算完全一致。
- 验证 1(抢队尾):2 张桌,X、Y 各 256 token。step 1 两人同批 prefill;step 2 decode 时 X 需新桌、空桌 0 → `running.pop()` 抢到 Y,Y 退桌回 waiting 队首,X 拿到桌 1;step 4 Y 重排 `num_scheduled=257, num_cached=0`。
- 验证 2(自抢撞 assert):1 张桌,单人 256 token。step 2 decode 需新桌、running 空 → 自抢 + break → `scheduled_seqs` 为空 → `AssertionError`(`scheduler.py:71`)。
- 验证 3(自抢但别人已排上):3 张桌,X、Y 各 256。step 2:X 拿走最后一张空桌,Y 自抢,批 = [X];step 4 Y 重排时 `num_scheduled=1, num_cached=256`,桌位 [1, 2]:自己退掉的桌 1 仍有哈希登记且未被复用,`can_allocate` 命中。

### archify

命令:`node ~/.claude/skills/archify/bin/archify.mjs validate workflow <json> --quality showcase --json`,几何诊断另加 `--layout-json`。

| 图 | 轮次 | 诊断 | 修法 |
|---|---|---|---|
| a | 1 | `layout/constraint`:d4 副标题需 173px,节点只有 168px | 缩短全部副标题(≤ 14 个中文字宽),节点宽 150 |
| a | 2 | `composition/desktop-readability`:viewBox 1480 宽,8px 副标题在 1440 桌面上投影 5.03px < 6px | 主路径边标签缩短或删除(`num_batched += scheduled` 等),节点宽 140 |
| a | 3 | 无 | 通过 |
| b | 1 | `composition/desktop-readability`:viewBox 1466 宽,7.1px 文字投影 4.5px | 同上缩短副标题,节点宽 150 |
| b | 2 | 无 | 通过 |

最终回执(两图相同结构):`ok: true`;9 项 artifact checks 全过(single_svg、finite_svg、orthogonal_arrows、label_route_clearance、relationship_crossings、relationship_corridors、container_border_runs、route_rhythm、legend_clearance);`composition.profile: showcase`,`status: pass`,`summary: {errors: 0, warnings: 0}`;`properCrossings: 0`,`ambiguousCorridors: 0`,`desktopReadabilityIssues: 0`。图 a 编译回执 `contract: readable-v2`,`viewBox [1157, 528]`,列中心 [118, 334, 502, 688, 856, 1062],`diagnostics: []`。图 b `viewBox` 宽 1466(边标签少、节点文字短,可读性检查通过)。通过后未再改动两份 JSON。指标里 `routesOverSuggestedBends`(a: 4,b: 5)与 `maxStretch`(a: 1.21,b: 1.87)只是建议值,不计入错误或警告。

## 五、疑问与冲突

1. **自抢在批里没人时会崩溃**:`scheduler.py:63-65` 自抢后 `break`,`:71` `assert scheduled_seqs` 直接抛异常(验证 2)。触发条件是单条序列占满全部桌位再要新桌。页面按当前代码如实写「崩溃而不是退避」,不猜作者意图。
2. **图 a 的两处简化**:prefill 循环的四个出口(`:30`、`:33-34`、`:37-38`、`:42-43`)在图上合成 ⑤→⑥ 与 ①→⑦ 两条边;⑫→⑩ 那条边假定批里已有人。都在页面第五节和本表第一节注明。
3. **示例甲没有规定 `num_kvcache_blocks`**:干跑取 16(9 张够坐、无抢占),页面已注明。若总览想统一,建议底稿第七节把它写死。
4. **底稿比喻表「preempt:下次从头重算」不完全准确**:验证 3 表明退桌不清哈希,整桌命中时只重算未命中部分。建议改为「下次重新入座,整桌还在就拼桌、只重算剩下的」。这一点与 03、04 页的拼桌内容有交叉,由主线程决定放哪页展开。
5. **底稿第三节「没有新客才给在座的上下一道菜」**:准确条件是「prefill 循环一个都没排上」,包含 waiting 非空但队首被空桌卡住(`can_allocate` 返回 −1)的情况。页面第五节已按代码写。
6. **分块 prefill 未完成时厨房仍采样**(`model_runner.py:216-218`),领班丢弃(`scheduler.py:86-87`)。是浪费但无副作用;页面写为事实,不评价。
7. **`eos` 在干跑里是 0 而不是 tokenizer 的值**:调度逻辑只比较相等,不影响轨迹;页面未出现具体 eos 数值。
8. **术语「TTFT」不在底稿术语表里**,本页术语卡新增;汇总时请合并。

## 六、对底稿与任务卡的改进建议(试跑反馈)

1. 底稿第七节的干跑提示应补两句:`nanovllm/__init__.py` 会导入 torch、`nanovllm/config.py` 会导入 transformers,需要各桩一个;`uv run --with xxhash --with numpy` 可在缺依赖的机器上零安装跑通。
2. 任务卡应预告 archify showcase 的两条硬约束,少走两轮:节点副标题 ≤ 约 14 个中文字宽(节点宽 140-150),主路径边标签 ≤ 6 个中文字或干脆不写,否则 6 列图的 viewBox 会超过约 1200px 触发 `composition/desktop-readability`。
3. 骨架「一、先看图」应明确允许两张图各自带图题、交互版链接和「怎么读」段,并规定编号在每张图内重新从 ① 开始、正文用「图 a 的 ③」指代。
4. 底稿第八节可加一条泳道选择指引:决策流(有分支竞争)按阶段分泳道,记账流(线性交接)按角色分泳道;本页两张图各用一种,效果都好。
5. 底稿第七节示例甲建议补全 `num_kvcache_blocks=16` 与「eos 不出现在轨迹中」两句,避免各页各取一值。
6. 比喻表 preempt 一行按第五节第 4 条修订;术语表补「TTFT」「自抢」。

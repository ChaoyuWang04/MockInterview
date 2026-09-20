# WebWalker：搜索引擎横着捞浅层，站内还得一层层点下去

<!-- release-date: 2025-01-13 -->

**本文依据**：`WebWalker: Benchmarking LLMs in Web Traversal`，arXiv 2501.07572v3（页眉 `[cs.CL] 10 Aug 2025`），21 页。封面右上印 `2025-01-14`，那是 PDF 排版日，不进 `release-date`。封面未印会议录用，本文不补。作者 Jialong Wu、Wenbiao Yin、Yong Jiang（通讯）等；封面机构 Tongyi Lab, Alibaba Group（PDF p. 1）。第一作者单位是 Alibaba Tongyi Lab。盘上是 v3。首发日取 arXiv 页面 `Submitted on 13 Jan 2025`（[arxiv.org/abs/2501.07572](https://arxiv.org/abs/2501.07572)），这是外部补充，不来自 PDF 正文。代码封面写明 `https://github.com/Alibaba-NLP/WebAgent`（PDF p. 1）。文中数字均标 PDF 页码；标「外部补充」的段落不来自本文。

## 一句话

RAG 接 Google / Bing，本质是 **水平搜索**：按查询捞一批浅层结果。官方站里真正有用的日程、地址、规则，往往埋在二、三、四级子页，点按钮才能走到（PDF p. 1）。作者因此拆成两件东西：**WebWalkerQA** 是基准，680 题、四个真实域、超过 1373 张网页，测的是从给定根站做 **Web Traversal**；**WebWalker** 是探索–批评多智能体框架，explorer 按 ReAct 点 HTML 按钮，critic 攒记忆、决定何时作答（PDF p. 1–2）。即使用 GPT-4o 当骨干，WebWalker 总分也只有 **37.50%**，没过 40%（PDF p. 7 表 3）。闭卷最强也只有个位数；商业搜索增强里最好的 Tongyi **40.73%**（PDF p. 9 表 4）。贯穿全文的轴不是「再做一个网页点选榜」，而是：**水平检索捞不到的深度，必须在站内一层层走下去；测的是走，不是刷搜索捷径。**

## 一、矛盾：搜索横着扫一遍，权威信息却在子页里

LLM 训完知识就冻住。接搜索引擎做 RAG，能补时效，但传统引擎对查询做的是横向检索，**追不进网站内部更深的内容**（PDF p. 1）。

已有网页基准走的是另一条路：Mind2Web、WebArena 一类，按 HTML 指令–动作做事。问题是噪声大、输入超长，长上下文吃不消；真实场景里证据还埋在多层交互后面，这些榜也抓不住（PDF p. 1）。

作者新开一个任务叫 **Web Traversal（网页遍历）**：给定与查询对应的 **初始网站**，系统化地走子页，把信息挖出来（PDF p. 1）。WebWalkerQA 用问答形式测文本推理，动作约束成 **点击**，专门看导航和找信息，而不是把整页 DOM 当操作台（PDF p. 1）。域选教育、会议、机构、游戏，因为官方源公开、路径更结构化、按钮可点、推理链清楚（PDF p. 1–2）。题分 **单源** 和 **多源**，对应人在站内「往下挖一条」和「两页拼起来」两种走法（PDF p. 2）。

WebWalker 是配套强基线：explorer + critic。explorer 建在 ReAct 的 thought–action–observation 上；critic 管记忆、根据探索结果作答（PDF p. 2）。作者自己的三条发现（PDF p. 2）：

1. 网页导航在要规划和推理的任务上仍费劲。
2. RAG 加 WebWalker，水平与垂直配合有效。
3. 站内垂直探索，是给 RAG 做推理时扩展的一条路。

贡献三条（PDF p. 2）：680 查询、四个真实场景、超过 1373 网页的 WebWalkerQA；用多智能体管长上下文记忆的 WebWalker；实验表明基准难，信息寻求任务里 **页内垂直探索有用**。

```mermaid
flowchart TB
  horiz["水平搜索：Google / Bing 捞浅层"] --> miss["官方子页里的日程、地址、规则捞不全"]
  root["给定根站 URL + 查询"] --> click["只允许点 HTML 按钮往下走"]
  click --> qa["WebWalkerQA：680 题，短实体答案"]
  miss --> qa
  qa --> ww["WebWalker：explorer 走，critic 记"]
  ww --> rag["记忆并进 RAG 文档，水平加垂直"]
```

上图根据 PDF p. 1–2、p. 5–6、p. 9–10 重画，是评测口径示意，不是实测曲线。

相关工作里，表 1 把 WebWalkerQA 和其他榜并排放：Mind2Web 100 页、WebArena 6 页、AssistantBench 525 页、MMInA 100 页、GAIA 页数标「-」；WebWalkerQA 是英中双语、QA、深度 / 宽度 / hop 都打勾、**1373** 页（PDF p. 3 表 1）。Depth 指给定站上要挖多深；Width 指是否要多源；Hop 指是否多步。正文写「对比见表 2」，那是笔误，表头是 Table 1（PDF p. 3）。和 MMInA、AssistantBench 最接近：都要跨多页、费时间；差别是 WebWalkerQA 从 **宽度** 上同时造单源和多源，模拟人的两种逛法（PDF p. 3）。网页 agent 两条线：小模型筛动作 / HTML 元素，或提示 LLM 加模块；视觉榜再用截图。WebWalker 专做信息寻求，对着 HTML 按钮推理，多智能体去权威源（PDF p. 3）。

## 二、WebWalkerQA：从根站造树，再人工把问答钉死

### 漏斗造题

为了又便宜又准，两段漏斗：先 GPT-4o 初标，再众包人做质控和过滤（PDF p. 4、图 2 p. 3）。流水线：

1. 从会议、机构、教育、游戏的官方根站出发，递归点子链、收子页。
2. 按角色造查询：只盯一页，或两页同时看。
3. 滤掉不像人话的题，只留 **短答案、含实体** 的 QA（PDF p. 4）。

单源模拟「一条信息埋在深层」；多源模拟「必须两页一起才能答」。多源不好被搜索引擎捷径钻空子（PDF p. 4）。人再改写、校准问答（PDF p. 4）。根站怎么来的：Google 搜「conference official website」一类再人工筛；教育域用各校计算机系官网（PDF p. 17 附录 E.1）。

### 680 题怎么切

得到 **680** 对问答（PDF p. 4）。类型（PDF p. 4）：

- 单源标 `single_source_i`，$i \in [2,4]$，是那张子页的深度。
- 多源标 `multi_source_i`，$i \in [2,8]$，是两张关联子页深度之和。答这题必须两页一起读。脚注：`multi_source_6` 可以是两张 3 级页，也可以是一张 2 级加一张 4 级（PDF p. 4）。

难度按 $i$ 切成 easy / medium / hard（PDF p. 4–5）：

- 单源：2 → easy，3 → medium，4 → hard。
- 多源：2–4 → easy，4–6 → medium，6–8 → hard。

表 2（PDF p. 4）：

| 类型 | Easy | Medium | Hard |
|---|---|---|---|
| 单源 | 80 | 140 | 120 |
| 多源 | 80 | 140 | 120 |

四域合计 $80+140+120$ 再乘 2 = **680**。域：conference **24.0%**、organization **7.9%**、education **46.3%**、game **24.0%**（PDF p. 5 图 3）。语言按 **根网页语言** 分：中文 **60.5%**、英文 **39.5%**（PDF p. 5 及脚注 3）。

附录图 8 给 JSON 样例：问 ACL 2025 Industry Track 投稿截止和会场地址；答 March 21, 2025 与 Brune-Kreisky-Platz 1；根 URL `https://2025.aclweb.org/`；Info 里 Hop 为 multi-source、Domain Conference、Language English、Difficulty Medium、两张源页、Golden_Path（PDF p. 17 图 8）。数据集作者写将放到 HuggingFace Datasets（PDF p. 17）。

### 什么叫「做对了」

形式：给定根 URL $U_{\mathrm{root}}$ 和查询 $Q$，靠遍历凑够信息再答（PDF p. 5）。指标两套：表现用问答准确率 **acc.**；效率用答对那些局里的动作次数 **A.C.**（successful agentic executions answering correctly）（PDF p. 5）。短答案仍难精确匹配，所以用 **GPT-4** 当裁判，拿预测和标准答案做 CoT 比对（PDF p. 5）。提示在附录 F / 图 9：当老师改测验，只看事实对错，标点措辞差可以忽略，学生多说只要不冲突就算对，输出 CORRECT / INCORRECT（PDF p. 20 图 9）。这是模型当裁判，不是精确匹配。

## 三、WebWalker：一个人点链接，一个人记账

框架两个角色（PDF p. 5–6 图 4）。

### Explorer：先想再点

explorer 跟页面上的 HTML 按钮互动。时刻 $t$ 收到观察 $O_t$，按策略 $\pi(A_t \mid H_t)$ 出动作 $A_t$（PDF p. 5）。

$$
O_t = (p_t, l_t),\quad l_t = \{\mathrm{button}_i\}_{i=1}^{K}
$$

$p_t$ 是当前页信息；$l_t$ 是可点子链，每个按钮带描述和 URL。动作是选一个子页 URL 去逛，**不含作答**（PDF p. 6）。观察用页面 markdown，加上 Beautiful Soup 抽的可点按钮和 URL（PDF p. 6）。历史

$$
H_t = (T_1, A_1, O_1, \ldots, O_{t-1}, T_t, A_t, O_t)
$$

一直走到 critic 决定作答，或步数打满（PDF p. 6）。主实验把 explorer 动作上限 **K = 15**（PDF p. 7）。

### Critic：先想再批

$\pi(A_t \mid H_t)$ 是隐式的，$H_t$ 可能很大。作者借结对编程的想法加 critic（PDF p. 6）。explorer 每走一步，critic 吃查询和当前观察，初始化并累加记忆 $M$；输入是 $Q$ 与 $(O_t, A_t)$，更新 $M$，判断信息够不够，够了再给答案（PDF p. 6）。附录提示分两段：先判这条观察对查询有没有用（JSON `usefulness` + `information`）；再判攒起来的信息够不够答（JSON `judge` + `answer`）（PDF p. 19–20）。Explorer 提示是标准 ReAct 挖按钮找源（PDF p. 19）。

实现：基座是 Qwen-Agent；生成 `top_p = 0.8`；页面 markdown 靠 crawl4ai（文中写成 ai4crawl）（PDF p. 16 附录 B）。Llama 系列初步实验里跟不好 ReAct 格式，主表没收（PDF p. 7 脚注 7）。

可迁移的不是「再训一个网页模型」，而是：**走的人不要同时答，答的人不要把整段 $H_t$ 当工作记忆**——长页噪声会让小模型几步就放弃（后文错误分析）。

## 四、主实验：最强骨干也过不了四成

基线：ReAct、Reflexion（PDF p. 6）。骨干要求上下文至少 **128K**、参数至少 **7B**；闭源 GPT-4o、Qwen-Plus；开源 Qwen2.5-{7,14,32,72}B-Instruct。作者写共验证九个模型，主表列出的是这六个骨干 × 三种方法（PDF p. 7）。全部零样本。

表 3 主结果（PDF p. 7）。Overall：

| 骨干 | 方法 | acc. | A.C. |
|---|---|---|---|
| GPT-4o | ReAct | 33.82 | 3.83 |
| GPT-4o | Reflexion | 35.29 | 4.27 |
| GPT-4o | WebWalker | **37.50** | 4.67 |
| Qwen-Plus | ReAct | 33.08 | 3.03 |
| Qwen-Plus | Reflexion | 33.23 | 4.32 |
| Qwen-Plus | WebWalker | 33.82 | 4.36 |
| Qwen-2.5-7B | ReAct | 16.02 | 2.99 |
| Qwen-2.5-7B | Reflexion | 19.11 | 4.07 |
| Qwen-2.5-7B | WebWalker | 19.85 | 3.94 |
| Qwen-2.5-14B | ReAct | 22.35 | 2.76 |
| Qwen-2.5-14B | Reflexion | 25.14 | 3.01 |
| Qwen-2.5-14B | WebWalker | 27.50 | 3.60 |
| Qwen-2.5-32B | ReAct | 25.44 | 2.93 |
| Qwen-2.5-32B | Reflexion | 23.26 | 3.00 |
| Qwen-2.5-32B | WebWalker | 26.02 | 3.90 |
| Qwen-2.5-72B | ReAct | 30.73 | 2.86 |
| Qwen-2.5-72B | Reflexion | 32.50 | 4.09 |
| Qwen-2.5-72B | WebWalker | 33.26 | 4.32 |

闭源整体强于开源；开源随规模涨。作者排序：WebWalker 优于 Reflexion，Reflexion 优于 ReAct（PDF p. 7）。A.C. 只统计答对的局；模型变大，A.C. 也涨，作者解读为长程找信息变强（PDF p. 7）。GPT-4o + WebWalker 仍 **不超过 40%**，用来论证基准难（PDF p. 7）。深度加深或源变多，准确率掉（PDF p. 7）。抽几格对照：GPT-4o WebWalker 单源 easy **55.00** / hard **30.00**，多源 easy **47.50** / hard **15.83**（PDF p. 7 表 3）。Qwen-2.5-7B ReAct 单源 medium 表中印成 `18.5 7`，按同行小数格式读成 **18.57**（PDF p. 7）。

图 5(a) 把 acc 对 A.C. 撒点：越右上，越有效、走得越久。加大模型或给每步加反思，能解一部分多步题（PDF p. 8）。图 5(b) 看域和语言：会议域相对好，作者猜按钮信息更直白、更好推；中英差不多，因为用的模型双语预训练和 SFT（PDF p. 8）。图本身是点图，正文没有再给各域精确百分比。

### 错在哪

错执行分成三类：拒答或定位错、推理错、超过最大步数 $K$（PDF p. 8）。图 6 是 WebWalker / ReAct × Qwen-14B / Qwen-Plus 的预测分布（PDF p. 8）。小参数 + ReAct 挖不深，几步就判，不管有没有找到，像「不耐烦」；加记忆管长上下文、再加大参数，作者认为这来自长上下文噪声干扰和模型本身能力，与 §5.2 一致（PDF p. 8）。还有一类推理错：黄金页已经进过访问列表，仍标错——说明页上推理本身也难（PDF p. 8）。附录 G.1 例：MRS 站点问 Inclusive Connections Lounge 从 2024-12-01 到 06 合计多少小时，标准答案 **66 hours**；先要找到页，再算时间（PDF p. 20–21 表 5）。

## 五、水平 RAG 对上垂直 Walker

表 4 测搜索增强系统能不能捞到深层（PDF p. 9）。

闭卷（无检索）：Gemini-1.5-Pro 总分 **8.08**，o1-preview **9.85**（PDF p. 9 表 4）。商业：Doubao **16.76**、Gemini-Search **27.94**、ERNIE-4.0-8K **28.97**、Kimi **37.35**、Tongyi **40.73**。开源：Naive RAG **20.73**、MindSearch **11.32**。各难度行平均：单源 easy 37.50 / medium 24.29 / hard 23.42；多源 easy 19.86 / medium 18.02 / hard 16.48（PDF p. 9 表 4）。

闭卷差，因为题建在会更新的官网，预训练有截止日期（PDF p. 9）。附录 G.2：问 2025 MRS Fall Meeting 地点和时间，标准 Boston, Massachusetts；November 30 to December 5, 2025；o1 说知识截止 2023 年 10 月，尚未宣布（PDF p. 21 表 6）。商业最好 Tongyi 也只到约 40%。ERNIE 作者猜中文搜索更强。开源里多源低于单源：搜索引擎一次或几次 **水平** 检索捞不齐（PDF p. 9）。越难、信息越深，越差。

**发现 (i)**：RAG 仍过不了要有效网页遍历的坎（PDF p. 9）。

标准 RAG 当成水平搜文档；WebWalker 当成垂直探索，可以嵌进去。做法：Qwen-2.5-Plus 上的 WebWalker 接 naive RAG，critic 的记忆 $M$ **追加**到相关文档再生成（PDF p. 9）。图 7(a) 各难度都涨，多源更明显（PDF p. 9–10）。图是柱状示意，正文没有再印精确百分点。

**发现 (ii)**：WebWalker 可以当 agentic RAG 里的垂直模块（PDF p. 10）。

先前工作用加检索文档数看 RAG 推理扩展。这里把 $K \in \{5,10,15,20,25\}$ 做垂直扩展（PDF p. 10）。图 7(b)：$K$ 越大越好，一定范围内垂直扩展可行。骨干写 Qwen-Plus（PDF p. 10）。

**发现 (iii)**：把「顺着链接往下挖」做大，可能是 RAG 垂直探索的方向（PDF p. 10）。

商业系统走业务 API；开源 MindSearch 是 WebPlanner + WebSearcher；Naive RAG 用 Google 查词，Top-10 链接拼给 Qwen-Plus（PDF p. 16–17 附录 C）。

## 六、限制：题少、没视觉、没训、根 URL 还是送的

附录 A（PDF p. 16）：

- **题量**：680 高质量对；另有约 **14k** 银标，人还没仔细核，可当训练补充。对照 AssistantBench 214、MMInA 1,050、GAIA 466。
- **多模态**：只用 HTML-DOM 解析可点按钮，截图留给未来。
- **Agent tuning**：纯提示、没额外训练；可用黄金轨迹微调。
- **和 RAG 更好接**：§6.2 实验里根 URL 是 **提供** 给 WebWalker 的。更完整的接法：先在 RAG 里改写查询、搜到可能的官网，再让 Walker 挖；检索知识和矿出来的信息一起当增强（PDF p. 16）。

WebWalker 可以单独给一张网页当检索助理，也可以接 RAG。作者认为在 agentic RAG 下，**点击** 这一动作很有效（PDF p. 16）。封面没写会议；仓库与 HuggingFace 数据集是否已按附录承诺上线，本文不根据后日网页回写。

## 七、可迁移启发

1. **水平榜和垂直榜不要混**：搜得到首页摘要，不等于走得到四级子页。测信息寻求，先钉根站、再只允许点。
2. **多源是搜索捷径的克星**：答案必须两页拼，单次 SERP 不够。
3. **走和答拆开**：explorer 不管最终答题，critic 只维护 $M$。长 $H_t$ 会让小模型早停。
4. **A.C. 只在答对的局上读**：步数涨可能是更能走远，也可能是乱逛；作者只在正确执行上计数。
5. **裁判是 GPT-4 CoT，不是精确匹配**：短实体仍可能被措辞放过或卡住；和 GAIA 准匹配不是同一口径。
6. **根 URL 是实验礼物**：线上 RAG 还要先改写、先搜到官网，再垂直挖。
7. **K 是推理时扩展旋钮**：和「多捞几篇文档」正交，但是延迟和噪声一起涨。

## 八、关键词回看

**Web Traversal**：从给定根站系统点下去，凑够信息再答，不是开放网络随便搜。**WebWalkerQA**：680 题、四域、英中、单源/多源、深度当难度。**WebWalker**：explorer（ReAct 点按钮）+ critic（记忆与作答）。**水平 vs 垂直**：搜索引擎横向捞文档，Walker 沿站内链接向下。**acc. / A.C.**：GPT-4 判对的准确率；答对局里的点击步数。**$K$**：explorer 最大步数，主实验 15，扩展实验到 25。

## 参考资料

- 原件：`readings/_src/评测与 Benchmark/WebWalker.pdf`（盘上 v3，21 页）
- arXiv：[2501.07572](https://arxiv.org/abs/2501.07572)（v1 Submitted on 13 Jan 2025）
- 代码（封面）：[Alibaba-NLP/WebAgent](https://github.com/Alibaba-NLP/WebAgent)
- 文中点名的相邻基准：Mind2Web、WebArena、AssistantBench、MMInA、GAIA（PDF p. 3 表 1）

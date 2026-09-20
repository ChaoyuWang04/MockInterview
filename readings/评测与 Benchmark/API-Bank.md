# API-Bank：把「何时调、调哪个、怎么规划」收成可执行评测，再拿合成数据训出 Lynx

<!-- release-date: 2023-04-14 -->

**本文依据**：`API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs`，arXiv 2304.08244v2（页眉 `[cs.CL] 25 Oct 2023`），15 页。作者 Minghao Li（Alibaba Group）、Yingxiu Zhao（Hong Kong University of Science and Technology）等；通讯作者 Yongbin Li（Alibaba Group）。封面机构为 Alibaba Group、HKUST、Peking University、Shenzhen Intelligent Strong Technology Co., Ltd（PDF p. 1）。封面与页眉均未印会议录用，本文不补。盘上是 v2。首发日取 arXiv 页面 Submission history 的 `[v1] Fri, 14 Apr 2023 14:05:32 UTC`（[arxiv.org/abs/2304.08244](https://arxiv.org/abs/2304.08244) 的 Submitted on 14 Apr 2023），这是外部补充，不来自 PDF 正文。数据与代码：`https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank`（PDF p. 1）。文中数字均标 PDF 页码；标「外部补充」的段落不来自本文。

## 一句话

LLM 会聊天、会少样本、会写代码，但训练语料会过时、也盖不住所有场景；接外部 API 被看成补过时知识与第三方服务的路（PDF p. 1）。作者先问三件事：现有模型用工具到底好不好；怎么把能力做上去；还卡在哪（PDF p. 1）。API-Bank 用可运行系统答第一问：**73** 个 API、人工标的 **314** 段对话、**753** 次调用，测规划、检索、调用（PDF p. 1–2）。第二问用 Multi-agent 合成训练集：跨约 **1,000** 域、**2,138** 个 API、**1,888** 段对话，从 Alpaca 微调出 Lynx（PDF p. 1–2）。表 3 上 GPT-3.5 比 GPT-3 会用工具，GPT-4 更会规划；Lynx 在 Call 正确率上比 Alpaca-7B 高约 **26** 个百分点，总正确率 **39.58%**，接近 GPT-3.5 的 **47.16%**，距 GPT-4 的 **60.24%** 仍约 **21** 个百分点（PDF p. 7–8）。错误分析用来答第三问（PDF p. 1、p. 8–9）。

贯穿全文的轴不是「再堆一批 API 名」，而是：**什么叫会用工具——何时调、调哪个、多步怎么排——以及这三档必须在可执行系统里对结果，而不是只对生成文本。**

## 一、矛盾：接上工具容易，测「会不会用」很难

作者把用工具比成灵长类进化里的里程碑，认为当时必须先把三问钉死，而不是只演示插件能搜网页（PDF p. 1）。旧做法把工具当成人类特有行为；新做法是让模型访问最新信息、对接第三方服务（PDF p. 1）。没有权威能力定义时，他们先访谈 **500** 名希望给 LLM 加工具的用户，再定评测范围必须覆盖规划、检索、调用，并同时照顾域多样性、API 多样性、API 真实性、评测真实性（PDF p. 1–2）。

调用发生在开放环境里：域和功能事先定不完；调用又像数学一样严——名字错、参数错、顺序错，用户需求就完不成（PDF p. 3）。因此基准不能只看「模型有没有吐出一段像 API 的字符串」，而要有一套能实时执行、再看系统状态变没变的功能系统（PDF p. 3）。

```mermaid
flowchart TB
  q["用户需求"] --> few["池子小：文档能塞进上下文"]
  q --> many["池子大：必须先检索"]
  few --> call["Call：已知 API，填槽调用"]
  many --> rc["Retrieve+Call：先搜再调一次"]
  many --> prc["Plan+Retrieve+Call：未知 API，连续规划多步"]
  call --> sys["可执行系统：改库 / 返回值"]
  rc --> sys
  prc --> sys
```

上图根据 PDF p. 2–3 图 1、图 2 重画，是能力分级示意，不是实测曲线。

## 二、什么叫「做对了」：三档能力，对的是执行后果

### 两维切成三档，不是四格都考

用户需求被切成两维（PDF p. 2）：

1. **池子里 API 少还是多。** 少到 2–3 个时，名字、定义、输入输出都能塞进提示，模型自己选。多到几百个时，长度不够，必须先检索再调。
2. **一轮里调一次还是多次。** 有人愿意把复杂需求拆成多句、每句一次调用；有人一句丢过来，指望模型在一轮里逐步调完。

两维本有四格：少 API 单次、少 API 多次、多 API 单次、多 API 多次（PDF p. 3 图 2）。实现时前两格难度接近——文档都在手里时，规划多步并不难——于是合并，剩下三档（PDF p. 3）：

1. **Call**：API 已知，按查询调用。
2. **Retrieval+Call**：API 未知，检索并调用**一个** API。
3. **Plan+Retrieval+Call**：API 未知，连续规划、检索并调用**多个** API。

数据标准四条：域尽量广；API 的名字、定义、参数要像真的；类型与用途要杂；评测必须有可运行系统，模型出调用、系统执行、再按执行对用户需求的影响打分（PDF p. 3）。

### 评测系统：73 个可跑 API，外加「API Search」

系统实现 **73** 个常用 API，从天气预报到调其他 AI（如文生图）（PDF p. 3–4）。同一框架由资深研发实现，共 **98** 人天（PDF p. 4）。库类 API 先建库并写入初始条目，这是对话能往下走的前提。访问外部信息（如搜索引擎）的 API 必须把检索结果钉死：记下测试对话里每条查询、在某一时刻的检索结果，硬编码进 API，保证可复现（PDF p. 4）。

另做一个特殊 API **API Search**，专门服务 Retrieval+Call 与 Plan+Retrieval+Call（PDF p. 4）。这两种设定下，模型事先不知道池子里有什么，必须先用 API Search 按用户查询找可能需要的 API。提示开头给出 API Search 的说明，且**每次调其他 API 之前都要先 Search**（PDF p. 4）。模型把需求压成几个关键词；Search 对关键词与池中全部 API 元信息做句向量，算余弦相似度，返回相似度最高的那条元信息（PDF p. 4）。

附录图 7 里检索工具在提示里叫 `ToolSearcher`，与正文「API Search」是同一角色（PDF p. 14）。图 6 是已知 API 的忘记密码两步调用；图 7 是先搜再拿 token 再 `AddReminder`；图 8 是一句「算金融分析师税后月薪」，先搜职业薪资、再搜税算器（PDF p. 13–15）。图 8 最后一句把年薪 **100,000** 税后 **70,000** 说成「monthly」，这是附录示例本身的笔误，不是评测指标（PDF p. 15）。

### 对话怎么标

三档都人工标（PDF p. 4）。

- **Call**：从池中随机抽 API，标注员按文档想一条能被这些 API 解决的查询，标调用、系统执行、再按输出标回复。不必是单轮：同一组 API 上可以连问多句，同时给出对话历史和调用历史（PDF p. 4）。
- **Retrieval+Call**：从池中取 **1–5** 个 API，看能否共同覆盖一个复杂需求；能则拆成若干简单查询，每条标该调哪个、输入参数、以及根据系统输出该说什么（PDF p. 4）。
- **Plan+Retrieval+Call**：不把复杂查询拆开，标一条调用链，以及最后一次执行后的回复（PDF p. 4）。

计算机专业学生标对话，每段两人讨论；另两人查格式、逻辑一致性和调用是否合理。平均每段 **8** 美元。标了 **400** 段，因各类问题丢掉 **21.5%**，留下 **314** 段、共 **753** 次 API 调用（PDF p. 4）。

### 指标：调用对不对，回复像不像

调用侧用 Accuracy：正确预测数 / 总预测数（PDF p. 4）。每次评测先初始化系统，各 API 的库回到默认值；再把预测调用与人工标注比一致性。一致性定义为：对库的查询或修改是否相同、返回结果是否相同（PDF p. 4）。调用之后的自然语言回复用 ROUGE-L（PDF p. 4）。

附录提示尽量短：一块测 API 调用，一块测回复；调用格式是 `[ApiName(key1='value1', ...)]`，并写死当前年是 2023（PDF p. 12 图 4–5）。Plan 档另有「一次只输出一次调用、以 `[` 开头 `]` 结尾、不要解释文字」的说明（PDF p. 15）。

## 三、训练集：人标太贵，就拆成五个 agent

评测集约 **8** 美元一段，按这个价做大规模训练划不来；招募来的标注员也只能想出约 **100** 个 API，撑不起多样性（PDF p. 5）。作者不用一条超长指令让模型一次写完（他们观察到 ChatGPT 按这种指令只有约 **5%** 可用，换成 GPT-4 也只到约 **25%**），而是把需求拆成逐步任务（PDF p. 5）。

五个 agent 都用给 ChatGPT 的专用提示，逐步生成（PDF p. 5 图 3）：

1. 生成若干域（如 healthcare、fitness）。
2. 按域生成候选 API；为了像真的，输入里加入 [Public APIs](https://github.com/public-apis/public-apis) 的例子（PDF p. 5）。这是外部仓库链接，出现在原文脚注。
3. 随机抽一个或多个 API，再抽一档能力，写出匹配该能力、能被这些 API 满足的查询。
4. 吃下域、API、能力、查询，做出调用、模拟执行、写回复。
5. Tester 按设计原则自动验收，实际丢掉 **35%** 条（PDF p. 5）。

不必再雇人，每段约 **0.1** 美元，相对人工标评测集约 **98%**（摘要与第 4 节同一口径，PDF p. 2、p. 5）。

## 四、规模怎么读：表 2 是训练/评测拆开，结论里的 2,202 是加总

第 5 节先给加总：基准共 **1,008** 域、**2,211** 个 API、**2,202** 段对话、**6,135** 轮（PDF p. 5–6）。按能力：Call **934**、Retrieve+Call **769**、Plan+Retrieve+Call **499**（PDF p. 6）。训练自动生成，评测人工标，域、API、对话内容都有偏移，用来看分布外泛化（PDF p. 6）。

表 2（PDF p. 6）把训练和评测拆开——读榜时以这张表为准，不要拿加总去对「评测只有 73 个 API」：

| 统计 | 训练 | 评测 |
|---|---|---|
| 域 | 1,000 | 8 |
| API | 2,138 | 73 |
| 对话 | 1,888 | 314 |
| 轮次 | 5,221 | 914 |
| 其中单次调用轮 | 3,147 | 363 |
| 其中多次调用轮 | 493 | 122 |
| Call 条数 | 720 | 214 |
| Retrieve+Call 条数 | 719 | 50 |
| Plan+Retrieve+Call 条数 | 449 | 50 |
| 段均轮次 | 2.76 | 2.91 |

加总对得上：域 1,000+8=1,008，API 2,138+73=2,211，对话 1,888+314=2,202，轮次 5,221+914=6,135。摘要写训练集约 **1,000** 个不同域，与表 2 训练列一致（PDF p. 1、p. 6）。结论再次用加总 **2,202 / 2,211 / 1,008**（PDF p. 9）。

质量：评测每条经四人审。训练随机抽 **100** 条给人评，Multi-agent 可用率 **94%**，相对单 agent 的 self-instruct 高 **89** 个百分点（PDF p. 6）。被 tester 滤掉的数据里，**78%** 确实不符合设计原则（PDF p. 6）。

表 1 把 API-Bank 和当时几份工具基准比：作者强调自己域最多（表中 **1,000**）、覆盖多轮与一轮多调、同时评调用与回复、三档能力都有（PDF p. 6）。表里 DATESET 对应 Schick 等（Toolformer 一线，1 域 1 API）；APIBench（Patil 等）90 域、1,645 API；ToolAlpaca 50 域、426 API；ToolBench1（Qin 等）49 域、16,464 API；ToolBench2（Xu 等）8 域、232 API；ToolQA 6 域、13 API（PDF p. 6）。「域最多」是相对表 1 他们填的数字；ToolBench1 的 API 数比 API-Bank 大，作者比的是域多样性、真实多轮、以及回复与三档是否同时覆盖，不是 API 个数冠军（PDF p. 6）。

测试集域分布见表 8（PDF p. 12）：Account Management 7、Information Query and Processing 22、Health Management 8、Schedule Management 19、Smart Home 6、Finance Management 6、Others 5。

## 五、数字怎么读：表 3 是主表；26 pts 与 24% 不是同一刀

Lynx 从 LLaMA-7B 出发、Alpaca-7B 初始化，在 API-Bank 训练集上微调 **3** 个 epoch，batch **256**，学习率 $2\times 10^{-5}$（PDF p. 7）。对照：GPT-3 Davinci；GPT-3.5-turbo 的 `gpt-3.5-turbo-0613`；GPT-4 的 `gpt-4-0613`；ChatGLM-6B；Alpaca-7B（52K 指令数据）（PDF p. 7）。表头写 ChatGLM-6B，引言写 ChatGLM-6.2B，以表 3 为准（PDF p. 2、p. 7）。评测提示在附录，默认零样本（PDF p. 7）。

表 3（PDF p. 7），Correctness 为百分数，Rouge 为 ROUGE-L：

| 设定 | 模型 | Call 正确率 | Call Rouge | R+C 正确率 | R+C Rouge | P+R+C 正确率 | P+R+C Rouge | 总正确率 | 总 Rouge |
|---|---|---|---|---|---|---|---|---|---|
| 零样本 | Alpaca-7B | 24.06% | 0.0204 | 5.19% | 0.0019 | 0.00% | 0.086 | 15.19% | 0.0318 |
| 零样本 | ChatGLM-6B | 23.62% | 0.2451 | 13.33% | 0.2173 | 0.00% | 0.1522 | 16.42% | 0.2191 |
| 零样本 | GPT-3 Davinci | 0.50% | 0.1035 | 1.48% | 0.091 | 0.00% | 0.0156 | 0.57% | 0.0814 |
| 零样本 | GPT-3.5-turbo | 59.40% | 0.4598 | 38.52% | 0.3758 | 22.00% | 0.3809 | 47.16% | 0.4267 |
| 零样本 | GPT-4 | 63.66% | 0.3691 | 37.04% | 0.351 | 70.00% | 0.4808 | 60.24% | 0.3910 |
| 微调 | Lynx-7B | 49.87% | 0.4332 | 30.37% | 0.2503 | 20.00% | 0.3425 | 39.58% | 0.3794 |

作者怎么读这张表（PDF p. 7–8）：

- 能力越难，多数模型分数越低。Call 近似填槽：Alpaca-7B 与 ChatGLM-6B 约 **20%** 正确率，说明小模型也有一点调用能力。
- GPT-3 Davinci 几乎不会用（总正确率 **0.57%**）。作者猜测调用强依赖指令理解，而 GPT-3 没做指令微调。
- GPT-3.5 在 Call 上比 Alpaca-7B 高约 **35** 个百分点（59.40−24.06），Rouge-L 高约 **0.44**。Retrieve+Call 相对纯 Call 再掉约 **21** 个百分点（59.40−38.52），Plan+Retrieve+Call 再掉约 **17** 个百分点（38.52−22.00）——选哪个、怎么规划超出「读懂指令」。
- GPT-4 相对 GPT-3.5：Call 高约 **4** 个百分点；Retrieve+Call 差不多；最难的 Plan+Retrieve+Call 从 **22%** 到 **70%**，作者写成「近 50%」的提升（PDF p. 8）。总正确率 GPT-4 也最高，但总 Rouge-L（**0.3910**）低于 GPT-3.5（**0.4267**）——调用对了不自动等于回复更像标注。
- Lynx 相对 Alpaca-7B：Call 正确率 49.87−24.06=**25.81** 个百分点，摘要写成「超过 26 pts」；Call 的 Rouge-L 从 0.0204 到 0.4332，文中写高 **0.41**（PDF p. 1、p. 8）。引言另写三档能力平均提升 **24%**、距 GPT-4 仍有 **21%** 差距：三档正确率差的算术平均约 (25.81+25.18+20.00)/3≈23.7；总正确率 60.24−39.58=20.66。两句话切的不是同一列，不要混成一个数（PDF p. 2、p. 7）。

作者还拿自己的人标评测集对比同期自动生成评测：GPT-3.5 在那些集上调用准确率能到 **80%–90%**，在 API-Bank 上仍有明显空间，因为评测是按设计原则手标、更像真实场景，对方评测偏窄域、self-instruct（PDF p. 8）。此处原文点名的是 APIBench 与 ToolAlpaca，不是后出的其他榜。

表 7：把 ToolAlpaca 训练集转成同一格式得到 **10,366** 条，微调 Alpaca 后 Call 准确率 **53.88**、Rouge **39.75**；Lynx 用 **6,184** 条训练样本得到 **54.64 / 39.80**。只比 Call、不比检索，因为 ToolAlpaca 流程不含 API Retrieval（PDF p. 9）。表 7 的 Rouge 量级是百分数写法，与表 3 的 0–1 小数不同，不要直接横比。

## 六、错在哪：Alpaca 不调、Lynx 幻觉、GPT-4 搜不准

六类错误定义在附录 A.1（PDF p. 12）：API Hallucination（预测里的 API 名对不上标注）；Has Exception（触发本不该有的 Python 异常）；Invalid Input Parameters；False API Call Format（解析不了）；No API Call；Missing Input Parameters。

表 4 Alpaca（PDF p. 8）：No API Call **36.77%**，False API Call Format **23.65%**，API Hallucination **15.93%**，Invalid Input Parameters **7.96%**，Miss Input Parameters **1.17%**。微调后「根本不调」大幅下降；作者认为 Alpaca 那 52K 指令里的调用模式和评测系统对不上，零样本提示又把格式全压在说明上（PDF p. 8）。

表 5 Lynx（PDF p. 8）：API Hallucination **61.38%**，Has Exception **16.40%**，Invalid Input Parameters **8.47%**，False API Call Format **6.88%**，No API Call **5.29%**，Miss Input Parameters **1.59%**。正文把最大类写成「API 名对不上」，占约 **61%**：有时造出与用户意图无关、提示里也没有的假 API——训练见过的名字在测试里幻出来（PDF p. 8）。参数问题（异常、非法参数、解析失败）合计约 **32%**：占位符当值、日期格式不合法、缺参、要股票代码却传公司名等（PDF p. 9）。还有少量伪造用户/AI 话语而不调用（PDF p. 9）。

表 6 GPT-4（PDF p. 8）：Failed API Retrieval **67.86%**，False API Call Format **17.86%**，Invalid Input Parameters 与 Miss Input Parameters 各 **7.14%**。主要问题是不会用 API Search 找到该用的 API，约占 **68%**；微调后的 Lynx 评测里没出现这类主导错误——微调比较教得会 Retrieve+Call 流水线，In-Context Learning 则难把调用控成预期格式（PDF p. 9）。GPT-4 第二常见是调用解析失败：有时一次吐多个调用，违反「每次停、等返回」的提示（PDF p. 9）。

三条后续方向（PDF p. 9）：（1）直接生成调用受 API 数量限制，外挂检索又容易幻觉、检索调用本身不准；（2）解码时要卡死参数定义；（3）Lynx 已有苗头，更大训练集会不会更好，文中没做。

## 七、限制、伦理，以及文中没写的

限制（PDF p. 10）：只做英语；只微调了 Lynx-7B，没探更大骨干，尽管 7B 已接近 GPT-3.5；公司内部用更大 LLM 训过可商用工具模型，因匿名原因不报在线数字，留待后续版本。伦理（PDF p. 10）：访谈事先告知用于产品与论文、不披露个人隐私；数据集里 API 均为原创实现、不侵犯现有商业软件；评测标注四人，时薪 **15** 美元，高于当地法定最低。

相关工作第 6 节把 Toolformer、WebGPT、ReAct、ChatGPT Plugins、TaskMatrix.AI 等收成「用外部模块补权重里的知识」，并再次把三问当作本文切口（PDF p. 6–7）。这些是背景引用，不是 API-Bank 自己的实验。

## 八、可迁移的几条

1. **先定能力再堆 API。** 池子大小和一轮步数决定你到底在测填槽、检索还是规划；四格合并成三档，是因为「文档都在手里时多步并不难」（PDF p. 3）。自己做工具评测时，先问用户会不会把几百个私有 API 丢过来、会不会一句下完复杂单。
2. **对执行，不对文风。** 一致性看库变没变、返回一不一致；回复另用 ROUGE-L。GPT-4 总正确率最高、总 Rouge 却低于 GPT-3.5，说明两条轴会分叉（PDF p. 4、p. 7）。
3. **检索必须可复现。** 外部搜索的结果要按测试查询钉死时刻、硬编码，否则同一条对话下次分数会漂（PDF p. 4）。
4. **合成数据要拆步骤、要有过滤器。** 一条指令生成工具数据，ChatGPT/GPT-4 可用率低；五步加 tester 丢掉 35%，人抽检 94% 可用（PDF p. 5–6）。代价是幻觉：Lynx 会把训练见过的 API 名造进测试（PDF p. 8）。
5. **指令微调几乎是调用的开关。** 175B 的 Davinci 总正确率 0.57%，7B 指令模型约 20% Call；格式错误靠微调降得最明显（PDF p. 7–8）。规划仍更像推理：GPT-4 的优势集中在 Plan+Retrieve+Call（PDF p. 8）。
6. **报提升时写清分母。** 「26 pts」是 Call 正确率差，「24%」是三档平均，「21%」是总正确率相对 GPT-4。同一段摘要里三个数字，混用会读成互相打架（PDF p. 1–2、p. 7）。

## 关键词回看

- **工具增强 LLM（tool-augmented LLM）**：生成之外再调外部 API，补过时知识和第三方服务（PDF p. 1）。
- **Call / Retrieve+Call / Plan+Retrieve+Call**：已知填槽；未知先搜再调一次；未知连续规划多步（PDF p. 3）。
- **API Search / ToolSearcher**：对关键词与 API 元信息做句向量余弦，返回最相似文档；未知 API 设定下每次真调用前必搜（PDF p. 4、p. 14）。
- **执行一致性**：预测与标注是否造成相同的库读写和相同返回，不是字符串全等（PDF p. 4）。
- **Multi-agent**：域 → API → 查询 → 调用与回复 → tester，用 ChatGPT 降标注成本（PDF p. 5）。
- **Lynx**：Alpaca-7B 初始化、在 API-Bank 训练集上微调的 7B 工具模型（PDF p. 7）。
- **API Hallucination**：调用名对不上标注，含造出提示里没有的 API（PDF p. 8、p. 12）。

## 参考资料

- 原论文：Li 等，`API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs`，arXiv:2304.08244v2。
- 仓库（原文给出）：`https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank`。
- 训练 API 真实性用的例子来源（原文脚注）：`https://github.com/public-apis/public-apis`。

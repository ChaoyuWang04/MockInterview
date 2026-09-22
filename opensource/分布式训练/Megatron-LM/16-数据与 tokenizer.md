# 数据与 tokenizer

这一页回答一个问题:磁盘里的文档怎样变成可重复抽取的训练样本,再成为各 DP rank 的 batch。说法都在源码基准 `fb6a123a09`(tag `core_v0.19.2`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

**文档长度、训练序列长度和 batch 大小是三件事。** 一篇长文档可能贡献多个样本,一个样本也可能跨过多篇短文档。还要让训练与验证分开、混合多个数据来源、恢复已经消费的样本位置,不能只把文件逐行送给模型。

tokenizer 负责把内容变成 token ID,数据集负责决定取哪些 ID、如何组织边界。预处理好的 ID 不能因为训练时换了 tokenizer 就自动变成另一套词表。

## 二、解法

**底层把内容与定位信息分开。** indexed dataset 的数据文件存 token 等序列内容,索引文件存序列长度、字节偏移、文档对应的序列范围等元数据。普通 GPT 预训练路径读的是已经编码的 ID,不是每个 iteration 都重新对原始文本分词。

在选定训练或验证的数据范围后,GPT 数据集建立三层映射:文档索引安排多轮数据中的文档顺序;样本索引用「文档序号、文档内偏移」标出每个样本的起止;shuffle 索引再把外部样本编号映射到实际样本。**打乱样本次序不需要重写整份 token 数据。**

![文档顺序为A再B,长度4的样本跨越文档边界并额外读取1个token,形成错开1位的输入与标签;shuffle先选样本,样本边界再定位文档。](/opensource/Megatron-LM/16a-indexed-sample.svg)

默认 GPT 样本多读 1 个 token,前面的部分作为输入,后移 1 位作为 labels。跨文档时分别读取文档尾部、中间完整文档与下一篇开头,再拼接。**拼起来不代表边界自动隔离**:是否重置位置、阻止跨文档 attention、屏蔽特定 loss 位置,由相应配置决定。EOD 是 tokenizer 定义的特殊 ID;预处理可以显式追加,不能假设任何来源都有它。

多数据源混合在外面再加一层映射:先选数据集,再选该数据集中的样本。到 DataLoader 层,sampler 根据已经消费的样本数、microbatch 大小和 DP rank 分配样本编号,随后组装各 rank 的 batch。模型并行中的数据分发属于后续训练执行,不要把每个 GPU 都理解成独立读取一套不同样本。

## 三、代价

**首次建索引与稳定读样本的成本不同。** 文档、样本和shuffle映射可以缓存到磁盘,后续通过内存映射加载。常规分布式构建先让 rank 0 建立所需数据集,经过 barrier 后其他参与构建的 rank 再加载;不需要数据集的 rank 也要按约定参与同步。绕过整个构建调用可能让其他rank一直等待。

缓存省的是重复构建工作,不是把训练数据都放进显存。文件系统读带宽、页缓存缺失、CPU拼接与mask生成、worker调度都可能成为开销。盲目增加worker也会增加内存和存储并发压力,应先区分启动慢与每步取batch慢。

**边界策略会改变训练语义。** 连续文档拼接、独立文档attention、SFT中只对回答计loss并不是同一种样本。仓库另有SFT和变长packing路径,会携带序列边界等信息;FIM则改变样本内容组织。它们不能仅凭输出长度相同就与普通GPT样本互换。

## 四、和同类常见做法不一样的地方

DeepSpeed引擎可以接收调用方的Dataset,并据batch与并行信息构造DataLoader。Megatron本章则进一步提供面向语言模型的样本组织:从已编码文档建立索引,再把样本分给DP rank。

| 对比维度 | DeepSpeed引擎的数据接入路径 | Megatron GPT数据路径 |
|---|---|---|
| 接入的数据抽象 | 调用方提供Dataset,可提供sampler与collate函数 | indexed dataset存内容,GPT数据集建立文档、样本与shuffle映射 |
| 样本语义由谁决定 | Dataset与collate定义具体训练样本 | GPT路径实现跨文档取片、额外token和移位labels,其他任务另有数据集 |
| DP分工如何接入 | DataLoader构造使用batch大小与数据并行信息 | sampler结合已消费样本数、microbatch与DP rank分配编号 |
| 启动时重点检查什么 | Dataset初始化、sampler及worker设置 | 还要检查索引缓存命中、构建rank与barrier参与关系 |

这里比较的是引擎入口与GPT数据管线,不概括DeepSpeed全部数据处理工具。两边都需要调用方保持tokenizer、词表和任务语义一致。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

| 参数与入口 | 默认 | 调整影响 | 怎么看 |
|---|---|---|---|
| CLI `--seq-length` | 无,需由模型/训练配置确定 | 改变GPT样本跨度与索引构建,不等于文档长度 | 样本长度、有效token数、索引缓存命中 |
| CLI `--data-cache-path` | 无 | 指定索引缓存目录;普通非mock GPT路径未指定时使用数据前缀下的缓存目录 | Build/Load indices日志、各rank路径可见性 |
| CLI `--dataloader-fast-cache-load` | 关 | 假定缓存已完整准备,跳过部分存在检查和构建同步 | 缓存缺失错误、启动时间;不用于第一次建缓存 |
| CLI `--dataloader-defer-npy-index-mmap` | 关 | 延迟索引文件mmap到首次访问;同样要求预建缓存 | 启动耗时与第一次取样本延迟 |
| CLI `--no-mmap-bin-files` | 未设置,默认mmap | 关闭数据文件mmap读取 | 缺页、读带宽、进程内存与batch等待 |
| CLI `--num-workers` | 2 | 改DataLoader并发,不是dataset builder线程数 | CPU占用、主机内存、batch等待时间 |
| CLI `--num-dataset-builder-threads` | 1 | 改数据集构建并发 | 索引构建时间与文件系统负载 |
| CLI `--reset-position-ids` / `--reset-attention-mask` | 关 | 在EOD之后分别重置位置或隔离attention | 构造含EOD的小样本核position与mask |
| CLI `--eod-mask-loss` | 关 | GPT mask构造器对输入序列中EOD位置置零,不要泛称删除所有跨文档训练目标 | 直接核tokens、labels、loss_mask的对应位置 |
| CLI `--dataloader-inter-document-masking` | 关 | 返回文档边界的 `cu_seqlens`,供支持该路径的attention隔离文档 | 边界数组与下游attention兼容性 |
| API `MegatronTokenizer.from_pretrained` | 路径/metadata按所选实现提供 | 由metadata选tokenizer适配器;暴露tokenize、detokenize、词表与chat template接口 | 特殊ID、词表大小、模板输出与既有数据一致性 |

**典型配置**。以下为已有可运行GPT训练命令的增量;假设数据已经用匹配的tokenizer预处理,模型与global batch不变。缓存目录由作业提供,不在这里写真实路径。

| 场景与前提 | 配置起点 | 验证目标 |
|---|---|---|
| 首次使用一套数据和样本长度,所有参与rank可访问同一缓存位置 | 设置 `--data-cache-path "$DATA_INDEX_CACHE"`,保持fast-cache关闭 | 成功构建后记录样本数、缓存文件与启动时间 |
| 上一组缓存完整且训练数据/配置未改变,启动检查开销明显 | 保持缓存路径,加 `--dataloader-fast-cache-load` | 确认实际命中原缓存,比较启动与首batch延迟 |
| 稳态取batch等待明显且CPU/存储仍有余量 | 从 `--num-workers 2` 对照 `--num-workers 4` | 固定有效token数比较step时间与内存,无改善就不继续增加 |

组合与推荐值是按语义推的起点,不是实测最优。

| 症状 | 先查 | 然后 |
|---|---|---|
| 首次启动长时间等待 | 索引是否仍在构建、所有rank是否进入builder同步 | 核缓存目录可见性;已有完整缓存后再考虑fast-cache |
| 稳态GPU等待batch | worker的CPU负载、读带宽与主机内存 | 固定样本配方对照worker数,分别测拼接/mask与读取开销 |
| 文档边界行为不符预期 | tokens中的EOD、position与attention/loss mask | 用含边界的小样本核对各独立开关,不要只看序列长度 |

预处理入口 `tools/preprocess_data.py` 调用统一tokenizer builder,写出indexed dataset;`--append-eod`决定是否追加EOD。训练侧选择tokenizer实现时,应核对预处理版本、特殊token ID与chat template,而不只核文件名相同。

## 六、常见误区

- **“三类索引就是三个数据副本。”** 文档顺序、样本边界与shuffle映射都引用底层内容;数据源混合还有自己的两类映射,不能把所有索引混叫一个shuffle。
- **“一个样本就是一篇文档。”** 普通GPT路径允许跨文档拼接,默认还多读1个token生成移位labels。
- **“有EOD就自动隔开attention。”** EOD本身是一个ID;position、attention与loss各有处理条件。
- **“fast-cache能自动修好缺失缓存。”** 它依赖缓存已准备好,不是更积极的重建模式。
- **“数据慢只能换更快磁盘。”** 还应测CPU拼接、mask、分词或模板处理所在路径以及worker等待;普通预tokenized GPT与在线处理的SFT路径成本不同。

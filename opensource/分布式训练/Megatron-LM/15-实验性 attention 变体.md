# 实验性 attention 变体

这一页回答一个问题:长序列attention怎样少看一些位置,以及这会怎样改变模型装配与训练边界。说法都在源码基准 `fb6a123a09`(tag `core_v0.19.2`)上核过,「当前」与「默认」都指这组基准。

## 一、核心问题

序列变长时,每个query都与全部可见key计算attention会扩大工作量。仅仅换更快的kernel并没有改变“看哪些位置”。稀疏与压缩attention进一步改变候选集合,但选择器怎么训练、压缩位置什么时候可见、模型怎样装配,都要一起处理。

Megatron在实验性attention目录中接入了两条路线:**DSA由indexer选择原始token位置;DSv4混合attention则把局部窗口与压缩KV结合**。它们改变了attention的计算结构,不是普通attention的一组通用加速开关。

## 二、解法

DSA的indexer从hidden state与低秩query产生自己的query、key和head权重,计算排序分数,在causal或sequence边界约束内选top-k位置。随后主attention用自己的Q/K重新计算这些位置的权重。**indexer分数负责选位置,不直接作为最终attention权重**。

离散top-k选择本身不能替代indexer训练。启用辅助KL loss后,indexer学习匹配主attention提供的teacher分布;indexer输入与teacher路径会detach,避免这条辅助梯度直接改动主干。实现还允许后续层复用前一计算层的索引,但不能跨PP stage取这份结果。

DSv4的CSA先用可学习的gated pooling产生压缩KV,再把压缩候选与最近的原始token窗口合并。4倍压缩会重叠相邻分组,并由独立indexer选择压缩位置;128倍压缩不构建这个indexer,而看全部因果有效的压缩位置;不压缩的层只保留窗口。即使压缩块已经在整段forward中算出,也要等其中对应分组结束,才能对该query可见。

![DSA在原始位置上选top-k;CSA示例先以相邻分组重叠的4倍压缩生成C0到C3,再选择C0和C2,与query15的局部窗口12到15合并。候选索引随后交给主attention重新计算权重。](/opensource/Megatron-LM/15a-selected-and-compressed-context.svg)

CSA的indexer有自己的压缩表示,用于检索;主attention使用另一个compressor生成的KV,两者按压缩位置对应。主attention还包含每个head的可学习attention sink:它参与softmax分母,不贡献一个普通token的value。图中的两路候选在同一次attention中竞争权重,不是先各自归一化再简单相加。

这些模块已经接到当前源码的真实构造链。GPT builder根据变体配置生成layer spec;DSA接吸收式MLA与稀疏core attention,DSv4接压缩core attention。HybridModel也能按层配置装配DSA、CSA、重压缩与窗口层。GDN共享实验性入口,其递归状态与混合层编排见14章,这里不重讲08章的普通attention投影。

## 三、代价

**少看位置要先付选择与压缩成本。** indexer评分、top-k、compressor和辅助loss都增加工作,选择过少也可能损失有效上下文。尤其DSA的PyTorch参考路径仍可能先形成完整分数再mask,语义稀疏不保证执行成本已经按top-k缩小。

**压缩比例改变信息表示。** 4倍路径不是每4个token直接取平均;它学习gated权重并重叠相邻分组。窗口保留局部细节,压缩候选承载较远上下文,但二者的组合仍需用主loss和验证结果检验。

**当前实现有明确适用范围。** DSv4原生路径要求Transformer Engine与单路TP、CP,不接受packed sequence、外部padding或document边界mask,也不支持推理context、core attention checkpoint和QKV linear offload。代码存在与完整生产训练能力是两回事;不要把普通attention的所有并行、内存优化和推理配置原样搬进来。

## 四、和同类常见做法不一样的地方

同样面对长序列,DeepSpeed Ulysses与本章的稀疏/压缩attention处理的是不同维度:前者重新分布计算,后者改变参与计算的候选表示。这个对照用于区分并行化与模型算法变化,不作性能排名。

| 对比维度 | DeepSpeed Ulysses路径 | Megatron本章DSA/CSA路径 |
|---|---|---|
| 改变什么 | 在序列与head切分布局间交换数据,交给attention计算 | DSA选择原始位置,CSA组合压缩位置与局部窗口 |
| 是否引入新的候选选择 | 布局交换本身不引入本章的top-k indexer或压缩KV | 选择器、compressor及辅助训练共同决定候选集合 |
| 主要增加的工作 | 跨rank布局交换及其通信约束 | indexer评分、top-k、压缩与辅助loss,还受后端限制 |
| 验证的重点 | 布局转换前后attention语义与通信代价 | 候选语义、训练质量及选择成本,不能只看稀疏比例 |

两种思路在概念上可以组合,但本基准的DSv4路径限制TP和CP均为1;不能由这个对照推导出它已支持Ulysses或任意序列并行。

## 五、调参与观测

这一页的参数全是启动配置,改了要重新拉起整个作业。

以下为参考训练CLI与其映射的Core字段。它们用于构造相应架构,不是把任意已有checkpoint无损转换成稀疏模型。

| 参数 / 在哪调 | 默认 | 调整的作用与代价 | 怎么看 |
|---|---|---|---|
| CLI `--experimental-attention-variant` | 无 | 本章为 `dsa`、`dsv4_hybrid`;改变实际模块spec | 模型构造结果与启动校验 |
| CLI `--dsa-indexer-n-heads` / `--dsa-indexer-head-dim` / `--dsa-indexer-topk` | 都为无 | 需要indexer的路径必须明确尺寸与候选数;增大增加选择成本 | indexer时间、有效候选数、主loss |
| CLI `--dsa-indexer-loss-coeff` | 无,执行按0处理 | 正系数启用indexer KL辅助训练 | `indexer loss`及主/验证loss |
| CLI `--dsa-indexer-use-sparse-loss` | 关 | 开后辅助loss在选中位置上计算;与看全部有效位置的teacher目标不同 | 质量与辅助loss成本,不要只比数值大小 |
| CLI `--dsa-indexer-topk-freq` / `--dsa-indexer-skip-topk-offset` | 1 / 0 | DSA跨层复用索引,省部分indexer计算;source必须先出现在同一PP stage | 实际计算层与stage边界 |
| CLI `--dsa-kernel-backend` | 原始无;DSA归一为 `none`,DSv4归一为 `cudnn` | 选择可选融合实现或PyTorch参考路径;未支持layout可能fallback,缺依赖会报错 | 后端日志与trace |
| CLI `--csa-compress-ratios` | 无 | DSv4逐层列表,仅0、4、128;decoder占前段,MTP用尾段 | 长度、层号与真实层类型 |
| CLI `--csa-window-size` | 128 | 增大保留更多原始局部位置,增加attention工作 | 局部窗口成本与质量 |
| CLI `--csa-dense-mode` | 关 | 关闭4倍压缩路径的indexer,看全部因果有效压缩位置;仍保留压缩与窗口 | 不要误当成原始全序列dense attention |
| CLI `--csa-compress-rotary-base` | 40000 | 压缩层使用的RoPE base,应与模型配方一致 | 位置语义与验证结果 |
| CLI `--output-projection-groups` / `--output-projection-lora-rank` | 8 / 1024 | DSv4分组低秩输出投影尺寸;head总维需整除group数 | checkpoint形状及初始化校验 |

**兼容条件先按变体分开。**

| 路径 | 当前边界 |
|---|---|
| DSA | 要求MLA、无linear bias、关闭RoPE fusion;CP大于1时仅支持all-gather方式。不要把DSv4的TP1/CP1限制直接套到DSA |
| DSv4 | 要求MLA、TE、TP1、CP1;禁用QK clipping与MLA down-projection fusion;value head维必须大于RoPE维。只接原生固定batch/sequence布局及隐式causal mask |
| DSv4融合后端 | 不支持 `tilelang`;`cudnn`要求SM90及以后且相关可选包齐全。SM90上,4倍压缩indexer加正loss系数时不能用dense indexer loss,需sparse loss或参考后端 |

DSv4在本基准中已经存在,不是仅在dev分支的占位实现。HybridModel的 `D`、`C`、`H`、`W` 分别对应DSA、4倍压缩、128倍压缩、仅窗口;这些符号的层计数方式按14章处理,不要直接照GPT block数套列表。

**配置起点**。假设已有4层BF16 GPT MLA decoder实验配方,sequence length为256,TP=CP=PP=1,使用TE,不启用MTP、packing、recompute或offload。下面是Hopper H100环境的增量,须满足已安装库的版本与硬件要求,不是完整训练命令;DSA与DSv4分开建模型。

| 场景 | CLI增量 | 取舍 |
|---|---|---|
| DSA参考路径核对选择与loss | `--experimental-attention-variant dsa --disable-bias-linear --no-rope-fusion --dsa-kernel-backend none --dsa-indexer-n-heads 4 --dsa-indexer-head-dim 128 --dsa-indexer-topk 16 --dsa-indexer-loss-coeff 0.1` | 不依赖融合DSA后端,先验证选择与辅助loss;参考路径不代表稀疏吞吐收益 |
| DSv4混合窗口/压缩参考路径 | `--experimental-attention-variant dsv4_hybrid --csa-compress-ratios '[0,4,128,4]' --csa-window-size 16 --dsa-kernel-backend none --dsa-indexer-n-heads 4 --dsa-indexer-head-dim 128 --dsa-indexer-topk 8 --dsa-indexer-loss-coeff 0.1` | 在同模型中区分三种层,先验证结构和数值;保持MLA down-projection fusion与QK clipping关闭 |

组合与推荐值是按语义推的起点,不是实测最优。

**看哪里。** 训练日志和TensorBoard记录 `indexer loss`,同时看主loss与验证loss。CSA提供 `compressor`、`indexer_before_topk`、`compressed_indices`、`sparse_attn_kernel` 等NVTX区域,可拆开压缩、选择与主attention成本。DSA在DEBUG级别可记录融合hook declined及fallback原因。配置了后端不等于每个layout都命中融合kernel;也别用更低的辅助loss替代端到端质量验证。

| 症状 | 先查 | 然后 |
|---|---|---|
| indexer存在却没有辅助训练信号 | loss系数是否仍为空或0,该层是否复用索引 | 显式设置系数并检查实际计算层 |
| top-k缩小但没加速 | 是否走参考路径或fallback,indexer耗时多大 | 看trace,再测试受支持融合后端 |
| DSv4启动即失败 | TP/CP、ratio列表、TE/硬件/依赖和mask类型 | 回到固定shape的参考配置逐项核对 |
| 开dense模式仍不是全序列attention | 该开关只跳过压缩indexer | 对照实际窗口和压缩位置集合 |
| 改PP布局后DSA索引复用失败 | source层是否跨stage或出现在后面 | 调整层布局或关闭跨层复用 |

## 六、常见误区

**把indexer的top-k分数直接画成attention权重。** 两者都给query-key打分,容易混在一起。真正传给主attention的是位置索引,主attention重算自己的分数并归一化。

**看到4倍压缩就画成互不相交的4-token平均池化。** 压缩比例描述输出位置数量,不代表每个位置只有4个输入;本实现还重叠前一组,并用可学习的gate做pooling。

**以为默认没填后端就一定是PyTorch参考路径。** DSA与DSv4的归一化默认不同。后者会选择cuDNN路径,因此新环境可能先报依赖或硬件错误,应查最终配置。

**看到训练模块和backward就认为所有训练组合都支持。** 这些入口证明存在可训练路径,不证明packing、并行、重计算或推理均兼容。先对照本章限制,再决定实验范围。

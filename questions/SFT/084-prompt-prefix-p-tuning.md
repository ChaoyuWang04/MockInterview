---
difficulty: 简单
topic: 提示微调/注入位置与实现
summary: Prompt、Prefix、P-Tuning 在 LLaMA 哪里注入,怎样处理位置与缓存
tags: [SFT, 提示微调, Prompt Tuning, Prefix Tuning, P-Tuning, 待校对]
company: 通义实验室
mastered: false
highfreq: false
---

## 题目

请结合 LLaMA 模型的源码结构,说明你对基于提示(prompt)的微调方法(如 Prompt Tuning、Prefix Tuning、P-Tuning)的理解及其实现原理。

## 要点

- 区分输入连续提示与各层注意力前缀,不要把它们当普通文本提示
- 说明冻结范围、训练参数、损失掩码与梯度路径
- 对应嵌入、Attention、RoPE和KV缓存的修改位置
- P-Tuning v2的深层注入与重参数化是不同选择,效果须实测

## 答案

**这些方法学习连续向量,区别在于向量从哪里影响冻结模型。** 普通文字 prompt 是离散 token,而软提示直接作为可训练向量参与计算,不必对应词表中的词。

### 注入位置与参数

| 方法 | 注入位置 | 主要训练对象 |
|---|---|---|
| Prompt Tuning | 输入嵌入序列前的软提示 | 长度 m、维度 d 的矩阵,约 md 参数 |
| Prefix-tuning | 各层注意力使用的前缀 K/V 状态 | 每层前缀及可选重参数化网络 |
| P-Tuning v1 | 输入中的连续提示,可与离散模板组合 | 提示向量及 LSTM/MLP 等编码器 |
| P-Tuning v2 | 把可训练提示推广到各层 | 深层提示,重参数化是否有益依任务决定 |

P-Tuning 原文也研究过连同语言模型一起训练的情形;谈 PEFT 配置时要明确基座冻结。冻结模型参数并不阻断梯度,训练仍需经过模型反向传播到提示。

### 对应 LLaMA 结构

本地 vLLM 的 LLaMA 实现可核对到 LlamaModel.embed_tokens、逐层 LlamaDecoderLayer、注意力 qkv_proj 及 rotary_emb。这些是结构定位依据,不是声称该推理文件已经实现软提示训练。

训练实现中,Prompt Tuning 应在 token embedding 后拼接软提示,同步调整 attention mask、位置编号及标签掩码,软提示位置不作为答案标签。Prefix-tuning 则给各层 K/V 增加前缀;必须明确 K 是否已经应用 RoPE,并统一位置、因果 mask 和缓存长度,生成时不能每步重复插入。GQA 要按 KV 头数量分配前缀,不能一律套 $2Lmd$。训练结束可预计算固定前缀状态,但每个生成 token 仍需关注它。

### 效果与选型

v2 的深层注入让高层直接得到可调信号,也增加适配容量;原论文改进主要在其模型和 NLU/序列标注任务上验证,不是证明所有 LLaMA 任务都更稳定。MLP 也并非总有收益。

输入软提示实现较简单、任务副本小;需要更直接影响中间层时可试 Prefix/v2,代价是更多前缀状态及注意力开销。LoRA 直接改权重增量且可合并消去分支,软提示通常仍占上下文或 K/V 位置。三者在同一 LLaMA 上应固定数据、训练预算和评测比较,没有可通用引用的效果排名;未做过实验就明确说明。

## 知识点

提示微调、Prompt Tuning、Prefix Tuning、P-Tuning。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q221。
- 依据:[Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243/)、[Prefix-tuning](https://aclanthology.org/2021.acl-long.353/)、[P-Tuning](https://arxiv.org/abs/2103.10385)、[P-Tuning v2](https://arxiv.org/abs/2110.07602)。
- 源码定位:projects/推理服务/vllm/vllm/model_executor/models/llama.py 的 LlamaModel、LlamaDecoderLayer 与 LlamaAttention;只核对架构接口。知识库「提示微调」正文待写。

## 追问

- Prompt Tuning 和 Prefix-tuning 在 LLaMA 上怎样比较效果与适用场景?
- P-Tuning v2 相比 v1 改了什么,是否保证稳定?
- 这些方法与 LoRA 各有什么优势和局限?

## Note

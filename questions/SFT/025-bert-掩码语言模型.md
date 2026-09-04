---
difficulty: 简单
topic: 预训练流程/掩码语言建模
summary: BERT 的 Encoder 架构、MLM与NSP怎样学习双向表示
tags: [真题, 待校对, BERT, MLM, 预训练]
company: 小红书、字节
mastered: false
highfreq: false
---

## 题目

请说明 BERT 的整体架构、输入表示和两个原始预训练任务,并解释双向 MLM 为什么适合理解任务、与 GPT 的因果语言模型有何区别。

## 要点

- BERT 是 Transformer Encoder-only,每层使用双向自注意力与 FFN
- 输入由 token、segment、position embedding 相加,并使用 `[CLS]`、`[SEP]`
- MLM 恢复被选中的 token,NSP 判断两段文本是否相邻
- 15% 与 80/10/10 是原论文设置,不是通用最优值
- 双向表示适合理解完整输入,因果目标更直接适合左到右生成

## 答案

**BERT 是只保留 Transformer Encoder 的双向表示模型。** 每个 token 能同时读取左右上下文,多层自注意力负责交换信息,FFN 逐位置变换表示。输入向量由 token、segment 和绝对 position embedding 相加;`[CLS]` 常承载整段表示,`[SEP]` 用于分隔句段。原论文的 Base 配置是 12 层、768 隐藏维、12 个头,Large 是 24 层、1024、16 个头;这些数字只属于两个原始配置。

### 两个原始预训练任务

MLM 随机选约 15% 的 token 作为预测目标;其中 80% 换成 `[MASK]`,10% 换成随机 token,10% 保持原样,三类都预测原词。隐藏目标迫使模型结合两侧语境,普通文本就能产生监督信号。NSP 则用 `[CLS]` 判断第二段是否紧接第一段,意在学习句间关系;后续 RoBERTa 移除 NSP 仍取得良好结果,说明它不是必需组件。

MLM 只监督被选位置,且 `[MASK]` 在实际输入中少见。比例太低时每条样本信号少,太高时上下文被破坏;15% 是实验配置,应按数据与目标验证。动态重采样 mask 能让同一句在不同 epoch 提供不同目标。

### 与 GPT 的差别

GPT 式因果模型只看左侧并预测下一个 token,训练信息流与逐 token 生成一致。BERT 可融合完整输入,常用于分类、抽取和检索表示。原生 BERT 不直接适合普通自回归生成,但“技术上不能生成”过强:改变 mask、增加 Decoder 或把其参数用于 Encoder-Decoder 后仍可参与生成。

## 知识点

BERT、Encoder-only、MLM、NSP、双向注意力、自回归语言建模。

- 依据:[BERT](https://arxiv.org/abs/1810.04805)、[RoBERTa](https://arxiv.org/abs/1907.11692)、[用 BERT checkpoint 初始化生成模型](https://arxiv.org/abs/1907.12461)。

## 追问

- 为什么 BERT 选择双向 Encoder,而 GPT 选择因果 Decoder?
- 15% 和 80/10/10 分别指什么,比例过高或过低会怎样?
- NSP 为什么受到质疑,后续模型怎样替代它?
- BERT 的绝对位置编码与 RoPE 有什么区别?
- `[CLS]` 直接做语义相似度为什么可能效果不好?

## Note

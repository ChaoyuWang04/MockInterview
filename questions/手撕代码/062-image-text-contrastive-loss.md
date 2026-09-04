---
difficulty: 中等
topic: 对比学习/图文对比损失
tags: [真题, 待校对, 手撕代码, CLIP, InfoNCE, 对比学习]
summary: 手写批内负样本的双向图文对比损失并解释温度
company: 小红书
mastered: false
highfreq: false
---

## 题目

给定一批图像嵌入和与其一一配对的文本嵌入，使用 PyTorch 实现批内负样本、温度缩放的图文对比损失。说明归一化、双向交叉熵、维度映射、温度和大规模负样本怎样处理。

## 要点

- 图像和文本先投影到相同维度并做 L2 归一化，点积才是余弦相似度。
- 相似度矩阵第 $i$ 行/列的正样本都是对角线 $(i,i)$，其余批内样本为负例。
- 图到文和文到图分别做交叉熵，再取平均，不能漏掉一个方向。
- 温度越小 logits 越尖锐，难负例梯度越大，也更容易放大噪声。
- 分布式 all-gather 要处理全局标签偏移、跨卡梯度和重复/假负例。

## 答案

**把 $B$ 对图文样本组成一个 $B\times B$ 相似度矩阵，然后要求每张图和每段文本都把配对对象排在对角线最高。**

```python
import torch
import torch.nn.functional as F

def image_text_contrastive_loss(image_emb, text_emb, temperature=0.07):
    if image_emb.ndim != 2 or text_emb.ndim != 2:
        raise ValueError("embeddings must be 2-D")
    if image_emb.shape != text_emb.shape:
        raise ValueError("project both modalities to the same (B, D) shape")
    if image_emb.shape[0] == 0 or image_emb.shape[1] == 0:
        raise ValueError("batch and embedding dimension must be non-empty")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    image = F.normalize(image_emb, dim=-1)
    text = F.normalize(text_emb, dim=-1)
    logits = image @ text.T / temperature       # (B, B)
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

对第 $i$ 张图，图到文损失为

$$
-\log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)},
$$

反方向对列做同样计算。两种模态原始维度不同时，应各自通过可训练投影层映射到共同维度，而不是截断向量。

大负样本可用多卡全局 batch、队列/记忆库或难负样本挖掘。队列里的嵌入可能陈旧；批内语义相同但未配对的样本可能是假负例。CLIP 还包含图像/文本编码器、投影头、可学习且通常受约束的 logit scale，以及大规模分布式训练；这段代码只实现核心损失。

## 知识点

InfoNCE、余弦相似度、批内负样本、双向交叉熵、温度、投影头、假负例。


## 追问

相关真题追问：

- 温度为什么会改变难负样本梯度，过小有什么风险？
- 两种模态维度不同时，为什么要用投影层而不是直接比较？
- 队列和全局 batch 扩大负样本时，怎样处理陈旧表示与跨卡标签？
- 这段损失与完整 CLIP 训练系统还差哪些部分？

## Note

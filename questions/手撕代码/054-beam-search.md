---
difficulty: 中等
topic: 解码算法/Beam Search
summary: 手写束搜索并解释剪枝、终止、长度归一化与适用边界
tags: [真题, 待校对, 手撕代码, Beam Search, 文本生成]
company: 蚂蚁金服、蚂蚁集团
mastered: false
highfreq: false
---

## 题目

使用 PyTorch 实现简化的 Beam Search：初始化候选，逐步扩展并保留全局 top-k，正确处理累计分数、EOS、长度归一化和最终选择。再说明怎样减少重复、与随机采样如何取舍，以及大词表下怎样降低 top-k 成本。

## 要点

- 每个 beam 的分数累加 log 概率；每轮在 `beam × vocabulary` 的全部扩展中选 top-k。
- EOS 序列进入完成集合，不再扩展；活跃候选为空或满足安全的提前停止条件才结束。
- 长度惩罚只用于比较候选，必须统一定义；不能直接拿不同长度的原始 log 概率比较。
- Beam Search 偏向高概率、确定性输出，不保证多样，也不等同于随机采样。
- 示例明确只支持 batch=1；批处理还要维护每个样本各自的 beam 与完成状态。

## 答案

**Beam Search 每一步保留累计分数最高的 k 条前缀，用更多计算换取比贪心搜索更大的搜索范围。** 下面假设模型接收 `(B,L)` 的 token，并返回最后位置的 logits；代码只演示 batch=1。

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def beam_search(model, prompt, beam_size, max_new_tokens, eos_id,
                length_penalty=0.7):
    if prompt.ndim != 2 or prompt.shape[0] != 1:
        raise ValueError("this simplified implementation requires batch size 1")
    if beam_size < 1 or max_new_tokens < 0 or length_penalty < 0:
        raise ValueError("invalid search parameters")
    # item: (token序列, 未归一化的累计log概率)
    active = [(prompt.clone(), 0.0)]
    finished = []

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in active:
            logp = F.log_softmax(model(seq).logits[:, -1, :], dim=-1)
            values, ids = torch.topk(logp[0], beam_size)
            for value, token in zip(values.tolist(), ids.tolist()):
                new_seq = torch.cat([seq, seq.new_tensor([[token]])], dim=1)
                candidates.append((new_seq, score + value, token == eos_id))

        if not candidates:
            break
        # 每个 beam 最多一个 EOS，因此全局 top-2k 足够找出 k 个非 EOS
        # 候选（如果确实存在），同时让 EOS 参与本轮排序。
        candidates.sort(key=lambda x: x[1], reverse=True)
        active = []
        for seq, score, ended in candidates[:2 * beam_size]:
            if ended:
                finished.append((seq, score))
            elif len(active) < beam_size:
                active.append((seq, score))
        # 完成集合只保留归一化分数最高的 beam_size 条，避免无限增长。
        finished.sort(
            key=lambda item: item[1] /
            max(1, item[0].shape[1] - prompt.shape[1]) ** length_penalty,
            reverse=True,
        )
        finished = finished[:beam_size]
        if not active:
            break

    pool = finished + active
    prompt_len = prompt.shape[1]
    def normalized(item):
        seq, score = item
        generated_len = max(1, seq.shape[1] - prompt_len)
        return score / (generated_len ** length_penalty)
    return max(pool, key=normalized)[0]
```

为保持示例简短，它没有实现批处理、KV cache 和严格的 early stopping。它将完成集合与活跃集合分开维护，因此可能继续搜索一些最终不会胜出的前缀；生产实现可比较“最佳活跃候选的理论上界”和已完成候选，在保证结果不变时提前结束，并避免对每个 beam 重算完整前缀。

重复可以用 repetition penalty、no-repeat n-gram 或多样性束搜索缓解，但这些约束会改变搜索目标。翻译、转写等输出较确定的任务常用 Beam Search；开放式创作通常更适合 temperature、top-k/top-p 采样。词表很大时，可先在每个 beam 内取较小 top-k，再合并候选，或使用分块/近似 top-k；输出投影本身仍可能是主要成本。

## 知识点

Beam Search、累计 log 概率、全局剪枝、EOS、长度惩罚、重复约束、随机采样。


## 追问

相关真题追问：

- 重复生成怎样用 repetition penalty、n-gram blocking 或 Diverse Beam Search 缓解？
- Beam Search、贪心和 temperature/top-p 采样分别适合什么任务？
- 词表很大时，怎样减少输出投影与 top-k 的计算和内存开销？
- 长度惩罚为什么需要统一作用于已完成和未完成候选？

## Note

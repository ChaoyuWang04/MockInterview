---
difficulty: 中等
topic: 博弈搜索/Minimax
tags: [面经, 待校对, 手撕代码, 博弈论, Minimax, Alpha-Beta]
summary: 用 Minimax 判断有限零和博弈胜负并说明剪枝与纳什边界
company: 拼多多
mastered: false
highfreq: false
---

## 题目

两人零和有限博弈中，双方都采用最优策略，怎样用 Minimax 判断必胜、必败或平局？说明 Alpha-Beta、评估函数、MCTS 的扩展，以及 Minimax 与纳什均衡在什么条件下相关。

## 要点

- 对确定性、完全信息、轮流行动的有限博弈，终局效用可设为胜 1、平 0、负 -1。
- MAX 节点取子状态最大值，MIN 节点取最小值；根值即双方最优下的结果。
- 记忆化处理合流状态；Alpha-Beta 保持结果不变，只剪去不可能影响决策的分支。
- 理想排序下 Alpha-Beta 可接近 $O(b^{d/2})$，最坏仍是 $O(b^d)$。
- 深度截断需要评估函数；MCTS 是大状态空间的近似搜索，不给一般性的必胜证明。

## 答案

**Minimax 从终局向前倒推：我方选对自己最有利的分支，对手也总选对我方最不利的分支。**

```python
def minimax(state, maximizing, alpha=float("-inf"), beta=float("inf")):
    if state.is_terminal():
        return state.utility()       # MAX胜=1，平=0，负=-1

    if maximizing:
        value = float("-inf")
        for nxt in state.successors():
            value = max(value, minimax(nxt, False, alpha, beta))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = float("inf")
    for nxt in state.successors():
        value = min(value, minimax(nxt, True, alpha, beta))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value
```

若根节点返回 1，MAX 有保证胜利的策略；返回 0，MAX 至少能保平但无法强制获胜；返回 -1，对手能强制获胜。真实游戏常有不同分差，这时效用可以是任意零和数值，判断动作仍取 argmax。

状态空间大时先做置换表/记忆化、对称状态归一化和更好的走法排序。深度受限时用评估函数近似非终局价值，评估误差会改变决策；迭代加深便于按时间预算返回当前最好结果。MCTS 用采样集中探索有希望的分支，适合分支大或评估困难的场景，但有限采样不等于最优性证明。

在有限两人零和博弈中，允许混合策略时 Minimax 定理给出双方相同的博弈值，并存在纳什均衡。上面的树搜索还额外假设确定性、完全信息、顺序行动；同时行动或不完全信息要改用策略分布和信息集方法。

## 知识点

零和博弈、Minimax、最优策略、Alpha-Beta、走法排序、评估函数、MCTS、混合策略纳什均衡。

- 面经原题：[B006-G01-Q238](../../docs/references/面经原题.md#b006-g01-q238)。
- 老师答案参考：[P009-Q238](../../docs/references/平台题/P009-LC-161-241.md#p009-q238)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 状态空间很大时，Alpha-Beta、记忆化、迭代加深和 MCTS 怎样取舍？
- 非终止状态的评估函数怎样训练和校准？
- Minimax 与纳什均衡的等价需要哪些“有限、两人、零和、混合策略”条件？

## Note

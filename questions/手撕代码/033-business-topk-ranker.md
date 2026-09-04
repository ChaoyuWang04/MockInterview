---
difficulty: 简单
topic: 业务算法/多目标排序TopK
summary: 把电商候选排序建模为可配置打分和稳定TopK并补充多样性重排
tags: [面经, 待校对, TopK, 推荐系统, 堆]
company: 美团
mastered: false
highfreq: false
---

## 题目

请根据一个具体的业务场景（如电商推荐排序、广告出价预估等），现场设计算法逻辑并手写实现核心代码，要求考虑边界条件、数据结构选择与时间复杂度优化。

## 要点

- 先明确目标、候选规模、展示位、延迟约束和离线评估指标。
- 打分权重、归一化尺度与冷启动先验都应配置或通过训练/实验确定。
- 大小为 K 的小根堆把全排序 $O(N\log N)$ 降为 $O(N\log K)$。
- 堆项必须有稳定的唯一 tie-breaker；多样性需要真实参与重排。

## 答案

以电商精排后的 Top-K 为例。下面把多目标分数参数化，并用输入次序处理同分，避免 Python 在分数相同时比较不可排序对象。

```python
from dataclasses import dataclass
from heapq import heappush, heapreplace
from math import isfinite


@dataclass(frozen=True)
class Item:
    item_id: str
    ctr: float | None
    cvr: float | None
    price: float
    category: str


def top_k_items(
    candidates: list[Item],
    k: int,
    weights: tuple[float, float, float],
    priors: tuple[float, float],
    gmv_scale: float,
) -> list[Item]:
    if k <= 0 or not candidates:
        return []
    if gmv_scale <= 0:
        raise ValueError("gmv_scale must be positive")

    w_ctr, w_cvr, w_gmv = weights
    prior_ctr, prior_cvr = priors
    heap: list[tuple[float, int, Item]] = []

    for order, item in enumerate(candidates):
        if item.price < 0:
            raise ValueError("price must be non-negative")
        ctr = prior_ctr if item.ctr is None else item.ctr
        cvr = prior_cvr if item.cvr is None else item.cvr
        score = w_ctr * ctr + w_cvr * cvr + w_gmv * (ctr * cvr * item.price / gmv_scale)
        if not isfinite(score):
            continue

        entry = (score, -order, item)
        if len(heap) < min(k, len(candidates)):
            heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapreplace(heap, entry)

    heap.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return [item for _, _, item in heap]
```

遍历 N 个候选并维护不超过 K 个元素的堆，时间 $O(N\log K)$、辅助空间 $O(K)$。`gmv_scale`、权重和缺失值先验来自离线训练或线上实验；不能把乘 10、除 100、新物品乘 1.2 等固定数字写死为通用规律。

多样性不是声明一个未使用的 `diversity_weight` 就完成了。一个可审计方案是先取 Top-M，再按类别上限、MMR 或子模目标贪心重排；规则、候选扩大倍数和不足 K 时的回填策略都要写进契约，并通过 A/B 测试评估收益与护栏指标。

## 知识点

- 多目标排序、Top-K 小根堆、同分键、缺失值、冷启动、多样性重排、A/B 测试。

- 面经原题：[B006-G01-Q077](../../docs/references/面经原题.md#b006-g01-q077)；老师答案参考：[P009-Q077](../../docs/references/平台题/P009-LC-001-080.md#p009-q077)。

## 追问

以下均为平台页面追问，不计入面经原题：

- 候选达到百万级时怎样分层、分片和归并？
- 如何实现实时个性化权重调整？
- 如何设计 A/B 测试验证排序效果？

## Note

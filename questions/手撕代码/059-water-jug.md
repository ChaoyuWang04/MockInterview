---
difficulty: 中等
topic: 图搜索/水桶量水
tags: [真题, 待校对, 手撕代码, BFS, 最大公约数, 状态搜索]
summary: 判断两桶量水是否可行并用 BFS 找到最少操作路径
company: 快手
mastered: false
highfreq: false
---

## 题目

容量为非负整数 $a,b$ 的两个水桶可以装满、倒空和相互倾倒。怎样量出非负整数 $c$ 升水？先明确目标是“某一桶恰有 $c$”还是“两桶总量为 $c$”，给出可解条件，并用 BFS 返回最少操作步骤。

## 要点

- 可解性的代数核心是 $\gcd(a,b)$ 必须整除 $c$，同时目标水量不能超过题意允许保存的容量。
- 若要求某一桶中恰有 $c$，需 $c\le\max(a,b)$；若允许两桶合计为 $c$，可放宽到 $c\le a+b$。
- 状态是当前水量 $(x,y)$，边是装满、倒空、A 倒 B、B 倒 A。
- BFS 在每个动作代价相同且状态有限时找到最少步数，并用前驱恢复路径。
- 多桶时状态维数和状态数急剧增长，通常需要数学约束、双向搜索或领域剪枝。

## 答案

**先用最大公约数判断是否可能，再用 BFS 在水量状态图中找最短操作序列。** 下例按“两桶中任一桶恰有 $c$”定义目标：

```python
from collections import deque
from math import gcd

def measure(a, b, c):
    if any(not isinstance(v, int) or isinstance(v, bool) for v in (a, b, c)):
        raise ValueError("capacities and target must be integers")
    if min(a, b, c) < 0:
        return None
    if c == 0:
        return [(0, 0)]
    if a == 0 and b == 0:
        return None
    if c > max(a, b) or c % gcd(a, b) != 0:
        return None

    start = (0, 0)
    q = deque([start])
    prev = {start: None}
    while q:
        x, y = q.popleft()
        if x == c or y == c:
            path = []
            cur = (x, y)
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]

        pour_ab = min(x, b - y)
        pour_ba = min(y, a - x)
        nxt = {(a, y), (x, b), (0, y), (x, 0),
               (x - pour_ab, y + pour_ab),
               (x + pour_ba, y - pour_ba)}
        for state in nxt:
            if state not in prev:
                prev[state] = (x, y)
                q.append(state)
    return None
```

状态最多 $(a+1)(b+1)$ 个，时间、空间都与可达状态数同阶。若只问是否可行，直接做 gcd 判定即可，无需 BFS。三升和五升量四升的一条路线是：装满 5 → 倒入 3 → 清空 3 → 把剩余 2 倒入 3 → 再装满 5 → 向 3 倒 1，五升桶剩 4。

## 知识点

贝祖定理、最大公约数、状态图、BFS、最短操作、目标条件。


## 追问

- 如果要求操作步数最少，怎样用 BFS 求解？
- 多于两个桶时，可解性与搜索规模怎样变化？
- 若每种操作成本不同，为什么要从 BFS 改成 Dijkstra？
- 怎样同时返回操作名和水量状态？

## Note

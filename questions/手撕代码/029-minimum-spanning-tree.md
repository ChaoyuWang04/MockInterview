---
difficulty: 简单
topic: 图算法/最小生成树
summary: 手写 Kruskal 与并查集，并比较 Prim 的复杂度和适用图
tags: [面经, 待校对, 手撕代码, 图算法, 最小生成树, 并查集]
company: 蚂蚁金服
mastered: false
highfreq: false
---

## 题目

手写最小生成树算法，可以选择 Kruskal 或 Prim。解释时间复杂度、适用图类型，以及并查集等关键数据结构。代码需要识别不连通图。

## 要点

- 最小生成树只对无向、带权、连通图定义；不连通时得到最小生成森林。
- Kruskal 按边权排序，每次加入不会成环的最小边；并查集负责连通性判断。
- 路径压缩和按大小/秩合并后，并查集操作均摊为 $O(\alpha(V))$，不是严格 $O(1)$。
- Kruskal 时间 $O(E\log E)$；邻接表+堆的 Prim 为 $O(E\log V)$；很稠密的图可用邻接矩阵 Prim $O(V^2)$。

## 答案

```python
class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def kruskal(n, edges):
    """edges: iterable of (weight, u, v), vertices are 0..n-1."""
    if n < 0:
        raise ValueError("n must be non-negative")
    dsu = DSU(n)
    total, chosen = 0, []
    for weight, u, v in sorted(edges):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("vertex out of range")
        if dsu.union(u, v):
            total += weight
            chosen.append((u, v, weight))
            if len(chosen) == n - 1:
                break
    if n > 0 and len(chosen) != n - 1:
        raise ValueError("graph is disconnected")
    return total, chosen
```

排序保证当前检查的是尚未处理的最小边。若它连接两个不同连通分量，根据割性质，总存在一棵最小生成树包含这条安全边。若两端已经连通，加入它只会形成环，跳过即可。

排序花 $O(E\log E)$，并查集总计 $O(E\alpha(V))$，所以整体由排序主导；空间是 $O(V+E)$（若输入边已在内存中，额外排序空间取决于语言实现）。

## 知识点

- 贪心、割性质、环性质、并查集、路径压缩、按大小合并、连通图。
- 面经原题：[B006-G01-Q060](../../docs/references/面经原题.md#b006-g01-q060)、[B006-G01-Q103](../../docs/references/面经原题.md#b006-g01-q103)。
- 老师答案参考：[P009-Q060](../../docs/references/平台题/P009-LC-001-080.md#p009-q060)、[P009-Q103](../../docs/references/平台题/P009-LC-081-160.md#p009-q103)。

## 追问

以下为 Q060 平台页面追问，Q103 页面没有附加追问；均不计入面经原题：

- 图中有负权边时，Kruskal 和 Prim 是否仍然适用？
- 路径压缩与按秩/大小合并怎样改善并查集复杂度？
- 地图导航场景应怎样选择 Kruskal、Prim 或其他路径算法？

## Note

---
difficulty: 简单
topic: 聚类/K-means
summary: 从零实现含 K-means++、空簇处理和可靠收敛判断的聚类
tags: [真题, 待校对, 手撕代码, K-means, 聚类]
company: 百度、蚂蚁金服
mastered: false
highfreq: false
---

## 题目

解释 K-means 的目标、初始化、簇分配、质心更新和收敛条件，并从零实现核心算法。说明时间复杂度、优缺点和适用场景，介绍 K-means++、Mini-Batch 等改进，并简要比较它与其他聚类方法。

## 要点

- 目标是最小化簇内平方和 $\sum_i\|x_i-\mu_{c_i}\|_2^2$。
- “分配到最近中心”和“按簇求均值”交替进行，每一步都不会增大目标，因此会收敛到局部最优或稳定划分。
- 必须处理空簇；直接对空切片求均值会产生 NaN。
- K-means++ 按到已选中心的最小平方距离采样；所有距离都为零时要有退化处理。
- 每轮时间 $O(nkd)$，总计 $O(Inkd)$；本实现的广播临时数组峰值为 $O(nkd)$，分块或改写距离公式可降到 $O(nk)$。

## 答案

```python
import numpy as np


def _kmeans_plus_plus(X, k, rng):
    n = len(X)
    centers = [X[rng.integers(n)].copy()]
    closest_sq = ((X - centers[0]) ** 2).sum(axis=1)

    for _ in range(1, k):
        total = float(closest_sq.sum())
        if total <= 0.0:  # 所有点都与已选中心重合
            idx = int(rng.integers(n))
        else:
            idx = int(rng.choice(n, p=closest_sq / total))
        centers.append(X[idx].copy())
        new_sq = ((X - centers[-1]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)
    return np.asarray(centers)


def kmeans(X, k, max_iter=300, tol=1e-4, seed=None):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("X must be a non-empty 2-D array")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if not 1 <= k <= len(X):
        raise ValueError("k must be between 1 and number of samples")
    if max_iter <= 0 or tol < 0:
        raise ValueError("invalid stopping parameters")

    rng = np.random.default_rng(seed)
    centers = _kmeans_plus_plus(X, k, rng)

    for iteration in range(1, max_iter + 1):
        sq_dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = sq_dist.argmin(axis=1)
        nearest_sq = sq_dist[np.arange(len(X)), labels]

        new_centers = centers.copy()
        empty = []
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                empty.append(cluster)

        # 空簇重置到当前误差最大的不同样本。
        if empty:
            candidates = np.argsort(nearest_sq)[::-1]
            used = set()
            cursor = 0
            for cluster in empty:
                while int(candidates[cursor]) in used:
                    cursor += 1
                idx = int(candidates[cursor])
                used.add(idx)
                new_centers[cluster] = X[idx]

        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        if shift <= tol:
            break

    sq_dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = sq_dist.argmin(axis=1)
    inertia = float(sq_dist[np.arange(len(X)), labels].sum())
    return centers, labels, inertia, iteration
```

`max_iter` 与 `tol` 是调用参数，不是普适常数。实际使用应多次随机重启，选 inertia 最小的结果。K-means++ 改善初始化但不保证全局最优；Mini-Batch 牺牲少量精度换吞吐；Elkan/Hamerly 用三角不等式减少距离计算。肘部法只能辅助选 k，还可结合轮廓系数和业务解释。

代码会把空簇重置到当前误差较大的样本，但当不同坐标数少于 k 时，无法保证最终每个簇都非空。若业务要求 k 个非空簇，应先限制 k 不超过不同点数，并在重新分配后复查空簇。

## 知识点

- Lloyd 迭代、簇内平方和、K-means++、空簇、局部最优、Mini-Batch。

## 追问

相关真题追问：

- K-means++ 为什么通常比随机初始化稳定？
- 除多次重启外，怎样缓解初值敏感？
- 样本量很大时怎样做 Mini-Batch 或分布式聚类？

## Note

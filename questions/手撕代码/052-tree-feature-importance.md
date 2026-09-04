---
difficulty: 简单
topic: 模型可解释性/树特征重要性
summary: 比较不纯度、排列与增益类特征重要性的含义和偏差
tags: [面经, 待校对, 手撕代码, 特征重要性, 随机森林, GBDT]
company: 哔哩哔哩
mastered: false
highfreq: false
---

## 题目

在随机森林、GBDT、XGBoost、LightGBM 等树模型中，常见特征重要性怎样计算？说明不纯度减少、排列重要性和 Gain/Cover/Frequency 的含义、优缺点、计算成本及解释边界。

## 要点

- MDI 汇总使用某特征切分带来的加权不纯度下降，计算快，但偏爱可切分点多的高基数/连续特征。
- Permutation Importance 在独立验证集打乱某列，观察预测分数下降；它是预测依赖，不是因果效应，也不保证无偏。
- 相关特征会互相替代，单列打乱可能低估双方；打乱还可能生成训练分布外的特征组合。
- XGBoost 的 Gain/Cover/Frequency 分别描述损失下降、覆盖样本/二阶权重和使用次数，回答的问题不同。
- 不同指标回答的问题不同：分裂次数多、覆盖样本多和带来较大损失下降不能互相替代。

## 答案

对节点 t，样本权重为 $N_t$，分裂后的左右节点为 L/R。不纯度下降可写为

$$
\Delta I_t=N_tI_t-N_LI_L-N_RI_R.
$$

某特征的 MDI 是所有使用该特征节点的 $\Delta I_t$ 之和，再归一化。分类树的 $I$ 可为 Gini/entropy，回归树可为 MSE；不能把所有提升树都简单归为 Gini。

排列重要性的框架可以手写为：

```python
import numpy as np


def permutation_importance(predict, X, y, score, repeats=5, seed=None):
    """score 越大越好；返回每列分数下降的均值和标准差。"""
    X = np.asarray(X)
    y = np.asarray(y)
    if y.ndim != 1 or X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must be 2-D and match y")
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    rng = np.random.default_rng(seed)
    baseline = score(y, predict(X))
    drops = np.empty((repeats, X.shape[1]), dtype=float)

    for repeat in range(repeats):
        for feature in range(X.shape[1]):
            shuffled = X.copy()
            shuffled[:, feature] = rng.permutation(shuffled[:, feature])
            drops[repeat, feature] = baseline - score(y, predict(shuffled))

    return drops.mean(axis=0), drops.std(axis=0, ddof=1 if repeats > 1 else 0)
```

应在独立验证/测试切分上计算，并报告重复打乱的方差。训练集上的重要性高可能只是过拟合或泄露；比较训练集和验证集只能提供诊断线索，不能单凭重要性证明泄露或因果关系。

XGBoost 一类库还常报告三组内部统计：**Gain** 汇总使用该特征分裂带来的目标下降，**Cover** 汇总这些分裂覆盖的样本权重或 Hessian，**Frequency/Weight** 统计该特征被用于分裂的次数。它们分别回答“降损失多少、覆盖多少、用了几次”；不同库对求和或平均的定义可能不同，使用时应先核对口径。

MDI/Gain 适合快速模型内部诊断；Permutation 适合回答“模型在这份评估分布上依赖该特征多少”。平台追问中的 SHAP 用于解释单样本相对基线的有正负贡献，全局重要性常汇总 `mean(abs(SHAP))`；它和本题这些全局统计不是同一个问题。模型、数据分布、相关性和评估指标变化时，重要性也会变化。

## 知识点

- MDI、Gini/MSE、Permutation Importance、Gain/Cover/Frequency、SHAP、相关特征、数据泄露。
- 面经原题：[B006-G01-Q157](../../docs/references/面经原题.md#b006-g01-q157)。
- 老师答案参考：[P009-Q157](../../docs/references/平台题/P009-LC-081-160.md#p009-q157)。

## 追问

以下均为平台页面追问，不计入面经原题：

- Permutation Importance 和 SHAP 什么时候各自更合适？
- 某特征全局重要性高，但一个样本的 SHAP 为负，怎样解释？
- 高基数类别特征怎样减少重要性偏差？

## Note

---
difficulty: 简单
topic: 集成学习/GBDT与XGBoost
summary: 解释 GBDT 负梯度提升、节点分裂及 XGBoost 的二阶扩展
tags: [真题, 待校对, 手撕代码, GBDT, XGBoost, 决策树]
company: 哔哩哔哩、小红书
mastered: false
highfreq: false
---

## 题目

同一题簇包含三个层次：

1. 解释 GBDT 的加法模型、前向分步训练和负梯度（伪残差）拟合。
2. 说明一棵新树怎样选择节点分裂特征与切分点。
3. 比较传统 GBDT 与 XGBoost 在二阶信息、正则化、分裂搜索、缺失值与工程实现上的差异。

## 要点

- 通用 GBDT 每轮拟合当前损失对模型输出的负梯度，不等于“所有 GBDT 都使用 Hessian 增益”。
- 平方误差下，负梯度恰好是残差 $y-F(x)$；其他损失下是伪残差。
- 传统回归树可按残差的平方误差下降选择切分；XGBoost 使用一阶梯度和二阶梯度构造正则化增益。
- XGBoost 是梯度提升树的一种具体实现与目标函数扩展，应与通用 GBDT 分层讲。
- 树之间依赖上一轮预测，轮次方向难以完全并行；单棵树内部的特征/候选切分统计可以并行。

## 答案

### 通用 GBDT

模型是树的加法：

$$
F_m(x)=F_{m-1}(x)+\eta f_m(x).
$$

第 $m$ 轮对每个样本计算

$$
r_{im}=-\left.\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}},
$$

再训练一棵回归树拟合 $(x_i,r_{im})$。平方误差下 $r_{im}=y_i-F_{m-1}(x_i)$。树节点可以枚举“特征 j + 阈值 s”，选择左右子节点残差平方和最小、也就是 SSE 降幅最大的切分。

```text
F <- 最优常数预测
repeat M rounds:
    r_i <- - d L(y_i, F(x_i)) / d F(x_i)
    tree <- CART.fit(X, r)              # 按残差 SSE 下降分裂
    为每个叶子求最优叶值 gamma
    F(x) <- F(x) + learning_rate * tree(x)
```

### XGBoost 的分裂增益

在当前节点内令 $g_i=\partial L/\partial \hat y_i$、$h_i=\partial^2L/\partial\hat y_i^2$。对候选左右集合，记 $G_L=\sum_{i\in L}g_i$、$H_L=\sum_{i\in L}h_i$，右侧同理。省略叶权 L1 正则时，常见增益是

$$
\mathrm{Gain}=\frac12\left[
\frac{G_L^2}{H_L+\lambda}+
\frac{G_R^2}{H_R+\lambda}-
\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}
\right]-\gamma.
$$

只在增益为正且满足最小叶权等约束时分裂。这里的 $\lambda$ 是叶权 L2 正则，$\gamma$ 是新增叶子的复杂度惩罚。叶值为 $-G/(H+\lambda)$。

传统 GBDT 常按具体伪残差训练 CART；XGBoost 把二阶近似、显式树复杂度正则、列/行采样、稀疏感知、缺失值默认方向和近似/直方图切分放进统一框架。LightGBM 也用梯度统计，但有自己的直方图、叶优先生长和类别特征处理。

GBDT 很适合结构化表格数据：树能直接表达阈值和特征交互，通常也不要求特征缩放。局限是各轮依赖上一轮结果，训练难以沿树的轮次完全并行，而且对树深、学习率、轮数和噪声较敏感。超高维稀疏原始文本、端到端感知或需要外推的任务，不一定是它最合适的场景。

若每个节点重新排序，代价较高；预排序或直方图将候选统计复用。精确渐进复杂度取决于树深、样本数、特征数、桶数和稀疏度，不能只写一个与实现无关的常数式。

## 知识点

- 加法模型、前向分步、伪残差、CART 分裂、二阶泰勒、正则化增益、直方图算法。

## 追问

相关真题追问：

- XGBoost 与 LightGBM 的候选切分搜索有什么不同？
- GBDT 怎样处理高基数类别特征和高维稀疏特征？
- 为什么二阶信息有用？只用一阶梯度是否仍能提升？
- GBDT+LR 的特征组合思路是什么？

## Note

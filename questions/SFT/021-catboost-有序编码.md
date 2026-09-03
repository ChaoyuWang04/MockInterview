---
difficulty: 简单
topic: 集成学习/有序目标编码
summary: CatBoost如何用有序目标统计减少泄漏,与有序Boosting有何区别
tags: [CatBoost, 类别特征, 目标编码, 梯度提升, 待校对]
company: 美团
mastered: false
highfreq: false
---

## 题目

CatBoost 在处理类别型特征时采用了哪些关键技术机制使其优于传统方法?请详细解释有序目标编码(Ordered Target Encoding)的实现原理,以及它是如何有效缓解目标泄漏和过拟合问题的。

## 要点

- 全量目标均值包含当前样本标签,会造成目标泄漏
- 随机排列后只用此前同类别样本的标签统计,结合先验平滑
- 区分有序目标统计与有序 Boosting,Plain 也可用前者
- 回答高基数特征与 LightGBM EFB 的不同作用,不承诺必然更优

## 答案

**CatBoost 用有序目标统计把类别转成数值,让当前样本的标签不进入自己的编码,再通过先验平滑减轻低频类别的噪声。** 相对直接使用全量类别均值,它缓解了目标泄漏;是否优于其他方法仍要在相同数据划分下比较。

随机排列训练样本,用 $H_i$ 表示排在样本 $i$ 之前、且类别与它相同的样本集合。以均值型编码为例:

$$
\mathrm{enc}_i=\frac{\sum_{j\in H_i}y_j+a\mu}{|H_i|+a}
$$

$y_j$ 是历史标签,$\mu$ 是先验均值,$a>0$ 控制平滑强度。没有同类历史时回退到先验;历史越多,编码越依赖观测。若一个用户 ID 只出现一次,全量均值就等于它的标签,有序统计则不能这样“偷看答案”。训练中采用多个排列,减少对单一顺序的依赖;预测用训练数据建立的统计,不使用测试标签。

### 两种有序机制不要混淆

| 机制 | 防止哪一种泄漏或偏移 |
|---|---|
| 有序目标统计 | 标签经类别编码泄漏到特征 |
| Ordered Boosting | 计算当前样本残差时,模型已用过该样本标签导致的预测偏移;改用此前样本训练的模型估计 |

**Plain 是普通梯度提升,仍可使用有序目标统计。** Ordered Boosting 更复杂,常对小数据有帮助;Plain 常更快,不是“生产必须用 Ordered”。当前官方参数说明也支持按数据、设备和损失选择。

高基数特征可以用目标统计和特征组合,避免完整 one-hot 的维度膨胀;但近乎唯一的 ID 缺少可迁移统计,平滑不能凭空补出规律,应按用户或时间留出评估。LightGBM 的 EFB 将互斥稀疏特征捆绑以减少计算,不是目标编码,也不直接解决标签泄漏;其原生类别分裂是另一个机制,不能只拿 EFB 判断两套模型的类别处理优劣。

## 知识点

目标统计、先验平滑、随机排列、预测偏移、Ordered Boosting、稀疏特征捆绑。

- 来源:[老师平台](https://course.terminiai.com/interview),P002-Q024。
- 依据:[CatBoost 原论文 §3–5](https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf)、[boosting_type 官方说明](https://catboost.ai/docs/en/references/training-parameters/common#boosting_type)、[LightGBM 官方特性](https://lightgbm.readthedocs.io/en/latest/Features.html)。

## 追问

- CatBoost 的 Ordered 模式与 Plain 模式有什么区别,各自适用什么场景?
- CatBoost 如何处理高基数类别特征(如用户 ID)?
- 与 LightGBM 的 EFB(互斥特征捆绑)相比,CatBoost 的类别特征处理有何优劣?

## Note

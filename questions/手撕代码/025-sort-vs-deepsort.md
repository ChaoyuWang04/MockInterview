---
difficulty: 中等
topic: 多目标跟踪/SORT与DeepSORT
tags: [面经, 待校对, 多目标跟踪, SORT, DeepSORT, ReID]
summary: 比较 SORT 与 DeepSORT 的运动、外观关联和 ID 保持机制
company: 大疆
mastered: false
highfreq: false
---

## 题目

比较 SORT 与 DeepSORT 的设计思想、状态估计、数据关联、外观建模和 ID 保持能力；说明遮挡、轻量 ReID 与级联匹配的影响，并给出适用边界。

## 要点

- 两者都依赖检测结果并用卡尔曼滤波预测轨迹、匈牙利算法做匹配。
- SORT 主要按运动预测与 IoU 关联，简单快，但遮挡或交叉时容易 ID switch。
- DeepSORT 加入 ReID 外观嵌入、马氏距离门控和 matching cascade，提高重现目标的识别能力。
- 级联匹配是 DeepSORT 的机制，不是“把 SORT 的级联直接搬到 DeepSORT”。
- DeepSORT 也会受检测漏检、外观域偏移、相似制服、长时间遮挡影响。

## 答案

**SORT 主要问“这个框离哪条预测轨迹最近”，DeepSORT 还问“它外观看起来像哪条轨迹”。**

```mermaid
flowchart LR
  A[当前帧检测框] --> B[卡尔曼预测轨迹]
  B --> C[运动门控]
  C --> D[外观距离与级联匹配]
  D --> E[IoU补充匹配]
  E --> F[更新 新建 删除轨迹]
```

SORT 用常速度卡尔曼模型预测位置，构造检测框与预测框的 IoU 代价，再用匈牙利算法做一对一分配。它没有长期外观记忆，因此快速运动、相互遮挡或目标交叉时容易换 ID，但成本低，适合遮挡少、检测稳定、延迟严格的场景。

DeepSORT 为检测框提取 ReID 向量，用外观余弦距离帮助找回刚被遮挡的轨迹；卡尔曼协方差给出马氏距离门控，排除运动上不可能的匹配。matching cascade 按轨迹多久未匹配来分层关联，让近期可靠轨迹优先；剩余目标还可用 IoU 补配。

换轻量 ReID 会降低计算和延迟，但若区分度、输入分辨率或训练域不够，ID switch 可能上升。严重遮挡可加入更强检测器、相机运动补偿、轨迹生命周期调节、时序外观聚合或更全局的数据关联。安全性和 ID 保持不能只由算法名字推断，需在目标域上看 HOTA、IDF1、ID switch 与速度。

## 知识点

Tracking-by-detection、卡尔曼滤波、IoU、匈牙利匹配、ReID、马氏距离、matching cascade、ID switch。

- 面经原题：[B006-G01-Q054](../../docs/references/面经原题.md#b006-g01-q054)、[B006-G01-Q177](../../docs/references/面经原题.md#b006-g01-q177)。
- 老师答案参考：[P009-Q054](../../docs/references/平台题/P009-LC-001-080.md#p009-q054)、[P009-Q177](../../docs/references/平台题/P009-LC-161-241.md#p009-q177)。

## 追问

以下均为平台页面追问，不计入面经原题：

- ReID 换成轻量模型后，怎样判断速度收益是否抵消 ID 保持下降？
- 严重遮挡下 SORT 和 DeepSORT 分别在哪一步失败？
- DeepSORT 的 matching cascade 为什么按轨迹新鲜度分层？

## Note

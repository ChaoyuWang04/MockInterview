---
difficulty: 简单
topic: 卷积与池化/池化算子
summary: 比较最大、平均、全局与自适应池化的计算和使用场景
tags: [面经, 待校对, 手撕代码, Pooling, CNN]
company: 网易
mastered: false
highfreq: false
---

## 题目

深度学习中有哪些常见池化操作？说明最大池化、平均池化、全局池化和自适应池化的计算方式、梯度特点、优缺点、适用场景，以及它们在典型网络中的位置。

## 要点

- 局部最大/平均池化都按窗口与步幅下采样，但保留的信息和梯度分配不同。
- 全局平均池化把每个通道的空间维压成一个数；算子本身无参数，但网络后续分类器仍可能有参数。
- 自适应池化由目标输出尺寸反推窗口边界，适合可变输入。
- MaxPool 的梯度只回到最大值位置；AvgPool 平均分配梯度。
- 池化带来一定局部平移鲁棒性，也会丢失精确位置；现代网络常用 stride convolution 替代部分池化。

## 答案

对二维特征图 $X\in\mathbb R^{C\times H\times W}$，窗口 $R_{ij}$ 上：

$$
y^{\text{max}}_{cij}=\max_{(u,v)\in R_{ij}}X_{cuv},\qquad
 y^{\text{avg}}_{cij}=\frac1{|R_{ij}|}\sum_{(u,v)\in R_{ij}}X_{cuv}.
$$

下面是不依赖深度学习框架的单通道 NumPy 示例：

```python
import numpy as np


def pool2d(x, kernel, stride=None, mode="max"):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not isinstance(kernel, int) or isinstance(kernel, bool) or kernel <= 0:
        raise ValueError("x must be 2-D and kernel a positive integer")
    stride = kernel if stride is None else stride
    if (not isinstance(stride, int) or isinstance(stride, bool) or stride <= 0
            or x.shape[0] < kernel or x.shape[1] < kernel):
        raise ValueError("invalid stride or kernel")

    out_h = 1 + (x.shape[0] - kernel) // stride
    out_w = 1 + (x.shape[1] - kernel) // stride
    out = np.empty((out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            window = x[i * stride:i * stride + kernel,
                       j * stride:j * stride + kernel]
            if mode == "max":
                out[i, j] = window.max()
            elif mode == "avg":
                out[i, j] = window.mean()
            else:
                raise ValueError("mode must be max or avg")
    return out


def global_average_pool(x):
    # x: (..., C, H, W)
    return np.asarray(x, dtype=float).mean(axis=(-2, -1))
```

MaxPool 强调最强响应，适合“是否出现某局部模式”；它对噪声尖峰敏感，且反传稀疏。AvgPool 保留整体平均响应，梯度更平滑，但可能冲淡强特征。Global Average Pooling 输出每通道一个数，经常在分类头前替代大尺寸展平；它减少的是空间到分类头的参数，不是把整网参数“降为零”。Adaptive Pooling 保证固定输出大小，常用于输入尺寸变化的模型接口。

在典型架构中，VGG 常在卷积 stage 之间用 MaxPool 降采样；常见 ResNet 在 stem 后做一次池化，并在分类头前用 Global Average Pooling。具体位置随模型版本而变，不能把一种布局推广到所有网络。

局部池化直接实现的时间与输出窗口覆盖元素总数成正比；内存主要是输出。反向传播还需保存 MaxPool 的最大位置或 AvgPool 的窗口信息。

## 知识点

- 下采样、局部平移鲁棒性、梯度路由、全局平均池化、自适应输出、stride convolution。
- 面经原题：[B006-G01-Q122](../../docs/references/面经原题.md#b006-g01-q122)。
- 老师答案参考：[P009-Q122](../../docs/references/平台题/P009-LC-081-160.md#p009-q122)。

## 追问

以下为平台页面追问，不计入面经原题：

- Vision Transformer 没有传统 MaxPool 时怎样改变 token/空间分辨率？

## Note

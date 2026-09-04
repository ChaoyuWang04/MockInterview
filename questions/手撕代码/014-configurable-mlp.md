---
difficulty: 简单
topic: 神经网络/可配置MLP
summary: 用 PyTorch 动态构建隐藏层、激活函数并正确初始化
tags: [真题, 待校对, PyTorch, MLP, 参数初始化]
company: 网易
mastered: false
highfreq: false
---

## 题目

请使用PyTorch或TensorFlow框架实现一个可灵活配置隐藏层数量、每层神经元数量及激活函数的多层感知机（MLP）模型，要求代码结构清晰、模块化，并说明模型设计的关键思路与实现步骤，包括网络构建、参数初始化和前向传播机制。

## 要点

- 验证每层宽度和激活配置。
- ModuleList 适合动态控制流，Sequential 适合纯串行结构。
- 初始化应与激活匹配；GELU 没有简单等同 ReLU 的固定 He gain。
- 未知激活应报错，不能静默替换。

## 答案

```python
import torch
from torch import nn


class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="relu"):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or any(d <= 0 for d in hidden_dims):
            raise ValueError("all dimensions must be positive")
        factories = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }
        if activation not in factories:
            raise ValueError(f"unsupported activation: {activation}")

        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            if activation == "relu" and i < len(dims) - 2:
                nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            if i < len(dims) - 2:
                layers.append(factories[activation]())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
```

设宽度为 $d_0,\ldots,d_L$，参数量和单样本主要计算量均为 $\Theta(\sum_i d_i d_{i+1})$。加入 BatchNorm、残差或多输出头时，ModuleList/显式 forward 往往更清楚。

## 知识点

- 动态网络构建、ModuleList/Sequential、激活函数、Xavier/He 初始化、配置校验、参数量。


## 追问

相关真题追问：

- 怎样加入多个输出 head？
- 何时加 BatchNorm、LayerNorm 或残差？
- ModuleList 与 Sequential 的注册和 forward 有什么区别？

## Note

---
difficulty: 简单
topic: 模型算子/Conv2D 复杂度
summary: 从输出尺寸推导普通、分组和深度可分离卷积计算量
tags: [面经, 待校对, Conv2D, 计算复杂度, FLOPs]
company: 小红书
mastered: false
highfreq: false
---

## 题目

对于二维卷积，输入为 $H\times W\times C$，卷积核为 $K_h\times K_w$，输出通道数为 $F$，请推导输出尺寸、MACs和FLOPs，并说明stride、padding、dilation、groups及实际硬件对延迟的影响。

## 要点

- 先算 $H'$、$W'$，再数每个输出元素的乘加
- 普通卷积MACs为 $H'W'F K_hK_wC$
- groups把每个卷积核看到的输入通道降为 $C/G$
- 声明MAC/FLOP口径，理论运算量不等于墙钟延迟

## 答案

设stride为 $(S_h,S_w)$，padding为 $(P_h,P_w)$，dilation为 $(D_h,D_w)$，则：

$$
H'=\left\lfloor\frac{H+2P_h-D_h(K_h-1)-1}{S_h}\right\rfloor+1
$$

$W'$同理。普通卷积的每个输出元素需要遍历 $K_hK_wC$ 个输入并累加；一共有 $H'W'F$ 个输出元素，所以：

$$
\mathrm{MACs}=H'W'F K_hK_wC
$$

若把一次乘法和一次加法各算一次浮点运算，则FLOPs约为 $2\times\mathrm{MACs}$；有的工具把一次FMA记作一次操作，比较数字前必须先统一口径。bias、激活等另计，通常不改变卷积主体的复杂度阶。

分组卷积有 $G$ 组，每个输出通道只看 $C/G$ 个输入通道：

$$
\mathrm{MACs}_{group}=H'W'F K_hK_w\frac{C}{G}
$$

深度可分离卷积先做 $G=C$ 的depthwise卷积，再做 $1\times1$ pointwise卷积，MACs约为 $H'W'(K_hK_wC+CF)$。

stride增大通常让输出变小；padding、dilation通过输出尺寸和有效感受野影响计算。实际延迟还受内存布局、访存、kernel启动、Tensor Core对齐、batch、融合和并行度影响，所以FLOPs下降不保证同比加速。应以同一设备上的profile验证算力、带宽和占用率瓶颈。

## 知识点

输出尺寸、MACs与FLOPs、分组卷积、深度可分离卷积、算术强度、kernel效率。

- 真实面经：[B002-G01-Q084](../../docs/references/面经原题.md#b002-g01-q084)、[B002-G01-Q157](../../docs/references/面经原题.md#b002-g01-q157)
- 老师答案参考：[P005-Q084](../../docs/references/平台题/P005-Infra-061-090.md#p005-q084)、[P005-Q157](../../docs/references/平台题/P005-Infra-151-180.md#p005-q157)

## 追问

- 页面参考追问：组卷积的复杂度怎样变化？
- 页面参考追问：stride、padding和dilation怎样影响输出尺寸和计算量？
- 页面参考追问：深度可分离卷积为何能降低计算量？
- 页面参考追问：除了理论FLOPs，哪些因素会影响卷积层实际延迟？
- 页面参考追问：怎样优化卷积执行以逼近理论加速比？

## Note

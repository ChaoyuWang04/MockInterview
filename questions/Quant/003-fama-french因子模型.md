---
difficulty: 中等
topic: 因子模型/Fama-French与因子回归
summary: 三因子五因子各含什么,α 怎样设定与解读
tags: [真题, 待校对, Quant, Fama-French, 因子模型, Alpha, Fama-MacBeth]
company:
mastered: false
highfreq: false
---

## 题目

Fama-French 三因子和五因子分别包含哪些因子,各自想解释什么?用它评估一个策略的超额收益时,回归怎么设定,$\alpha$ 又该怎么解读?

## 要点

- 三因子 = MKT + SMB + HML;五因子加 RMW(盈利)与 CMA(投资)
- 动量走的是另一条线:Carhart 四因子加 MOM/UMD
- 时间序列回归的截距 $\alpha$ 才是因子解释不掉的超额收益
- 五因子里 RMW/CMA 常吸走 HML 的显著性,报 $\alpha$ 必须说明用了哪个模型
- Fama-MacBeth 回答的是另一个问题,且只修截面相关不修时序自相关

## 答案

**三因子(Fama & French, 1993)= 市场超额收益 MKT + 规模 SMB(小减大)+ 价值 HML(高减低账面市值比)。五因子(2015)在其上加盈利能力 RMW(robust minus weak)和投资 CMA(conservative minus aggressive)。** 动量不在这条线上,它由 Carhart(1997)的四因子模型以 MOM/UMD 补入。

评估策略时做时间序列回归:

$$
r_{p,t}-r_{f,t}=\alpha+\beta_{\rm MKT}{\rm MKT}_t+\beta_{\rm SMB}{\rm SMB}_t+\beta_{\rm HML}{\rm HML}_t+\varepsilon_t
$$

**关注的是截距 $\alpha$**:它是已知因子暴露解释不掉的那部分收益。$\beta$ 大不代表策略好,只说明它在赚某个已知风险溢价——那部分买对应指数就能拿到,不该按主动管理定价。

三个容易被追的点:

1. **换模型会改变结论。** 五因子加入后 HML 常常不再显著,说明一部分价值溢价可由盈利能力和投资解释。所以报 $\alpha$ 必须写清用的是三因子、四因子还是五因子。
2. **$\alpha$ 的标准误要修。** 残差存在异方差和自相关,直接用 OLS 标准误会高估显著性,见 [OLS 假设与失效](001-ols-假设与失效.md)。
3. **时序回归与截面回归回答的不是同一个问题。** 时序回归估的是这只组合的 $\beta$ 和 $\alpha$;若要问"每单位因子暴露对应多少风险溢价",要用 Fama-MacBeth 两步法:先按个股时序回归估 $\beta$,再在每个时点对 $\beta$ 做截面回归得到因子收益序列,最后对该序列取均值与 $t$ 值。它修正的是**截面相关**,不修时序自相关,长持有期时需另行处理。

边界:因子的构造口径会影响结果——同一因子用 2×2 与 2×3 分组得到的平均收益不同,比较时必须口径一致。$\alpha$ 显著也可能来自数据挖掘,要看样本外表现和多重检验校正。

## 知识点

MKT/SMB/HML/RMW/CMA/MOM、时间序列 $\alpha$ 与截面风险溢价、Fama-MacBeth 两步法、因子构造口径、多重检验。

参考:Fama & French (1993) 三因子、Carhart (1997) 四因子、Fama & French (2015) 五因子;因子收益序列可取自 Kenneth French 数据库。

## 追问

- 为什么加入 RMW 和 CMA 后 HML 可能变得不显著?
- 时间序列 $\alpha$ 与 Fama-MacBeth 得到的风险溢价分别回答什么问题?
- 策略 $\beta$ 很高、$\alpha$ 接近零,该怎样评价它?
- 因子暴露随时间漂移时,固定窗口回归会出什么问题?

## Note

# ProcTHOR：用程序化生成把可交互房屋扩到上万套，RGB 智能体在六项具身任务上零样本与微调都站住

<!-- release-date: 2022-06-14 -->

**本文依据**：`ProcTHOR: Large-Scale Embodied AI Using Procedural Generation`，13 页 letter。作者 Matt Deitke^{†ψ}、Eli VanderBilt^{†}、Alvaro Herrasti^{†}、Luca Weihs^{†}、Jordi Salvador^{†}、Kiana Ehsani^{†}、Winson Han^{†}、Eric Kolve^{†}、Ali Farhadi^{ψ}、Aniruddha Kembhavi^{†ψ}、Roozbeh Mottaghi^{†ψ}。^{†} PRIOR @ Allen Institute for AI，^{ψ} University of Washington, Seattle。第一作者第一单位是 Allen Institute for AI（PRIOR）。封面印 `36th Conference on Neural Information Processing Systems (NeurIPS 2022).`。封面无 arXiv 行。官方 abs：`https://arxiv.org/abs/2206.06994`，v1 提交日 **2022-06-14**。项目页写在第 1 页：`procthor.allenai.org`。文中数字都标 PDF 页码。本文只读这一份 PDF。

## 一句话

视觉和语言能靠海量无任务数据预训练，具身智能（Embodied AI，E-AI）却还在几十到一千套场景里过拟合（PDF p. 1）。手工搭房太慢，扫真房又难交互。ProcTHOR 架在 AI2-THOR 上，按房间规格程序化采样平面、结构、门、灯光、家具和材质，生成任意多套可物理交互的房屋（PDF p. 1–3）。作者放出 **10,000** 套训练房（ProcTHOR-10K），外加 **1,000** 验证、**1,000** 测试；智能体只用 RGB、CNN+RNN、不做显式建图、没有人工任务监督（PDF p. 1–2）。截至 **2022-06-14 上午 10 点 PT**，在六项导航 / 重排 / 机械臂基准上报当时 SoTA，含 Habitat 2022 ObjectNav、AI2-THOR Rearrangement 2022、RoboTHOR；并按 PDF 所写给出 **0-shot**（只在 ProcTHOR 上预训练、下游训练数据不用）数字，常超过当时用过下游训练集的系统（PDF p. 1–2、p. 8–9）。另交 ArchitecTHOR：艺术家设计的 **10** 套评测房（PDF p. 2、p. 8）。

它解决的不是「再扫一千套真房」，而是：**可交互房屋能不能程序化扩到上万，让简单 RGB 策略在未见布局和外观上站住。**

## 一、矛盾：模拟器变强了，场景数量没跟上

作者把视觉 / NLP 的 CLIP、DALL-E、GPT-3、Flamingo 对照具身社区：任务长、规划远，场景却远少于那些预训练语料（PDF p. 1）。近年模拟器已经能物理、机械臂、物体状态、可变形、流体、真–仿对照，但扩到上万场景仍难（PDF p. 1）。两条旧路：

1. **艺术家手工**（AI2-THOR 一类）：资产、摆放、贴图、灯光都靠人，再扩几个数量级不现实（PDF p. 1）。
2. **真房扫描**（Habitat 一类）：专用相机走一遍再拼接；物体往往不能开合、不能抓，也难加状态机（PDF p. 1–3）。

因此缺口是：**要交互、要物理、要能无限采样**，而不是再做一套更漂亮的静态网格。

```mermaid
flowchart LR
  Spec["房间规格"] --> Cut["切外轮廓"]
  Cut --> Rooms["递归分房间"]
  Rooms --> Conn["按约束连通"]
  Conn --> Door["采样门与灯光"]
  Door --> Floor["地板大件与 SAG"]
  Floor --> Wall["墙面物件"]
  Wall --> Surf["台面小件"]
  Surf --> Mat["材质与灯光"]
  Mat --> JSON["JSON 房屋描述"]
  JSON --> THOR["运行时载入 AI2-THOR"]
```

上图是机制示意，根据 PDF 图 2 与第 3 节（p. 3–6）重画，不是实测时间轴。

## 二、相关工作：平台各有专长，缺「可交互 + 可无限生成」

**平台。** AI2-THOR / ManipulaTHOR / RoboTHOR 走 Unity，重交互、状态和物理。Habitat 用扫描房，早期物体不可交互；Habitat 2.0 加了交互，但平面少、场景偏合成。iGibson 偏真实外观、交互有限；iGibson 2.0 转家务与状态。ThreeDWorld 偏液体与可变形。VirtualHome 用程序模拟人活动。RLBench、RoboSuite、Sapien 偏细操作（PDF p. 2–3）。ProcTHOR 自称的切口是：**程序化生成多样可交互场景，专门给大规模训练和数据增强用**（PDF p. 3）。

**大规模数据。** 图像、图文、3D、驾驶、抓取都有大规模集；可交互具身库很少。最近对照三条（PDF p. 3）：

- **HM3D**：约 **1,000** 套扫描房。静态、无物理、只适合导航；扩库要再扫再清洗。
- **OpenRooms**：同源扫描、可交互，但困在已扫房屋上，给一件物体标材质约 **1** 分钟。
- **Megaverse**：程序化、仿真极快，但外观像游戏。ProcTHOR 要对齐真实房屋的外观、物理和交互复杂度。

**场景合成。** 平面生成改编 Lopes 等人的约束生长；摆放靠近共现与迭代放置文献。作者说：接到自家物体库、生成超出静态房屋数据集分布的复杂房时，程序化比深度学习更稳、更灵活；细节与深度学习的限制写在附录，**本 13 页正文没有展开**（PDF p. 3）。

## 三、生成流水线：规格进，JSON 出，运行时才进 THOR

ProcTHOR 继承 AI2-THOR 的资产库、机器人和物理。给定规格（例如 1 卧 + 1 卫），多阶段条件采样：平面 → 外墙 → 灯光与门 → 大件 / 小件 / 墙件 → 颜色纹理 → 摆放（PDF p. 3）。房屋存成 JSON，训练时再载入，内存开销极低；可预先生成数据集，也可每轮动态采样（PDF p. 6）。

五条产品属性（PDF p. 3–6）：

### 多样性

- **平面。** 先迭代切边界得到外轮廓（矩形到复杂多边形），再用 Lopes 递归算法按房间切分，最后用用户约束决定连通（卧室常连卫、卫生间常单入口）（PDF p. 4）。
- **资产。** **108** 类、**1,633** 个可交互实例；不少继承 AI2-THOR，窗、门、台面等新件由 3D 设计师补。实例按 train/val/test 切；可拾放，部分多状态（灯开/关），部分带刚体部件（微波炉门）（PDF p. 1、p. 4）。
- **材质。** 墙：**40** 种常见纯色或 **122** 种砖 / 瓷砖一类贴图；地板 **55** 种；天花从墙材质里抽。物体材质只在类内随机，保证看起来仍是那一类（PDF p. 5）。还支持**动态材质随机**：每次载入场景可改单件颜色与材质（PDF p. 2）。
- **摆放。** 软标注：房间归属（沙发在客厅不在卫生间）、位置约束（冰箱靠墙、电视不上地）。**语义资产组**（Semantic Asset Group，SAG）：共现组，如四人餐桌，必须相关采样。先铺地板大件与 SAG（留出导航与操作空间），再墙面（窗、画），再台面（杯子上厨房台）（PDF p. 5）。正文写 **18** 个 SAG，约 **2,000 万** 组实例；第 4 节把组合数写成 **1,930 万**（PDF p. 6–7）。
- **灯光。** 一盏方向光（太阳）加多盏点光（灯泡），改颜色、强度、位置，模拟室内照明和一天中的时刻（PDF p. 5）。

### 交互、可定制、规模

物体可改位置和状态；带臂智能体可操作，多智能体可互操作。扫描库 HM3D 做不到这些（PDF p. 6）。用几行规格可生成教室、图书馆、办公室一类非住宅（PDF p. 6）。当前用 **16** 种场景规格做种子，号称超过 **1,000 亿** 种平面布局（PDF p. 6）。

## 四、ProcTHOR-10K：一小时采一万套，房间数从 1 到 10

训练 **10,000** 套；另有 **1,000** 验证、**1,000** 测试。资产切分在附录。房屋全可导航：不交互也能走遍每间（PDF p. 6）。对照当时规模：AI2-iTHOR **120** 景、RoboTHOR **89**、iGibson **15**、HM3D **1,000** 静态、Habitat 2.0 **105** 种平面（PDF p. 7）。这 10K 在本地工作站 **4** 块 NVIDIA RTX A5000 上 **1 小时**生成（PDF p. 7）。

规格例：1 卧连 1 卫、1 厨、1 厅（图 2）。房间数 **1–10**。物体来自 **95** 类（冰箱、台面、床、马桶、植物、门洞、窗等）。SAG 例：四人餐桌、床加两枕头（PDF p. 7）。图 10 给房间物体数、房屋面积（小 / 中 / 大桶）、房间数直方图；正文**没有**把直方图读成一张数字表（PDF p. 6–7）。作者强调分布比 iTHOR / RoboTHOR（偏单间）和 Gibson / HM3D（偏大宅）更宽（PDF p. 7）。

渲染速度是大规模训练的硬条件。表 1 在 **8** 块 NVIDIA Quadro RTX 8000 上测；1 GPU 用 **15** 进程，8 GPU 用 **120** 进程均分（PDF p. 7）：

| 算力 | 导航 FPS 小房 | 导航 FPS 大房 | 隔离交互 FPS 小 | 隔离交互 FPS 大 | 环境查询 FPS 小 | 环境查询 FPS 大 |
|---|---:|---:|---:|---:|---:|---:|
| 8 GPU | 8,599±359 | 3,208±127 | 6,488±250 | 2,861±107 | 480,205±19,684 | 433,587±18,729 |
| 1 GPU | 1,427±74 | 6,280±40 | 1,265±71 | 597±37 | 160,622±2,846 | 157,567±2,689 |
| 1 进程 | 240±69 | 115±19 | 180±42 | 93±15 | 14,825±199 | 14,916±186 |

导航是移动 / 旋转，交互是推物体，查询是问智能体尺寸一类。作者称帧率与 iTHOR、RoboTHOR 相当，尽管房屋更大；细节在附录（PDF p. 7）。注意：表里 **1 GPU、大房、导航** 印成 **6,280±40**，高于同列小房 **1,427**，也高于 8 GPU 大房；正文未解释，本文按印刷值抄，不自行改成 628。

## 五、ArchitecTHOR：十套艺术家房，专门测「像真房子吗」

iTHOR 是单间，RoboTHOR 像宿舍迷宫，都不像真宅。作者请专业 3D 艺术家做 **10** 套评测房（5 val、5 test）（PDF p. 2、p. 8）。验证房：**4–8** 间，每套 **121±26** 个物体，典型面积 **111±26 m²**。ProcTHOR-10K 方差更大：**1–10** 间，**76±48** 个物体，**96±74 m²**（PDF p. 8）。

## 六、实验：故意用简单网络，只喂 RGB

任务：ObjectNav（导航到某类物体）在 ProcTHOR、ArchitecTHOR、RoboTHOR、HM3D、AI2-iTHOR；操作两项——ArmPointNav（机械臂把物体从源点搬到 3D 坐标目标）和 1-phase 房间重排（改物体位置或状态，达到目标场景）（PDF p. 8）。

模型一律 CNN 编码视觉 + GRU 吃时间。ObjectNav 与重排用 EmbCLIP 一类 CLIP 架构；ArmPointNav 用 **3** 层卷积，作者发现比 CLIP 编码器更有效。训练框架 AllenAct；超参在附录（PDF p. 8）。全部实验**只用 RGB**，不用深度和其他模态（PDF p. 8）。

**0-shot**：只在 ProcTHOR 上训，下游基准训练集不用。微调：再用该基准自己的训练场景。0-shot 难在外观、布局、物体分布都和 ProcTHOR 不同：ArchitecTHOR 与 iTHOR 是高保真艺术家光影；HM3D 是扫描；RoboTHOR 墙板地板纹理很特殊（PDF p. 8）。

截至 **2022-06-14 10:00 PT**，微调后在三个公开榜当时第一：Habitat 2022 ObjectNav、AI2-THOR Rearrangement 2022、RoboTHOR ObjectNav。作者强调：架构很简单、只有 RGB；别人常加建图、视觉里程计和深度（PDF p. 2、p. 8）。

表 2（PDF p. 9）。Success 是成功率，SPL 是成功加权路径长度。重排另报 % Fixed Strict；ArmPointNav 另报拾取成功率：

| 任务 | 基准 | 方法 | Success | 第二指标 |
|---|---|---|---:|---:|
| ObjectNav | RoboTHOR | EmbCLIP（在 RoboTHOR 上训） | 47.0% | SPL 0.200 |
| ObjectNav | RoboTHOR | ProcTHOR 0-shot | 55.0% | 0.237 |
| ObjectNav | RoboTHOR | ProcTHOR + 微调 | 65.2% | 0.288 |
| ObjectNav | Habitat 2022 / HM3D | MLNLC（榜） | 52.0% | 0.280 |
| ObjectNav | Habitat 2022 / HM3D | FusionNav AIRI（榜） | 54.0% | 0.270 |
| ObjectNav | Habitat 2022 / HM3D | ProcTHOR 0-shot | 9.00% | 0.055 |
| ObjectNav | Habitat 2022 / HM3D | ProcTHOR + 微调 | 53.0% | 0.270 |
| ObjectNav | Habitat 2022 / HM3D | ProcTHOR + Large + 0-shot | 13.2% | 0.077 |
| ObjectNav | Habitat 2022 / HM3D | ProcTHOR + Large + 微调 | 54.4% | 0.318 |
| ObjectNav | AI2-iTHOR | EmbCLIP（在 iTHOR 上训） | 68.4% | 0.516 |
| ObjectNav | AI2-iTHOR | ProcTHOR 0-shot | 75.7% | 0.644 |
| ObjectNav | AI2-iTHOR | ProcTHOR + 微调 | 77.5% | 0.621 |
| ObjectNav | ArchitecTHOR | EmbCLIP（在 iTHOR 上训） | 18.5% | 0.118 |
| ObjectNav | ArchitecTHOR | ProcTHOR | 31.4% | 0.195 |
| 重排 1-phase 2022 | AI2-THOR | EmbCLIP | 7.10% | Fixed Strict 0.190 |
| 重排 1-phase 2022 | AI2-THOR | ProcTHOR 0-shot | 3.80% | 0.156 |
| 重排 1-phase 2022 | AI2-THOR | ProcTHOR + 微调 | 7.40% | 0.245 |
| ArmPointNav | ManipulaTHOR | iTHOR-SimpleConv（RGB 重训全 iTHOR） | 29.2% | Pickup 73.4 |
| ArmPointNav | ManipulaTHOR | ProcTHOR 0-shot | 37.9% | 74.8 |

引言把 RoboTHOR 微调相对旧 SoTA 的 SPL 增益写成 **8.8** 点：0.288 − 0.200 = 0.088（PDF p. 2、p. 9）。Habitat 微调相对下一档提交的 SPL 增益写成 **>3** 点：Large 微调 0.318 对 FusionNav 的 0.270（PDF p. 2、p. 9）。重排 Prop Fixed Strict **0.19 → 0.245**（PDF p. 2）。ArchitecTHOR 成功率 **18.5% → 31.4%**（PDF p. 2）。iTHOR 上 0-shot 已超过在 iTHOR 上训过的 EmbCLIP；微调成功率 **77.5%**（PDF p. 2）。ArmPointNav 的 0-shot 在只用 RGB 时超过旧 SoTA（PDF p. 2）。

读表时几条不能混：

- Habitat **0-shot 很弱**（成功率 9.00%，Large 也才 13.2%），和 RoboTHOR / iTHOR 的 0-shot 不是同一量级。摘要写「强 0-shot、常超过用过下游数据的旧 SoTA」对 RoboTHOR、iTHOR、ArmPointNav 成立；对 Habitat 和重排 **不成立**——重排 0-shot Fixed Strict **0.156 < 0.190**（PDF p. 1、p. 9）。
- iTHOR 微调成功率升到 77.5%，但 SPL **0.621 低于** 0-shot 的 **0.644**（PDF p. 9）。正文没解释。
- Habitat 的 Large 是更大 CLIP 骨干加更宽 RNN，细节在补充材料（PDF p. 9）。
- ArchitecTHOR 一行没有写 0-shot / 微调标签，只写 ProcTHOR **31.4% / 0.195**（PDF p. 9）。

规模消融（表 3，**不用**材质增强；训到训练成功率 80%）（PDF p. 9）：

| 训练房屋数 | ArchitecTHOR SPL / SR | RoboTHOR 0-shot SPL / SR | HM3D 0-shot SPL / SR | iTHOR 0-shot SPL / SR |
|---|---|---|---|---|
| 10 | 0.077 / 11.3% | 0.040 / 8.53% | 0.007 / 1.60% | 0.249 / 28.7% |
| 100 | 0.102 / 18.6% | 0.076 / 20.9% | 0.050 / 10.4% | 0.352 / 42.0% |
| 1,000 | 0.122 / 17.2% | 0.157 / 33.1% | 0.027 / 4.65% | 0.456 / 53.0% |
| 10,000 | 0.185 / 27.0% | 0.210 / 44.5% | 0.060 / 9.70% | 0.554 / 64.9% |

主趋势是房屋越多越好。例外：ArchitecTHOR 从 100 到 1,000 成功率 **18.6% → 17.2%**；HM3D 从 100 到 1,000 **10.4% → 4.65%** 再回升。作者仍写「测试随房屋数上升」，并认为还可以再扩（PDF p. 2、p. 9）。表 3 的 10K 数字与表 2 的 0-shot **不是同一列设定**（表 3 关掉材质增强，且 ArchitecTHOR 成功率 27.0% 对表 2 的 31.4%）。

## 七、作者自己划的边界

清单把局限、社会影响、许可证、算力、随机种子稳健性全部指到**附录**（PDF p. 13）。本 13 页正文能直接看到的边界是：

- 生成算法细节、资产切分、训练超参、与深度学习场景合成的对比，都不在正文（PDF p. 3、p. 6–8）。
- Habitat 扫描域上 0-shot 仍然差；重排 0-shot 低于 EmbCLIP 基线（PDF p. 9）。
- 表 1 个别单元格与「大房更慢」的直觉冲突，正文未注释（PDF p. 7）。
- 开源承诺写在引言：将开源 ProcTHOR，并发布本工作代码（PDF p. 2）。本 PDF 不是仓库快照。

## 八、可迁移的几件事

1. **可交互规模优先于扫描真实感。** 一万套能抓、能开合的合成房，比一千套静态扫描更能喂简单策略。迁移时先问任务要不要改物体状态。
2. **规格 + 分阶段约束，比端到端生成房子更好接资产库。** 平面、连通、SAG、墙面、台面拆开，坏了只换一段。
3. **证明数据规模，可以冻住小网络。** CNN+RNN、只用 RGB、不建图，让房屋数从 10 走到 10K。若一上来就上地图模块，分不清是数据还是架构。
4. **0-shot 要按域拆开报。** 同引擎的 RoboTHOR / iTHOR 可以「预训练即超 SoTA」；扫描 HM3D 必须微调。摘要里的「强 0-shot」不能当成每一行都强。
5. **评测房要另做，不要用生成器自己的 val。** ArchitecTHOR 专门卡「像不像真宅」；生成分布方差大，不等于覆盖艺术家布局。

## 关键词回看

- **ProcTHOR**：在 AI2-THOR 上程序化生成可交互房屋的框架。
- **ProcTHOR-10K**：采样出的 10,000 套训练房（另有 1K/1K val/test）。
- **ArchitecTHOR**：10 套艺术家评测宅。
- **SAG（语义资产组）**：必须一起采样的共现家具组。
- **0-shot（本文口径）**：只在 ProcTHOR 预训练，不用下游训练集。
- **ObjectNav / ArmPointNav / 1-phase Rearrangement**：找类导航、坐标搬运、一阶段房间复原。
- **EmbCLIP**：CLIP 视觉编码的具身基线；本文多数任务拿它当对照。

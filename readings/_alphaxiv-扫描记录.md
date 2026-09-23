# alphaXiv 扫描记录

下划线开头,不出现在网页。每日扫描 alphaXiv 热榜的历史:一篇论文在首发 7 天内**第一次**达到票数线的那天记一行,之后不再重复记。规则见 [10-材料解读流程](../docs/10-材料解读流程.md) 第二节「来源二」。

- `npm run papers:feed -- --write` 追加新行,判定与去向先填脚本初判;核实后由维护者或定时任务改成终判
- **票数**是达标那天扫到的 `public_total_votes`(网页点赞按钮上的数),之后不回写。它是两个解读库「按时间」视图里同一时段内的排序依据
- **去向**写 `reports/<公司>/<材料>` 或 `readings/<方向>/<材料>` 时,页面按这个路径把票数挂到卡片上;没入库的写 `待定`、`C · 待拍板` 或 `不收:理由`
- 判定取值:`A · 名单内` / `B · 名单外过闸门三` / `C · 待拍板` / `D · 已在库` / `待核 · 名单外`(脚本初判,核实后改成 B 或 C)

| 达标日 | 编号 | 标题 | 机构 | 票数 | 判定 | 去向 |
|---|---|---|---|---|---|---|
| 2026-09-23 | 2609.19969 | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | Deepseek | 570 | D · 已在库 | reports/DeepSeek/DeepSeek-V4.1-Flash |
| 2026-09-23 | 2609.mimo-scaling-reinforcement-learning | MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement | Xiaomi | 174 | A · 名单内(Xiaomi) | reports/Xiaomi/MiMo-V2.6 |
| 2026-09-23 | 2609.18207 | Reinforcement Learning for Real-Time Vision-Language-Action Policies | Stanford University | 168 | A · 名单内(Stanford) | reports/Stanford/Real-Time-EXPO-FT |
| 2026-09-23 | 2609.20807 | Score Centering Stabilizes Off-policy Reinforcement Learning | Together AI | 158 | A · 名单内(TogetherAI) | reports/TogetherAI/Score-Centering |
| 2026-09-23 | 2609.20519 | SoL-Pi: Recursively Scaling Auto-Research Loops for Efficient Agent Harness | NVIDIA、Nanyang Technological University、Massachusetts Institute of Technology | 127 | A · 名单内(NVIDIA) | reports/NVIDIA/SoL-Pi |
| 2026-09-23 | 2609.20800 | JEPA-Anything: Learning Predictive Models across Different Worlds | Phi AI Labs | 118 | A · 名单内(CUHK) | reports/CUHK/JEPA-Anything |
| 2026-09-23 | 2609.19138 | In-Context Robot Learning with VLM Agents | Morphi Robot、Shanghai Innovation Institute、Huazhong University of Science and Technology、Fudan University | 97 | B · 名单外 | readings/世界模型与 Agent/GPT-Policy |
| 2026-09-23 | 2609.20612 | What Does Privileged Information Add to On-Policy Self-Distillation? | National University of Singapore | 65 | A · 名单内(NUS) | reports/NUS/Privileged-Info-OPSD |
| 2026-09-23 | 2609.21561 | On Repulsive and Attractive Teachers: Separating Correctness from Behavior in Self-Distillation | ETH Zürich、Max Planck Institute for Intelligent Systems | 58 | A · 名单内(ETH) | reports/ETH/Repulsive-Self-Distillation |
| 2026-09-23 | 2609.stable-unstable-singularities-navier-stokes | Stable and Unstable Singularities in Navier-Stokes | (alphaXiv 未标) | 54 | 不收 | 不收:数学分析论文(Navier-Stokes 奇点理论),非计算机研究 |
| 2026-09-23 | 2609.20784 | RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning | Zhejiang University、Alibaba Group | 53 | A · 名单内(ZJU) | reports/ZJU/RetireOPD |
| 2026-09-23 | 2609.reinforcing-agents-collective-skills | Reinforcing Agents with Collective Skills | NVIDIA | 52 | A · 名单内(NVIDIA) | reports/NVIDIA/Skill2Env |
| 2026-09-23 | 2609.19107 | How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents | New York University、Q Labs | 52 | A · 名单内(NYU) | reports/NYU/Model-Growth-Scaling-Exponents |
| 2026-09-23 | 2609.19134 | ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments | AItonomyFoundation、PhAI-Labs | 47 | A · 名单内(Oxford) | reports/Oxford/ScienceIDE |
| 2026-09-23 | 2609.22068 | CodeMidas: Scaling Agentic Coding RL Environments from Code Itself | Xiaomi、Peking University、University of Hong Kong、Renmin University of China | 46 | A · 名单内(Xiaomi) | reports/Xiaomi/CodeMidas |
| 2026-09-23 | 2609.ier-opd | 1% of Tokens Can Be Enough: On Gradient Estimation in On-Policy Distillation | Ant Group、MBZUAI | 44 | A · 名单内(MBZUAI) | reports/MBZUAI/IER-OPD |
| 2026-09-23 | 2609.20804 | An Empirical Study of Harness Design for Coding Agents | University of Massachusetts Amherst、Emory University、Zoom Video Communications | 43 | B · 名单外 | readings/Agent 训练与工具使用/Coding-Harness-Design |
| 2026-09-23 | 2609.18708 | Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening | Shanghai Jiao Tong University、Shanghai Artificial Intelligence Laboratory、Westlake University、Nanjing University | 41 | A · 名单内(SJTU) | reports/SJTU/Value-Flattening |
| 2026-09-23 | 2609.20794 | PosteriorBench: From Point Estimates to Posterior Matching in Evaluating Generative Inverse Solvers | California Institute of Technology、National Taiwan University、Lawrence Berkeley National Laboratory、Stanford University | 36 | A · 名单内(Caltech) | reports/Caltech/PosteriorBench |
| 2026-09-23 | 2609.20649 | DexTouch-WM: Learning Action-Conditioned Tactile World Models from Human Touch for Dexterous Robot Manipulation | The Hong Kong University of Science and Technology (Guangzhou)、Xspark AI、Peking University、University of Hong Kong | 34 | A · 名单内(HKUST) | reports/HKUST/DexTouch-WM |
| 2026-09-23 | 2609.19101 | Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations | Goodfire | 32 | B · 名单外 | readings/可解释性与对齐/Reward-Hacking-Probes |
| 2026-09-23 | 2609.19644 | ScientistTwo: Pioneering the Human Knowledge Frontier with Autonomous AI | Google Cloud AI Research、University of Waterloo | 30 | A · 名单内(Google) | reports/Google/ScientistTwo |

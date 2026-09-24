# 新项目包装手册（Name + Logo + 描述一键生成）

> 2026-07 五个项目（Syncopate / Laminar / Switchboard / Halftone / Darkroom）包装沉淀的标准流程。
> 用法：把新项目仓库路径和本文档一起丢给 Claude，说「按 playbook 包装」，即可产出全套。

---

## 0. 总览：一个项目 = 六件产物

| # | 产物 | 用在哪 |
|---|---|---|
| 1 | 项目名（单个英文词） | 仓库名、logo、主页卡片标题 |
| 2 | README 副标题式长介绍（1-2 句英文） | 仓库 README `<p align="center">` |
| 3 | GitHub About 精简版（≤160 字符） | 仓库侧栏 About、主页项目卡片 description |
| 4 | Topics（空格分隔、一键复制） | 仓库 Topics |
| 5 | 16:9 Banner 生图提示词 | 生成 `images/logo.jpg` |
| 6 | 主页卡片条目（en + zh JSON） | 本仓库 `src/i18n/messages/*/collections.json` |

---

## 1. 起名方法论

**公式：从另一个学科借一个优雅的单词，该词的经典含义与项目核心机制构成一句话能讲清的隐喻。**

- 气质要求：学术正式 + 一点点科幻，避免 `XxxRL` / `Async-Xxx` 这类无记忆点的合成词
- 检验标准：① 一句话讲清隐喻（"切分音之于节拍器 = fully-async 之于 batch barrier"）；② 隐喻的多个侧面能映射到项目的多个机制（越深越好）；③ ML 圈无重名知名框架；④ 大写排在 logo 上视觉稳定
- 已用词族（借鉴气质，不要撞学科重复太近）：
  - **Syncopate**（音乐·切分音）= 打破同步节拍 → async agentic RL
  - **Laminar**（流体·层流）= 无气泡时间线 → GPU 执行气泡消除
  - **Switchboard**（通信·电话交换台）= token↔专家接线，接线员=CPU 开销，直拨=IBGDA → MoE all-to-all
  - **Halftone**（印刷·半色调）= 半墨印出无损照片 → int8 KV cache 量化
  - **Darkroom**(摄影·暗房) = 显影 latent image（摄影术语双关 diffusion latent）→ diffusion RL
- 每次给 1 个主推荐 + 2 个备选，说明取舍

**仓库改名约定**：GitHub 上改成 `新名_旧名`（如 `Halftone_QuantizedKVCache`），旧 URL 自动重定向。

## 2. 介绍文案三层

以 AdCampaignAgent 的文案为格式母版：`定位名词短语, featuring 三个核心要素. 量化结果/研究承诺.`

- **长介绍**（README 副标题）：featuring 后面塞 3-4 个带实测数字的要素；已完成项目收尾用实测数字，进行中项目收尾用研究承诺（"Quantifies when... and at what scale..."——比虚报结果更有力）
- **精简版**（GitHub About + 主页卡片）：长介绍压到 ≤160 字符，保留最硬的 1-2 个数字
- **中文版**（主页 zh 卡片）：地道翻译精简版，术语保留英文（GRPO、rollout、staleness 不翻）

**Topics**：10-12 个，小写连字符，前几个放大流量主题词（reinforcement-learning、cuda…），中间放差异化关键词（项目独有的机制词），不放 machine-learning 这类纯噪声泛词。空格分隔输出方便一键粘贴。

## 3. Banner 生图提示词模板

风格锚点：**深空 navy + CRT scanline + 左视觉右排印的 mission-control 风**。左半是项目机制的视觉隐喻（必须是「论文示意图级别准确」的图形，不是装饰），右半四层文字。生图用文字渲染强的模型（Nano Banana / GPT-Image），文字逐字检查，2-3 次内能对。

```
A 16:9 hero banner for an AI research project, retro-futuristic
mission-control aesthetic. Background: deep ink-navy space gradient with a
very subtle CRT scanline and pixel-grid texture, sparse tiny stars.

LEFT HALF — the visual: [机制隐喻图形：用发光几何图形把项目核心机制画成
示意图，如节奏条/流线/接线板/半色调渐变/胶片显影序列。加 2-4 个 monospace
HUD 小标注（thin leader lines），其中一个用 amber 十字准星标出关键指标]。
Colors: cyan #7FDBFF, teal, muted violet, one amber accent; everything
glowing softly against navy.

RIGHT HALF — the typography, left-aligned in a column:
top: a thin horizontal rule, then small letter-spaced monospace caps
"[领域词1 · 领域词2 · 领域词3]" in cyan;
center: huge bold condensed sans-serif title "[项目名]" — render "[前半]"
in off-white and "[后半]" in bright cyan, with a subtle [与隐喻呼应的]
texture inside the letterforms;
below it: two lines of monospace subtitle in off-white:
"[一句话定位]"
"[一句话卖点/口号]";
bottom: another thin horizontal rule, then a small monospace caps strip in
muted gray-cyan: "[硬件/模型 · 实测数字1 · 实测数字2 · 关键词]".

Overall: academic and restrained, NASA-mission-patch-meets-research-paper,
slight sci-fi glow but no spaceships, no robots, no human figures, no
circuit-board clichés. Clean spacing, precise alignment, high contrast,
crisp text.
```

要点：标题双色拆分（前半 off-white 后半 cyan）；副标题第二行可以是双关口号（"Half the Memory, None the Wiser"），好的口号能直接复用为博客标题。

## 4. 项目仓库侧操作清单

1. 原自用中文 README 改名 `ForChaoyu.md` 保留，模板 README 复制为 `README.md`
2. 按项目实际信息重写 README：banner `width="100%"` 不带 height；实测数字进表格；**诚实标注已知缺口**（对 PhD 申请是加分项）；方法论/反直觉发现单独成节；Roadmap 已完成打勾+展望；无 HuggingFace 的项目删 HF 徽章；Report Bug / Request Feature 旧链接和 LinkedIn 保留不动（统一后续处理）
3. 补 `LICENSE`（MIT）+ `images/logo.jpg`（生成的 banner）
4. 检查 gitignore（权重/数据/第三方版权材料不入库；注意 git 无法 re-include 被排除目录内的文件，需 `dir/*` + `!dir/file` 写法）
5. commit + push → GitHub 改名 `新名_旧名` → `git remote set-url` + sed 更新 README 内链接 → 再 push
6. 网页端手动填 About（精简版）+ Topics

## 5. 主页侧操作清单（本仓库）

1. banner 复制为 `public/<Name>.jpg`
2. `src/i18n/messages/{en,zh}/collections.json` 的 `projects.items` 各加一条（**中英成对**）：
   ```json
   {
     "title": "<新名_旧名>",
     "href": "https://github.com/ChaoyuWang04/<新名_旧名>",
     "dates": "<真实起止，如 Jun 2026 – Present / 2026 年 6 月 – 至今>",
     "active": true,
     "description": "<精简版介绍（en）/ 中文翻译（zh）>",
     "links": [{ "type": "Github", "href": "<仓库URL>", "icon": "github" }],
     "image": "/<Name>.jpg",
     "video": ""
   }
   ```
3. 排序：新项目按时间倒序插在前面；有 HF 的加 HuggingFace link
4. `node scripts/validate-static-assets.mjs` 过了再 push（大小写必须精确匹配）
5. 其余站点维护细节见项目记忆（缩进差异、中文 "至今"、区块隐藏逻辑等）

---

*五个项目的完整产物实例（对照参考）：各仓库的 README + 本仓库 collections.json 现有条目。*

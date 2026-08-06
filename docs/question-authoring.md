# 题目编写规范

每道题一个 md 文件,路径:`questions/<分类名>/<序号>-<简称>.md`。

- 分类文件夹名就是页面上显示的分类名(中文英文皆可)
- 文件名的序号前缀(`001-`)决定题目顺序;简称用小写英文/拼音或中文短语,连字符分隔,不超过 5 个词(如 `003-grpo-kl.md`、`001-手写-self-attention.md`)
- 以 `_` 或 `.` 开头的文件/文件夹会被忽略(所以 `_template.md` 不算题)

## frontmatter 字段

```yaml
---
difficulty: 中等             # 可选:简单 | 中等 | 困难
topic: GRPO/KL散度           # 必填:去重索引,1-2 段
summary: GRPO 的 KL 惩罚为何放 reward   # 必填:一句话题干,≤50 字
tags: [RL, GRPO]             # 可选:知识点标签
company: 字节、美团           # 可选:出题公司,多家用、分隔
mastered: false              # 程序写回,新题保持 false
---
```

### topic:三层主题定位(去重的核心)

完整层级 = `分类文件夹 / topic 第一段 / topic 第二段`,例如 `RL → GRPO → KL散度` 写成:文件放在 `questions/RL/`,frontmatter 写 `topic: GRPO/KL散度`。综述型题目可以只有一段(如 `topic: 解码策略/综述`)。

`npm run topics` 会打印全库主题树——**加题前先看树,同一考点不重复建题**;问法角度确实不同才新增,并用不同的叶子名区分。

`topic` 与 `summary` 有测试守护(`tests/samples.test.ts`):缺失、超过 2 段、summary 超 50 字都会让 `npm test` 报错并指出文件名。

### 「待校对」标签约定

答案由 AI 代写(截图无答案)的题,tags 里加 `待校对`,页面上会显示 chip;复核确认后手动删掉该标签。

## 六个分区(关键约定)

正文用 `## ` 二级标题分区,分区名**只能**是以下六个:

| 分区 | 必填 | 说明 |
|---|---|---|
| `## 题目` | ✅ | 题干,默认显示 |
| `## 要点` | | 展开后显示为 KEY POINTS |
| `## 答案` | ✅ | 展开后显示为 REFERENCE ANSWER |
| `## 知识点` | | 展开后显示为 KNOWLEDGE POINT |
| `## 追问` | | 展开后显示为追问题目 |
| `## Note` | | 你的笔记,页面上编辑,程序写回 |

**`## ` 只保留给这六个分区名**,题目/答案内部的小标题请用 `###` 及以下。围栏代码块里的 `## ` 不受影响(程序做了围栏感知)。

## 长度红线

- 题干 ≤ 3 句;summary ≤ 50 字
- 答案 ≤ 约 500 字或等价条目(一张对比表 + 几组要点)
- 超限**不要硬塞**:拆成更细的 topic 叶子另建一题,把关联问法放进「追问」分区

## 支持的 markdown 能力

- **表格**(GFM):正常 `| a | b |` 语法
- **数学公式**(KaTeX):行内 `$E=mc^2$`;**块级公式的 `$$` 必须独占一行**,写在同一行会被当成行内公式(不居中、字号小):

  ```markdown
  $$
  P(w_i) = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
  $$
  ```

- **Mermaid 图**:语言标注为 `mermaid` 的代码块自动渲染成图
- **代码高亮**:标注语言的代码块(`python`、`ts` 等)

## 知识库文章

`knowledge/<分类>/<topic第一段>.md`,一篇文章 = 一个 topic 主题的整体讲解(题目是切片,文章是全景)。

- **文件名必须与题目 frontmatter `topic` 的第一段完全一致**(如 `knowledge/RL/GRPO.md` 对应 `topic: GRPO/...`)——刷题页的「📚 知识库」入口和文章页的关联题目列表都靠这个字符串匹配;文章名的受控词表与施工进度见 [kb-roadmap.md](kb-roadmap.md)
- **粒度:尽量细碎**,一篇只讲一个可独立成文的知识点(GQA、RoPE、ZeRO 各一篇);同主题再单开一篇「XX总览」做横向对比与串联
- 无 frontmatter,正文以 `# 标题` 开头;markdown 能力与题目一致(表格/KaTeX/mermaid,块级公式 `$$` 独行)
- **完整性优先**:覆盖该知识点的全部细节与公式,同时每个术语/公式都配通俗直白的类比或解释(风格样板:`knowledge/RL/GRPO.md` 与 `docs/references/frontier-llm-architecture-handbook-2026.md`)
- **可视化**:流程/结构优先用 mermaid;mermaid 表达不了的(如几何示意、论文截图)留占位符独立一行:`> 🖼️ 占位:<想要的图的描述>`,用户后期替换为图片
- **文末必须有 `## 相关文献`**:该主题的核心论文列表,格式 `- 论文名 — [arXiv:xxxx.xxxxx](https://arxiv.org/abs/xxxx.xxxxx)`;不确定的编号必须联网核实,**严禁凭记忆编造链接**;无 arxiv 的资源(博客/文档)给原始 URL
- 建议骨架:是什么(一句话)→ 动机 → 核心机制(公式)→ 细节与常见坑 → 与近亲方法对比 → 面试考点串联 → 相关文献
- 导入新题发现文章未覆盖的点,**增量补进对应小节**(规则见 [import-workflow.md](import-workflow.md))

## 批量出题

把本规范和 `questions/_template.md` 一起交给 AI 批量生成即可;截图批量导入的协作流程见 [import-workflow.md](import-workflow.md)。写完跑 `npm test` 验证全部可解析、索引字段合规。

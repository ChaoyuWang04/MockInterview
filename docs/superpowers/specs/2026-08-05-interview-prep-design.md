# 本地大模型面试刷题系统 — 设计文档

日期:2026-08-05
状态:已获用户批准

## 1. 背景与目标

复刻老师线上刷题平台的核心体验,做一个**本地、单人、极简**的大模型面试刷题系统。所有内容(题目、答案、备注)与状态(已掌握)持久化为 markdown 纯文本文件,可用 git 管理、随意备份迁移。

### 范围内

- 主页:总题数/已掌握大数字 + 分类卡片(题数、进度条、已掌握数)
- 刷题页:单题展示,答案点击展开(空格键快捷),←/→ 切换题目,右上角分类进度
- 标记已掌握(写回文件 frontmatter)
- Note 备注区,页面编辑、写回文件的 `## Note` 分区
- Markdown 渲染完整支持:GFM 表格、KaTeX 数学公式、Mermaid 图、代码高亮

### 范围外(明确不做)

- 刷题计划/每日目标/连胜天数/待复习队列等计划类功能
- 网页内新增/编辑题目(题目内容直接用编辑器改 md 文件)
- 多用户、登录、部署、数据库
- 搜索(如未来需要另立计划)

## 2. 技术栈

| 用途 | 选择 |
|---|---|
| 运行时 | Node.js 20+ LTS |
| 框架 | Next.js 15(App Router)+ React 19 + TypeScript |
| 样式 | Tailwind CSS |
| frontmatter 解析 | gray-matter |
| Markdown 渲染 | react-markdown + remark-gfm + remark-math + rehype-katex |
| Mermaid | mermaid(客户端动态 import,` ```mermaid ` 代码块渲染为 SVG) |
| 代码高亮 | rehype-highlight |
| 测试 | Vitest(仅覆盖 lib/questions.ts) |

使用方式:`npm run dev` → 浏览器打开 `http://localhost:3000`。无数据库、无环境变量、无部署。

## 3. 数据格式

### 3.1 文件路径即身份

`questions/<分类名>/<序号>-<简称>.md`,例:`questions/RAG/001-检索优化技术.md`。

- 分类文件夹名直接用中文/英文原名(`RAG`、`手撕代码`),即页面显示名,无需配置文件(中文名出现在 URL 中,`[category]` 路由参数需 `decodeURIComponent`)
- 分类排序:文件夹名排序;题目排序:文件名排序(序号前缀保证顺序)
- 题目的唯一标识 = `分类名/文件名`,不在 frontmatter 里重复存 id

### 3.2 单题文件结构

```markdown
---
difficulty: 简单        # 简单 | 中等 | 困难
tags: [RAG, 查询扩展]
company: 字节           # 可选
mastered: false         # 程序写回
---

## 题目

(必填,支持全部 markdown 语法)

## 要点

(可选)

## 答案

(必填,表格/公式/mermaid/代码块自由使用)

## 知识点

(可选)

## 追问

(可选)

## Note

(可选;用户在页面编辑,程序写回本分区)
```

frontmatter 字段中仅 `mastered` 必需(缺省视为 `false`);`difficulty`、`tags`、`company` 均可选,缺省时页面不渲染对应元素。

### 3.3 分区约定(关键约束)

顶层 `## ` 二级标题**保留**给六个固定分区名:`题目`、`要点`、`答案`、`知识点`、`追问`、`Note`。题目/答案内部的小标题必须用 `###` 及以下层级。这是程序切分文件与写回 Note 的依据,写入题目模板与编写规范文档。

### 3.4 写回策略

- 「已掌握」→ 改写 frontmatter 的 `mastered` 字段。**必须用针对 `mastered:` 行的定点字符串替换**,不得用 gray-matter 整体重新序列化 frontmatter(后者会重排格式、丢失行内注释,违反验收标准 3)
- Note → 替换 `## Note` 分区内容(无该分区则追加到文件末尾;保存空内容 = 清空该分区正文,保留分区标题)
- 两者均为「读取 → 修改 → 写临时文件 → rename 原子替换」,不改动文件其余内容

## 4. 目录结构

```
interviewprep/
├── docs/                            # 项目文档(独立文件夹)
│   ├── superpowers/specs/           #   设计文档
│   ├── question-authoring.md        #   题目编写规范(分区约定、frontmatter 字段)
│   └── maintenance.md               #   维护指南(加题/备份/升级依赖)
├── questions/                       # 题库内容(纯内容,与代码完全分离)
│   ├── _template.md                 #   新题模板
│   └── <分类名>/<序号>-<简称>.md
├── app/                             # Next.js 页面 + API
│   ├── layout.tsx
│   ├── globals.css
│   ├── page.tsx                     #   主页
│   ├── [category]/page.tsx          #   刷题页
│   └── api/question/route.ts        #   唯一写接口(PATCH)
├── components/
│   ├── Markdown.tsx                 #   统一渲染器
│   ├── Mermaid.tsx
│   ├── CategoryCard.tsx
│   ├── QuestionView.tsx             #   刷题页主体(客户端组件)
│   └── NoteEditor.tsx
├── lib/
│   ├── questions.ts                 #   核心模块:扫描/解析/序列化/原子写回
│   └── types.ts
├── tests/questions.test.ts
├── README.md
└── package.json / tsconfig.json / next.config.ts
```

原则:**内容(questions/)、文档(docs/)、代码(app/components/lib/)严格分离**。题库文件夹不含任何代码,未来换框架时题库原样搬走。

## 5. 页面与数据流

### 5.1 主页 `/`(服务端组件)

请求时实时扫描 `questions/`(dev 模式无缓存,加题刷新即见):

- 大数字:已掌握 / 总题数
- 分类卡片网格:分类名、题数、已掌握数、进度条;点击进入 `/<分类名>`

### 5.2 刷题页 `/[category]`(服务端组件加载 + 客户端组件交互)

服务端一次性读取该分类全部题目(解析后的结构化数据)传给客户端组件 `QuestionView`,之后的题目切换纯客户端完成,无需再请求。

单题视图(对齐老师系统,去掉计划类元素):

1. 顶栏:返回主页链接;右上角「<分类> 掌握 x/N」进度
2. 题头:分类 · 难度 · #n/N
3. 题目 markdown(默认显示)+ 标签行(tags、company)
4. 「展开答案 / 收起答案」按钮,**空格键**切换
5. 展开后依次渲染:要点、答案、知识点、追问(缺省的分区不渲染)
6. 「标记已掌握 / 取消掌握」按钮 → PATCH API → 成功后更新本地状态与进度
7. Note 区:显示已有内容,「编辑」→ textarea →「保存」→ PATCH API
8. **← / →** 方向键或按钮切换上一题/下一题(空格/方向键在 textarea 聚焦时不触发)

进入分类页固定从第 1 题开始,当前题号只存客户端状态,刷新后回到第 1 题——这是有意的极简设计,不加 `?q=` 之类的位置参数。

### 5.3 写接口(唯一 API)

`PATCH /api/question`,body:`{ category, file, mastered?, note? }`

- 校验:category 必须存在于 `questions/` 下的真实文件夹列表、file 必须存在于该文件夹的真实文件列表(白名单校验,杜绝路径穿越);拒绝一切不合法参数,返回 400
- 写入:见 3.4 原子写回
- 读操作不设 API,服务端组件直接调用 `lib/questions.ts`

## 6. Markdown 渲染管线

`components/Markdown.tsx` 统一封装,全站唯一入口:

- react-markdown + remark-gfm(表格/删除线/任务列表)
- remark-math + rehype-katex(`$...$`、`$$...$$`),katex CSS 全局引入
- rehype-highlight(代码高亮)
- ` ```mermaid ` 代码块拦截 → `Mermaid.tsx` 客户端组件(动态 import mermaid,渲染 SVG;渲染失败显示原始代码块与错误信息,不崩页面)

## 7. 错误处理

| 场景 | 行为 |
|---|---|
| frontmatter 解析失败 / 缺少必填分区 | 该题在页面显示红色错误卡片并注明文件名与原因;不悄悄跳过、不影响其他题目;统计中不计入已掌握 |
| PATCH 写回失败 | 页面 toast 提示,前端状态回滚不变 |
| Mermaid 语法错误 | 显示原始代码 + 错误信息 |
| 空分类文件夹 / 空题库 | 主页/刷题页显示引导文案(指向模板与编写规范) |

## 8. 测试策略

Vitest 单元测试,只覆盖最值得测的 `lib/questions.ts`:

- 解析 ↔ 序列化往返:内容不丢失、不变形
- Note 替换:只改 Note 分区,其余字节不变;无 Note 分区时正确追加
- mastered 切换:只改 frontmatter 对应字段
- 分区切分:正文含 `###` 子标题、代码块内含 `## ` 时不误切
- 路径校验:非法 category/file 被拒绝

页面交互以浏览器手动验证为主(实现完成后逐项过验收清单),不做 e2e。

## 9. 维护方式

- **加题**:复制 `questions/_template.md` 到分类文件夹,改内容,刷新页面即生效;新分类 = 新建文件夹
- **版本管理**:整个仓库(含题库与 note)git 管理
- **批量出题**:让 AI 按 `_template.md` 与 `docs/question-authoring.md` 规范批量生成 md 文件
- **文档**:README(上手)、question-authoring.md(编写规范)、maintenance.md(维护指南)随实现一并交付

## 10. 验收标准

1. `npm run dev` 一条命令启动,主页正确显示实时统计与分类卡片
2. 刷题页完整支持:展开/收起答案(含空格键)、←/→ 切题、标记已掌握、Note 编辑保存
3. 标记已掌握与保存 Note 后,对应 md 文件内容正确变化且其余内容零改动(git diff 验证)
4. 含表格、KaTeX 公式、mermaid 图、代码块的题目全部正确渲染
5. 全部 Vitest 通过

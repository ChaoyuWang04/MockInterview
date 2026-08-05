# 本地大模型面试刷题系统

本地、单人、极简的刷题系统:**markdown 文件即数据库**。题目、答案、笔记、掌握状态全部存在 `questions/` 下的纯文本文件里,可以用 git 管理、随意备份迁移。

## 启动

```bash
npm run dev
```

浏览器打开 <http://localhost:3000>。首次使用需先 `npm install`(要求 Node.js 20+)。

## 使用

- **主页**:已掌握 / 总题数 + 分类卡片,点卡片进入刷题
- **刷题页**:
  - `空格` 或点击「展开答案」:展开/收起答案
  - `←` / `→`:上一题 / 下一题
  - 「标记已掌握」:写回题目文件 frontmatter 的 `mastered` 字段
  - NOTE 区「编辑」:写笔记,保存后写回文件的 `## Note` 分区
  - 右侧题目导航栏(宽屏显示):题号按 topic 主题分组,绿色 = 已掌握,点击直达;方便按主题集中复习

## 加题三步

1. 复制 `questions/_template.md` 到分类文件夹,如 `questions/RAG/003-新题.md`(新分类 = 新建文件夹)
2. 按模板写内容(格式规范见 [docs/question-authoring.md](docs/question-authoring.md))
3. 刷新页面即生效

## 目录结构

```
questions/     题库(纯内容,与代码完全分离)
  _template.md   新题模板
  <分类>/<序号>-<简称>.md
app/           Next.js 页面 + PATCH API
components/    Markdown 渲染器 / 刷题视图等组件
lib/           核心模块:扫描、解析、定点写回
tests/         Vitest 单元测试(npm test)
docs/          文档与设计资料
```

## 技术栈

Next.js 15(App Router)· React 19 · TypeScript · Tailwind CSS 4 · react-markdown(GFM 表格 + KaTeX 公式 + 代码高亮)· Mermaid · gray-matter · Vitest

维护指南见 [docs/maintenance.md](docs/maintenance.md)。

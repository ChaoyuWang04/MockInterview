# 本地大模型面试刷题系统

本地、单人、极简的刷题系统:**markdown 文件即数据库**。题目、答案、笔记、掌握状态全部存在 `questions/` 下的纯文本文件里,可以用 git 管理、随意备份迁移。

## 启动

**日常使用**(生产模式常驻,终端任意目录可用):

```bash
leet-start
```

打开 <http://localhost:3000>;停止用 `leet-stop`,状态用 `leet-status`。改内容(题目/笔记/文章)不用重启,改代码用 `leet-rebuild`。详见 [docs/server-daemon.md](docs/server-daemon.md)。

**开发调试**(热更新,端口 3001,与常驻服务互不干扰):

```bash
npm run dev
```

首次使用需先 `npm install`(要求 Node.js 20+)。

## 使用

- **主页**:已掌握 / 总题数 + 分类卡片,点卡片进入刷题
- **分类页(列表视图)**:进入分类先看到题目清单——按主题分组,每行是 summary + 公司 + 难度 + 「高」高频开关,已掌握的打 ✓ 并灰化;点任意一行进入刷题界面,`Esc` 回列表。高频题自动排到组内最前(组内再按 简单→中等→困难,组的顺序不变)
- **刷题页**:
  - `空格` 或点击「展开答案」:展开/收起答案
  - `←` / `→`:上一题 / 下一题
  - 「标记已掌握」:写回题目文件 frontmatter 的 `mastered` 字段
  - NOTE 区「编辑」:写笔记,保存后写回文件的 `## Note` 分区
  - 右侧题目导航栏(宽屏显示):题号按 topic 主题分组,绿色 = 已掌握、右上红点 = 高频,点击直达;顺序与列表视图一致
- **知识库**(主页入口或 `/kb`):每篇文章对应一个 topic 主题的整体讲解,题目页遇到有文章的主题会出现「📚 知识库」入口;文章放 `knowledge/<分类>/<topic>.md`
- **LeetCode 热题 100**(主页入口或 `/leetcode`):官方 17 专题清单;点「高」标记高频题(自动排到组内最前,同桶内按简单→困难),点 📝 开悬浮小窗记解题技巧(纯文本自动保存),窗内可直达 LeetCode US / 力扣中国;数据在 `leetcode/`,说明见 [docs/leetcode-hot100.md](docs/leetcode-hot100.md)
- **开源项目解读**(主页入口或 `/opensource`):按主题分组的项目解读,阅读器第 0 节是全景总览、之后每节一个子系统,←/→ 翻节;解读文档在 `opensource/`,对应源码仓在 `projects/`(git 不追踪),规范见 [docs/opensource-workflow.md](docs/opensource-workflow.md)

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

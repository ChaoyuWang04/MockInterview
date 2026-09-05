# 本地大模型面试准备系统

本地、单人、Markdown 即数据的学习系统。项目有六个功能模块:**题库、知识库、开源解读、报告解读、LeetCode 和模拟面试**。内容与学习记录都以纯文本保存,可以直接编辑、用 git 追踪并随时迁移。

> 接手维护?先读 [docs/00-START.md](docs/00-START.md)(守则 + 文档地图)。

## 启动

**日常使用**(生产模式常驻,终端任意目录可用):

```bash
leet-start
```

打开 <http://localhost:3000>;停止用 `leet-stop`,状态用 `leet-status`。改内容(题目/笔记/文章)不用重启,改代码用 `leet-rebuild`。详见 [docs/08-常驻服务.md](docs/08-常驻服务.md)。

**开发调试**(热更新,端口 3001,与常驻服务互不干扰):

```bash
npm run dev
```

首次使用需先 `npm install`(要求 Node.js 20+)。

## 六个功能模块

| 模块 | 入口 | 数据位置 | 说明 |
|---|---|---|---|
| 题库 | `/`、`/<分类>` | `questions/` | 真题、答案、掌握状态、高频标记和 Note |
| 知识库 | `/kb` | `knowledge/` | 按学习路径组织的文章,题目通过 `topic` 关联 |
| 开源解读 | `/opensource` | `opensource/` | 基于 `projects/` 中真实源码写成的项目导读 |
| 报告解读 | `/reports` | `reports/` | 按公司整理 Technical Report 与技术论文的完整长文解读 |
| LeetCode | `/leetcode` | `leetcode/` | Hot 100、补充题、高频标记和轻量笔记 |
| 模拟面试 | `/interview` | `interview/` | 使用题库和知识库出题、判卷并保存复盘 |

除模拟面试的云端判卷和可选云端语音识别外,内容浏览与编辑均为本地文件操作。

## 添加内容

- 新增、合并或从截图导入真题:走 [真题入库流程](docs/02-题库导入流程.md),不能跳过去重和知识库同步。
- 修改题目格式或答案:遵守 [题目编写规范](docs/03-题目写作规范.md)。
- 编写知识库文章:先查 [知识库地图](docs/04-知识库地图.md),再按 [知识库写作契约](docs/05-知识库写作契约.md) 写作。
- 解读开源项目:见 [开源解读流程](docs/06-开源解读流程.md)。
- 解读一份 Technical Report 或技术论文:见 [报告解读流程](docs/10-基模报告流程.md)。
- 维护 LeetCode 清单:见 [LeetCode 清单](docs/07-LeetCode清单.md)。

## 目录结构

```
questions/     真题库
knowledge/     知识库
opensource/    开源项目解读
projects/      解读依据的源码快照(git 不追踪)
reports/       Technical Report 与技术论文的长文解读
papers/        报告 PDF 原件(默认不追踪 PDF)
leetcode/      LeetCode 清单与笔记
interview/     简历、画像、会话复盘与本地缓存
app/           Next.js 页面和 API
components/    页面组件
lib/           内容读取、写回和业务逻辑
scripts/       维护、索引与服务脚本
tests/         自动检查
docs/          当前手册、参考资料与历史设计存档
```

## 技术栈

Next.js 15(App Router)· React 19 · TypeScript · Tailwind CSS 4 · react-markdown(GFM 表格 + KaTeX 公式 + 代码高亮)· Mermaid · gray-matter · Vitest

维护指南见 [docs/09-日常维护.md](docs/09-日常维护.md)。

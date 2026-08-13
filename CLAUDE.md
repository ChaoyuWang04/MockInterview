# 本地大模型面试刷题系统

markdown 文件即数据库的本地刷题应用(Next.js 15 App Router)。题库在 `questions/`,一题一文件,分类即文件夹;知识库在 `knowledge/`,一篇文章 = 一个 topic 主题的整体讲解(文件名 = topic 第一段,页面路由 `/kb`);开源项目解读在 `opensource/<主题>/<项目>/NN-*.md`(路由 `/opensource`),对应源码仓在 `projects/`(git 不追踪,仅作阅读材料)。

## 常用命令

- `npm run dev` — 启动应用(localhost:3000)
- `npm run topics` — 打印主题树(批量导入前的去重索引)
- `npm test` — 全部验证(题库可解析 + topic/summary 合规 + 知识库文章非空),秒级

## 关键文档(按任务查)

- **用户发题目截图要求入库**:严格按 [docs/import-workflow.md](docs/import-workflow.md) 执行(去重判定、分类映射、汇总表;验证只跑 `npm test`,不开浏览器)
- **出题/改题格式**:[docs/question-authoring.md](docs/question-authoring.md)(六分区、topic/summary、长度红线、块级公式 `$$` 独行)
- **知识库扩建/续建**:[docs/kb-roadmap.md](docs/kb-roadmap.md)(需求清单、进度状态、文章标准摘要;架构类参考 docs/references/ 下的架构手册)
- **开源项目解读(新写/续写/加项目)**:[docs/opensource-workflow.md](docs/opensource-workflow.md)(必须读 projects/ 下真实代码后写作、分页标准、项目清单与状态)
- **日常维护**:[docs/maintenance.md](docs/maintenance.md)
- 设计与实现背景:`docs/superpowers/` 下的 spec 与 plan

## 硬约束

- 题目正文的 `## ` 二级标题只保留给六分区名(题目/要点/答案/知识点/追问/Note),内部小标题用 `###` 及以下
- 修改 `lib/questions.ts` 写回逻辑时:mastered/Note 必须定点字符串替换,**禁止**用 gray-matter 整体重新序列化 frontmatter(会破坏用户手写格式与行内注释)
- 不要主动添加:刷题计划/待复习队列/搜索/网页内题目编辑器/URL 位置参数(用户明确排除的功能)

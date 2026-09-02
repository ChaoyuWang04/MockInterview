# 本地大模型面试刷题系统

markdown 文件即数据库的本地学习系统(Next.js 15 App Router),五个模块:题库 `questions/` · 知识库 `knowledge/` · 开源解读 `opensource/`(源码仓在 `projects/`,git 不追踪)· LeetCode 清单 `leetcode/` · 常驻服务 `scripts/leet-*`。

## 先读这个

**[docs/00-START.md](docs/00-START.md)** —— 守则 + 文档地图,按任务查该看哪份手册。任何工作开始前先看它。

## 常用命令

- `leet-start` / `leet-stop` / `leet-rebuild` — 常驻生产服务(localhost:**3000**,改代码后要 rebuild)
- `npm run dev` — 开发服务(localhost:**3001**,与常驻服务互不干扰)
- `npm run topics` — 打印题库主题树(去重索引)
- `npm test` — 全量验证(题库可解析 / topic·summary 合规 / 文章名唯一),秒级
- `npm run kb:lint` — 知识库写作契约检查(标记/公式/标题层级/mermaid 规模/代码长度/收尾/断链/篇幅/考点表规模),秒级

## 硬约束

- 题目正文的 `## ` 二级标题只保留给六分区名(题目/要点/答案/知识点/追问/Note),内部小标题用 `###` 及以下
- 修改 `lib/questions.ts` 写回逻辑时:mastered/highfreq/Note 必须定点字符串替换,**禁止**用 gray-matter 整体重新序列化 frontmatter(会破坏用户手写格式与行内注释)
- 知识库文章名全局唯一(它是题目 `topic` 第一段的匹配键)
- 不编造:论文编号联网核实,项目实现去 `projects/` 下确认
- 不要主动添加:刷题计划/待复习队列/搜索/网页内题目编辑器/URL 位置参数(用户明确排除的功能)
- 旧稿标 `⚠️ 旧版` 的文章,**「重写」= 按新标准重做语言/数字/结构/边界,不是修 lint 违规**(判据见 `docs/05-知识库写作契约.md` 第二节)
- 用户发来**真实面经**时走 `docs/10-面经双写流程.md`:题库与知识库**两边都要写**,且原文先逐字存档到 `docs/references/面经原题.md`

# 维护指南

## 日常习惯

- **题库即数据**:`questions/` 里的一切(包括笔记和掌握状态)都是纯文本,勤 `git commit`,历史就是你的刷题记录
- 备份 = 推到私有远端仓库,或直接拷贝整个项目文件夹;换电脑 `git clone` + `npm install` 即恢复

## 常见问题

| 现象 | 原因与处理 |
|---|---|
| 页面显示红色错误卡片 | 卡片上写着文件名和原因(frontmatter 语法错 / 缺「题目」「答案」分区),照着改文件即可 |
| 新分类/新题不显示 | 检查文件(夹)名是否以 `_` 或 `.` 开头(会被忽略)、文件是否 `.md` 结尾;然后刷新页面 |
| 答案里的小标题变成了分区 | 正文里用了顶层 `## `,改成 `###` 及以下 |
| 端口被占用 | `lsof -ti:3000 | xargs kill`,或 `npm run dev -- -p 3001` |
| Mermaid 图显示失败 | 页面会显示原始代码和错误信息,按提示修语法 |

## 依赖升级

```bash
npm outdated        # 查看可升级项
npm update          # 小版本升级
npm test && npm run build   # 升级后跑测试与构建确认
```

大版本升级(next 16、react 20 等)建议单独开分支验证。

## 测试

```bash
npm test
```

- `tests/questions.test.ts`:核心解析/写回逻辑的单元测试
- `tests/samples.test.ts`:真实题库回归防护——任何解析失败的题会带文件名报错

## 设计资料

- 设计文档:`docs/superpowers/specs/2026-08-05-interview-prep-design.md`
- 实现计划:`docs/superpowers/plans/2026-08-05-interview-prep.md`

// 用力扣官方 GraphQL 返回的 JSON 覆盖 leetcode/hot100.md
// 用法:见 docs/leetcode-hot100.md
//   1) curl 拉取官方数据到 /tmp/hot100.json
//   2) node scripts/hot100-sync.mjs /tmp/hot100.json
import fs from 'node:fs'
import path from 'node:path'

const src = process.argv[2]
if (!src) {
  console.error('用法: node scripts/hot100-sync.mjs <官方 JSON 路径>')
  process.exit(1)
}

const raw = JSON.parse(fs.readFileSync(src, 'utf8'))
const groups = raw?.data?.studyPlanV2Detail?.planSubGroups
if (!Array.isArray(groups) || groups.length === 0) {
  console.error('JSON 结构不符:找不到 data.studyPlanV2Detail.planSubGroups')
  process.exit(1)
}

const DIFF = { EASY: '简单', MEDIUM: '中等', HARD: '困难' }
const lines = [
  '# LeetCode 热题 100',
  '',
  '数据来源:力扣官方学习计划 `top-100-liked`(17 个分组,100 题)。',
  '校准方式见 docs/leetcode-hot100.md(一条 curl + `npm run hot100:sync` 可用官方数据覆盖本文件)。',
  '格式:每个 `## ` 是一个分组;表格列固定为 `题号 | 标题 | slug | 难度`。slug 同时用于 leetcode.com 与 leetcode.cn。',
]

let total = 0
for (const g of groups) {
  lines.push('', `## ${g.name}`, '', '| 题号 | 标题 | slug | 难度 |', '|---|---|---|---|')
  for (const q of g.questions ?? []) {
    const title = q.translatedTitle || q.title
    const diff = DIFF[q.difficulty] ?? q.difficulty
    lines.push(`| ${q.frontendQuestionId ?? q.questionFrontendId} | ${title} | ${q.titleSlug} | ${diff} |`)
    total++
  }
}

const out = path.join(process.cwd(), 'leetcode', 'hot100.md')
fs.writeFileSync(out, lines.join('\n') + '\n', 'utf8')
console.log(`已写入 ${out}:${groups.length} 个分组 / ${total} 题`)
console.log('接着跑 npm test 校验结构(应为 17 组 100 题)')

// 报告库存与解读状态:读 reports/index.md,状态由文件推导(docs/10-基模报告流程.md 第七节)
//   ✅ reports/<公司>/<报告>.md 已发布   🚧 存在 _<报告>.md 草稿   ⬜ 只有原件或线索
// 用法:npm run reports:status        有告警(已发布未登记 / 公司不一致 / PDF 未登记)时退出码为 1
// lib/reports.ts 是同一套解析规则的 TS 版本;tests/reports.test.ts 钉住两边不漂移
import fs from 'node:fs'
import path from 'node:path'

const REPORTS = path.join(process.cwd(), 'reports')
const PAPERS = path.join(process.cwd(), 'papers')

const visible = (name) => !name.startsWith('.') && !name.startsWith('_')

function subdirs(root) {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((e) => e.isDirectory() && visible(e.name))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 与 lib/reports.ts 的 parseReportIndex 同规则:`## 方向` 分组,三列表格,行序即顺序 */
export function parseIndex(root = REPORTS) {
  const file = path.join(root, 'index.md')
  if (!fs.existsSync(file)) throw new Error('reports/index.md 不存在')
  const topics = []
  const seenTopics = new Set()
  const seenEntries = new Set()
  let current = null
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
  for (const [i, line] of lines.entries()) {
    const at = `index.md 第 ${i + 1} 行`
    const heading = line.match(/^##\s+(.+?)\s*$/)
    if (heading) {
      if (seenTopics.has(heading[1])) throw new Error(`${at}: 主题「${heading[1]}」重复`)
      seenTopics.add(heading[1])
      current = { title: heading[1], entries: [] }
      topics.push(current)
      continue
    }
    if (!line.startsWith('|')) continue
    const cells = line.split('|').slice(1, -1).map((c) => c.trim())
    if (cells[0] === '报告' || cells.every((c) => /^:?-+:?$/.test(c))) continue
    if (!current) throw new Error(`${at}: 表格行出现在任何主题之前`)
    if (cells.length < 3) throw new Error(`${at}: 列数不足,需要 报告|公司|一句话 三列`)
    const [slug, company, summary] = cells
    if (!slug || !company) throw new Error(`${at}: 报告或公司为空`)
    if (!visible(slug) || !visible(company)) throw new Error(`${at}: 不能登记以 _ 或 . 开头的草稿或隐藏名`)
    const key = `${company}/${slug}`
    if (seenEntries.has(key)) throw new Error(`${at}: ${key} 重复登记`)
    seenEntries.add(key)
    current.entries.push({ slug, company, summary })
  }
  return topics
}

/** 推导状态并收集告警;warnings 的措辞与 lib/reports.ts 的 checkReportIndex 完全一致 */
export function collect(root = REPORTS, papers = PAPERS) {
  const published = new Set()
  const drafts = new Set()
  for (const company of subdirs(root)) {
    for (const name of fs.readdirSync(path.join(root, company))) {
      if (!name.endsWith('.md')) continue
      if (visible(name)) published.add(`${company}/${name.slice(0, -3)}`)
      else if (name.startsWith('_')) drafts.add(`${company}/${name.slice(1, -3)}`)
    }
  }

  const registered = new Set()
  const bySlug = new Map()
  const topics = parseIndex(root).map((topic) => ({
    title: topic.title,
    rows: topic.entries.map((entry) => {
      const key = `${entry.company}/${entry.slug}`
      registered.add(key)
      bySlug.set(entry.slug, [...(bySlug.get(entry.slug) ?? []), entry.company])
      const status = published.has(key) ? '✅' : drafts.has(key) ? '🚧' : '⬜'
      return { ...entry, status }
    }),
  }))

  const warnings = []
  for (const key of [...published].sort((a, b) => a.localeCompare(b, 'zh-CN'))) {
    if (registered.has(key)) continue
    const [company, slug] = key.split('/')
    const elsewhere = bySlug.get(slug)
    warnings.push(
      elsewhere
        ? `${company}/${slug}.md 在 index.md 登记的公司是 ${elsewhere.join('、')},与目录不一致`
        : `${company}/${slug}.md 未在 index.md 登记`,
    )
  }

  const unregisteredPdfs = []
  for (const company of subdirs(papers)) {
    for (const name of fs.readdirSync(path.join(papers, company))) {
      if (!name.endsWith('.pdf')) continue
      const key = `${company}/${name.slice(0, -4)}`
      if (!registered.has(key)) unregisteredPdfs.push(`papers/${key}.pdf`)
    }
  }

  return { topics, warnings, unregisteredPdfs }
}

function main() {
  const { topics, warnings, unregisteredPdfs } = collect()
  const total = { '✅': 0, '🚧': 0, '⬜': 0 }
  for (const topic of topics) {
    const count = { '✅': 0, '🚧': 0, '⬜': 0 }
    for (const row of topic.rows) count[row.status]++
    console.log(`\n## ${topic.title}  ✅${count['✅']} 🚧${count['🚧']} ⬜${count['⬜']}`)
    for (const row of topic.rows) {
      total[row.status]++
      console.log(`${row.status} ${row.company}/${row.slug}  ${row.summary}`)
    }
  }
  console.log(`\n合计 ✅${total['✅']} 🚧${total['🚧']} ⬜${total['⬜']}`)

  const problems = [...warnings, ...unregisteredPdfs.map((f) => `${f} 有原件但索引没有登记`)]
  if (problems.length > 0) {
    console.log('\n告警:')
    for (const p of problems) console.log(`- ${p}`)
    process.exit(1)
  }
}

if (import.meta.url === `file://${process.argv[1]}`) main()

// 日常研读库存与解读状态:读 readings/index.md,状态由文件推导(docs/10-材料解读流程.md 第八节)
//   ✅ readings/<方向>/<材料>.md 已发布   🚧 存在 _<材料>.md 草稿   ⬜ 只有原件或线索
// 用法:npm run readings:status      有告警时退出码为 1
// 与报告解读的差别只有一处:**目录是方向**(等于索引的 `## ` 标题),机构只是第二列的元数据。
// lib/readings.ts 是同一套解析规则的 TS 版本;tests/readings.test.ts 钉住两边不漂移
import fs from 'node:fs'
import path from 'node:path'

const READINGS = path.join(process.cwd(), 'readings')
const SRC = path.join(READINGS, '_src')

const visible = (name) => !name.startsWith('.') && !name.startsWith('_')

function subdirs(root) {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((e) => e.isDirectory() && visible(e.name))
    .map((e) => e.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 与 lib/readings.ts 的 parseReadingIndex 同规则:`## 方向` 分组,三列表格,行序即顺序 */
export function parseIndex(root = READINGS) {
  const file = path.join(root, 'index.md')
  if (!fs.existsSync(file)) throw new Error('readings/index.md 不存在')
  const topics = []
  const seenTopics = new Set()
  const seenSlugs = new Set()
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
    if (cells[0] === '材料' || cells.every((c) => /^:?-+:?$/.test(c))) continue
    if (!current) throw new Error(`${at}: 表格行出现在任何主题之前`)
    if (cells.length < 3) throw new Error(`${at}: 列数不足,需要 材料|机构|一句话 三列`)
    const [slug, org, summary] = cells
    if (!slug || !org) throw new Error(`${at}: 材料或机构为空`)
    if (!visible(slug) || !visible(org)) throw new Error(`${at}: 不能登记以 _ 或 . 开头的名字`)
    if (seenSlugs.has(slug)) throw new Error(`${at}: ${slug} 重复登记`)
    seenSlugs.add(slug)
    current.entries.push({ slug, org, summary })
  }
  return topics
}

/** 推导状态并收集告警;措辞与 lib/readings.ts 的 checkReadingIndex 保持一致 */
export function collect(root = READINGS, src = SRC) {
  const published = new Set()
  const drafts = new Set()
  for (const topic of subdirs(root)) {
    for (const name of fs.readdirSync(path.join(root, topic))) {
      if (!name.endsWith('.md')) continue
      const key = `${topic}/${name.replace(/^_/, '').slice(0, -3)}`
      ;(name.startsWith('_') ? drafts : published).add(key)
    }
  }

  const registered = new Map()
  const topics = parseIndex(root).map((topic) => ({
    title: topic.title,
    rows: topic.entries.map((entry) => {
      const key = `${topic.title}/${entry.slug}`
      registered.set(entry.slug, topic.title)
      const status = published.has(key) ? '✅' : drafts.has(key) ? '🚧' : '⬜'
      return { ...entry, status }
    }),
  }))

  const warnings = []
  const titles = new Set(topics.map((t) => t.title))
  for (const dir of subdirs(root)) {
    if (!titles.has(dir)) {
      warnings.push(`readings/${dir}/ 是目录,但 index.md 里没有同名的 \`## ${dir}\``)
    }
  }
  for (const key of [...published].sort((a, b) => a.localeCompare(b, 'zh-CN'))) {
    const [dir, slug] = key.split('/')
    const at = registered.get(slug)
    if (at === dir) continue
    warnings.push(
      at
        ? `${dir}/${slug}.md 在 index.md 登记在方向「${at}」下,与目录不一致`
        : `${dir}/${slug}.md 未在 index.md 登记`,
    )
  }

  const releaseDateOf = (topic, slug) => {
    const file = path.join(root, topic, `${slug}.md`)
    if (!fs.existsSync(file)) return null
    for (const line of fs.readFileSync(file, 'utf8').split(/\r?\n/)) {
      const match = line.trim().match(/^<!-- release-date: (\d{4}-\d{2}-\d{2}) -->$/)
      if (match) return match[1]
    }
    return null
  }
  for (const topic of topics) {
    let previous = null
    for (const row of topic.rows) {
      if (row.status !== '✅') continue
      const releaseDate = releaseDateOf(topic.title, row.slug)
      if (!releaseDate) continue
      if (previous) {
        const dateOrder = releaseDate.localeCompare(previous.releaseDate)
        if (dateOrder > 0) {
          warnings.push(
            `${topic.title}: ${row.slug} (${releaseDate}) 排在 ${previous.slug} (${previous.releaseDate}) 后面,已发布卡片必须按首发日从新到旧`,
          )
        } else if (dateOrder === 0 && row.slug < previous.slug) {
          warnings.push(
            `${topic.title}: ${row.slug} 与 ${previous.slug} 同日 ${releaseDate},同日必须按 slug 升序`,
          )
        }
      }
      previous = { slug: row.slug, releaseDate }
    }
  }

  // 原件在 _src/<方向>/ 下,同样要有索引行;网页与转录文本原件没有文件,靠索引登记,不在这里查
  const unregisteredSources = []
  for (const topic of subdirs(src)) {
    for (const name of fs.readdirSync(path.join(src, topic))) {
      const slug = name.replace(/\.(pdf|srt|vtt|md)$/, '')
      if (slug === name) continue
      if (registered.get(slug) !== topic) unregisteredSources.push(`readings/_src/${topic}/${name}`)
    }
  }

  return { topics, warnings, unregisteredSources }
}

function main() {
  const { topics, warnings, unregisteredSources } = collect()
  const total = { '✅': 0, '🚧': 0, '⬜': 0 }
  for (const topic of topics) {
    const count = { '✅': 0, '🚧': 0, '⬜': 0 }
    for (const row of topic.rows) count[row.status]++
    console.log(`\n## ${topic.title}  ✅${count['✅']} 🚧${count['🚧']} ⬜${count['⬜']}`)
    for (const row of topic.rows) {
      total[row.status]++
      console.log(`${row.status} ${row.slug}  [${row.org}]  ${row.summary}`)
    }
  }
  console.log(`\n合计 ✅${total['✅']} 🚧${total['🚧']} ⬜${total['⬜']}`)

  const problems = [...warnings, ...unregisteredSources.map((f) => `${f} 有原件但索引没有登记`)]
  if (problems.length > 0) {
    console.log('\n告警:')
    for (const p of problems) console.log(`- ${p}`)
    process.exit(1)
  }
}

if (import.meta.url === `file://${process.argv[1]}`) main()

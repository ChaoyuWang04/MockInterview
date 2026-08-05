// 打印题库主题树:npm run topics
// 用途:批量导入截图前的去重索引——按 分类 → topic 第一段 → 叶子 一屏看全
import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

const root = path.join(process.cwd(), 'questions')
const visible = (n) => !n.startsWith('.') && !n.startsWith('_')

const cats = fs
  .readdirSync(root, { withFileTypes: true })
  .filter((d) => d.isDirectory() && visible(d.name))
  .map((d) => d.name)
  .sort((a, b) => a.localeCompare(b, 'zh-CN'))

let total = 0
for (const cat of cats) {
  const dir = path.join(root, cat)
  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith('.md') && visible(f))
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
  console.log(`\n${cat} (${files.length} 题)`)
  const groups = new Map()
  for (const file of files) {
    total++
    let data = {}
    try {
      data = matter.read(path.join(dir, file)).data
    } catch {
      // 解析失败的文件也要出现在索引里,避免漏判
    }
    const segs = String(data.topic ?? '').split('/').filter(Boolean)
    const head = segs[0] ?? '(未分层)'
    const leaf = segs.slice(1).join('/') || '—'
    if (!groups.has(head)) groups.set(head, [])
    groups.get(head).push({
      leaf,
      summary: data.summary ?? '(无 summary)',
      file,
      mastered: data.mastered === true,
    })
  }
  for (const [head, items] of [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0], 'zh-CN'))) {
    console.log(`  ${head}`)
    for (const it of items) console.log(`    · ${it.leaf} — ${it.summary} [${it.file}]${it.mastered ? ' ✓' : ''}`)
  }
}
console.log(`\n共 ${cats.length} 分类 · ${total} 题`)

// 知识库写作契约的自动检查(docs/05-知识库写作契约.md §九 自检表)
// 用法:npm run kb:lint        有硬性违规时退出码为 1
import fs from 'node:fs'
import path from 'node:path'

const ROOT = path.join(process.cwd(), 'knowledge')

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    if (e.name.startsWith('.') || e.name.startsWith('_')) return []
    const p = path.join(dir, e.name)
    return e.isDirectory() ? walk(p) : e.name.endsWith('.md') ? [p] : []
  })
}

/** 数一张 mermaid 图的节点:按箭头切段,每段取标签外的首个标识符 */
function mermaidNodes(block) {
  const ids = new Set()
  for (const raw of block.split('\n')) {
    const line = raw.trim()
    if (!line || /^(flowchart|graph|subgraph|end|%%|style|classDef)/.test(line)) continue
    for (const seg of line.split(/--+>?\|?[^|]*?\|?|==+>/)) {
      const m = seg.trim().match(/^([A-Za-z_][A-Za-z_0-9]*)/)
      if (m) ids.add(m[1])
    }
  }
  return ids.size
}

const files = walk(ROOT)
const titles = new Set(files.map((f) => path.basename(f, '.md').replace(/^\d+-/, '')))
const errors = []   // 已成文文章的违规 → 阻断
const warns = []    // 提醒 → 不阻断
const legacyIssues = [] // 旧稿的违规 → 不阻断,作为重写工作清单
let placeholders = 0
let legacy = 0
let done = 0
let keypoints = 0

for (const file of files) {
  const rel = path.relative(process.cwd(), file)
  const text = fs.readFileSync(file, 'utf8')
  const lines = text.split('\n')
  const isPlaceholder = text.includes('🚧 占位')
  const isLegacy = text.includes('⚠️ 旧版')
  const isKey = text.includes('🔴 重点考点')
  if (isPlaceholder) placeholders++
  else if (isLegacy) legacy++
  else done++
  if (isKey) keypoints++

  // 旧稿是已知不合规的存量,违规单列成重写清单,不阻断
  const E = (msg) => (isLegacy ? legacyIssues : errors).push(`${rel}: ${msg}`)
  const W = (msg) => warns.push(`${rel}: ${msg}`)

  if (!lines[0].startsWith('# ')) E('首行不是 H1 标题')
  if (text.startsWith('---\n')) E('不应有 frontmatter')

  // 占位稿只查到这里,其余规则对成文/旧稿生效
  if (isPlaceholder) continue

  // 公式:块级 $$ 必须独占一行
  const inline = lines.filter((l) => /\$\$.+\$\$/.test(l)).length
  if (inline) E(`${inline} 处块级公式写在同一行($$公式$$),必须让 $$ 独占一行`)

  // 标题层级
  const h4 = lines.filter((l) => l.startsWith('#### ')).length
  if (h4) E(`${h4} 个四级标题,契约要求最深到三级`)

  // mermaid 规模
  for (const m of text.matchAll(/```mermaid\n([\s\S]*?)```/g)) {
    const n = mermaidNodes(m[1])
    if (n > 8) E(`mermaid 图有 ${n} 个节点,超出上限 8(拆图或改用表格)`)
    if (/subgraph[\s\S]*subgraph/.test(m[1])) E('mermaid 出现嵌套 subgraph')
  }

  // 代码块长度
  let inBlock = false
  let count = 0
  let maxCode = 0
  for (const l of lines) {
    if (/^```/.test(l)) {
      if (inBlock) { maxCode = Math.max(maxCode, count); inBlock = false } else { inBlock = true; count = 0 }
      continue
    }
    if (inBlock) count++
  }
  if (maxCode > 25) E(`最长代码块 ${maxCode} 行,超出上限 25`)

  // 收尾
  if (!text.includes('## 相关文献')) E('缺少「## 相关文献」')
  if (isKey && !text.includes('面试考点串联')) E('带 🔴 标记但缺少「面试考点串联」')

  // 跨篇引用的目标必须存在(只查「见/引/参见 XX 篇」这种全名写法)
  for (const m of text.matchAll(/(?:见|引|参见)\s*[「『]?([A-Za-z0-9一-龥]{3,20}?)[」』]?\s*篇/g)) {
    const name = m[1]
    if (titles.has(name)) continue
    // 允许指向开源解读模块与本章总览这类非文章目标
    if (/开源解读|本章|本篇|各分|子领域/.test(name)) continue
    if ([...titles].some((t) => t.endsWith(name) || name.endsWith(t))) {
      W(`跨篇引用「${name} 篇」用了简写,建议写全名(便于 grep 审计)`)
    } else {
      W(`跨篇引用「${name} 篇」找不到同名文章(可能是句子片段的误报,请人工确认)`)
    }
  }

  // 篇幅
  const n = lines.length
  if (!isLegacy && (n < 100 || n > 300)) W(`篇幅 ${n} 行,契约建议 120–260`)
}

// 文章名全局唯一(00-总览 是各章 hub 页,豁免)
const seen = new Map()
for (const f of files) {
  const t = path.basename(f, '.md').replace(/^\d+-/, '')
  if (t === '总览') continue
  if (seen.has(t)) errors.push(`文章名重复:${t}(${seen.get(t)} 与 ${path.relative(process.cwd(), f)})`)
  seen.set(t, path.relative(process.cwd(), f))
}

console.log(`知识库 ${files.length} 篇:成文 ${done} · 旧稿 ${legacy} · 占位 ${placeholders} · 重点考点 ${keypoints}\n`)
if (legacyIssues.length) {
  console.log(`📋 ${legacyIssues.length} 条旧稿违规(不阻断,重写这些篇时一并修):`)
  for (const l of legacyIssues) console.log('   ' + l)
  console.log()
}
if (warns.length) {
  console.log(`⚠️  ${warns.length} 条提醒(不阻断):`)
  for (const w of warns) console.log('   ' + w)
  console.log()
}
if (errors.length) {
  console.log(`❌ ${errors.length} 条契约违规:`)
  for (const e of errors) console.log('   ' + e)
  process.exit(1)
}
console.log('✅ 契约检查通过')

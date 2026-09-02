// 打印模拟面试的候选池规模:npm run interview:index
// 用途:知识库每写完一篇,重跑一次就能看到池子变大;也是 M0 门禁的可见指标。
//
// ⚠️ 真逻辑在 lib/interview/corpus.ts,这里是它的 .mjs 镜像(脚本跑不了 .ts)。
//    两边由 tests/interview.test.ts 的「防漂移」用例钉住,改一边必须改另一边。
import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'

const EXCLUDED_CATEGORIES = ['手撕代码']
const visible = (n) => !n.startsWith('.') && !n.startsWith('_')

function walk(dir, acc = []) {
  if (!fs.existsSync(dir)) return acc
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!visible(e.name)) continue
    const p = path.join(dir, e.name)
    if (e.isDirectory()) walk(p, acc)
    else if (e.name.endsWith('.md')) acc.push(p)
  }
  return acc
}

export function extractExamPoints(body) {
  const start = body.search(/^##\s+.*面试考点串联/m)
  if (start < 0) return []
  const section = body.slice(start).split(/^## /m)[1] ?? ''
  const out = []
  for (const raw of section.split('\n')) {
    const line = raw.trim()
    if (line.startsWith('|')) {
      const cells = line.split('|').slice(1, -1).map((c) => c.trim())
      if (cells.length < 2) continue
      if (/^-+$/.test(cells[0].replace(/[:\s]/g, ''))) continue
      if (/^(高频)?问法$/.test(cells[0])) continue
      if (!cells[0]) continue
      out.push({ ask: cells[0], where: cells[1] })
      continue
    }
    const m = line.match(/^\d+\.\s*(.+?)\s*→\s*[「『"]?(.*?)[」』"]?\s*$/)
    if (m && m[1]) out.push({ ask: m[1], where: m[2] || '' })
  }
  return out
}

export function collect(cwd = process.cwd()) {
  const kRoot = path.join(cwd, 'knowledge')
  const qRoot = path.join(cwd, 'questions')

  const articles = []
  for (const file of walk(kRoot)) {
    if (path.basename(file) === '00-总览.md') continue // hub 页,不参与匹配
    const body = fs.readFileSync(file, 'utf8')
    const segments = path.relative(kRoot, file).split(path.sep)
    const state = body.includes('🚧 占位')
      ? 'placeholder'
      : body.includes('⚠️ 旧版')
        ? 'legacy'
        : 'ready'
    const examPoints = extractExamPoints(body)
    articles.push({
      title: path.basename(file, '.md').replace(/^\d+-/, ''),
      chapter: segments[0] ?? '',
      state,
      keypoint: body.includes('🔴 重点考点'),
      examPoints,
      usableAsSource: state !== 'placeholder' && examPoints.length > 0,
    })
  }
  const byTitle = new Map(articles.map((a) => [a.title, a]))

  const questions = []
  const excluded = {}
  let unmatched = 0
  const candidates = []
  for (const file of walk(qRoot)) {
    const category = path.relative(qRoot, file).split(path.sep)[0]
    const { data } = matter(fs.readFileSync(file, 'utf8'))
    const topic = String(data.topic ?? '')
    const article = topic.split('/')[0]?.trim() ?? ''
    questions.push({ category, article })
    if (EXCLUDED_CATEGORIES.includes(category)) {
      excluded[category] = (excluded[category] ?? 0) + 1
      continue
    }
    const hit = byTitle.get(article)
    if (!hit) {
      unmatched++
      continue
    }
    candidates.push({ kind: 'question', chapter: hit.chapter, article: hit.title })
  }

  const covered = new Set(candidates.map((c) => c.article))
  for (const a of articles) {
    if (!a.usableAsSource || covered.has(a.title)) continue
    for (let i = 0; i < a.examPoints.length; i++) {
      candidates.push({ kind: 'exam-point', chapter: a.chapter, article: a.title })
    }
  }

  const byChapter = {}
  for (const c of candidates) byChapter[c.chapter] = (byChapter[c.chapter] ?? 0) + 1

  const stats = {
    题目总数: questions.length,
    参与出题的题目: candidates.filter((c) => c.kind === 'question').length,
    排除的分类: excluded,
    匹配不到文章的题目: unmatched,
    文章总数: articles.length,
    可出题文章: articles.filter((a) => a.usableAsSource).length,
    考点行总数: articles.reduce((s, a) => s + a.examPoints.length, 0),
    候选池大小: candidates.length,
    按章节: byChapter,
  }
  return { stats, articles }
}

/**
 * 生成全域热词表给云端 STT 用。
 * 只喂云端:本地 Whisper 解码上下文只有 448 token,灌全域会溢出并从前面截断,反而更差。
 */
function writeHotwords(articles) {
  const terms = new Set()
  for (const a of articles) terms.add(a.title)
  for (const file of walk(path.join(process.cwd(), 'questions'))) {
    const t = fs.readFileSync(file, 'utf8')
    for (const m of t.match(/\b[A-Z][A-Za-z0-9]{2,20}\b/g) ?? []) terms.add(m)
  }
  for (const file of walk(path.join(process.cwd(), 'knowledge'))) {
    const t = fs.readFileSync(file, 'utf8')
    for (const m of t.match(/\b[A-Z][A-Za-z0-9]{2,20}\b/g) ?? []) terms.add(m)
  }
  const out = [...terms].filter((x) => x.length > 2).sort()
  const dir = path.join(process.cwd(), 'interview')
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'hotwords.txt'), out.join('\n') + '\n', 'utf8')
  return out.length
}

function main() {
  const { stats: s, articles } = collect()
  if (process.argv.includes('--json')) {
    console.log(JSON.stringify(s))
    return
  }

  const byState = { ready: 0, legacy: 0, placeholder: 0 }
  for (const a of articles) byState[a.state]++

  console.log('\n模拟面试候选池')
  console.log('─'.repeat(52))
  console.log(`题库    ${s.题目总数} 道 → 参与出题 ${s.参与出题的题目} 道`)
  for (const [cat, n] of Object.entries(s.排除的分类)) {
    console.log(`        排除 ${cat} ${n} 道(口头面试念不了代码题)`)
  }
  if (s.匹配不到文章的题目) {
    console.log(`        ⚠️  ${s.匹配不到文章的题目} 道 topic 匹配不到文章,已跳过`)
  }
  console.log(
    `知识库  ${s.文章总数} 篇(成文 ${byState.ready} · 旧稿 ${byState.legacy} · 占位 ${byState.placeholder})`,
  )
  console.log(`        可出题 ${s.可出题文章} 篇 · 考点行 ${s.考点行总数} 条`)
  console.log('─'.repeat(52))
  console.log(`候选池  ${s.候选池大小} 个可问点`)
  for (const [ch, n] of Object.entries(s.按章节).sort()) {
    console.log(`        ${ch.padEnd(16)} ${n}`)
  }
  const n = writeHotwords(articles)
  console.log(`热词    ${n} 个 → interview/hotwords.txt(只给云端 STT 用)`)
  console.log(
    `\n提示:写完一篇知识库文章(删掉 🚧 占位 + 补考点表)重跑本命令,池子和热词都会自动变大。\n`,
  )
}

if (import.meta.url === `file://${process.argv[1]}`) main()

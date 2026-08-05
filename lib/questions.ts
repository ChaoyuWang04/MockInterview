import fs from 'node:fs'
import path from 'node:path'
import matter from 'gray-matter'
import { SECTION_NAMES, type SectionName, type Question, type Stats } from './types'

export function questionsRoot(): string {
  return path.join(process.cwd(), 'questions')
}

function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

export function listCategories(root = questionsRoot()): string[] {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((d) => d.isDirectory() && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

export function listQuestionFiles(category: string, root = questionsRoot()): string[] {
  const dir = path.join(root, category)
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => d.isFile() && d.name.endsWith('.md') && isVisible(d.name))
    .map((d) => d.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 白名单校验:category/file 必须真实存在于扫描结果中,天然杜绝路径穿越 */
export function isValidRef(category: string, file: string, root = questionsRoot()): boolean {
  return listCategories(root).includes(category) && listQuestionFiles(category, root).includes(file)
}

interface FenceState {
  char: string
  len: number
}

function fenceOpen(line: string): FenceState | null {
  const m = line.match(/^\s*(`{3,}|~{3,})/)
  return m ? { char: m[1][0], len: m[1].length } : null
}

function fenceClose(line: string, fence: FenceState): boolean {
  const m = line.match(/^\s*(`{3,}|~{3,})\s*$/)
  return !!m && m[1][0] === fence.char && m[1].length >= fence.len
}

function reservedHeading(line: string): SectionName | null {
  const m = line.match(/^##\s+(.+?)\s*$/)
  if (m && (SECTION_NAMES as readonly string[]).includes(m[1])) return m[1] as SectionName
  return null
}

/** 按保留的 ## 分区标题切分正文(围栏代码块内的 ## 不算) */
export function splitSections(body: string): Partial<Record<SectionName, string>> {
  const acc: Partial<Record<SectionName, string[]>> = {}
  let fence: FenceState | null = null
  let current: SectionName | null = null
  for (const line of body.split('\n')) {
    if (fence) {
      if (fenceClose(line, fence)) fence = null
      if (current) acc[current]!.push(line)
      continue
    }
    const open = fenceOpen(line)
    if (open) {
      fence = open
      if (current) acc[current]!.push(line)
      continue
    }
    const heading = reservedHeading(line)
    if (heading) {
      current = heading
      acc[current] = []
      continue
    }
    if (current) acc[current]!.push(line)
  }
  const out: Partial<Record<SectionName, string>> = {}
  for (const name of SECTION_NAMES) {
    const lines = acc[name]
    if (lines !== undefined) out[name] = lines.join('\n').trim()
  }
  return out
}

export function loadQuestion(category: string, file: string, root = questionsRoot()): Question {
  const base: Question = { category, file, meta: { tags: [], mastered: false }, sections: {} }
  let raw: string
  try {
    raw = fs.readFileSync(path.join(root, category, file), 'utf8')
  } catch (e) {
    return { ...base, error: `读取文件失败:${(e as Error).message}` }
  }
  let data: Record<string, unknown>
  let content: string
  try {
    const parsed = matter(raw)
    data = parsed.data
    content = parsed.content
  } catch (e) {
    return { ...base, error: `frontmatter 解析失败:${(e as Error).message}` }
  }
  const q: Question = {
    category,
    file,
    meta: {
      difficulty: typeof data.difficulty === 'string' ? data.difficulty : undefined,
      tags: Array.isArray(data.tags) ? data.tags.map(String) : [],
      company: typeof data.company === 'string' ? data.company : undefined,
      mastered: data.mastered === true,
    },
    sections: splitSections(content),
  }
  if (!q.sections['题目']) q.error = '缺少必填分区「## 题目」'
  else if (!q.sections['答案']) q.error = '缺少必填分区「## 答案」'
  return q
}

export function loadCategory(category: string, root = questionsRoot()): Question[] {
  return listQuestionFiles(category, root).map((f) => loadQuestion(category, f, root))
}

export function getStats(root = questionsRoot()): Stats {
  const categories = listCategories(root).map((name) => {
    const qs = loadCategory(name, root)
    return { name, total: qs.length, mastered: qs.filter((q) => !q.error && q.meta.mastered).length }
  })
  return {
    total: categories.reduce((s, c) => s + c.total, 0),
    mastered: categories.reduce((s, c) => s + c.mastered, 0),
    categories,
  }
}

function atomicWrite(filePath: string, content: string): void {
  const tmp = `${filePath}.${process.pid}.tmp`
  fs.writeFileSync(tmp, content, 'utf8')
  fs.renameSync(tmp, filePath)
}

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---/
// 三组:键与空白 / 现值 / 行尾(空白+注释),定点替换保注释
const MASTERED_LINE_RE = /^(mastered:[ \t]*)([^\s#]*)([ \t]*(?:#[^\n]*)?)$/m

/** 定点改写 frontmatter 的 mastered 行,绝不整体重新序列化 */
export function setMastered(category: string, file: string, mastered: boolean, root = questionsRoot()): void {
  const filePath = path.join(root, category, file)
  const raw = fs.readFileSync(filePath, 'utf8')
  const value = String(mastered)
  const m = raw.match(FRONTMATTER_RE)
  let next: string
  if (m) {
    const fm = m[1]
    const updated = MASTERED_LINE_RE.test(fm)
      ? fm.replace(MASTERED_LINE_RE, `$1${value}$3`)
      : `${fm}\nmastered: ${value}`
    next = `---\n${updated}\n---` + raw.slice(m[0].length)
  } else {
    next = `---\nmastered: ${value}\n---\n\n${raw}`
  }
  atomicWrite(filePath, next)
}

/** 替换 ## Note 分区正文;无该分区则追加;空内容清空正文保留标题 */
export function saveNote(category: string, file: string, note: string, root = questionsRoot()): void {
  const filePath = path.join(root, category, file)
  const raw = fs.readFileSync(filePath, 'utf8')
  const trimmed = note.trim()
  const lines = raw.split('\n')

  // 跳过 frontmatter
  let i = 0
  if (lines[0]?.trim() === '---') {
    i = 1
    while (i < lines.length && lines[i].trim() !== '---') i++
    i++
  }

  // 围栏感知扫描,定位 ## Note 标题行与分区结束(下一个保留 ## 或 EOF)
  let fence: FenceState | null = null
  let noteStart = -1
  let noteEnd = lines.length
  for (; i < lines.length; i++) {
    const line = lines[i]
    if (fence) {
      if (fenceClose(line, fence)) fence = null
      continue
    }
    const open = fenceOpen(line)
    if (open) {
      fence = open
      continue
    }
    const heading = reservedHeading(line)
    if (!heading) continue
    if (noteStart >= 0) {
      noteEnd = i
      break
    }
    if (heading === 'Note') noteStart = i
  }

  let next: string
  if (noteStart >= 0) {
    const before = lines.slice(0, noteStart + 1).join('\n')
    const after = lines.slice(noteEnd).join('\n')
    const body = trimmed ? `\n\n${trimmed}\n` : '\n'
    next = after ? `${before}${body}\n${after}` : `${before}${body}`
  } else {
    if (!trimmed) return // 没有分区且内容为空:无事可做
    const base = raw.endsWith('\n') ? raw : `${raw}\n`
    next = `${base}\n## Note\n\n${trimmed}\n`
  }
  atomicWrite(filePath, next)
}

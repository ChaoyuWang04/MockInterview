import fs from 'node:fs'
import path from 'node:path'

export interface LcProblem {
  id: string
  title: string
  slug: string
  difficulty: string
}

export interface LcGroup {
  name: string
  problems: LcProblem[]
}

export function leetcodeRoot(): string {
  return path.join(process.cwd(), 'leetcode')
}

function notesDir(root = leetcodeRoot()): string {
  return path.join(root, 'notes')
}

/** 解析 hot100.md:`## 分组` + 四列表格(题号|标题|slug|难度) */
export function listHot100(root = leetcodeRoot()): LcGroup[] {
  const file = path.join(root, 'hot100.md')
  if (!fs.existsSync(file)) return []
  const groups: LcGroup[] = []
  let current: LcGroup | null = null
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    const heading = line.match(/^##\s+(.+?)\s*$/)
    if (heading) {
      current = { name: heading[1], problems: [] }
      groups.push(current)
      continue
    }
    if (!current || !line.startsWith('|')) continue
    const cells = line.split('|').slice(1, -1).map((c) => c.trim())
    if (cells.length < 4) continue
    const [id, title, slug, difficulty] = cells
    if (!/^\d+$/.test(id)) continue // 跳过表头与分隔行
    current.problems.push({ id, title, slug, difficulty })
  }
  return groups.filter((g) => g.problems.length > 0)
}

export function isValidSlug(slug: string, root = leetcodeRoot()): boolean {
  return listHot100(root).some((g) => g.problems.some((p) => p.slug === slug))
}

export function getNote(slug: string, root = leetcodeRoot()): string {
  const file = path.join(notesDir(root), `${slug}.md`)
  if (!fs.existsSync(file)) return ''
  return fs.readFileSync(file, 'utf8')
}

/** 全部已存在的笔记,页面一次性加载(每篇都很小) */
export function getAllNotes(root = leetcodeRoot()): Record<string, string> {
  const dir = notesDir(root)
  if (!fs.existsSync(dir)) return {}
  const out: Record<string, string> = {}
  for (const f of fs.readdirSync(dir)) {
    if (!f.endsWith('.md') || f.startsWith('.')) continue
    out[f.slice(0, -3)] = fs.readFileSync(path.join(dir, f), 'utf8')
  }
  return out
}

/** 整文件覆盖写回(笔记文件里没有别的内容需要保护);内容为空则删除文件 */
export function saveNote(slug: string, note: string, root = leetcodeRoot()): void {
  const dir = notesDir(root)
  const file = path.join(dir, `${slug}.md`)
  if (!note.trim()) {
    if (fs.existsSync(file)) fs.unlinkSync(file)
    return
  }
  fs.mkdirSync(dir, { recursive: true })
  const tmp = `${file}.${process.pid}.tmp`
  fs.writeFileSync(tmp, note.endsWith('\n') ? note : `${note}\n`, 'utf8')
  fs.renameSync(tmp, file)
}

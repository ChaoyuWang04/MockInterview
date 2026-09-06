import fs from 'node:fs'
import path from 'node:path'

export interface KbArticle {
  /** 显示名 = 文件名去掉 NN- 前缀与 .md,同时也是与题目 topic 第一段匹配的键 */
  title: string
  /** 相对 knowledge/ 的路径段,如 ['01-模型结构', 'RoPE.md'] */
  segments: string[]
  /** 是否仍是占位稿(正文含 🚧 占位 标记) */
  placeholder: boolean
  /** 是否为写作契约确立之前的旧稿,待按新标准重写(正文含 ⚠️ 旧版 标记) */
  legacy: boolean
}

export interface KbFolder {
  title: string
  segments: string[]
  folders: KbFolder[]
  articles: KbArticle[]
}

export function knowledgeRoot(): string {
  return path.join(process.cwd(), 'knowledge')
}

function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

/** 去掉用于排序的 NN- 前缀:`01-模型结构` → `模型结构` */
export function stripOrder(name: string): string {
  return name.replace(/^\d+-/, '')
}

/** 目录内按原始文件名排序:NN- 前缀决定顺序,00-总览 自然排第一 */
function sortedEntries(dir: string): fs.Dirent[] {
  if (!fs.existsSync(dir)) return []
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((d) => isVisible(d.name))
    .sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
}

function buildFolder(absDir: string, segments: string[]): KbFolder {
  const folders: KbFolder[] = []
  const articles: KbArticle[] = []
  for (const entry of sortedEntries(absDir)) {
    const next = [...segments, entry.name]
    if (entry.isDirectory()) {
      folders.push(buildFolder(path.join(absDir, entry.name), next))
    } else if (entry.name.endsWith('.md')) {
      const content = fs.readFileSync(path.join(absDir, entry.name), 'utf8')
      articles.push({
        title: stripOrder(entry.name.slice(0, -3)),
        segments: next,
        placeholder: content.includes('🚧 占位'),
        legacy: content.includes('⚠️ 旧版'),
      })
    }
  }
  return {
    title: segments.length ? stripOrder(segments[segments.length - 1]) : '知识库',
    segments,
    folders,
    articles,
  }
}

/** 整棵知识库树(任意层嵌套) */
export function listKbTree(root = knowledgeRoot()): KbFolder {
  return buildFolder(root, [])
}

export function flattenArticles(folder: KbFolder): KbArticle[] {
  return [...folder.articles, ...folder.folders.flatMap(flattenArticles)]
}

export function countArticles(folder: KbFolder): number {
  return flattenArticles(folder).length
}

/**
 * 白名单式读取:逐段校验路径都真实存在于扫描结果里,天然杜绝路径穿越。
 * segments 形如 ['01-模型结构', 'RoPE.md'] 或去掉 .md 的 ['01-模型结构', 'RoPE']。
 */
export function getArticleBySegments(segments: string[], root = knowledgeRoot()): string | null {
  if (segments.length === 0) return null
  let dir = root
  for (let i = 0; i < segments.length - 1; i++) {
    const hit = sortedEntries(dir).find((e) => e.isDirectory() && e.name === segments[i])
    if (!hit) return null
    dir = path.join(dir, hit.name)
  }
  const last = segments[segments.length - 1]
  const file = sortedEntries(dir).find(
    (e) => e.isFile() && (e.name === last || e.name === `${last}.md`),
  )
  if (!file) return null
  return fs.readFileSync(path.join(dir, file.name), 'utf8')
}

/**
 * 按文章名全库查找(题目 topic 第一段 → 文章)。
 * 因为知识库的分章不再与题库分类一一对应,匹配只认文章名,不认所在文件夹。
 */
export function findArticle(title: string, root = knowledgeRoot()): KbArticle | null {
  return flattenArticles(listKbTree(root)).find((a) => a.title === title) ?? null
}

/** 文章页链接:/kb/<各段编码> */
export function articleHref(article: KbArticle): string {
  return `/kb/${article.segments.map(encodeURIComponent).join('/')}`
}

/** 题库页用:topic 第一段 → 文章链接 */
export function kbLinksFor(topics: string[], root = knowledgeRoot()): Record<string, string> {
  const all = flattenArticles(listKbTree(root))
  const out: Record<string, string> = {}
  for (const t of new Set(topics)) {
    const hit = all.find((a) => a.title === t)
    if (hit) out[t] = articleHref(hit)
  }
  return out
}

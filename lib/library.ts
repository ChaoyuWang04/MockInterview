/**
 * 两个解读库(reports 报告解读、readings 日常研读)共用的原语。
 *
 * 两边真正相同的只有三件事:文章自身的校验(一级标题 + release-date)、
 * 目录扫描与排序、索引表格的逐行解析。**一致性检查不在此列**——报告解读校验
 * 「第二列 = 目录名」,日常研读校验「方向标题 = 目录名」,是两套不同的不变量,
 * 各自实现在各自的模块里。规则见 docs/10-材料解读流程.md 第一、八节。
 */
import fs from 'node:fs'
import path from 'node:path'

export interface ArticleSummary {
  slug: string
  title: string
  releaseDate: string
}

/** 索引表的一行。badge 是第二列,含义随库而定:报告解读是公司(同时是目录名),日常研读是机构(纯元数据) */
export interface RawIndexEntry {
  slug: string
  badge: string
  summary: string
}

export interface RawIndexTopic {
  title: string
  entries: RawIndexEntry[]
}

/** 以 _ 或 . 开头的目录与文件不进网页,可用作未完成草稿;readings/_src 也靠这一条被跳过 */
export function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

export function indexPath(root: string): string {
  return path.join(root, 'index.md')
}

/** 只认整行的 release-date 注释;正文里以行内代码提到 `release-date` 的段落不会命中 */
function releaseDateLineIndexes(lines: string[]): number[] {
  return lines.flatMap((line, index) =>
    /<!--\s*release-date\b/.test(line) ? [index] : [],
  )
}

/** 该注释是排序元数据、不计入正文,渲染前按行剥掉,其余内容原样保留 */
export function stripReleaseDate(content: string): string {
  const lines = content.split(/\r?\n/)
  const [index] = releaseDateLineIndexes(lines)
  if (index === undefined) return content
  lines.splice(index, 1)
  return lines.join('\n')
}

function isValidCalendarDate(year: number, month: number, day: number): boolean {
  if (year < 1 || month < 1 || month > 12 || day < 1) return false
  const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0)
  const daysInMonth = [31, leapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  return day <= daysInMonth[month - 1]
}

export function validatePublishedArticle(
  file: string,
  content: string,
): Pick<ArticleSummary, 'title' | 'releaseDate'> {
  const titleMatch = content.match(/^#\s+(.+)$/m)
  if (!titleMatch?.[1].trim()) throw new Error(`${file}: 缺少一级标题`)

  const rawBody = content.replace(titleMatch[0], '').trim()
  if (!rawBody) throw new Error(`${file}: 正文为空`)

  const lines = content.split(/\r?\n/)
  const titleIndex = lines.findIndex((line) => /^#\s+(.+)$/.test(line))
  const releaseDateIndexes = releaseDateLineIndexes(lines)

  if (releaseDateIndexes.length === 0) throw new Error(`${file}: 缺少 release-date`)
  if (releaseDateIndexes.length > 1) throw new Error(`${file}: release-date 重复`)

  const releaseDateIndex = releaseDateIndexes[0]
  const firstNonBlankAfterTitle = lines.findIndex(
    (line, index) => index > titleIndex && line.trim() !== '',
  )
  if (releaseDateIndex !== firstNonBlankAfterTitle) {
    throw new Error(`${file}: release-date 必须是一级标题后的第一个非空行`)
  }

  const releaseDateMatch = lines[releaseDateIndex]
    .trim()
    .match(/^<!-- release-date: (\d{4})-(\d{2})-(\d{2}) -->$/)
  if (!releaseDateMatch) throw new Error(`${file}: release-date 格式错误`)

  const [, yearText, monthText, dayText] = releaseDateMatch
  if (!isValidCalendarDate(Number(yearText), Number(monthText), Number(dayText))) {
    throw new Error(`${file}: release-date 日期非法`)
  }

  const body = lines
    .filter((_, index) => index !== titleIndex && index !== releaseDateIndex)
    .join('\n')
    .trim()
  if (!body) throw new Error(`${file}: 正文为空`)

  return {
    title: titleMatch[1].trim(),
    releaseDate: `${yearText}-${monthText}-${dayText}`,
  }
}

/** 库根下的可见一级目录。报告解读拿到的是公司,日常研读拿到的是方向 */
export function listDirectories(root: string): string[] {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && isVisible(entry.name))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 某个目录下的已发布文章,按首发日从新到旧、同日按 slug 升序 */
export function listArticles(root: string, directory: string): ArticleSummary[] {
  if (!listDirectories(root).includes(directory)) return []
  const dir = path.join(root, directory)
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && entry.name.endsWith('.md') && isVisible(entry.name))
    .map((entry) => entry.name)
    .map((file) => {
      const slug = file.replace(/\.md$/, '')
      const content = fs.readFileSync(path.join(dir, file), 'utf8')
      return { slug, ...validatePublishedArticle(file, content) }
    })
    .sort((a, b) => {
      const dateOrder = b.releaseDate.localeCompare(a.releaseDate)
      if (dateOrder !== 0) return dateOrder
      if (a.slug === b.slug) return 0
      return a.slug < b.slug ? -1 : 1
    })
}

export function readArticleBody(root: string, directory: string, slug: string): string {
  return stripReleaseDate(fs.readFileSync(path.join(root, directory, `${slug}.md`), 'utf8'))
}

/**
 * 逐行解析索引表:`## 方向` 分组,表格三列。
 * 行序就是页面顺序,这里不做任何重排;已发布行的排序由各库自己的一致性检查守。
 * 状态不在索引里:是否已发布由文件是否存在决定。
 */
export function parseIndex(
  root: string,
  spec: {
    slugHeader: string
    badgeHeader: string
    label: string
    /** 判重键,**跨方向全局**生效。报告解读按 公司/报告,日常研读按材料名 */
    dedupKey: (entry: RawIndexEntry) => string
  },
): RawIndexTopic[] {
  const file = indexPath(root)
  if (!fs.existsSync(file)) {
    throw new Error(`${spec.label}/index.md 不存在:索引是页面分组与排序的唯一数据源`)
  }
  const topics: RawIndexTopic[] = []
  const seenTopics = new Set<string>()
  const seenEntries = new Set<string>()
  let current: RawIndexTopic | null = null
  const lines = fs.readFileSync(file, 'utf8').split(/\r?\n/)
  for (const [index, line] of lines.entries()) {
    const at = `index.md 第 ${index + 1} 行`
    const heading = line.match(/^##\s+(.+?)\s*$/)
    if (heading) {
      const title = heading[1]
      if (seenTopics.has(title)) throw new Error(`${at}: 主题「${title}」重复`)
      seenTopics.add(title)
      current = { title, entries: [] }
      topics.push(current)
      continue
    }
    if (!line.startsWith('|')) continue
    const cells = line.split('|').slice(1, -1).map((cell) => cell.trim())
    // 表头与分隔行
    if (cells[0] === spec.slugHeader || cells.every((cell) => /^:?-+:?$/.test(cell))) continue
    if (!current) throw new Error(`${at}: 表格行出现在任何主题之前`)
    if (cells.length < 3) {
      throw new Error(
        `${at}: 列数不足,需要 ${spec.slugHeader}|${spec.badgeHeader}|一句话 三列`,
      )
    }
    const [slug, badge, summary] = cells
    if (!slug || !badge) throw new Error(`${at}: ${spec.slugHeader}或${spec.badgeHeader}为空`)
    if (!isVisible(slug) || !isVisible(badge)) {
      throw new Error(`${at}: 不能登记以 _ 或 . 开头的草稿或隐藏名`)
    }
    const entry = { slug, badge, summary }
    const key = spec.dedupKey(entry)
    if (seenEntries.has(key)) throw new Error(`${at}: ${key} 重复登记`)
    seenEntries.add(key)
    current.entries.push(entry)
  }
  return topics
}

/** 某个方向内,已发布行必须按首发日从新到旧、同日按 slug 升序 */
export function checkTopicOrder(
  topics: RawIndexTopic[],
  publishedOf: (entry: RawIndexEntry, topicTitle: string) => ArticleSummary | undefined,
  /** 报错里怎么称呼这一行。报告解读用 公司/报告,日常研读用材料名 */
  labelOf: (entry: RawIndexEntry) => string = (entry) => entry.slug,
): string[] {
  const errors: string[] = []
  for (const topic of topics) {
    let previous: { slug: string; label: string; releaseDate: string } | null = null
    for (const entry of topic.entries) {
      const summary = publishedOf(entry, topic.title)
      if (!summary) continue
      const label = labelOf(entry)
      if (previous) {
        const dateOrder = summary.releaseDate.localeCompare(previous.releaseDate)
        if (dateOrder > 0) {
          errors.push(
            `${topic.title}: ${label} (${summary.releaseDate}) 排在 ${previous.label} (${previous.releaseDate}) 后面,已发布卡片必须按首发日从新到旧`,
          )
        } else if (dateOrder === 0 && entry.slug < previous.slug) {
          errors.push(
            `${topic.title}: ${label} 与 ${previous.label} 同日 ${summary.releaseDate},同日必须按 slug 升序`,
          )
        }
      }
      previous = { slug: entry.slug, label, releaseDate: summary.releaseDate }
    }
  }
  return errors
}

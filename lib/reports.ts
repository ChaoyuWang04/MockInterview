import fs from 'node:fs'
import path from 'node:path'

export interface ReportSummary {
  slug: string
  title: string
  releaseDate: string
}

export interface ReportArticle extends ReportSummary {
  company: string
  /** 所属方向,来自 reports/index.md;索引缺失或未登记时为 null */
  topic: string | null
  content: string
}

/** reports/index.md 的一行:报告名 = 文件名去 .md,公司 = 目录名 */
export interface ReportIndexEntry {
  slug: string
  company: string
  summary: string
}

/** reports/index.md 的一个 `## 方向`,条目按行序 */
export interface ReportIndexTopic {
  title: string
  entries: ReportIndexEntry[]
}

/** 页面卡片:索引行 + 文章自带的标题与首发日 */
export interface ReportCard extends ReportSummary, ReportIndexEntry {}

export interface ReportTopic {
  title: string
  reports: ReportCard[]
}

export function reportsRoot(): string {
  return path.join(process.cwd(), 'reports')
}

export function reportIndexPath(root = reportsRoot()): string {
  return path.join(root, 'index.md')
}

function isVisible(name: string): boolean {
  return !name.startsWith('.') && !name.startsWith('_')
}

export function listReportCompanies(root = reportsRoot()): string[] {
  if (!fs.existsSync(root)) return []
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && isVisible(entry.name))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
}

/** 只认整行的 release-date 注释;正文里以行内代码提到 `release-date` 的段落不会命中 */
function releaseDateLineIndexes(lines: string[]): number[] {
  return lines.flatMap((line, index) =>
    /<!--\s*release-date\b/.test(line) ? [index] : [],
  )
}

/** 该注释是排序元数据、不计入正文,渲染前按行剥掉,其余内容原样保留 */
function stripReleaseDate(content: string): string {
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

function validatePublishedReport(
  file: string,
  content: string,
): Pick<ReportSummary, 'title' | 'releaseDate'> {
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

export function listReports(company: string, root = reportsRoot()): ReportSummary[] {
  if (!listReportCompanies(root).includes(company)) return []
  const dir = path.join(root, company)
  return fs
    .readdirSync(dir, { withFileTypes: true })
    .filter(
      (entry) => entry.isFile() && entry.name.endsWith('.md') && isVisible(entry.name),
    )
    .map((entry) => entry.name)
    .map((file) => {
      const slug = file.replace(/\.md$/, '')
      const content = fs.readFileSync(path.join(dir, file), 'utf8')
      return { slug, ...validatePublishedReport(file, content) }
    })
    .sort((a, b) => {
      const dateOrder = b.releaseDate.localeCompare(a.releaseDate)
      if (dateOrder !== 0) return dateOrder
      if (a.slug === b.slug) return 0
      return a.slug < b.slug ? -1 : 1
    })
}

export function isValidReport(company: string, slug: string, root = reportsRoot()): boolean {
  return listReports(company, root).some((report) => report.slug === slug)
}

export function getReport(
  company: string,
  slug: string,
  root = reportsRoot(),
): ReportArticle | null {
  const summary = listReports(company, root).find((report) => report.slug === slug)
  if (!summary) return null
  return {
    ...summary,
    company,
    topic: findReportTopic(company, slug, root),
    content: stripReleaseDate(fs.readFileSync(path.join(root, company, `${slug}.md`), 'utf8')),
  }
}

/**
 * 解析 reports/index.md:`## 方向` 分组,表格三列 报告|公司|一句话。
 * 行序就是页面顺序(方向内从整体报告排到单点机制),这里不做任何重排。
 * 状态不在索引里:是否已发布由文件是否存在决定,见 listReportTopics。
 */
export function parseReportIndex(root = reportsRoot()): ReportIndexTopic[] {
  const file = reportIndexPath(root)
  if (!fs.existsSync(file)) {
    throw new Error('reports/index.md 不存在:报告索引是页面分组与排序的唯一数据源')
  }
  const topics: ReportIndexTopic[] = []
  const seenTopics = new Set<string>()
  const seenEntries = new Set<string>()
  let current: ReportIndexTopic | null = null
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
    if (cells[0] === '报告' || cells.every((cell) => /^:?-+:?$/.test(cell))) continue
    if (!current) throw new Error(`${at}: 表格行出现在任何主题之前`)
    if (cells.length < 3) throw new Error(`${at}: 列数不足,需要 报告|公司|一句话 三列`)
    const [slug, company, summary] = cells
    if (!slug || !company) throw new Error(`${at}: 报告或公司为空`)
    if (!isVisible(slug) || !isVisible(company)) {
      throw new Error(`${at}: 不能登记以 _ 或 . 开头的草稿或隐藏名`)
    }
    const key = `${company}/${slug}`
    if (seenEntries.has(key)) throw new Error(`${at}: ${key} 重复登记`)
    seenEntries.add(key)
    current.entries.push({ slug, company, summary })
  }
  return topics
}

/** 已发布文章与索引的一致性:每篇已发布文章必须在索引里恰有一行,且公司列等于目录名 */
export function checkReportIndex(root = reportsRoot()): string[] {
  const registered = new Set<string>()
  const bySlug = new Map<string, string[]>()
  for (const topic of parseReportIndex(root)) {
    for (const entry of topic.entries) {
      registered.add(`${entry.company}/${entry.slug}`)
      bySlug.set(entry.slug, [...(bySlug.get(entry.slug) ?? []), entry.company])
    }
  }
  const errors: string[] = []
  for (const company of listReportCompanies(root)) {
    for (const report of listReports(company, root)) {
      if (registered.has(`${company}/${report.slug}`)) continue
      const elsewhere = bySlug.get(report.slug)
      errors.push(
        elsewhere
          ? `${company}/${report.slug}.md 在 index.md 登记的公司是 ${elsewhere.join('、')},与目录不一致`
          : `${company}/${report.slug}.md 未在 index.md 登记`,
      )
    }
  }
  return errors
}

/**
 * 页面数据:按索引顺序返回各方向及其已发布文章;未解读条目不出现,没有已发布文章的方向也不出现。
 * 索引与文件不一致时直接抛错,和 release-date 契约一样让问题带文件名暴露。
 */
export function listReportTopics(root = reportsRoot()): ReportTopic[] {
  const errors = checkReportIndex(root)
  if (errors.length > 0) throw new Error(errors.join('\n'))
  const cache = new Map<string, ReportSummary[]>()
  const summariesOf = (company: string): ReportSummary[] => {
    if (!cache.has(company)) cache.set(company, listReports(company, root))
    return cache.get(company) ?? []
  }
  return parseReportIndex(root)
    .map((topic) => ({
      title: topic.title,
      reports: topic.entries.flatMap((entry): ReportCard[] => {
        const summary = summariesOf(entry.company).find((report) => report.slug === entry.slug)
        return summary ? [{ ...summary, ...entry }] : []
      }),
    }))
    .filter((topic) => topic.reports.length > 0)
}

/** 某篇文章所属的方向;索引不存在时返回 null,不让文章页因为索引缺失而打不开 */
export function findReportTopic(
  company: string,
  slug: string,
  root = reportsRoot(),
): string | null {
  if (!fs.existsSync(reportIndexPath(root))) return null
  for (const topic of parseReportIndex(root)) {
    if (topic.entries.some((entry) => entry.company === company && entry.slug === slug)) {
      return topic.title
    }
  }
  return null
}

export function countReports(root = reportsRoot()): number {
  return listReportCompanies(root).reduce(
    (total, company) => total + listReports(company, root).length,
    0,
  )
}

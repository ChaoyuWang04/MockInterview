import fs from 'node:fs'
import path from 'node:path'

export interface ReportSummary {
  slug: string
  title: string
  releaseDate: string
}

export interface ReportArticle extends ReportSummary {
  company: string
  content: string
}

export function reportsRoot(): string {
  return path.join(process.cwd(), 'reports')
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
    content: stripReleaseDate(fs.readFileSync(path.join(root, company, `${slug}.md`), 'utf8')),
  }
}

export function countReports(root = reportsRoot()): number {
  return listReportCompanies(root).reduce(
    (total, company) => total + listReports(company, root).length,
    0,
  )
}

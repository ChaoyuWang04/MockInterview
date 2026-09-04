import fs from 'node:fs'
import path from 'node:path'

export interface ReportSummary {
  slug: string
  title: string
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

function validatePublishedReport(file: string, content: string): string {
  const titleMatch = content.match(/^#\s+(.+)$/m)
  if (!titleMatch?.[1].trim()) throw new Error(`${file}: 缺少一级标题`)

  const body = content.replace(titleMatch[0], '').trim()
  if (!body) throw new Error(`${file}: 正文为空`)
  return titleMatch[1].trim()
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
    .sort((a, b) => a.localeCompare(b, 'zh-CN'))
    .map((file) => {
      const slug = file.replace(/\.md$/, '')
      const content = fs.readFileSync(path.join(dir, file), 'utf8')
      return { slug, title: validatePublishedReport(file, content) }
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
    content: fs.readFileSync(path.join(root, company, `${slug}.md`), 'utf8'),
  }
}

export function countReports(root = reportsRoot()): number {
  return listReportCompanies(root).reduce(
    (total, company) => total + listReports(company, root).length,
    0,
  )
}

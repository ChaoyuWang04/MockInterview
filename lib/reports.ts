/**
 * 报告解读库:必看的报告与论文,目录键是**公司**,方向只存在于 reports/index.md。
 * 共用原语见 lib/library.ts;日常研读库见 lib/readings.ts。规则见 docs/10-材料解读流程.md。
 */
import fs from 'node:fs'
import path from 'node:path'

import {
  type ArticleSummary,
  checkTopicOrder,
  indexPath,
  listArticles,
  listDirectories,
  parseIndex,
  readArticleBody,
} from './library'

export type ReportSummary = ArticleSummary

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

/** reports/index.md 的一个 `## 方向`;已发布行必须按首发日从新到旧,同日按 slug 升序 */
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

const INDEX_SPEC = {
  slugHeader: '报告',
  badgeHeader: '公司',
  label: 'reports',
  // 同一家公司下的同一份报告只能登记一次,跨方向也算重复
  dedupKey: (entry: { slug: string; badge: string }) => `${entry.badge}/${entry.slug}`,
}

export function reportsRoot(): string {
  return path.join(process.cwd(), 'reports')
}

export function reportIndexPath(root = reportsRoot()): string {
  return indexPath(root)
}

export function listReportCompanies(root = reportsRoot()): string[] {
  return listDirectories(root)
}

export function listReports(company: string, root = reportsRoot()): ReportSummary[] {
  return listArticles(root, company)
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
    content: readArticleBody(root, company, slug),
  }
}

export function parseReportIndex(root = reportsRoot()): ReportIndexTopic[] {
  return parseIndex(root, INDEX_SPEC).map((topic) => ({
    title: topic.title,
    entries: topic.entries.map((entry) => ({
      slug: entry.slug,
      company: entry.badge,
      summary: entry.summary,
    })),
  }))
}

/** 已发布文章与索引的一致性:每篇已发布文章必须在索引里恰有一行,且**公司列等于目录名**;每个方向的已发布行必须按首发日从新到旧,同日按 slug 升序 */
export function checkReportIndex(root = reportsRoot()): string[] {
  const registered = new Set<string>()
  const bySlug = new Map<string, string[]>()
  const topics = parseReportIndex(root)
  for (const topic of topics) {
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

  const cache = new Map<string, Map<string, ReportSummary>>()
  const publishedOf = (company: string, slug: string): ReportSummary | undefined => {
    if (!cache.has(company)) {
      cache.set(company, new Map(listReports(company, root).map((r) => [r.slug, r])))
    }
    return cache.get(company)?.get(slug)
  }
  errors.push(
    ...checkTopicOrder(
      topics.map((topic) => ({
        title: topic.title,
        entries: topic.entries.map((entry) => ({
          slug: entry.slug,
          badge: entry.company,
          summary: entry.summary,
        })),
      })),
      (entry) => publishedOf(entry.badge, entry.slug),
      (entry) => `${entry.badge}/${entry.slug}`,
    ),
  )
  return errors
}

/**
 * 页面数据:按索引顺序返回各方向及其已发布文章;未解读条目不出现,没有已发布文章的方向也不出现。
 * 索引与文件不一致、或某方向已发布行不是从新到旧时直接抛错,和 release-date 契约一样让问题带文件名暴露。
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

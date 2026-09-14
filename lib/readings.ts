/**
 * 日常研读库:关注的论文、网页与视频字幕。
 *
 * 与报告解读的唯一结构差别是**目录键**:这里的目录是**方向**(等于 readings/index.md 的
 * `## ` 标题,逐字一致),第二列的机构只是卡片条幅用的元数据,不建目录。
 * 原件放在 readings/_src/<方向>/,以 _ 开头,页面扫描自动跳过。
 * 共用原语见 lib/library.ts;规则见 docs/10-材料解读流程.md。
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

export type ReadingSummary = ArticleSummary

export interface ReadingArticle extends ReadingSummary {
  /** 方向,同时是目录名 */
  topic: string
  /** 机构,卡片条幅;索引未登记时为 null */
  org: string | null
  content: string
}

/** readings/index.md 的一行:材料 = 文件名去 .md,机构 = 纯元数据 */
export interface ReadingIndexEntry {
  slug: string
  org: string
  summary: string
}

export interface ReadingIndexTopic {
  title: string
  entries: ReadingIndexEntry[]
}

export interface ReadingCard extends ReadingSummary, ReadingIndexEntry {}

export interface ReadingTopic {
  title: string
  readings: ReadingCard[]
}

const INDEX_SPEC = {
  slugHeader: '材料',
  badgeHeader: '机构',
  label: 'readings',
  // 材料名全库唯一:同一篇不该在两个方向下各登记一次
  dedupKey: (entry: { slug: string }) => entry.slug,
}

export function readingsRoot(): string {
  return path.join(process.cwd(), 'readings')
}

export function readingIndexPath(root = readingsRoot()): string {
  return indexPath(root)
}

/** 方向目录。`_src`(原件)与其他下划线开头的目录被 isVisible 挡在外面 */
export function listReadingTopicDirs(root = readingsRoot()): string[] {
  return listDirectories(root)
}

export function listReadings(topic: string, root = readingsRoot()): ReadingSummary[] {
  return listArticles(root, topic)
}

export function isValidReading(topic: string, slug: string, root = readingsRoot()): boolean {
  return listReadings(topic, root).some((reading) => reading.slug === slug)
}

export function getReading(
  topic: string,
  slug: string,
  root = readingsRoot(),
): ReadingArticle | null {
  const summary = listReadings(topic, root).find((reading) => reading.slug === slug)
  if (!summary) return null
  return {
    ...summary,
    topic,
    org: findReadingOrg(topic, slug, root),
    content: readArticleBody(root, topic, slug),
  }
}

export function parseReadingIndex(root = readingsRoot()): ReadingIndexTopic[] {
  return parseIndex(root, INDEX_SPEC).map((topic) => ({
    title: topic.title,
    entries: topic.entries.map((entry) => ({
      slug: entry.slug,
      org: entry.badge,
      summary: entry.summary,
    })),
  }))
}

/**
 * 一致性检查。比报告解读多一条:**方向目录与索引的 `## ` 必须双向一致**——
 * 有目录就必须有对应标题,反过来不要求(方向可以先建标题、还没有文章)。
 */
export function checkReadingIndex(root = readingsRoot()): string[] {
  const topics = parseReadingIndex(root)
  const titles = new Set(topics.map((topic) => topic.title))
  const errors: string[] = []

  for (const dir of listReadingTopicDirs(root)) {
    if (!titles.has(dir)) {
      errors.push(`readings/${dir}/ 是目录,但 index.md 里没有同名的 \`## ${dir}\``)
    }
  }

  const registered = new Map<string, string>()
  for (const topic of topics) {
    for (const entry of topic.entries) registered.set(entry.slug, topic.title)
  }
  for (const dir of listReadingTopicDirs(root)) {
    for (const reading of listReadings(dir, root)) {
      const at = registered.get(reading.slug)
      if (at === dir) continue
      errors.push(
        at
          ? `${dir}/${reading.slug}.md 在 index.md 登记在方向「${at}」下,与目录不一致`
          : `${dir}/${reading.slug}.md 未在 index.md 登记`,
      )
    }
  }

  const cache = new Map<string, Map<string, ReadingSummary>>()
  const publishedOf = (slug: string, topicTitle: string): ReadingSummary | undefined => {
    if (!cache.has(topicTitle)) {
      cache.set(topicTitle, new Map(listReadings(topicTitle, root).map((r) => [r.slug, r])))
    }
    return cache.get(topicTitle)?.get(slug)
  }
  errors.push(
    ...checkTopicOrder(
      topics.map((topic) => ({
        title: topic.title,
        entries: topic.entries.map((entry) => ({
          slug: entry.slug,
          badge: entry.org,
          summary: entry.summary,
        })),
      })),
      (entry, topicTitle) => publishedOf(entry.slug, topicTitle),
    ),
  )
  return errors
}

export function listReadingTopics(root = readingsRoot()): ReadingTopic[] {
  const errors = checkReadingIndex(root)
  if (errors.length > 0) throw new Error(errors.join('\n'))
  const cache = new Map<string, ReadingSummary[]>()
  const summariesOf = (topic: string): ReadingSummary[] => {
    if (!cache.has(topic)) cache.set(topic, listReadings(topic, root))
    return cache.get(topic) ?? []
  }
  return parseReadingIndex(root)
    .map((topic) => ({
      title: topic.title,
      readings: topic.entries.flatMap((entry): ReadingCard[] => {
        const summary = summariesOf(topic.title).find((r) => r.slug === entry.slug)
        return summary ? [{ ...summary, ...entry }] : []
      }),
    }))
    .filter((topic) => topic.readings.length > 0)
}

/** 某篇材料的机构;索引不存在或未登记时返回 null,不让文章页因为索引缺失而打不开 */
export function findReadingOrg(
  topic: string,
  slug: string,
  root = readingsRoot(),
): string | null {
  if (!fs.existsSync(readingIndexPath(root))) return null
  for (const entry of parseReadingIndex(root)) {
    if (entry.title !== topic) continue
    const hit = entry.entries.find((e) => e.slug === slug)
    if (hit) return hit.org
  }
  return null
}

export function countReadings(root = readingsRoot()): number {
  return listReadingTopicDirs(root).reduce(
    (total, topic) => total + listReadings(topic, root).length,
    0,
  )
}

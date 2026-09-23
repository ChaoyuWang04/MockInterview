/**
 * 两个解读库索引页的数据:把索引卡片整理成「按方向」「按时间」两种分组,附上票数与已读标记。
 * 页面组件只负责切换与勾选,见 components/LibraryIndex.tsx。
 */
import type { LibraryCard, LibraryGroup } from '@/components/LibraryIndex'

import { votesByArticle } from './alphaxiv'
import { listReadMarks } from './read-marks'
import { listReadingTopics, readingsRoot } from './readings'
import { listReportTopics, reportsRoot } from './reports'
import { groupByAge, todayInShanghai } from './timeline'

export interface LibraryViews {
  total: number
  byTopic: LibraryGroup[]
  byTime: LibraryGroup[]
  initialRead: string[]
}

function build(byTopic: LibraryGroup[], root: string): LibraryViews {
  const cards = byTopic.flatMap((group) => group.cards)
  const published = new Set(cards.map((card) => card.key))
  const byTime = groupByAge(cards, todayInShanghai()).map((group) => ({
    title: group.label,
    cards: group.items,
  }))
  return {
    total: cards.length,
    byTopic,
    byTime,
    // 已下线或改名的文章留在已读文件里无害,但不必传给页面
    initialRead: [...listReadMarks(root)].filter((key) => published.has(key)),
  }
}

const href = (library: string, dir: string, slug: string) =>
  `/${library}/${encodeURIComponent(dir)}/${encodeURIComponent(slug)}`

export function reportViews(): LibraryViews {
  const votes = votesByArticle('reports')
  const byTopic = listReportTopics().map((topic) => ({
    title: topic.title,
    cards: topic.reports.map((report): LibraryCard => {
      const key = `${report.company}/${report.slug}`
      return {
        key,
        href: href('reports', report.company, report.slug),
        title: report.title,
        badge: report.company,
        topic: topic.title,
        releaseDate: report.releaseDate,
        votes: votes.get(key) ?? null,
      }
    }),
  }))
  return build(byTopic, reportsRoot())
}

export function readingViews(): LibraryViews {
  const votes = votesByArticle('readings')
  const byTopic = listReadingTopics().map((topic) => ({
    title: topic.title,
    cards: topic.readings.map((reading): LibraryCard => {
      const key = `${topic.title}/${reading.slug}`
      return {
        key,
        href: href('readings', topic.title, reading.slug),
        title: reading.title,
        badge: reading.org,
        topic: topic.title,
        releaseDate: reading.releaseDate,
        votes: votes.get(key) ?? null,
      }
    }),
  }))
  return build(byTopic, readingsRoot())
}

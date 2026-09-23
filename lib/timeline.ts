/**
 * 两个解读库「按时间」视图:按首发日距今分五段,段内按达标当天的 alphaXiv 票数从高到低。
 *
 * 票数来自 readings/_alphaxiv-扫描记录.md;没有票数的(热榜接入之前的老文章、手动收的材料)
 * 排在同段有票数的后面,再按首发日从新到旧、同日按键名升序。这个视图是为了看新文章,
 * 不为老文章的顺序做额外照顾。
 */

export const TIMELINE_BUCKETS = [
  { label: '三天内', maxDays: 3 },
  { label: '七天内', maxDays: 7 },
  { label: '一个月内', maxDays: 30 },
  { label: '六个月内', maxDays: 182 },
  { label: '六个月以外', maxDays: Infinity },
] as const

export interface TimelineItem {
  key: string
  releaseDate: string
  votes: number | null
}

export interface TimelineGroup<T extends TimelineItem> {
  label: string
  items: T[]
}

/** 两个 YYYY-MM-DD 之间相差的天数;首发日在未来(时区差)按 0 天算 */
export function daysBetween(from: string, to: string): number {
  const ms = Date.parse(`${to}T00:00:00Z`) - Date.parse(`${from}T00:00:00Z`)
  return Math.max(0, Math.round(ms / 86_400_000))
}

/** 本地日历上的「今天」,按北京时间取,避免服务器在 UTC 跨日时把昨天当今天 */
export function todayInShanghai(now = new Date()): string {
  return new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Shanghai' }).format(now)
}

function compare(a: TimelineItem, b: TimelineItem): number {
  if (a.votes !== null || b.votes !== null) {
    if (a.votes === null) return 1
    if (b.votes === null) return -1
    if (a.votes !== b.votes) return b.votes - a.votes
  }
  const dateOrder = b.releaseDate.localeCompare(a.releaseDate)
  if (dateOrder !== 0) return dateOrder
  if (a.key === b.key) return 0
  return a.key < b.key ? -1 : 1
}

/** 空段不返回 */
export function groupByAge<T extends TimelineItem>(items: T[], today: string): TimelineGroup<T>[] {
  const groups = TIMELINE_BUCKETS.map((bucket) => ({ ...bucket, items: [] as T[] }))
  for (const item of items) {
    const age = daysBetween(item.releaseDate, today)
    const group = groups.find((bucket) => age <= bucket.maxDays)
    group?.items.push(item)
  }
  return groups
    .filter((group) => group.items.length > 0)
    .map((group) => ({ label: group.label, items: [...group.items].sort(compare) }))
}

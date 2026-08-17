// 纯函数模块:不碰 fs,服务端与客户端组件都能引用
import type { LcProblem } from './leetcode'
import type { Question } from './types'

const DIFFICULTY_ORDER: Record<string, number> = { 简单: 0, 中等: 1, 困难: 2 }

/**
 * 统一的排序规则(LeetCode 清单与题库列表共用):
 * 高频项整体排在前面,两个分桶内部都按 简单 → 中等 → 困难。
 * 全是高频或全不是高频时,自然退化为纯难度排序。
 * sort 是稳定排序,同桶同难度保持原始顺序。
 */
export function sortByFreqThenDifficulty<T>(
  items: T[],
  isHighFreq: (item: T) => boolean,
  difficultyOf: (item: T) => string | undefined,
): T[] {
  return [...items].sort((a, b) => {
    const ha = isHighFreq(a) ? 0 : 1
    const hb = isHighFreq(b) ? 0 : 1
    if (ha !== hb) return ha - hb
    return (DIFFICULTY_ORDER[difficultyOf(a) ?? ''] ?? 9) - (DIFFICULTY_ORDER[difficultyOf(b) ?? ''] ?? 9)
  })
}

/** LeetCode 清单:高频集合来自 high-freq.md */
export function sortProblems(problems: LcProblem[], highFreq: Set<string>): LcProblem[] {
  return sortByFreqThenDifficulty(problems, (p) => highFreq.has(p.slug), (p) => p.difficulty)
}

/** 题库:高频标记来自每题 frontmatter 的 highfreq */
export function sortQuestions(questions: Question[]): Question[] {
  return sortByFreqThenDifficulty(questions, (q) => q.meta.highfreq, (q) => q.meta.difficulty)
}

/** 按 topic 第一段分组,保持传入顺序;列表视图与右侧导航共用 */
export function groupByTopic<T>(items: T[], topicOf: (item: T) => string | undefined) {
  const groups: { name: string; items: { item: T; index: number }[] }[] = []
  const seen = new Map<string, number>()
  items.forEach((item, index) => {
    const name = topicOf(item)?.split('/')[0]?.trim() || '未分层'
    let gi = seen.get(name)
    if (gi === undefined) {
      gi = groups.length
      seen.set(name, gi)
      groups.push({ name, items: [] })
    }
    groups[gi].items.push({ item, index })
  })
  return groups
}

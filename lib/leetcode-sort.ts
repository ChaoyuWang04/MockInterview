// 纯函数模块:不碰 fs,服务端与客户端组件都能引用
import type { LcProblem } from './leetcode'

const DIFFICULTY_ORDER: Record<string, number> = { 简单: 0, 中等: 1, 困难: 2 }

/**
 * 排序规则:高频题整体排在前面,两个分桶内部都按 简单 → 中等 → 困难。
 * 全是高频或全不是高频时,自然退化为纯难度排序。
 * sort 是稳定排序,同桶同难度保持清单原始顺序。
 */
export function sortProblems(problems: LcProblem[], highFreq: Set<string>): LcProblem[] {
  return [...problems].sort((a, b) => {
    const ha = highFreq.has(a.slug) ? 0 : 1
    const hb = highFreq.has(b.slug) ? 0 : 1
    if (ha !== hb) return ha - hb
    return (DIFFICULTY_ORDER[a.difficulty] ?? 9) - (DIFFICULTY_ORDER[b.difficulty] ?? 9)
  })
}
